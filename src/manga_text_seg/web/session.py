"""The session/run store: FR-022, FR-022a and FR-065 in one place.

The service is single-process and in-memory: a session is issued on the first
request and lives until the process exits (FR-022a). What outlives the process
is the *run* — a directory under the configured run root — because that is the
deliverable (FR-063, FR-066).

Nothing here is persisted. ``run_id`` is the only handle a later request needs,
so a restarted process can still address an existing run through
:meth:`SessionStore.create_run` with an explicit id; the session that produced
it simply no longer exists, which is why the run is owned and checked rather
than assumed (FR-065).
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Session ids and run ids are both ``<prefix>-<hex>``. The prefix is for a
#: human reading ``outputs/end-to-end/``; nothing parses it.
_RUN_PREFIX = "run"
_SESSION_PREFIX = "sess"
_HEX_CHARS = 12


class RunExists(RuntimeError):
    """A run_id that already exists on disk was reused without overwrite (FR-065)."""


@dataclass
class Session:
    """One app session and the run it owns.

    ``state`` is the FR-022 app state (manga, page, method, runs, language) and
    is deliberately a plain dict: the routes that write it land in T024, and a
    typed model here would have to be widened for every key they add.
    """

    session_id: str
    run_id: str
    run_dir: Path
    state: dict[str, Any] = field(default_factory=dict)


class SessionStore:
    """Issues sessions, resumes them by id, and hands out their runs.

    ``run_root`` is the configured ``outputs/end-to-end`` equivalent (FR-086) —
    the store never derives it.
    """

    def __init__(self, run_root: Path) -> None:
        self.run_root = Path(run_root)
        self._sessions: dict[str, Session] = {}
        # ponytail: one lock for the whole store. Sessions are created once per
        # browser tab and touched a handful of times per request; per-session
        # locks would be more code guarding a dict write.
        self._lock = threading.Lock()

    # --- sessions ----------------------------------------------------------

    def resume_or_create(self, session_id: str | None) -> Session:
        """Return the session ``session_id`` names, or issue a new one (FR-022a).

        An id this store never issued is *not* resumed. Adopting a caller's
        string would put attacker-chosen text into the store, where it could
        later collide with a genuinely issued id; issuing a fresh one keeps the
        id space entirely server-side.
        """
        if session_id is not None:
            existing = self._sessions.get(session_id)
            if existing is not None:
                return existing

        with self._lock:
            # Re-check under the lock: two concurrent first requests with the
            # same id must not each mint a session.
            if session_id is not None and session_id in self._sessions:
                return self._sessions[session_id]

            run_id, run_dir = self.create_run()
            session = Session(
                session_id=f"{_SESSION_PREFIX}-{uuid.uuid4().hex[:_HEX_CHARS]}",
                run_id=run_id,
                run_dir=run_dir,
            )
            self._sessions[session.session_id] = session
            return session

    def set_state(self, session_id: str, **values: Any) -> None:
        """Merge ``values`` into a session's FR-022 app state.

        An unknown session is ignored rather than raising: a request carrying a
        stale id must not fail the whole call, and the next request will simply
        be issued a new session.
        """
        session = self._sessions.get(session_id)
        if session is not None:
            session.state.update(values)

    def owns(self, session_id: str | None, run_id: str | None) -> bool:
        """Whether ``session_id`` is the session that owns ``run_id`` (FR-065)."""
        session = self._sessions.get(session_id) if session_id else None
        return session is not None and run_id is not None and session.run_id == run_id

    # --- runs --------------------------------------------------------------

    def create_run(
        self, run_id: str | None = None, *, overwrite: bool = False
    ) -> tuple[str, Path]:
        """Create the run directory and return ``(run_id, run_dir)``.

        FR-065: an id that already exists is refused unless ``overwrite`` was
        given explicitly. The refusal happens *before* anything is created, so a
        refused call cannot have emptied the run it collided with.

        Only the directory is made here. Filling it — ``config.json`` and the
        FR-063 layout — is the run materialization in T017.
        """
        if run_id is None:
            # An auto-generated id must be free *on disk*, not merely unused in
            # this process: runs outlive the process (FR-066). A collision here
            # is regenerated, never refused — the caller asked for a new run,
            # not for this particular id.
            candidate = f"{_RUN_PREFIX}-{uuid.uuid4().hex[:_HEX_CHARS]}"
            while (self.run_root / candidate).exists():
                candidate = f"{_RUN_PREFIX}-{uuid.uuid4().hex[:_HEX_CHARS]}"
        else:
            candidate = run_id
            if not overwrite and (self.run_root / candidate).exists():
                raise RunExists(
                    f"run '{candidate}' already exists; pass overwrite to reuse "
                    f"it (FR-065)"
                )

        run_dir = self.run_root / candidate
        run_dir.mkdir(parents=True, exist_ok=True)
        return candidate, run_dir
