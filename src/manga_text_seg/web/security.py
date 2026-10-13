"""The FR-013c/FR-013d/FR-008 guards: bind, origin, and artifact resolution.

Two separate defences live here:

* :func:`assert_loopback_bind` runs once at startup and refuses to serve at all
  on a non-loopback address (FR-013c).
* :func:`host_allowed`, :func:`origin_allowed` and :func:`resolve_artifact` run
  per request.

``resolve_artifact`` is the only way a caller-supplied reference becomes a
filesystem path. It takes a *run-relative reference* (``<root name>/<tail>``),
never a caller-supplied absolute path, and returns a path only when it is
inside an allow-listed root and outside every forbidden root (FR-013d, FR-008).
"""

from pathlib import Path, PurePosixPath, PureWindowsPath
from urllib.parse import urlsplit

from ..config import LOOPBACK_HOSTS, WebConfig


class NonLoopbackBind(RuntimeError):
    """The service was asked to bind somewhere other than loopback (FR-013c)."""


class ArtifactRefused(Exception):
    """A reference did not resolve inside an allow-listed artifact root.

    ``reason`` is a short machine-readable slug — ``unknown_root``,
    ``absolute``, ``empty``, ``traversal`` or ``forbidden`` — so the route layer
    can report which refusal happened without parsing the message.
    """

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def assert_loopback_bind(config: WebConfig) -> None:
    """Refuse to start unless the configured bind is a loopback name (FR-013c).

    ``load_web_config`` already rejects a non-loopback host, so reaching the
    raise means the config was built by something other than the loader — which
    is exactly when a second check earns its keep.
    """
    host = config.server.host
    if host not in LOOPBACK_HOSTS:
        raise NonLoopbackBind(
            f"refusing to bind to non-loopback host '{host}' (FR-013c); "
            f"allowed: {sorted(LOOPBACK_HOSTS)}"
        )


def _split_host(value: str) -> tuple[str, int | None]:
    """Split a ``Host`` header value into a lowercase name and optional port."""
    text = (value or "").strip()
    if text.startswith("["):  # IPv6 literal, e.g. [::1]:8765
        end = text.find("]")
        if end == -1:
            return "", None
        name, rest = text[1:end], text[end + 1 :]
    elif text.count(":") == 1:
        name, _, rest = text.partition(":")
    else:
        name, rest = text, ""
    port = rest[1:] if rest.startswith(":") else rest
    return name.strip().lower(), int(port) if port.isdigit() else None


def host_allowed(config: WebConfig, host_header: str | None) -> bool:
    """Whether a ``Host`` header names this service on loopback (FR-013c).

    The name must match a loopback name *exactly* — ``127.0.0.1.evil.example``
    is a different host, not a prefix match — and a stated port must be the
    configured one.
    """
    name, port = _split_host(host_header or "")
    if name not in LOOPBACK_HOSTS:
        return False
    return port is None or port == config.server.port


def origin_allowed(config: WebConfig, origin: str | None) -> bool:
    """Whether an ``Origin`` header is this same-origin loopback service.

    An absent ``Origin`` is allowed: a same-origin navigation may omit it, and
    the ``Host`` guard still applies. Anything else must be plain ``http`` on a
    loopback name at the configured port — ``https`` there would mean the
    request came from some other page, not from ours.
    """
    if origin is None:
        return True

    parts = urlsplit(origin.strip())
    if parts.scheme != "http" or not parts.hostname:
        return False
    if parts.hostname.lower() not in LOOPBACK_HOSTS:
        return False
    return parts.port is None or parts.port == config.server.port


def resolve_artifact(config: WebConfig, reference: str) -> Path:
    """Resolve a run-relative artifact reference to an absolute path.

    ``reference`` is ``<root name>/<tail>``, where the root name is one of
    ``config.artifact_roots``. The returned path is guaranteed to be inside
    that root and outside every forbidden root; anything else raises
    :class:`ArtifactRefused`.
    """
    text = (reference or "").strip().replace("\\", "/")

    # A caller-supplied filesystem path is never a valid reference (FR-013d).
    # Both flavours are checked: on Windows "/etc/passwd" has no drive and so
    # is not "absolute" there, but it is still not a reference we accept.
    if PurePosixPath(text).is_absolute() or PureWindowsPath(text).is_absolute():
        raise ArtifactRefused("absolute", f"'{reference}' is not a relative reference")

    # Lexical, not realpath, and checked before the root lookup: any '..' is
    # refused outright rather than resolved, so a symlink or a climb out of the
    # root can never widen what a reference reaches.
    if ".." in PurePosixPath(text).parts:
        raise ArtifactRefused(
            "traversal", f"'{reference}' escapes its artifact root (FR-013d)"
        )

    root_name, _, tail = text.partition("/")
    root = config.artifact_roots.get(root_name)
    if root is None:
        raise ArtifactRefused(
            "unknown_root", f"'{root_name}' is not an allow-listed artifact root"
        )
    if not tail:
        raise ArtifactRefused("empty", f"'{reference}' names no artifact")

    resolved = root / tail
    for forbidden in config.forbidden_roots:
        if resolved.is_relative_to(forbidden):
            raise ArtifactRefused(
                "forbidden", f"'{reference}' is under a forbidden root (FR-008)"
            )
    return resolved
