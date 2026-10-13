"""Unit tests for the session/run store — FR-022a reload, FR-065 ownership (T011).

FR-022a: the backend issues the session id on the first request and the id
survives a page reload, so a reload does not abandon the session's run.
FR-065: the app auto-creates exactly one new run per app session, two sessions
never share a run, and a run_id that already exists on disk is refused unless
an explicit overwrite was asked for.
FR-022: the app's state (manga, page, method, runs, language) lives in the
backend and survives interactions within the session.

These are unit tests on the store, so no app is built; the routes that call it
land in T024.
"""

import dataclasses

import pytest


@pytest.fixture()
def store(web_bundle):
    """A ``SessionStore`` rooted at the fixture's configured run root.

    ``web_bundle`` is requested as a factory rather than built here so a test
    can ask for the variant it needs.
    """
    from manga_text_seg.web.session import SessionStore

    return SessionStore(web_bundle()["web_config"].run_root)


# --- FR-022a: issue, then resume --------------------------------------------


def test_first_request_issues_a_session_and_a_run(store):
    session = store.resume_or_create(None)

    assert session.session_id
    assert session.run_id
    assert session.run_dir.is_dir()


def test_reload_with_the_issued_id_resumes_the_same_run(store):
    """FR-022a: the id issued on the first request is the id that resumes it."""
    first = store.resume_or_create(None)

    again = store.resume_or_create(first.session_id)

    assert again.session_id == first.session_id
    assert again.run_id == first.run_id
    assert again.run_dir == first.run_dir


def test_unknown_id_does_not_resume_another_sessions_run(store):
    """An id the store never issued is not a session, so a fresh one is issued.

    Adopting a caller-supplied identifier would put an attacker-chosen string
    into the store, where it could later collide with a genuinely issued id.
    """
    first = store.resume_or_create(None)

    stranger = store.resume_or_create("not-an-issued-id")

    assert stranger.session_id != "not-an-issued-id"
    assert stranger.run_id != first.run_id


def test_session_state_survives_across_lookups(store):
    """FR-022: the app's state lives in the backend, not in the bundle."""
    session = store.resume_or_create(None)
    store.set_state(
        session.session_id, manga="WebManga", page="001", method="unetpp_efficientnetv2"
    )

    again = store.resume_or_create(session.session_id)

    assert again.state["manga"] == "WebManga"
    assert again.state["page"] == "001"
    assert again.state["method"] == "unetpp_efficientnetv2"


def test_state_is_scoped_to_its_own_session(store):
    """FR-022: one session's selections never leak into another's."""
    first = store.resume_or_create(None)
    second = store.resume_or_create(None)
    store.set_state(first.session_id, language="vi")

    assert "language" not in store.resume_or_create(second.session_id).state


# --- FR-065: one run per session, no sharing --------------------------------


def test_two_sessions_get_two_distinct_runs(store):
    """FR-065: each app session owns its own run."""
    first = store.resume_or_create(None)
    second = store.resume_or_create(None)

    assert first.session_id != second.session_id
    assert first.run_id != second.run_id
    assert first.run_dir != second.run_dir
    assert first.run_dir.is_dir() and second.run_dir.is_dir()


def test_a_session_owns_only_its_own_run(store):
    """FR-065: run ownership is checked, not assumed from the id alone."""
    first = store.resume_or_create(None)
    second = store.resume_or_create(None)

    assert store.owns(first.session_id, first.run_id)
    assert not store.owns(first.session_id, second.run_id)
    assert not store.owns(second.session_id, first.run_id)
    assert not store.owns("not-an-issued-id", first.run_id)


def test_run_directory_lives_under_the_configured_run_root(store):
    """FR-063/FR-066: the run is a self-contained directory named by its id."""
    session = store.resume_or_create(None)

    assert session.run_dir.parent == store.run_root
    assert session.run_dir.name == session.run_id


def test_created_runs_are_unique(store):
    run_ids = {store.resume_or_create(None).run_id for _ in range(8)}

    assert len(run_ids) == 8


# --- FR-065: a colliding run_id is refused unless overwrite was asked for ----


def test_colliding_run_id_is_refused(store):
    """FR-065: an existing run_id is refused, not silently reused."""
    from manga_text_seg.web.session import RunExists

    existing = store.resume_or_create(None)

    with pytest.raises(RunExists):
        store.create_run(run_id=existing.run_id)


def test_colliding_run_id_is_allowed_with_explicit_overwrite(store):
    """FR-065: the refusal is lifted by an explicit overwrite option."""
    existing = store.resume_or_create(None)

    run_id, run_dir = store.create_run(run_id=existing.run_id, overwrite=True)

    assert run_id == existing.run_id
    assert run_dir == existing.run_dir


def test_a_refused_collision_leaves_the_existing_run_untouched(store):
    """The refusal must not have half-created or emptied the existing run."""
    from manga_text_seg.web.session import RunExists

    existing = store.resume_or_create(None)
    marker = existing.run_dir / "config.json"
    marker.write_text('{"kept": true}', encoding="utf-8")

    with pytest.raises(RunExists):
        store.create_run(run_id=existing.run_id)

    assert marker.read_text(encoding="utf-8") == '{"kept": true}'


# --- FR-086: nothing hardcodes the run root ---------------------------------


def test_an_alternative_run_root_is_honoured(web_bundle):
    """FR-086: nothing hardcodes ``outputs/end-to-end``."""
    from manga_text_seg.web.session import SessionStore

    built = web_bundle()
    elsewhere = dataclasses.replace(
        built["web_config"], run_root=built["root"] / "somewhere-else"
    )

    session = SessionStore(elsewhere.run_root).resume_or_create(None)

    assert session.run_dir.parent == elsewhere.run_root
    assert session.run_dir.is_dir()
