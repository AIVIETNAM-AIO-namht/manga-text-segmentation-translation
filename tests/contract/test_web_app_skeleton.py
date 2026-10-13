"""Contract tests for the app skeleton: bind, static root, health, `not_ready` (T013).

Four things the skeleton must get right before any route exists:

1. **FR-013c** — the process refuses to start on a non-loopback bind, and the
   per-request `Host`/`Origin` guard is actually wired into the app (the unit
   tests in `test_web_security.py` cover the guard functions; this file covers
   the middleware that calls them).
2. **FR-013a** — `GET /` serves the configured bundle verbatim, and replacing
   the UI is a directory swap with no code change (SC-014).
3. The `not_ready` rule from plan.md: a missing or empty static root still
   starts, and `/` returns a `not_ready` envelope naming the expected root —
   never a built-in UI.
4. The health endpoint answers.

`web_client` binds the `Host` header to the configured loopback origin, so a
refusal here is the guard firing, not a client artifact.
"""

import dataclasses

import pytest


# --- FR-013c: the bind, and the per-request guard ---------------------------


def test_app_builds_for_the_configured_loopback_bind(web_app):
    assert web_app is not None


def test_app_refuses_a_non_loopback_bind(web_bundle):
    """FR-013c: a public bind is refused at startup, not merely at request time."""
    from manga_text_seg.web.app import create_app
    from manga_text_seg.web.security import NonLoopbackBind

    config = web_bundle()["web_config"]
    public = dataclasses.replace(
        config, server=dataclasses.replace(config.server, host="0.0.0.0")
    )

    with pytest.raises(NonLoopbackBind):
        create_app(public)


def test_non_loopback_host_header_is_refused(web_client):
    """SC-015 case 1, through the wired app rather than the guard function."""
    response = web_client.get("/", headers={"Host": "evil.example:8765"})

    assert response.status_code == 400
    assert response.json()["error"]["category"] == "invalid_request"


def test_foreign_origin_is_refused(web_client):
    """SC-015 case 2, through the wired app."""
    response = web_client.get("/", headers={"Origin": "http://evil.example"})

    assert response.status_code == 400
    assert response.json()["error"]["category"] == "invalid_request"


def test_loopback_origin_is_accepted(web_client):
    server = web_client.app.state.fixture["web_config"].server

    response = web_client.get(
        "/", headers={"Origin": f"http://{server.host}:{server.port}"}
    )

    assert response.status_code == 200


# --- FR-013a / SC-014: the static bundle ------------------------------------


def test_root_serves_the_fixture_bundle(web_client):
    """FR-013a: the bundle comes from the configured static root, verbatim."""
    response = web_client.get("/")

    assert response.status_code == 200
    assert "fixture-bundle" in response.text


def test_static_root_subpaths_are_served(web_client):
    """plan.md: the directory is mounted for `/` *and its subpaths*."""
    response = web_client.get("/index.html")

    assert response.status_code == 200
    assert "fixture-bundle" in response.text


def test_replacing_the_bundle_needs_no_code_change(web_bundle):
    """SC-014: pointing the app at another directory is the whole swap."""
    from fastapi.testclient import TestClient

    from manga_text_seg.web.app import create_app

    built = web_bundle()
    config = built["web_config"]
    server = config.server
    swapped = dataclasses.replace(config, static_root=built["alt_static_root"])

    with TestClient(
        create_app(swapped), base_url=f"http://{server.host}:{server.port}"
    ) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "fixture-bundle-alt" in response.text


# --- plan.md: a missing or empty static root still starts -------------------


def _not_ready_client(config):
    from fastapi.testclient import TestClient

    from manga_text_seg.web.app import create_app

    server = config.server
    return TestClient(
        create_app(config), base_url=f"http://{server.host}:{server.port}"
    )


def test_missing_static_root_still_starts_and_reports_not_ready(web_bundle):
    """The service starts anyway; `/` says why, naming the root it expected."""
    built = web_bundle(missing_static_root=True)

    with _not_ready_client(built["web_config"]) as client:
        response = client.get("/")

    assert response.status_code == 503
    payload = response.json()
    assert payload["error"]["category"] == "not_ready"
    # Asserted against the decoded field, not `json.dumps(payload)`: the JSON
    # text escapes Windows backslashes, so a raw path is never a substring.
    assert str(built["static_root"]) in payload["error"]["message"]


def test_empty_static_root_reports_not_ready(web_bundle, tmp_path):
    """An empty directory is the same condition as a missing one."""
    built = web_bundle()
    empty = tmp_path / "empty-static"
    empty.mkdir()
    config = dataclasses.replace(built["web_config"], static_root=empty)

    with _not_ready_client(config) as client:
        response = client.get("/")

    assert response.status_code == 503
    assert response.json()["error"]["category"] == "not_ready"


def test_not_ready_never_substitutes_a_built_in_ui(web_bundle):
    """The refusal is an envelope, not a fallback page."""
    built = web_bundle(missing_static_root=True)

    with _not_ready_client(built["web_config"]) as client:
        response = client.get("/")

    assert response.headers["content-type"].startswith("application/json")
    assert "<html" not in response.text.lower()


# --- The health endpoint ----------------------------------------------------


def test_health_endpoint_answers(web_client):
    response = web_client.get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_reports_the_static_root_condition(web_bundle, tmp_path):
    """The one question the endpoint exists to answer during a demo.

    The second app points at a root that was never created rather than
    rebuilding with ``missing_static_root=True``: ``web_bundle`` is
    function-scoped on one ``tmp_path``, so the first build already made
    ``static/`` and the flag would only decline to re-create it.
    """
    built = web_bundle()
    assert _not_ready_client(built["web_config"]).get("/api/health").json() == {
        "status": "ok",
        "static_root_ready": True,
    }

    absent = dataclasses.replace(built["web_config"], static_root=tmp_path / "never")
    with _not_ready_client(absent) as client:
        assert client.get("/api/health").json()["static_root_ready"] is False
