"""``serve`` — the local web application (T015, FR-013, FR-078).

Spec 005 adds exactly one thing to the CLI: a subcommand that starts the
backend.  Two properties matter, and neither is visible from the function in
isolation.

The subparser has to exist in ``argparse``.  A command function no argument can
reach is not a command, and only the real entry point can prove it — so one
test spawns the CLI and asks it for help, which is the same thing a user does.

The backend has to start on the bind the *config* names.  FR-086 keeps the host
and port out of the source, so the way to prove that is to watch what is handed
to the server: the test replaces ``uvicorn.run`` (which otherwise blocks for the
lifetime of the process) and inspects the call.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from manga_text_seg import cli

ROOT = Path(__file__).resolve().parents[2]


def _invoke_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the installed entry point, the way a user does (FR-078)."""
    environment = os.environ.copy()
    environment.setdefault("MANGA_TEXT_SEG_INTEGRATION", "1")
    return subprocess.run(
        ["uv", "run", "--locked", "--python", "3.12", "manga-text-seg", *arguments],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def served(web_bundle, monkeypatch):
    """``cmd_serve`` with the server replaced by a recorder.

    Patched by name rather than through ``cli.uvicorn``: the import is lazy, so
    the attribute does not exist until the command runs, and a test that forced
    the import itself would stop proving the import is inside the command.
    """
    built = web_bundle()
    calls: list[tuple[object, dict]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kwargs: calls.append((app, kwargs)))

    def _serve(config_path: Path | str) -> int:
        return cli.cmd_serve(str(config_path))

    return built, _serve, calls


def test_serve_is_reachable_through_the_command_line():
    """The subparser exists and dispatches (FR-078).

    Both spellings are checked because they fail differently: a missing
    subparser shows up in the top-level listing, while a subparser that is
    registered but wired to the wrong function shows up only in its own help.
    """
    top = _invoke_cli("--help")
    assert top.returncode == 0, top.stderr
    assert "serve" in top.stdout

    sub = _invoke_cli("serve", "--help")
    assert sub.returncode == 0, sub.stderr
    assert "usage: manga-text-seg serve" in sub.stdout


def test_serve_starts_the_backend_on_the_configured_bind(served):
    """Host and port come from the config, not from the source (FR-086)."""
    built, serve, calls = served
    config = built["web_config"]

    assert serve(built["web_config_path"]) == 0

    assert len(calls) == 1
    app, kwargs = calls[0]
    assert kwargs == {"host": config.server.host, "port": config.server.port}


def test_serve_hands_the_server_the_app_for_that_config(served):
    """The app served is built from the same config, not a default one.

    Compared by value, not identity: ``serve`` loads the config from the path
    it was given, so it is necessarily a second instance of the same frozen
    dataclass. A default config would still differ here, which is the point.
    """
    built, serve, calls = served

    serve(built["web_config_path"])

    app, _ = calls[0]
    assert app.state.config == built["web_config"]


def test_serve_serves_the_configured_static_root(served):
    """End to end: what ``serve`` starts answers with the bundle's own bytes."""
    from fastapi.testclient import TestClient

    built, serve, calls = served
    serve(built["web_config_path"])
    app, kwargs = calls[0]

    client = TestClient(app, base_url=f"http://{kwargs['host']}:{kwargs['port']}")
    response = client.get("/")

    assert response.status_code == 200
    assert "fixture-bundle" in response.text


def test_serve_refuses_a_missing_config(served, tmp_path):
    """A config that is not there is an error, never a silent default."""
    _, serve, calls = served

    with pytest.raises(FileNotFoundError):
        serve(tmp_path / "absent.json")

    assert calls == []


def test_serve_refuses_a_config_that_is_not_loopback(served, tmp_path):
    """FR-013c is enforced where the bind is chosen, before anything starts."""
    built, serve, calls = served
    data = json.loads(Path(built["web_config_path"]).read_text(encoding="utf-8"))
    data["server"]["host"] = "0.0.0.0"
    broken = tmp_path / "not-loopback.json"
    broken.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(cli.ConfigError):
        serve(broken)

    assert calls == []


def test_the_default_web_config_is_the_shipped_one():
    """``serve`` with no ``--config`` reads the repo's own config (FR-086)."""
    assert (ROOT / cli.DEFAULT_WEB_CONFIG).is_file()
