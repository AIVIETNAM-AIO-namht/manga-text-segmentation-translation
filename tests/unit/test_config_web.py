"""Unit tests for load_web_config() (T001, FR-086, FR-013c, FR-013d).

Mirrors test_config_inpaint.py's shape: happy-path tests load the real, committed
configs/web.json; refusal tests load a perturbed copy via write_web.
Written before the implementation exists — must fail (ImportError) until T001.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg.config import ConfigError, load_web_config

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_CONFIG = REPO_ROOT / "configs" / "web.json"

ARTIFACT_ROOT_NAMES = {
    "segmentation",
    "inpainting",
    "translation",
    "deliverables",
    "raw_images",
    "groundtruth",
}

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


@pytest.fixture()
def write_web(tmp_path: Path):
    """Write a perturbed copy of configs/web.json and return its path."""

    def _write(mutate) -> Path:
        with open(WEB_CONFIG, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        # The mutator edits doc in place; its return value is ignored, so
        # dict.update()/pop() work without being re-returned.
        mutate(doc)
        path = tmp_path / "web.json"
        path.write_text(json.dumps(doc, indent=4), encoding="utf-8")
        return path

    return _write


class TestValidWebConfigLoads:
    """The shipped configs/web.json loads and keeps every path out of code."""

    def test_loopback_bind_only(self) -> None:
        cfg = load_web_config(WEB_CONFIG, repo_root=REPO_ROOT)
        assert cfg.server.host in LOOPBACK_HOSTS
        assert 1024 <= cfg.server.port <= 65535

    def test_static_root_resolves_against_repo_root(self) -> None:
        cfg = load_web_config(WEB_CONFIG, repo_root=REPO_ROOT)
        assert cfg.static_root.is_absolute()
        assert cfg.static_root != REPO_ROOT
        assert cfg.static_root.is_relative_to(REPO_ROOT)

    def test_artifact_roots_resolve_under_repo_root(self) -> None:
        cfg = load_web_config(WEB_CONFIG, repo_root=REPO_ROOT)
        assert set(cfg.artifact_roots) == ARTIFACT_ROOT_NAMES
        for name, root in cfg.artifact_roots.items():
            assert root.is_absolute(), f"{name} did not resolve"
            assert root.is_relative_to(REPO_ROOT), name

    def test_benchmark_source_images_are_allow_listed(self) -> None:
        cfg = load_web_config(WEB_CONFIG, repo_root=REPO_ROOT)
        expected = (
            REPO_ROOT / "data" / "raw" / "Manga109s_released_2026_05_21" / "images"
        ).resolve()
        assert cfg.artifact_roots["raw_images"] == expected

    def test_forbidden_root_never_sits_inside_an_allow_listed_root(self) -> None:
        cfg = load_web_config(WEB_CONFIG, repo_root=REPO_ROOT)
        assert cfg.forbidden_roots, "FR-008 needs at least one forbidden root"
        for forbidden in cfg.forbidden_roots:
            for name, root in cfg.artifact_roots.items():
                assert not forbidden.is_relative_to(root), (
                    f"forbidden root {forbidden} would be served by '{name}'"
                )

    def test_run_root(self) -> None:
        cfg = load_web_config(WEB_CONFIG, repo_root=REPO_ROOT)
        assert cfg.run_root == (REPO_ROOT / "outputs" / "end-to-end").resolve()

    def test_session_names(self) -> None:
        cfg = load_web_config(WEB_CONFIG, repo_root=REPO_ROOT)
        assert cfg.session.header_name == "X-Manga-Session"
        assert cfg.session.cookie_name == "manga_session"

    def test_provider_block(self) -> None:
        cfg = load_web_config(WEB_CONFIG, repo_root=REPO_ROOT)
        assert cfg.provider.name
        assert cfg.provider.model
        assert cfg.provider.secret_env_var is not None
        assert cfg.provider.secret_env_var.isupper()
        assert cfg.provider.timeout_seconds > 0
        assert cfg.provider.max_attempts >= 1
        assert cfg.provider.max_backoff_seconds >= cfg.provider.initial_backoff_seconds
        assert cfg.provider.cache_root.is_absolute()


class TestWebConfigRefusals:
    """A malformed web configuration fails fast with a clear ConfigError."""

    @pytest.mark.parametrize(
        "missing_key",
        [
            "server",
            "static_root",
            "artifact_roots",
            "forbidden_roots",
            "run_root",
            "session",
            "provider",
        ],
    )
    def test_missing_top_level_key(self, write_web, missing_key: str) -> None:
        path = write_web(lambda doc: doc.pop(missing_key))
        with pytest.raises(ConfigError, match=missing_key):
            load_web_config(path)

    @pytest.mark.parametrize(
        "host", ["0.0.0.0", "::", "192.168.1.10", "example.test", ""]
    )
    def test_non_loopback_bind_is_refused(self, write_web, host: str) -> None:
        path = write_web(lambda doc: doc["server"].update(host=host))
        with pytest.raises(ConfigError, match=r"server\.host"):
            load_web_config(path)

    @pytest.mark.parametrize("port", [0, 80, 70000, "8765", True])
    def test_invalid_port_is_refused(self, write_web, port) -> None:
        path = write_web(lambda doc: doc["server"].update(port=port))
        with pytest.raises(ConfigError, match=r"server\.port"):
            load_web_config(path)

    def test_static_root_escaping_the_repo_is_refused(self, write_web) -> None:
        path = write_web(lambda doc: doc.update(static_root="../outside/"))
        with pytest.raises(ConfigError, match="static_root"):
            load_web_config(path)

    def test_artifact_root_covering_the_forbidden_root_is_refused(
        self, write_web
    ) -> None:
        path = write_web(
            lambda doc: doc["artifact_roots"].update(segmentation="data/")
        )
        with pytest.raises(ConfigError, match="no-need-to-read"):
            load_web_config(path)

    @pytest.mark.parametrize("missing_key", ["header_name", "cookie_name"])
    def test_missing_session_key(self, write_web, missing_key: str) -> None:
        path = write_web(
            lambda doc: doc["session"].pop(missing_key)
        )
        with pytest.raises(ConfigError, match=missing_key):
            load_web_config(path)

    def test_invalid_session_header_name_is_refused(self, write_web) -> None:
        path = write_web(
            lambda doc: doc["session"].update(header_name="Bad Header\n")
        )
        with pytest.raises(ConfigError, match="header_name"):
            load_web_config(path)

    def test_literal_credential_field_is_refused(self, write_web) -> None:
        path = write_web(
            lambda doc: doc["provider"].update(api_key="sk-live-not-allowed")
        )
        with pytest.raises(ConfigError, match="credential"):
            load_web_config(path)

    @pytest.mark.parametrize(
        "base_url",
        [
            "file:///tmp/secret",
            "https://user:password@example.test/v1",
            "https://[invalid/v1",
            "https://example.test:bad/v1",
        ],
    )
    def test_malformed_provider_base_url_is_refused(
        self, write_web, base_url: str
    ) -> None:
        path = write_web(lambda doc: doc["provider"].update(base_url=base_url))
        with pytest.raises(ConfigError, match="base_url"):
            load_web_config(path)

    def test_secret_env_var_must_name_an_env_var(self, write_web) -> None:
        path = write_web(
            lambda doc: doc["provider"].update(secret_env_var="literal-secret-value")
        )
        with pytest.raises(ConfigError, match="secret_env_var"):
            load_web_config(path)

    def test_missing_file_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_web_config(tmp_path / "absent.json")
