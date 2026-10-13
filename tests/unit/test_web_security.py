"""Unit tests for the security guards — SC-015's four refusal cases (T009).

SC-015 names four refusals the service must make:

1. a non-loopback ``Host`` header,
2. a foreign ``Origin`` header,
3. a path escaping an allow-listed artifact root,
4. any path under a forbidden root (``data/no-need-to-read/``, FR-008).

These are unit tests on the guard functions, so they run without an app; the
middleware that calls them is wired in T014.
"""

import dataclasses

import pytest

from manga_text_seg.web import security


@pytest.fixture()
def guard_config(web_bundle):
    """The fixture's ``WebConfig``, built without an app.

    ``web_bundle`` is requested as a factory rather than built here so the test
    controls which variant it asks for.
    """
    return web_bundle()["web_config"]


# --- Case 1: non-loopback Host ----------------------------------------------


def test_loopback_host_is_accepted(guard_config):
    server = guard_config.server

    assert security.host_allowed(guard_config, f"{server.host}:{server.port}")
    assert security.host_allowed(guard_config, "localhost")


def test_non_loopback_host_is_refused(guard_config):
    """Case 1: a Host that is not the configured loopback name is refused."""
    for host in ("evil.example:8765", "0.0.0.0:8765", "192.168.1.10:8765", ""):
        assert not security.host_allowed(guard_config, host), host


def test_loopback_prefix_is_not_a_match(guard_config):
    """A lookalike host must not pass by prefix — `127.0.0.1.evil.example`."""
    assert not security.host_allowed(guard_config, "127.0.0.1.evil.example:8765")


# --- Case 2: foreign Origin -------------------------------------------------


def test_loopback_origin_is_accepted(guard_config):
    server = guard_config.server

    assert security.origin_allowed(guard_config, f"http://{server.host}:{server.port}")


def test_absent_origin_is_accepted(guard_config):
    """A same-origin navigation may send no Origin at all."""
    assert security.origin_allowed(guard_config, None)


def test_foreign_origin_is_refused(guard_config):
    """Case 2: an Origin from another site is refused."""
    for origin in ("http://evil.example", "https://127.0.0.1:8765", "null"):
        assert not security.origin_allowed(guard_config, origin), origin


# --- Cases 3 and 4: artifact reference resolution ---------------------------


def test_allow_listed_reference_resolves_inside_its_root(guard_config):
    reference = "inpainting/run-a/unetpp_efficientnetv2/telea/WebManga/001.png"

    resolved = security.resolve_artifact(guard_config, reference)

    assert resolved == guard_config.artifact_roots["inpainting"] / (
        "run-a/unetpp_efficientnetv2/telea/WebManga/001.png"
    )


def test_unknown_root_name_is_refused(guard_config):
    with pytest.raises(security.ArtifactRefused) as excinfo:
        security.resolve_artifact(guard_config, "not_a_root/page.png")

    assert excinfo.value.reason == "unknown_root"


def test_traversal_out_of_the_root_is_refused(guard_config):
    """Case 3: a reference that climbs out of its root is refused."""
    for reference in (
        "inpainting/../../no-need-to-read/secret.png",
        "inpainting/run-a/../../../etc/passwd",
        "../inpainting/page.png",
        "inpainting/run-a/..",
    ):
        with pytest.raises(security.ArtifactRefused) as excinfo:
            security.resolve_artifact(guard_config, reference)
        assert excinfo.value.reason == "traversal", reference


def test_absolute_reference_is_refused(guard_config):
    """A caller-supplied filesystem path is never a valid reference."""
    for reference in ("C:/Windows/win.ini", "/etc/passwd", "//server/share/x.png"):
        with pytest.raises(security.ArtifactRefused) as excinfo:
            security.resolve_artifact(guard_config, reference)
        assert excinfo.value.reason == "absolute", reference


def test_reference_naming_no_artifact_is_refused(guard_config):
    """A bare root name selects nothing — there is no file to serve."""
    for reference in ("inpainting/", "inpainting"):
        with pytest.raises(security.ArtifactRefused) as excinfo:
            security.resolve_artifact(guard_config, reference)
        assert excinfo.value.reason == "empty", reference


def test_forbidden_root_is_refused_even_when_reachable(guard_config):
    """Case 4 (FR-008): a path under a forbidden root is refused.

    ``load_web_config`` already refuses a config whose forbidden root sits
    inside an allow-listed root, so this state is unreachable through the
    loader. It is constructed directly here because the guard is the second
    line of defence and must refuse on its own.
    """
    forbidden = guard_config.artifact_roots["inpainting"] / "no-need-to-read"
    config = dataclasses.replace(
        guard_config, forbidden_roots=(forbidden,),
    )

    with pytest.raises(security.ArtifactRefused) as excinfo:
        security.resolve_artifact(config, "inpainting/no-need-to-read/secret.png")

    assert excinfo.value.reason == "forbidden"


def test_forbidden_root_check_covers_nested_paths(guard_config):
    """The forbidden refusal is by subtree, not by exact match."""
    forbidden = guard_config.artifact_roots["deliverables"] / "no-need-to-read"
    config = dataclasses.replace(guard_config, forbidden_roots=(forbidden,))

    with pytest.raises(security.ArtifactRefused) as excinfo:
        security.resolve_artifact(
            config, "deliverables/no-need-to-read/deep/nested/file.png"
        )

    assert excinfo.value.reason == "forbidden"


# --- The bind assertion -----------------------------------------------------


def test_loopback_bind_assertion_passes_for_the_fixture(guard_config):
    security.assert_loopback_bind(guard_config)


def test_loopback_bind_assertion_refuses_a_public_bind(guard_config):
    """FR-013c: the process refuses to start on a non-loopback host."""
    config = dataclasses.replace(
        guard_config,
        server=dataclasses.replace(guard_config.server, host="0.0.0.0"),
    )

    with pytest.raises(security.NonLoopbackBind):
        security.assert_loopback_bind(config)
