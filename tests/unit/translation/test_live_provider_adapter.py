"""T042 live-adapter checks: request shaping and response parsing, no network."""

from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.request

import pytest

from manga_text_seg.config import (
    TranslationCacheConfig,
    TranslationProviderConfig,
    TranslationRetryConfig,
)
from manga_text_seg.translation.translate import (
    MissingProviderSecretError,
    OpenAICompatibleTranslationProvider,
    ProviderHttpError,
    TranslationProviderError,
    _provider_ssl_context,
    default_provider_adapter,
)

SECRET = "sk-live-test-value-must-not-leak"


def _live_config(**overrides) -> TranslationProviderConfig:
    """A live-provider config, spelled out here rather than read from disk.

    The adapter is vendor-neutral: it posts to whatever ``base_url`` the config
    names. Pinning this fixture to ``configs/translation.json`` would couple a
    unit test to a deploy-time choice of gateway, so the fixture carries its own.
    """
    values = dict(
        name="example-gateway",
        model="example-model",
        version=None,
        base_url="https://gateway.example.test/v1",
        secret_env_var="ANTHROPIC_AUTH_TOKEN",
        timeout_seconds=12.5,
        retry=TranslationRetryConfig(3, 1.0, 30.0, (429, 500, 502, 503, 504)),
        failure_policy="keep_ocr_text",
        deterministic=False,
        cache=TranslationCacheConfig(True, False, None),
    )
    values.update(overrides)
    return TranslationProviderConfig(**values)


@pytest.fixture()
def provider():
    return OpenAICompatibleTranslationProvider(_live_config())


def _capture(monkeypatch, body: bytes):
    """Intercept urlopen; return the list the request object lands in."""
    seen = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return body

    def fake_urlopen(request, timeout=None, context=None):
        seen.append((request, timeout, context))
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    return seen


def test_request_targets_chat_completions_with_bearer_credential(provider, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SECRET)
    seen = _capture(monkeypatch, b'{"choices":[{"message":{"content":"Xin chao"}}]}')

    assert provider.translate("こんにちは", "vi") == "Xin chao"

    request, timeout, context = seen[0]
    assert request.full_url == "https://gateway.example.test/v1/chat/completions"
    assert request.method == "POST"
    assert request.get_header("Authorization") == f"Bearer {SECRET}"
    assert request.get_header("Anthropic-version") is None
    assert timeout == provider.config.timeout_seconds
    assert context is None or isinstance(context, ssl.SSLContext)
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["model"] == "example-model"
    assert payload["messages"][0]["role"] == "user"
    assert "こんにちは" in payload["messages"][0]["content"]


def test_trailing_slash_on_the_base_url_does_not_double_the_path(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SECRET)
    seen = _capture(monkeypatch, b'{"choices":[{"message":{"content":"ok"}}]}')
    slashed = OpenAICompatibleTranslationProvider(
        _live_config(base_url="https://gateway.example.test/v1/")
    )

    slashed.translate("こんにちは", "vi")

    assert seen[0][0].full_url == "https://gateway.example.test/v1/chat/completions"


def test_missing_credential_is_refused_before_any_request(provider, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    seen = _capture(monkeypatch, b"{}")

    with pytest.raises(MissingProviderSecretError, match="ANTHROPIC_AUTH_TOKEN"):
        provider.translate("こんにちは", "vi")
    assert seen == []


def test_http_status_becomes_a_status_bearing_provider_error(provider, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SECRET)

    def failing_urlopen(request, timeout=None, context=None):
        raise urllib.error.HTTPError(request.full_url, 429, "rate limited", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    with pytest.raises(ProviderHttpError) as excinfo:
        provider.translate("こんにちは", "vi")
    assert excinfo.value.status_code == 429
    assert SECRET not in str(excinfo.value)


@pytest.mark.parametrize(
    "body",
    [
        b"not json",
        b"[]",
        b'{"choices":[]}',
        b'{"choices":[{"message":{"content":""}}]}',
        b'{"choices":[{"message":{"content":null}}]}',
        b'{"choices":[{"message":{}}]}',
    ],
)
def test_unusable_responses_raise_a_provider_error(provider, monkeypatch, body):
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", SECRET)
    _capture(monkeypatch, body)
    with pytest.raises(TranslationProviderError):
        provider.translate("こんにちは", "vi")


def test_adapter_selection_keys_on_the_credential_not_the_vendor_name():
    """A gateway the code has never heard of still gets the live adapter.

    The seam is ``secret_env_var``: ``None`` means the network-free fixture
    (FR-060), anything else means a live call to the configured ``base_url``.
    """
    live = default_provider_adapter(_live_config())
    fixture = default_provider_adapter(_live_config(secret_env_var=None))

    assert isinstance(live, OpenAICompatibleTranslationProvider)
    assert not isinstance(fixture, OpenAICompatibleTranslationProvider)
    assert fixture.deterministic is True


def test_an_operator_supplied_trust_bundle_is_never_overridden(monkeypatch):
    """``SSL_CERT_FILE`` is the operator's decision; the adapter must defer to it."""
    monkeypatch.setenv("SSL_CERT_FILE", "C:/operator/ca-bundle.pem")

    assert _provider_ssl_context() is None


def test_a_working_default_trust_store_is_never_overridden(monkeypatch):
    """When the interpreter can verify on its own, certifi is not substituted in."""
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(
        ssl,
        "get_default_verify_paths",
        lambda: ssl.DefaultVerifyPaths(
            cafile=__file__, capath=None, openssl_cafile_env="SSL_CERT_FILE",
            openssl_cafile=__file__, openssl_capath_env="SSL_CERT_DIR",
            openssl_capath=None,
        ),
    )

    assert _provider_ssl_context() is None
