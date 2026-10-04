"""Per-region translation policy, mockable provider seam, and shared cache.

Spec 004 does not define a concrete provider's wire request/response schema.
This module therefore supplies the adapter contract and deterministic fixture
provider without inventing a live endpoint or authentication protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import socket
import time
import urllib.error
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from ..config import TranslationConfig
from .runs import ErrorReport


class TranslationProviderError(RuntimeError):
    """A provider failure that is safe to expose after credential scrubbing."""


class ProviderHttpError(TranslationProviderError):
    """Status-bearing provider failure for adapter implementations and fixtures."""

    def __init__(
        self, status_code: int, message: str = "provider request failed"
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


class MissingProviderSecretError(RuntimeError):
    """Raised before work starts when a configured credential is unavailable."""


class SecretLeakError(RuntimeError):
    """A provider response or input contains the configured credential value."""


class TranslationProviderAdapter(Protocol):
    """One independent provider request for one region's text."""

    def translate(self, text: str, target_language: str) -> str: ...


@dataclass(frozen=True)
class TranslationRequest:
    region_id: int
    source_text: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.region_id, int)
            or isinstance(self.region_id, bool)
            or self.region_id < 1
        ):
            raise ValueError("region_id must be a positive integer")
        if not isinstance(self.source_text, str):
            raise ValueError("source_text must be a string")


@dataclass(frozen=True)
class TranslationResult:
    region_id: int
    source_text: str
    translated_text: str
    target_language: str
    provider: str
    model: str
    version: str | None
    cache_status: str
    processing_time_seconds: float
    status: str
    error: str | None
    failure_policy: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.region_id, int)
            or isinstance(self.region_id, bool)
            or self.region_id < 1
        ):
            raise ValueError("region_id must be a positive integer")
        for label in (
            "source_text",
            "translated_text",
            "target_language",
            "provider",
            "model",
        ):
            if not isinstance(getattr(self, label), str):
                raise ValueError(f"{label} must be a string")
        if self.version is not None and not isinstance(self.version, str):
            raise ValueError("version must be a string or null")
        if self.cache_status not in {"hit", "miss"}:
            raise ValueError("cache_status must be 'hit' or 'miss'")
        if self.status not in {"ok", "failed", "skipped"}:
            raise ValueError("status must be 'ok', 'failed', or 'skipped'")
        if self.failure_policy not in {"keep_ocr_text", "exclude_from_rendering"}:
            raise ValueError("failure_policy is invalid")
        if (
            not isinstance(self.processing_time_seconds, (int, float))
            or isinstance(self.processing_time_seconds, bool)
            or not math.isfinite(float(self.processing_time_seconds))
            or self.processing_time_seconds < 0
        ):
            raise ValueError(
                "processing_time_seconds must be a finite non-negative number"
            )
        if self.status == "failed" and not self.error:
            raise ValueError("failed translations require an error")


@dataclass(frozen=True)
class TranslationCacheRecord:
    cache_key: str
    translated_text: str
    provider: str
    model: str
    version: str | None
    fetched_at: str


class FixtureTranslationProvider:
    """Deterministic, network-free provider used by unit and quickstart fixtures."""

    deterministic = True

    def __init__(self, responses: Mapping[str, str | BaseException]) -> None:
        self.responses = dict(responses)
        self.calls: list[tuple[str, str]] = []

    def translate(self, text: str, target_language: str) -> str:
        self.calls.append((text, target_language))
        if text not in self.responses:
            raise TranslationProviderError("fixture has no response for this input")
        response = self.responses[text]
        if isinstance(response, BaseException):
            raise response
        return response


def translation_cache_key(
    text: str, target_language: str, provider: str, model: str
) -> str:
    """Return the contract's SHA-256 over canonical JSON of its four inputs."""
    canonical = json.dumps(
        [text, target_language, provider, model],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def write_translations_json(
    path: str | Path, results: Sequence[TranslationResult]
) -> Path:
    """Persist one validated translation record for each supplied region."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ids = [result.region_id for result in results]
    if len(ids) != len(set(ids)):
        raise ValueError("translations must contain at most one result per region_id")
    destination.write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    return destination


class TranslationService:
    """Apply per-region retries, failure policy, cache, provenance and secret checks."""

    def __init__(
        self,
        config: TranslationConfig,
        provider: TranslationProviderAdapter,
        *,
        error_report: ErrorReport | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.provider = provider
        self.error_report = error_report
        self._sleeper = sleeper
        secret_env_var = config.provider.secret_env_var
        if secret_env_var is not None and not os.environ.get(secret_env_var):
            raise MissingProviderSecretError(
                "required provider credential environment variable "
                f"{secret_env_var!r} is not set"
            )

    @property
    def provenance(self) -> dict[str, Any]:
        provider = self.config.provider
        return {
            "name": provider.name,
            "model": provider.model,
            "version": provider.version,
            "deterministic": bool(
                getattr(self.provider, "deterministic", provider.deterministic)
            ),
            "secret_env_var": provider.secret_env_var,
        }

    def _secret(self) -> str | None:
        name = self.config.provider.secret_env_var
        return os.environ.get(name) if name is not None else None

    def _sanitize(self, message: str) -> str:
        secret = self._secret()
        return message.replace(secret, "[REDACTED]") if secret else message

    def _has_secret(self, value: str) -> bool:
        secret = self._secret()
        return bool(secret and secret in value)

    def translate_regions(
        self,
        requests: Sequence[TranslationRequest],
        *,
        image_id: str | None = None,
        output_path: str | Path | None = None,
    ) -> tuple[TranslationResult, ...]:
        """Translate each request independently; optionally write JSON output."""
        region_ids = [request.region_id for request in requests]
        if len(region_ids) != len(set(region_ids)):
            raise ValueError("translation input contains duplicate region_id values")
        results = tuple(
            self.translate_one(request, image_id=image_id) for request in requests
        )
        if output_path is not None:
            write_translations_json(output_path, results)
        return results

    def translate_one(
        self, request: TranslationRequest, *, image_id: str | None = None
    ) -> TranslationResult:
        if not isinstance(request, TranslationRequest):
            raise TypeError("request must be a TranslationRequest")
        provider = self.config.provider
        source_text = request.source_text
        if self._has_secret(source_text):
            safe_source = self._sanitize(source_text)
            return self._failed_result(
                request.region_id,
                safe_source,
                "source text contains configured credential material; request refused",
                cache_status="miss",
                image_id=image_id,
            )

        key = translation_cache_key(
            source_text,
            self.config.target_language,
            provider.name,
            provider.model,
        )
        cache_enabled = provider.cache.enabled and not provider.cache.bypass
        cache_path = provider.cache.root / f"{key}.json"
        if cache_enabled:
            cached = self._read_cache(cache_path, key)
            if cached is not None:
                if self._has_secret(cached.translated_text):
                    return self._failed_result(
                        request.region_id,
                        source_text,
                        "cached response contained configured credential material",
                        cache_status="hit",
                        image_id=image_id,
                    )
                return TranslationResult(
                    region_id=request.region_id,
                    source_text=source_text,
                    translated_text=cached.translated_text,
                    target_language=self.config.target_language,
                    provider=provider.name,
                    model=provider.model,
                    version=provider.version,
                    cache_status="hit",
                    processing_time_seconds=0.0,
                    status="ok",
                    error=None,
                    failure_policy=provider.failure_policy,
                )

        started = time.perf_counter()
        translated_text, error = self._request_with_retries(request, image_id=image_id)
        elapsed = time.perf_counter() - started
        if error is not None:
            return self._failed_result(
                request.region_id,
                source_text,
                error,
                cache_status="miss",
                processing_time_seconds=elapsed,
                image_id=image_id,
            )

        assert translated_text is not None
        if self._has_secret(translated_text):
            return self._failed_result(
                request.region_id,
                source_text,
                "provider response contained configured credential material; "
                "response refused",
                cache_status="miss",
                processing_time_seconds=elapsed,
                image_id=image_id,
            )

        result = TranslationResult(
            region_id=request.region_id,
            source_text=source_text,
            translated_text=translated_text,
            target_language=self.config.target_language,
            provider=provider.name,
            model=provider.model,
            version=provider.version,
            cache_status="miss",
            processing_time_seconds=elapsed,
            status="ok",
            error=None,
            failure_policy=provider.failure_policy,
        )
        if provider.cache.enabled and not provider.cache.bypass:
            self._write_cache(
                cache_path,
                TranslationCacheRecord(
                    cache_key=key,
                    translated_text=translated_text,
                    provider=provider.name,
                    model=provider.model,
                    version=provider.version,
                    fetched_at=datetime.now(UTC).isoformat(),
                ),
            )
        return result

    def _request_with_retries(
        self, request: TranslationRequest, *, image_id: str | None
    ) -> tuple[str | None, str | None]:
        retry = self.config.provider.retry
        for attempt in range(1, retry.max_attempts + 1):
            try:
                text = self.provider.translate(
                    request.source_text, self.config.target_language
                )
                if not isinstance(text, str) or not text.strip():
                    raise TranslationProviderError(
                        "provider returned an empty or invalid translation"
                    )
                return text, None
            except Exception as exc:
                safe_reason = self._sanitize(str(exc)) or type(exc).__name__
                status_code = _status_code(exc)
                timed_out = _is_timeout(exc)
                rate_limited = status_code == 429
                if timed_out or rate_limited:
                    self._record_error(
                        "api_timeout_or_rate_limit",
                        request.region_id,
                        safe_reason,
                        image_id,
                    )
                retryable = timed_out or (
                    status_code is not None and status_code in retry.retry_status_codes
                )
                if retryable and attempt < retry.max_attempts:
                    delay = min(
                        retry.initial_backoff_seconds * (2 ** (attempt - 1)),
                        retry.max_backoff_seconds,
                    )
                    self._sleeper(delay)
                    continue
                return None, safe_reason
        return None, "provider failed after configured retry attempts"

    def _failed_result(
        self,
        region_id: int,
        source_text: str,
        reason: str,
        *,
        cache_status: str,
        processing_time_seconds: float = 0.0,
        image_id: str | None,
    ) -> TranslationResult:
        provider = self.config.provider
        safe_reason = self._sanitize(reason)
        self._record_error("translation_failure", region_id, safe_reason, image_id)
        keep_text = provider.failure_policy == "keep_ocr_text"
        return TranslationResult(
            region_id=region_id,
            source_text=source_text,
            translated_text=source_text if keep_text else "",
            target_language=self.config.target_language,
            provider=provider.name,
            model=provider.model,
            version=provider.version,
            cache_status=cache_status,
            processing_time_seconds=processing_time_seconds,
            status="failed" if keep_text else "skipped",
            error=safe_reason,
            failure_policy=provider.failure_policy,
        )

    def _record_error(
        self,
        category: str,
        region_id: int,
        reason: str,
        image_id: str | None,
    ) -> None:
        if self.error_report is None:
            return
        if image_id is None:
            raise ValueError("image_id is required when recording translation errors")
        self.error_report.add(
            category,
            image_id=image_id,
            region_id=region_id,
            stage="translate",
            reason=self._sanitize(reason) or "provider error",
        )

    @staticmethod
    def _read_cache(path: Path, key: str) -> TranslationCacheRecord | None:
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            record = TranslationCacheRecord(**raw)
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            return None
        if record.cache_key != key:
            return None
        return record

    @staticmethod
    def _write_cache(path: Path, record: TranslationCacheRecord) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(record), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


def _status_code(error: Exception) -> int | None:
    if isinstance(error, urllib.error.HTTPError):
        return error.code
    value = getattr(error, "status_code", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_timeout(error: Exception) -> bool:
    if isinstance(error, (TimeoutError, socket.timeout)):
        return True
    if isinstance(error, urllib.error.URLError):
        return isinstance(error.reason, (TimeoutError, socket.timeout))
    return False
