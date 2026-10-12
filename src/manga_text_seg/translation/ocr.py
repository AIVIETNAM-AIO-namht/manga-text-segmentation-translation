"""OCR adapter and editable per-region OCR artifacts for Spec 004."""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from PIL import Image

from ..config import TranslationOcrConfig
from .regions import TextRegion, crop_region


class OcrEngineUnavailableError(RuntimeError):
    """The selected OCR engine is not installed or cannot be initialized."""


# Compatibility alias retained for existing consumers of the adapter module.
OcrEngineUnavailable = OcrEngineUnavailableError


class Recognizer(Protocol):
    """Callable OCR backend; test recognizers need no manga-ocr installation."""

    def __call__(
        self, image: Image.Image
    ) -> str | dict[str, Any] | tuple[Any, ...]: ...


@dataclass(frozen=True)
class OcrResult:
    region_id: int
    crop_path: str
    text: str
    confidence: float | None
    engine: str
    engine_version: str | None
    preprocessing: dict[str, Any] | None
    processing_time_seconds: float
    status: str
    error: str | None
    manually_edited: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.region_id, int)
            or isinstance(self.region_id, bool)
            or self.region_id < 1
        ):
            raise ValueError("region_id must be a positive integer")
        if not isinstance(self.crop_path, str) or not self.crop_path:
            raise ValueError("crop_path must be a non-empty string")
        if not isinstance(self.text, str):
            raise ValueError("text must be a string")
        if self.confidence is not None and (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not math.isfinite(float(self.confidence))
        ):
            raise ValueError("confidence must be a finite number or null")
        if self.status not in {"ok", "empty", "failed"}:
            raise ValueError("status must be 'ok', 'empty', or 'failed'")
        if (
            not isinstance(self.processing_time_seconds, (int, float))
            or isinstance(self.processing_time_seconds, bool)
            or not math.isfinite(float(self.processing_time_seconds))
            or self.processing_time_seconds < 0
        ):
            raise ValueError("processing_time_seconds must be non-negative")
        if self.status == "failed" and not self.error:
            raise ValueError("failed OCR results require an error message")


@dataclass(frozen=True)
class ManualEditRecord:
    region_id: int
    original_text: str
    edited_text: str
    edited_at: str
    invalidated_by_reextraction: bool = False


@dataclass(frozen=True)
class OcrRun:
    engine: str
    engine_version: str | None
    preprocessing: dict[str, Any] | None
    results: tuple[OcrResult, ...]
    manual_edits: tuple[ManualEditRecord, ...]
    invalidation_reports: tuple[dict[str, Any], ...]
    path: Path


def _load_manga_ocr() -> tuple[Recognizer, str | None]:
    """Import and initialize manga-ocr only when the real adapter is used."""
    try:
        from manga_ocr import MangaOcr
    except ImportError as exc:
        raise OcrEngineUnavailableError(
            "manga-ocr is not installed; install the optional OCR extra with "
            "`pip install -e '.[ocr]'`"
        ) from exc

    try:
        version = importlib.metadata.version("manga-ocr")
    except importlib.metadata.PackageNotFoundError:
        version = None
    return MangaOcr(), version


def _as_pil_image(image: np.ndarray) -> Image.Image:
    if image.ndim == 2:
        return Image.fromarray(image)
    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError("original image must be grayscale, RGB/BGR, or RGBA/BGRA")
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(image)


def _preprocess(image: Image.Image, config: TranslationOcrConfig) -> Image.Image:
    result = image
    if config.grayscale:
        result = result.convert("L")
    if config.upscale_factor != 1.0:
        size = (
            max(1, round(result.width * config.upscale_factor)),
            max(1, round(result.height * config.upscale_factor)),
        )
        result = result.resize(size, Image.Resampling.LANCZOS)
    return result


def _result_text(value: Any) -> tuple[str, float | None]:
    confidence: float | None = None
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict):
        text = value.get("text")
        confidence = value.get("confidence")
    elif isinstance(value, tuple) and len(value) == 2:
        text, confidence = value
    else:
        raise TypeError(
            "recognizer must return text, (text, confidence), or a result mapping"
        )
    if not isinstance(text, str):
        raise TypeError("recognizer result text must be a string")
    if confidence is not None and (
        not isinstance(confidence, (int, float))
        or isinstance(confidence, bool)
        or not math.isfinite(float(confidence))
    ):
        raise TypeError("recognizer confidence must be a finite number or null")
    return text, None if confidence is None else float(confidence)


def _region_signature(regions: Sequence[TextRegion]) -> list[dict[str, Any]]:
    return [
        {
            "region_id": region.region_id,
            "bbox": list(region.bbox),
            "polygon": [list(point) for point in region.polygon],
        }
        for region in sorted(regions, key=lambda item: item.region_id)
    ]


def _load_document(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read existing OCR artifact {path}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise ValueError("ocr.json must contain a results array")
    document.setdefault("manual_edits", [])
    document.setdefault("stale_region_ids", [])
    return document


def _record_direct_edits(document: dict[str, Any]) -> bool:
    """Recognize text-only hand edits by comparing with immutable OCR baselines."""
    baselines = document.get("recognized_text", {})
    if not isinstance(baselines, dict):
        return False
    edits = [ManualEditRecord(**entry) for entry in document["manual_edits"]]
    latest = {edit.region_id: edit for edit in edits}
    changed = False
    for result in document["results"]:
        region_id = result.get("region_id")
        baseline = baselines.get(str(region_id))
        text = result.get("text")
        if (
            not isinstance(baseline, str)
            or not isinstance(text, str)
            or text == baseline
        ):
            continue
        previous = latest.get(region_id)
        if (
            previous is not None
            and not previous.invalidated_by_reextraction
            and previous.edited_text == text
        ):
            result["manually_edited"] = True
            continue
        edit = ManualEditRecord(
            region_id=region_id,
            original_text=baseline,
            edited_text=text,
            edited_at=datetime.now(timezone.utc).isoformat(),
        )
        document["manual_edits"].append(asdict(edit))
        latest[region_id] = edit
        result["manually_edited"] = True
        result["status"] = "empty" if text == "" else "ok"
        result["error"] = None
        changed = True
    return changed


def _edited_result(
    record: dict[str, Any], edit: ManualEditRecord, *, crop_path: str
) -> OcrResult:
    return OcrResult(
        region_id=edit.region_id,
        crop_path=crop_path,
        text=edit.edited_text,
        confidence=record.get("confidence"),
        engine=record["engine"],
        engine_version=record.get("engine_version"),
        preprocessing=record.get("preprocessing"),
        processing_time_seconds=record["processing_time_seconds"],
        status="empty" if edit.edited_text == "" else "ok",
        error=None,
        manually_edited=True,
    )


class OcrAdapter:
    """Recognize region crops with an injectable, lazily initialized engine."""

    def __init__(
        self,
        recognizer: Recognizer | None = None,
        *,
        engine: str | None = None,
        engine_version: str | None = None,
        error_report: Any | None = None,
    ) -> None:
        if engine is not None and (not isinstance(engine, str) or not engine.strip()):
            raise ValueError("engine must be a non-empty string when specified")
        self._recognizer = recognizer
        self.engine = engine
        self.engine_version = engine_version
        self.error_report = error_report

    def _resolve_recognizer(self) -> Recognizer:
        if self._recognizer is not None:
            return self._recognizer
        if self.engine != "manga-ocr":
            raise OcrEngineUnavailableError(
                f"OCR engine {self.engine!r} needs an injected adapter implementation"
            )
        try:
            self._recognizer, version = _load_manga_ocr()
        except OcrEngineUnavailableError:
            raise
        except Exception as exc:
            raise OcrEngineUnavailableError(
                f"manga-ocr could not be initialized: {exc}"
            ) from exc
        self.engine_version = self.engine_version or version
        return self._recognizer

    def run(
        self,
        original_image: np.ndarray | str | Path,
        regions: Sequence[TextRegion],
        page_dir: str | Path,
        *,
        config: TranslationOcrConfig,
        image_id: str | None = None,
        signature_regions: Sequence[TextRegion] | None = None,
    ) -> OcrRun:
        """Write pre-inpaint crops and `ocr.json`, isolating errors per region."""
        if not isinstance(config, TranslationOcrConfig):
            raise TypeError("config must be a TranslationOcrConfig")
        if self.engine is None:
            self.engine = config.engine
        page = Path(page_dir)
        crops_dir = page / "crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(original_image, (str, Path)):
            image = cv2.imread(str(original_image), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(
                    f"cannot read original pre-inpaint image: {original_image}"
                )
        elif isinstance(original_image, np.ndarray):
            image = original_image
        else:
            raise TypeError("original_image must be the original image array or path")
        if image.ndim not in {2, 3} or image.shape[:2] != (1170, 1654):
            raise ValueError(
                "original pre-inpaint image must have aligned size 1654x1170"
            )

        output_path = page / "ocr.json"
        previous = _load_document(output_path)
        signature = _region_signature(signature_regions or regions)
        old_signature = previous.get("region_set") if previous else None
        old_regions = {
            item.get("region_id"): item
            for item in old_signature or []
            if isinstance(item, dict) and isinstance(item.get("region_id"), int)
        }
        current_regions = {item["region_id"]: item for item in signature}
        unchanged_region_ids = {
            region_id
            for region_id, item in current_regions.items()
            if old_regions.get(region_id) == item
        }
        if previous is not None:
            _record_direct_edits(previous)
        manual_edits = [
            ManualEditRecord(**entry)
            for entry in (previous.get("manual_edits", []) if previous else [])
        ]
        invalidation_reports: list[dict[str, Any]] = []
        if previous and old_signature != signature:
            updated_edits = []
            for edit in manual_edits:
                if (
                    not edit.invalidated_by_reextraction
                    and edit.region_id not in unchanged_region_ids
                ):
                    edit = ManualEditRecord(
                        region_id=edit.region_id,
                        original_text=edit.original_text,
                        edited_text=edit.edited_text,
                        edited_at=edit.edited_at,
                        invalidated_by_reextraction=True,
                    )
                    invalidation_reports.append(
                        {
                            "region_id": edit.region_id,
                            "reason": "region set changed during re-extraction",
                        }
                    )
                updated_edits.append(edit)
            manual_edits = updated_edits

        active_edits = {
            edit.region_id: edit
            for edit in manual_edits
            if not edit.invalidated_by_reextraction
        }
        old_results = {
            record["region_id"]: record
            for record in (previous.get("results", []) if previous else [])
        }
        preprocessing = (
            {
                "enabled": True,
                "grayscale": config.grayscale,
                "upscale_factor": config.upscale_factor,
            }
            if config.preprocessing_enabled
            else None
        )
        recognizer: Recognizer | None = None
        engine_error: str | None = None
        results: list[OcrResult] = []

        for region in sorted(regions, key=lambda item: item.region_id):
            crop = crop_region(image, region)
            crop_path = crops_dir / f"{region.region_id}.png"
            if not cv2.imwrite(str(crop_path), crop):
                message = f"could not write OCR crop: {crop_path}"
                result = self._failure(
                    region.region_id,
                    f"crops/{region.region_id}.png",
                    preprocessing,
                    message,
                )
                self._record_error(image_id, region.region_id, message)
                results.append(result)
                continue
            relative_crop_path = crop_path.relative_to(page).as_posix()

            if (
                region.region_id in unchanged_region_ids
                and region.region_id in active_edits
            ):
                old = old_results.get(region.region_id)
                if old is not None:
                    results.append(
                        _edited_result(
                            old,
                            active_edits[region.region_id],
                            crop_path=relative_crop_path,
                        )
                    )
                    continue

            if engine_error is None and recognizer is None:
                try:
                    recognizer = self._resolve_recognizer()
                except Exception as exc:
                    engine_error = str(exc)

            if engine_error is not None:
                result = self._failure(
                    region.region_id,
                    relative_crop_path,
                    preprocessing,
                    engine_error,
                )
                self._record_error(image_id, region.region_id, engine_error)
                results.append(result)
                continue

            pil_crop = _as_pil_image(crop)
            ocr_input = (
                _preprocess(pil_crop, config)
                if config.preprocessing_enabled
                else pil_crop
            )
            started = time.perf_counter()
            try:
                text, confidence = _result_text(recognizer(ocr_input))
                elapsed = time.perf_counter() - started
                status = "ok" if text.strip() else "empty"
                error = None if status == "ok" else "recognizer returned empty text"
                result = OcrResult(
                    region_id=region.region_id,
                    crop_path=relative_crop_path,
                    text=text,
                    confidence=confidence,
                    engine=self.engine,
                    engine_version=self.engine_version,
                    preprocessing=preprocessing,
                    processing_time_seconds=elapsed,
                    status=status,
                    error=error,
                    manually_edited=False,
                )
                if status == "empty":
                    self._record_error(
                        image_id, region.region_id, error or "empty OCR result"
                    )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                message = f"{type(exc).__name__}: {exc}"
                result = self._failure(
                    region.region_id,
                    relative_crop_path,
                    preprocessing,
                    message,
                    elapsed,
                )
                self._record_error(image_id, region.region_id, message)
            results.append(result)

        run = OcrRun(
            engine=self.engine,
            engine_version=self.engine_version,
            preprocessing=preprocessing,
            results=tuple(results),
            manual_edits=tuple(manual_edits),
            invalidation_reports=tuple(invalidation_reports),
            path=output_path,
        )
        old_baselines = previous.get("recognized_text", {}) if previous else {}
        recognized_text = {
            str(result.region_id): (
                old_baselines.get(str(result.region_id), result.text)
                if result.region_id in unchanged_region_ids
                and result.region_id in active_edits
                else result.text
            )
            for result in results
        }
        _write_run(run, signature, recognized_text)
        return run

    def _failure(
        self,
        region_id: int,
        crop_path: str,
        preprocessing: dict[str, Any] | None,
        message: str,
        elapsed: float = 0.0,
    ) -> OcrResult:
        return OcrResult(
            region_id=region_id,
            crop_path=crop_path,
            text="",
            confidence=None,
            engine=self.engine,
            engine_version=self.engine_version,
            preprocessing=preprocessing,
            processing_time_seconds=elapsed,
            status="failed",
            error=message,
            manually_edited=False,
        )

    def _record_error(self, image_id: str | None, region_id: int, reason: str) -> None:
        if self.error_report is None:
            return
        if image_id is None:
            raise ValueError(
                "image_id is required when writing OCR failures to ErrorReport"
            )
        self.error_report.add(
            "ocr_failure",
            image_id=image_id,
            region_id=region_id,
            stage="ocr",
            reason=reason,
        )


def _write_run(
    run: OcrRun,
    region_set: list[dict[str, Any]],
    recognized_text: dict[str, str],
) -> None:
    document = {
        "engine": run.engine,
        "engine_version": run.engine_version,
        "preprocessing": run.preprocessing,
        "region_set": region_set,
        "recognized_text": recognized_text,
        "results": [asdict(result) for result in run.results],
        "manual_edits": [asdict(edit) for edit in run.manual_edits],
        "invalidation_reports": list(run.invalidation_reports),
        "stale_region_ids": [],
    }
    run.path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def apply_manual_edit(
    ocr_path: str | Path,
    region_id: int,
    edited_text: str,
    *,
    edited_at: str | None = None,
) -> ManualEditRecord:
    """Apply an auditable human edit to an existing `ocr.json` result."""
    if not isinstance(edited_text, str):
        raise TypeError("edited_text must be a string")
    path = Path(ocr_path)
    document = _load_document(path)
    if document is None:
        raise FileNotFoundError(path)
    results = document["results"]
    target = next(
        (item for item in results if item.get("region_id") == region_id), None
    )
    if target is None:
        raise KeyError(f"OCR result for region_id {region_id} was not found")
    previous_edits = [ManualEditRecord(**entry) for entry in document["manual_edits"]]
    original_text = target["text"]
    timestamp = edited_at or datetime.now(timezone.utc).isoformat()
    edit = ManualEditRecord(region_id, original_text, edited_text, timestamp)
    document["manual_edits"] = [
        *[asdict(item) for item in previous_edits],
        asdict(edit),
    ]
    target["text"] = edited_text
    target["status"] = "empty" if edited_text == "" else "ok"
    target["error"] = None
    target["manually_edited"] = True
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return edit


def load_ocr_results(path: str | Path) -> tuple[OcrResult, ...]:
    """Load OCR records while applying the newest non-invalidated manual edits."""
    document = _load_document(Path(path))
    if document is None:
        raise FileNotFoundError(path)
    if _record_direct_edits(document):
        Path(path).write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    edits = {
        edit.region_id: edit
        for edit in (ManualEditRecord(**entry) for entry in document["manual_edits"])
        if not edit.invalidated_by_reextraction
    }
    stale_region_ids = set(document.get("stale_region_ids", []))
    loaded: list[OcrResult] = []
    for raw in document["results"]:
        if raw.get("region_id") in stale_region_ids:
            continue
        result = OcrResult(**raw)
        edit = edits.get(result.region_id)
        if edit is not None:
            result = _edited_result(raw, edit, crop_path=result.crop_path)
        loaded.append(result)
    return tuple(loaded)
