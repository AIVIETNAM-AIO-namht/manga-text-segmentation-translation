"""Spec 004 translation package; module imports are intentionally left to their callers.

Module responsibilities and functional-requirement ranges:

* ``runs``: output addressing and run/error metadata (FR-046–FR-059).
* ``intake``: Spec 1–3 input admission (FR-001–FR-012).
* ``regions``: deterministic mask-to-region extraction (FR-013–FR-021).
* ``ocr``: OCR adapter and editable OCR records (FR-022–FR-028).
* ``translate``: provider, retries, secrets, and cache (FR-029–FR-036).
* ``render``: deterministic text rendering and font handling (FR-037–FR-045).
* ``pipeline``: orchestration, CLI integration, and operability (FR-046–FR-061).

No submodule is imported here, so package import has no OCR, provider, or rendering
runtime side effects.
"""
