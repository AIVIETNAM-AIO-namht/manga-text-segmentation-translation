"""T052: inpainting reads input artifacts without changing source data."""

from __future__ import annotations

import builtins
from pathlib import Path

import cv2

from manga_text_seg.inpaint import process_sample
from manga_text_seg.manifest import load_manifest, save_manifest


def test_inpainting_preserves_inputs_and_never_opens_excluded_data(
    tmp_path, monkeypatch, synthetic_manifest, inpaint_fixture, inpaint_config
):
    manifest = synthetic_manifest()
    manifest_file = tmp_path / "manifest.json"
    save_manifest(manifest, manifest_file)
    persisted_manifest = manifest_file.read_bytes()
    source = inpaint_fixture("classical_baseline", "ARMS/001")
    config = inpaint_config({"classical_baseline": source}, algorithms=("telea",))
    source_files = sorted(path for path in source.rglob("*") if path.is_file())
    before = {path: path.read_bytes() for path in source_files}

    excluded = tmp_path / "data" / "no-need-to-read"
    excluded.mkdir(parents=True)
    sentinel = excluded / "must-not-open.txt"
    sentinel.write_text("private fixture sentinel", encoding="utf-8")

    def guard(path):
        if "no-need-to-read" in str(path).lower():
            raise AssertionError(f"Excluded data was opened: {path}")

    original_open = builtins.open
    original_cv_read = cv2.imread
    original_path_open = Path.open

    def audited_open(file, *args, **kwargs):
        guard(file)
        return original_open(file, *args, **kwargs)

    def audited_cv_read(file, *args, **kwargs):
        guard(file)
        return original_cv_read(file, *args, **kwargs)

    def audited_path_open(self, *args, **kwargs):
        guard(self)
        return original_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", audited_open)
    monkeypatch.setattr(cv2, "imread", audited_cv_read)
    monkeypatch.setattr(Path, "open", audited_path_open)

    process_sample(
        manifest, "ARMS", "001", method="classical_baseline", run_id="invariance",
        config=config, experiment_id="fixture", model_repository=None,
        algorithms=("telea",),
    )

    assert {path: path.read_bytes() for path in source_files} == before
    assert manifest_file.read_bytes() == persisted_manifest
    assert not manifest.gt_root.exists()
    assert sentinel.is_file()
