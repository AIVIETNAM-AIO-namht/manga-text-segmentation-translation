"""T057: inpainting operates on masks and does not recompute segmentation metrics."""

import json

from manga_text_seg.inpaint import process_sample


def test_pipeline_does_not_call_segmentation_metrics_or_emit_forbidden_scores(
    monkeypatch, inpaint_fixture, synthetic_manifest, inpaint_config
):
    from manga_text_seg import metrics

    def forbidden(*args, **kwargs):
        raise AssertionError("inpainting must not recompute segmentation metrics")

    monkeypatch.setattr(metrics, "compute_metrics", forbidden)
    manifest = synthetic_manifest()
    source = inpaint_fixture("classical_baseline", "ARMS/001")
    config = inpaint_config({"classical_baseline": source})
    process_sample(
        manifest, "ARMS", "001", method="classical_baseline", run_id="no-metrics",
        config=config, experiment_id="fixture", model_repository=None,
    )

    forbidden_terms = ("PSNR", "SSIM", "synthetic ground truth", "single-score ranking")
    outputs = [path for path in config.output_root.rglob("*") if path.is_file()]
    textual_outputs = [
        path.read_text(encoding="utf-8")
        for path in outputs if path.suffix.lower() in {".json", ".txt", ".md", ".csv"}
    ]
    serialized = json.dumps(textual_outputs, ensure_ascii=False)
    assert all(term.lower() not in serialized.lower() for term in forbidden_terms)
