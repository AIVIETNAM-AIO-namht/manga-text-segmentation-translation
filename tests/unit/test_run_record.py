"""The run record an admitted result leaves behind (T024, US1 scenario 7, FR-044, FR-056, FR-059).

Scenario 7 asks that inspecting the run record name the method, checkpoint
identity, repository and revision, both licences, checkpoint-load evidence,
device, input size, threshold, preprocessing, postprocessing, seed, timings and
package versions — *each attributed to the environment that produced them*.

That last clause is the whole point of FR-052: a run is assembled from results
produced elsewhere, so there is no single environment to record once. The record
carries one provenance record per method and no environment field of its own.
"""
from __future__ import annotations

import json

from manga_text_seg.imaging import load_raw_page
from manga_text_seg.receipt import admit


def _masks(manifest) -> dict:
    """A deterministic binary mask per page: no adapter, no checkpoint (FR-050)."""
    return {
        pair.image_id: (load_raw_page(pair.raw_path) > 127).astype("uint8") * 255
        for pair in manifest.pairs
    }


def _scenario_7_sidecar(image_id: str, doc: dict) -> dict:
    """Add the scenario-7 fields a real runner's sidecar would carry."""
    doc.update(
        {
            "input_size": [1024, 1024],
            "threshold": 0.5,
            "preprocessing": {"resize": "pad-to-multiple-8", "normalise": "imagenet"},
            "postprocessing": {"morphology": "none"},
            # FR-024a and FR-024b: the padding multiple actually inverted, the
            # threshold actually applied, and the label collapse actually used —
            # each verified against the produced mask by the adapter that
            # produced it, and recorded here so the run says which rule scored.
            "padding_multiple": 8,
            "effective_threshold": 0.5,
            "label_collapse": {
                "rule": "argmax over 6 classes; {1,2} -> 255, {0,3,4,5} -> 0",
                "verified": True,
            },
            "inference_time_seconds": 1.25,
        }
    )
    return doc


def _on_device(provenance: dict, *, device: str, packages: dict) -> dict:
    record = json.loads(json.dumps(provenance))
    record["device"] = {"type": "cpu", "name": device}
    record["packages"] = packages
    return record


def _record(tmp_path) -> dict:
    return json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8"))


def test_run_record_names_everything_scenario_7_names(
    exported, make_handoff, default_provenance, tmp_path
):
    page_list = exported["page_list"]
    handoff = make_handoff(
        page_list,
        _masks(exported["manifest"]),
        sidecar=_scenario_7_sidecar,
        provenance=_on_device(
            default_provenance, device="fixture-cpu", packages={"numpy": "2.2.6"}
        ),
    )
    admit(
        handoff,
        page_list=page_list,
        identity_record=exported["identity_record"],
        run_dir=tmp_path / "run",
        manifest=exported["manifest"],
    )

    record = _record(tmp_path)
    assert record["run_id"] == "run"
    assert record["page_list_identity"] == page_list["page_list_identity"]
    assert record["methods_admitted"] == ["standin"]

    # The provenance half of scenario 7, per method.
    provenance = record["method_provenance"]["standin"]
    assert provenance["repository"]["url"] == "https://example.invalid/standin"
    assert provenance["repository"]["revision"] == "0" * 40
    assert provenance["checkpoint"]["identity"] == "standin.bin"
    assert provenance["code_license"] == "MIT"
    assert provenance["weight_license"] == "MIT"
    assert provenance["checkpoint_load_evidence"]["evidenced"] is True
    assert provenance["seed"] == 42
    assert provenance["interpreter"] == "3.11.0"
    assert provenance["packages"] == {"numpy": "2.2.6"}

    # ...attributed to the environment that produced it, not to the run.
    assert record["method_device"]["standin"]["name"] == "fixture-cpu"
    assert "device" not in record
    assert "packages" not in record

    # The configuration half, lifted from the sidecars the runner returned.
    configuration = record["method_configuration"]["standin"]
    assert configuration["input_size"] == [1024, 1024]
    assert configuration["threshold"] == 0.5
    assert configuration["preprocessing"] == {
        "resize": "pad-to-multiple-8",
        "normalise": "imagenet",
    }
    assert configuration["postprocessing"] == {"morphology": "none"}
    # FR-024a and FR-024b: the run says which padding multiple was inverted,
    # which threshold actually scored, and which collapse rule produced the
    # binary mask — the three the method's own adapter verified per page.
    assert configuration["padding_multiple"] == 8
    assert configuration["effective_threshold"] == 0.5
    assert configuration["label_collapse"] == {
        "rule": "argmax over 6 classes; {1,2} -> 255, {0,3,4,5} -> 0",
        "verified": True,
    }

    assert record["admission_timestamps"]["standin"]
    assert record["started_at"]
    assert record["nondeterminism_sources"]


def test_a_second_admission_leaves_the_first_untouched(
    exported, make_handoff, default_provenance, tmp_path
):
    """FR-059: a result arriving later must not invalidate the ones already in."""
    page_list = exported["page_list"]
    masks = _masks(exported["manifest"])
    run_dir = tmp_path / "run"

    admit(
        handoff=make_handoff(page_list, masks),
        page_list=page_list,
        identity_record=exported["identity_record"],
        run_dir=run_dir,
        manifest=exported["manifest"],
    )
    stored = run_dir / "standin" / "masks"
    before = {p: p.read_bytes() for p in sorted(stored.rglob("*.png"))}

    second_provenance = json.loads(json.dumps(default_provenance))
    second_provenance["method"] = "second"
    admit(
        handoff=make_handoff(
            page_list, masks, method="second", provenance=second_provenance
        ),
        page_list=page_list,
        identity_record=exported["identity_record"],
        run_dir=run_dir,
        manifest=exported["manifest"],
    )

    assert {p: p.read_bytes() for p in sorted(stored.rglob("*.png"))} == before

    record = _record(tmp_path)
    assert record["methods_admitted"] == ["second", "standin"]
    assert set(record["method_provenance"]) == {"second", "standin"}
    assert record["started_at"]
