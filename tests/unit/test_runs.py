"""Tests for addressing + run scaffolding in manga_text_seg.runs (T005).

Written before the implementation exists — must fail until T007-T008 land.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from manga_text_seg.config import DilationConfig, InpaintingConfig, MaskProcessingConfig
from manga_text_seg.runs import (
    ERROR_CATEGORIES,
    AlignmentRecord,
    ErrorReport,
    InpaintRun,
    MaskRejection,
    MethodSource,
    RunError,
    SampleCounts,
    SampleMetadata,
    create_run,
    sample_paths,
    write_run_record,
    write_sample_metadata,
)

VALID_METHODS = [
    "classical_baseline",
    "manga_text_segmentation",
    "comic_text_detector",
    "unetpp_efficientnetv2",
]
VALID_ALGORITHMS = ["telea", "ns"]


class TestAddressing:
    """FR-026, FR-029: paths resolve from (run_id, method, algorithm, image_id) alone."""

    @pytest.mark.parametrize("method", VALID_METHODS)
    @pytest.mark.parametrize("algorithm", VALID_ALGORITHMS)
    def test_resolves_without_any_index_lookup(
        self, tmp_path: Path, method: str, algorithm: str
    ) -> None:
        paths = sample_paths(tmp_path, "run-2026-09-27", method, algorithm, "ARMS/000")

        expected_image = (
            tmp_path / "run-2026-09-27" / method / algorithm / "ARMS" / "000.png"
        )
        expected_meta = (
            tmp_path / "run-2026-09-27" / method / algorithm / "ARMS" / "000.json"
        )
        assert paths.image_path == expected_image
        assert paths.metadata_path == expected_meta

    def test_ablation_variant_uses_separated_namespace(self, tmp_path: Path) -> None:
        paths = sample_paths(
            tmp_path,
            "ablation-run-1",
            "classical_baseline",
            "telea",
            "ARMS/000",
            ablation=True,
        )
        assert "ablation" in paths.image_path.parts
        assert paths.image_path == (
            tmp_path
            / "ablation"
            / "ablation-run-1"
            / "classical_baseline"
            / "telea"
            / "ARMS"
            / "000.png"
        )

    def test_rejects_unknown_method(self, tmp_path: Path) -> None:
        with pytest.raises(RunError, match="method"):
            sample_paths(tmp_path, "run-1", "not_a_real_method", "telea", "ARMS/000")

    def test_rejects_unknown_algorithm(self, tmp_path: Path) -> None:
        with pytest.raises(RunError, match="algorithm"):
            sample_paths(
                tmp_path,
                "run-1",
                "classical_baseline",
                "not_a_real_algorithm",
                "ARMS/000",
            )


class TestRunScaffoldingAndNonOverwrite:
    """FR-027, SC-006: a run ID never silently overwrites a previous run."""

    def test_creates_run_tree(self, tmp_path: Path) -> None:
        run_dir = create_run(tmp_path, "run-A")
        assert run_dir.is_dir()
        assert run_dir == tmp_path / "run-A"

    def test_reusing_run_id_is_refused_without_overwrite(self, tmp_path: Path) -> None:
        create_run(tmp_path, "run-A")
        with pytest.raises(RunError, match="run-A"):
            create_run(tmp_path, "run-A")

    def test_explicit_overwrite_is_honoured(self, tmp_path: Path) -> None:
        run_dir = create_run(tmp_path, "run-A")
        (run_dir / "marker.txt").write_text("first")
        create_run(tmp_path, "run-A", overwrite=True)
        # Overwrite is explicit permission to recreate the run — the directory
        # exists again regardless of what create_run does with old contents,
        # but it must not raise.
        assert run_dir.is_dir()

    def test_two_run_ids_coexist_untouched(self, tmp_path: Path) -> None:
        run_a = create_run(tmp_path, "run-A")
        run_b = create_run(tmp_path, "run-B")
        (run_a / "marker.txt").write_text("A")
        (run_b / "marker.txt").write_text("B")

        assert (run_a / "marker.txt").read_text() == "A"
        assert (run_b / "marker.txt").read_text() == "B"


REQUIRED_METADATA_KEYS = {
    "image_id",
    "method",
    "algorithm",
    "model_repository",
    "experiment_id",
    "input_image_path",
    "prediction_mask_path",
    "mask_size",
    "alignment",
    "dilation",
    "inpaint_algorithm",
    "inpaint_radius",
    "output_format",
    "processing_time_seconds",
    "status",
}


def _metadata(**overrides) -> SampleMetadata:
    fields = dict(
        image_id="ARMS/000",
        method="classical_baseline",
        algorithm="telea",
        model_repository=None,
        experiment_id="default",
        input_image_path="data/raw/ARMS/000.jpg",
        prediction_mask_path="outputs/segmentation/default/adaptive/masks/ARMS/000.png",
        mask_size=(1654, 1170),
        alignment=AlignmentRecord("pass-through", (1654, 1170), (1654, 1170)),
        dilation=DilationConfig(True, "ellipse", (3, 3), 1),
        inpaint_algorithm="INPAINT_TELEA",
        inpaint_radius=3.0,
        output_format="png",
        processing_time_seconds=0.12,
        status="ok",
    )
    fields.update(overrides)
    return SampleMetadata(**fields)


class TestSampleMetadataWriter:
    """FR-025: the sidecar carries exactly the contract's field list."""

    def test_writes_exactly_the_contract_fields(self, tmp_path: Path) -> None:
        path = tmp_path / "ARMS" / "000.json"
        write_sample_metadata(path, _metadata())

        doc = json.loads(path.read_text(encoding="utf-8"))
        assert set(doc) == REQUIRED_METADATA_KEYS | {"error_message"}
        assert doc["error_message"] is None
        assert doc["alignment"] == {
            "rule": "pass-through",
            "pre_size": [1654, 1170],
            "post_size": [1654, 1170],
        }
        assert doc["dilation"]["kernel_size"] == [3, 3]

    def test_disabled_dilation_is_recorded_as_disabled(self, tmp_path: Path) -> None:
        path = tmp_path / "000.json"
        write_sample_metadata(path, _metadata(dilation=DilationConfig(False)))
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["dilation"]["enabled"] is False

    def test_output_is_byte_identical_across_writes(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        write_sample_metadata(a, _metadata())
        write_sample_metadata(b, _metadata())
        assert a.read_bytes() == b.read_bytes()

    def test_failed_status_requires_error_message(self) -> None:
        with pytest.raises(RunError, match="error_message"):
            _metadata(status="failed")

    def test_enabled_dilation_requires_kernel_fields(self) -> None:
        with pytest.raises(RunError, match="dilation"):
            _metadata(dilation=DilationConfig(True))

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(RunError, match="method"):
            _metadata(method="not_a_method")

    def test_rejects_malformed_image_id(self) -> None:
        with pytest.raises(RunError, match="image_id"):
            _metadata(image_id="ARMS-000")

    def test_rejects_inconsistent_algorithm_pair(self) -> None:
        with pytest.raises(RunError, match="algorithm"):
            _metadata(algorithm="telea", inpaint_algorithm="INPAINT_NS")


class TestErrorReport:
    """FR-011, FR-012: every failure is recorded by category; none stops the batch."""

    def test_has_the_nine_fr012_categories_plus_missing_metadata(self) -> None:
        assert ERROR_CATEGORIES[:9] == (
            "missing_source_image",
            "missing_prediction_mask",
            "image_id_mismatch",
            "wrong_size",
            "empty_mask",
            "non_binary_mask",
            "corrupt_image",
            "inpainting_failure",
            "output_write_failure",
        )
        assert "missing_metadata" in ERROR_CATEGORIES

    def test_records_entry_with_identifier_category_reason(self) -> None:
        report = ErrorReport()
        report.add("ARMS/000", "wrong_size", "mask is 100x100")
        assert len(report) == 1

    def test_unknown_category_is_refused(self) -> None:
        with pytest.raises(RunError, match="category"):
            ErrorReport().add("ARMS/000", "made_up", "x")

    def test_written_report_lists_every_category_even_when_empty(
        self, tmp_path: Path
    ) -> None:
        report = ErrorReport()
        report.add("ARMS/000", "empty_mask", "no text pixel")
        path = tmp_path / "errors.json"
        report.write(path)

        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["total"] == 1
        assert set(doc["by_category"]) == set(ERROR_CATEGORIES)
        assert doc["by_category"]["empty_mask"]["count"] == 1
        assert doc["by_category"]["wrong_size"] == {"count": 0, "entries": []}
        assert doc["by_category"]["empty_mask"]["entries"] == [
            {
                "image_id": "ARMS/000",
                "category": "empty_mask",
                "reason": "no text pixel",
                "method": None,
            }
        ]

    def test_output_does_not_depend_on_insertion_order(self, tmp_path: Path) -> None:
        first, second = ErrorReport(), ErrorReport()
        first.add("ARMS/000", "wrong_size", "a")
        first.add("ARMS/001", "wrong_size", "b")
        second.add("ARMS/001", "wrong_size", "b")
        second.add("ARMS/000", "wrong_size", "a")
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        first.write(a)
        second.write(b)
        assert a.read_bytes() == b.read_bytes()

    def test_empty_report_is_still_written(self, tmp_path: Path) -> None:
        path = tmp_path / "errors.json"
        ErrorReport().write(path)
        assert json.loads(path.read_text(encoding="utf-8"))["total"] == 0


PAGE_LIST_IDENTITY = {"sha256": "abc123", "pages": 390}
INPUT_IMAGE_IDENTITY = {"ARMS/000": {"sha256": "def456", "size": [1654, 1170]}}


def _run(**overrides) -> InpaintRun:
    fields = dict(
        run_id="run-A",
        method_sources={
            "classical_baseline": MethodSource(
                "outputs/segmentation/default/adaptive/",
                "adaptive",
                "highest IoU of the six Spec 1 methods (research R1)",
            ),
            "comic_text_detector": MethodSource(
                "deliverables/drill/comic_text_detector/", "drill"
            ),
        },
        mask_processing_config=MaskProcessingConfig(
            DilationConfig(True, "ellipse", (3, 3), 1)
        ),
        inpainting_config=InpaintingConfig(3.0, ("telea", "ns"), "png"),
        page_list_identity=PAGE_LIST_IDENTITY,
        input_image_identity=INPUT_IMAGE_IDENTITY,
        counts={"classical_baseline": {"telea": SampleCounts(2, 1, 1)}},
    )
    fields.update(overrides)
    return InpaintRun(**fields)


class TestRunRecord:
    """FR-027: run.json records what the run consumed and how it was configured."""

    def test_writes_run_json_in_the_run_directory(self, tmp_path: Path) -> None:
        path = write_run_record(tmp_path, _run())
        assert path == tmp_path / "run.json"
        assert path.is_file()

    def test_identities_are_quoted_verbatim(self, tmp_path: Path) -> None:
        doc = json.loads(write_run_record(tmp_path, _run()).read_text(encoding="utf-8"))
        assert doc["page_list_identity"] == PAGE_LIST_IDENTITY
        assert doc["input_image_identity"] == INPUT_IMAGE_IDENTITY

    def test_methods_are_listed_in_the_fixed_fr026_order(self, tmp_path: Path) -> None:
        doc = json.loads(write_run_record(tmp_path, _run()).read_text(encoding="utf-8"))
        # comic_text_detector was declared after classical_baseline, and the
        # fixed order puts it there regardless of dict insertion order.
        assert doc["methods"] == ["classical_baseline", "comic_text_detector"]
        assert list(doc["method_sources"]) == doc["methods"]

    def test_records_the_resolved_classical_method_and_why(
        self, tmp_path: Path
    ) -> None:
        doc = json.loads(write_run_record(tmp_path, _run()).read_text(encoding="utf-8"))
        source = doc["method_sources"]["classical_baseline"]
        assert source["resolved_identity"] == "adaptive"
        assert source["reason"]

    def test_records_both_configs_once(self, tmp_path: Path) -> None:
        doc = json.loads(write_run_record(tmp_path, _run()).read_text(encoding="utf-8"))
        assert doc["mask_processing_config"]["dilation"]["kernel_size"] == [3, 3]
        assert doc["inpainting_config"]["algorithms"] == ["telea", "ns"]
        assert doc["inpainting_config"]["radius"] == 3.0

    def test_records_counts_and_ablation_flag(self, tmp_path: Path) -> None:
        doc = json.loads(
            write_run_record(tmp_path, _run(ablation=True)).read_text(encoding="utf-8")
        )
        assert doc["counts"]["classical_baseline"]["telea"] == {
            "attempted": 2,
            "succeeded": 1,
            "failed": 1,
        }
        assert doc["ablation"] is True

    def test_carries_no_timestamp(self, tmp_path: Path) -> None:
        doc = json.loads(write_run_record(tmp_path, _run()).read_text(encoding="utf-8"))
        assert not any("time" in key or "date" in key for key in doc)

    def test_output_is_byte_identical_across_writes(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a", tmp_path / "b"
        pa, pb = write_run_record(a, _run()), write_run_record(b, _run())
        assert pa.read_bytes() == pb.read_bytes()

    def test_rejects_unknown_method(self) -> None:
        with pytest.raises(RunError, match="method"):
            _run(method_sources={"not_a_method": MethodSource("x", "x")}, counts={})

    def test_rejects_counts_for_a_method_not_processed(self) -> None:
        with pytest.raises(RunError, match="counts"):
            _run(counts={"unetpp_efficientnetv2": {"telea": SampleCounts(1, 1, 0)}})

    def test_rejects_inconsistent_counts(self) -> None:
        with pytest.raises(RunError, match="attempted"):
            SampleCounts(attempted=3, succeeded=1, failed=1)


def test_error_entry_records_the_method():
    report = ErrorReport()
    report.add("ARMS/000", "wrong_size", "1000x1000", method="comic_text_detector")

    entry = report.to_dict()["by_category"]["wrong_size"]["entries"][0]
    assert entry["method"] == "comic_text_detector"


def test_error_entry_method_defaults_to_none():
    report = ErrorReport()
    report.add("ARMS/000", "wrong_size", "1000x1000")

    entry = report.to_dict()["by_category"]["wrong_size"]["entries"][0]
    assert entry["method"] is None


def test_error_report_refuses_an_unknown_method():
    with pytest.raises(RunError):
        ErrorReport().add("ARMS/000", "wrong_size", "x", method="made_up")


def test_same_image_failing_under_two_methods_keeps_both_entries():
    report = ErrorReport()
    report.add("ARMS/000", "empty_mask", "r", method="unetpp_efficientnetv2")
    report.add("ARMS/000", "empty_mask", "r", method="classical_baseline")

    block = report.to_dict()["by_category"]["empty_mask"]
    assert block["count"] == 2
    assert [e["method"] for e in block["entries"]] == [
        "classical_baseline",
        "unetpp_efficientnetv2",
    ]


def test_add_rejection_records_a_mask_rejection():
    report = ErrorReport()
    rejection = MaskRejection("non_binary_mask", "value 128", "ARMS/002")

    report.add_rejection(rejection, method="classical_baseline")

    entry = report.to_dict()["by_category"]["non_binary_mask"]["entries"][0]
    assert entry == {
        "image_id": "ARMS/002",
        "category": "non_binary_mask",
        "reason": "value 128",
        "method": "classical_baseline",
    }


def test_add_rejection_refuses_one_without_an_image_id():
    with pytest.raises(RunError):
        ErrorReport().add_rejection(
            MaskRejection("empty_mask", "r"), method="classical_baseline"
        )


def test_rejections_do_not_stop_the_batch(tmp_path):
    """The caller records a rejection and carries on (FR-011)."""
    report = ErrorReport()
    outcomes = []
    for image_id in ("ARMS/000", "ARMS/001", "ARMS/002"):
        try:
            if image_id == "ARMS/001":
                raise MaskRejection("empty_mask", "no text pixel", image_id)
            outcomes.append(image_id)
        except MaskRejection as rejection:
            report.add_rejection(rejection, method="classical_baseline")

    assert outcomes == ["ARMS/000", "ARMS/002"]
    assert len(report) == 1
    report.write(tmp_path / "errors.json")
    assert json.loads((tmp_path / "errors.json").read_text())["total"] == 1
