"""A deviation that changes the output is a different method (T029, FR-055).

``contracts/returned-result.md`` rule 7: *"Deviations are recorded, not
absorbed.  A hand-off whose provenance declares a deviation that
``changes_output`` is a **different method** and is not admissible as the pinned
one (FR-055)."*

The rule is conditional, not a completeness rule.  ``deviation`` is an optional
property of ``contracts/provenance.schema.json`` — its absence is a conforming
record, and a workaround that does *not* change the mask is admissible, recorded
as a deviation rather than hidden.  Only ``applied`` together with
``changes_output`` makes the result inadmissible as the pinned method, which is
why this is not an entry in ``provenance.REQUIRED_KEYS``: adding one there would
refuse every delivered hand-off, none of which carries the key at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from manga_text_seg.provenance import deviation_refusal
from manga_text_seg.receipt import ReceiptError, admit, validate_handoff

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = (
    REPO_ROOT
    / "specs"
    / "002-deep-learning-segmentation-benchmark"
    / "contracts"
    / "provenance.schema.json"
)


def _with_deviation(provenance: dict, **deviation) -> dict:
    """A copy of ``provenance`` carrying a deviation — the fixture has none."""
    return {**provenance, "deviation": deviation}


@pytest.fixture()
def page_list(exported) -> dict:
    return exported["page_list"]


def _masks(page_list: dict, aligned_mask: np.ndarray) -> dict[str, np.ndarray]:
    return {page["image_id"]: aligned_mask.copy() for page in page_list["pages"]}


def _admit(exported, handoff: Path, run_dir: Path):
    return admit(
        handoff,
        page_list=exported["page_list"],
        identity_record=exported["identity_record"],
        run_dir=run_dir,
        manifest=exported["manifest"],
    )


class TestTheSchemaLeavesItOptional:
    """The premise the conditional rule rests on, read out of the contract."""

    def test_deviation_is_not_a_required_property(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert "deviation" in schema["properties"]
        assert "deviation" not in schema["required"]

    def test_changes_output_defaults_to_false(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        assert schema["properties"]["deviation"]["properties"]["changes_output"]["default"] is False


class TestTheRule:
    """``deviation_refusal`` returns a reason, or ``None`` when admissible."""

    def test_a_record_with_no_deviation_is_admissible(self, default_provenance):
        assert deviation_refusal(default_provenance) is None

    def test_a_declared_non_deviation_is_admissible(self, default_provenance):
        record = _with_deviation(default_provenance, applied=False)
        assert deviation_refusal(record) is None

    def test_a_workaround_that_does_not_change_the_output_is_admissible(
        self, default_provenance
    ):
        """Recorded, not absorbed — but still the pinned method (rule 7)."""
        record = _with_deviation(
            default_provenance,
            applied=True,
            what="pinned numpy<2 in the runner's environment",
            why="an upstream dtype deprecation",
            changes_output=False,
        )
        assert deviation_refusal(record) is None

    def test_changes_output_absent_means_false(self, default_provenance):
        record = _with_deviation(
            default_provenance, applied=True, what="a version pin"
        )
        assert deviation_refusal(record) is None

    def test_a_deviation_that_changes_the_output_is_inadmissible(self, default_provenance):
        record = _with_deviation(
            default_provenance,
            applied=True,
            what="re-exported the checkpoint through a different serialiser",
            why="a torch version the pinned pickle will not load under",
            changes_output=True,
        )

        reason = deviation_refusal(record)

        assert reason is not None
        # The refusal says *what* was changed, so the deviation is named rather
        # than absorbed, and cites the requirement that makes it inadmissible.
        assert "re-exported the checkpoint" in reason
        assert "FR-055" in reason

    def test_changes_output_true_without_applied_is_still_inadmissible(
        self, default_provenance
    ):
        """``changes_output: true`` with ``applied`` false is a contradiction.

        The conservative reading is the one that does not admit a method whose
        output may have been altered on the strength of a bookkeeping field.
        """
        record = _with_deviation(
            default_provenance, applied=False, changes_output=True
        )
        assert deviation_refusal(record) is not None

    def test_a_non_object_deviation_is_refused_not_crashed_on(self, default_provenance):
        record = {**default_provenance, "deviation": "a workaround, honest"}
        assert deviation_refusal(record) is not None


class TestAdmission:
    """The rule is enforced at the door, not only available as a function."""

    def test_a_handoff_carrying_an_output_changing_deviation_is_refused(
        self, exported, page_list, aligned_mask, make_handoff, default_provenance, tmp_path
    ):
        run_dir = tmp_path / "deliverables" / "testrun"
        provenance = _with_deviation(
            default_provenance,
            applied=True,
            what="a different sigmoid cutoff, 0.35 instead of 0.5",
            why="the pinned threshold under-segments thin strokes",
            changes_output=True,
        )
        handoff = make_handoff(
            page_list, _masks(page_list, aligned_mask), provenance=provenance
        )

        with pytest.raises(ReceiptError) as caught:
            _admit(exported, handoff, run_dir)

        assert caught.value.check == "deviation"
        assert "FR-055" in caught.value.reason
        assert not run_dir.exists(), "a refusal must write nothing"

    def test_the_same_handoff_without_the_deviation_is_admitted(
        self, exported, page_list, aligned_mask, make_handoff, default_provenance, tmp_path
    ):
        """The refusal is the deviation's doing, not something else in the fixture."""
        run_dir = tmp_path / "deliverables" / "testrun"
        handoff = make_handoff(page_list, _masks(page_list, aligned_mask))

        _admit(exported, handoff, run_dir)

        assert (run_dir / "standin" / "provenance.json").is_file()

    def test_a_workaround_that_leaves_the_output_alone_still_admits(
        self, exported, page_list, aligned_mask, make_handoff, default_provenance, tmp_path
    ):
        run_dir = tmp_path / "deliverables" / "testrun"
        provenance = _with_deviation(
            default_provenance,
            applied=True,
            what="pinned the runner's interpreter to 3.10",
            why="an upstream typing import",
            changes_output=False,
        )
        handoff = make_handoff(
            page_list, _masks(page_list, aligned_mask), provenance=provenance
        )

        _admit(exported, handoff, run_dir)

        stored = json.loads(
            (run_dir / "standin" / "provenance.json").read_text(encoding="utf-8")
        )
        # Recorded, not absorbed: the deviation survives into the run.
        assert stored["deviation"]["applied"] is True

    def test_the_check_runs_before_anything_is_written(
        self, exported, page_list, aligned_mask, make_handoff, default_provenance
    ):
        """``validate_handoff`` writes nothing and refuses the deviation."""
        provenance = _with_deviation(
            default_provenance, applied=True, changes_output=True
        )
        handoff = make_handoff(
            page_list, _masks(page_list, aligned_mask), provenance=provenance
        )

        with pytest.raises(ReceiptError) as caught:
            validate_handoff(
                handoff,
                page_list=page_list,
                identity_record=exported["identity_record"],
            )

        assert caught.value.check == "deviation"
