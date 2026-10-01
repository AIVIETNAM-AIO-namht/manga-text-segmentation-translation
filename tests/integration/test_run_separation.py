"""T033: run IDs have separate trees and overwrite is explicit."""

import pytest

from manga_text_seg.runs import RunError, create_run


def test_runs_coexist_and_refusal_preserves_the_previous_run(tmp_path):
    first = create_run(tmp_path, "first")
    marker = first / "completed.json"
    marker.write_text('{"complete":true}', encoding="utf-8")
    second = create_run(tmp_path, "second")
    second_marker = second / "sample.png"
    second_marker.write_bytes(b"second run")

    with pytest.raises(RunError):
        create_run(tmp_path, "first")
    assert marker.read_text(encoding="utf-8") == '{"complete":true}'
    assert second_marker.read_bytes() == b"second run"

    replaced = create_run(tmp_path, "first", overwrite=True)
    assert replaced == first
    assert list(replaced.iterdir()) == []
    assert second_marker.read_bytes() == b"second run"
