"""PNG output smoke test for the five-panel board renderer."""

import cv2

from manga_text_seg.boards import write_board
from tests.unit.test_boards import FIXTURE_SAMPLES


def test_write_board_creates_labelled_five_column_png(tmp_path):
    destination = write_board(FIXTURE_SAMPLES[0], tmp_path / "board.png")
    image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    assert image is not None
    assert image.shape[1] == 5 * 420
    assert image.shape[0] > 0
