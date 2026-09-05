# -*- coding: utf-8 -*-
"""EodColorDetector tests on synthetic frames."""
import numpy as np
import pytest

from modules.zw_opencv_module.detectors.eod_color import (
    EodColorDetector, normalize_error,
)

pytest.importorskip("cv2")

W, H = 320, 320

RED_CAND = {"id": 1, "name": "red", "lab": [0, 80, 40, 80, 10, 80]}
GREEN_CAND = {"id": 2, "name": "green", "lab": [0, 90, -120, -10, 0, 80]}


def make_frame(blobs):
    """blobs: list of (bgr, x, y, w, h)."""
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    for bgr, x, y, w, h in blobs:
        frame[y:y + h, x:x + w] = bgr
    return frame


def test_error_normalization_signs():
    # LAB red ~ (60, 90, 70)
    frame = make_frame([((60, 60, 200), 220, 200, 60, 60)])   # right-bottom quadrant
    det = EodColorDetector([RED_CAND], min_area_px=100)
    result = det.detect(frame)
    assert result.target_found
    assert result.error_x > 0, "blob right of center must give positive error_x"
    assert result.error_y > 0, "blob below center must give positive error_y"


def test_centered_blob_zero_error():
    frame = make_frame([((60, 60, 200), W // 2 - 30, H // 2 - 30, 60, 60)])
    det = EodColorDetector([RED_CAND], min_area_px=100)
    result = det.detect(frame)
    assert result.target_found
    assert abs(result.error_x) < 50 and abs(result.error_y) < 50


def test_expected_color_priority():
    frame = make_frame([
        ((60, 60, 200), 20, 20, 100, 100),    # red, big
        ((60, 200, 60), 200, 200, 40, 40),    # green, small
    ])
    det = EodColorDetector([RED_CAND, GREEN_CAND], min_area_px=100)
    result = det.detect(frame, expected_id=2)
    assert result.target_found and result.dev_id == 2, "expected color must win over larger blob"
    assert result.center[0] > 150


def test_grab_zone_ready_bottom_anchor():
    # blob bottom edge inside the default grab zone (0.35..0.65, 0.45..0.85)
    frame = make_frame([((60, 60, 200), W // 2 - 30, 200, 60, 60)])   # bottom edge y=260 -> 0.81
    det = EodColorDetector([RED_CAND], min_area_px=100,
                           grab_zone=(0.35, 0.45, 0.65, 0.85))
    result = det.detect(frame, zone_anchor="bottom")
    assert result.ready

    frame_high = make_frame([((60, 60, 200), W // 2 - 30, 100, 60, 60)])  # bottom y=160 -> 0.5 inside!
    # y=100..160 bottom=160/320=0.5 -> inside zone (0.45..0.85): still ready
    result2 = det.detect(frame_high, zone_anchor="bottom")
    assert result2.ready


def test_edge_touching_blob_rejected():
    frame = make_frame([((60, 60, 200), 0, 0, 100, 100)])   # touches top-left corner
    det = EodColorDetector([RED_CAND], min_area_px=100, edge_margin_px=4)
    result = det.detect(frame)
    assert not result.target_found


def test_normalize_error_clamps():
    # right edge of a 320-wide frame -> +5000, left edge -> -5000
    assert normalize_error(320, 160, 160) == 5000
    assert normalize_error(0, 160, 160) == -5000
    assert normalize_error(480, 160, 160) == 5000        # clamped
    assert normalize_error(-160, 160, 160) == -5000      # clamped
    assert normalize_error(160, 160, 160) == 0
    assert normalize_error(160.0, 160.0, 0) == 0         # degenerate half-width


def test_empty_frame_returns_not_found():
    det = EodColorDetector([RED_CAND], min_area_px=100)
    result = det.detect(np.zeros((H, W, 3), dtype=np.uint8))
    assert not result.target_found
    assert result.error_x == 0 and result.error_y == 0
    result = det.detect(None)
    assert not result.target_found
