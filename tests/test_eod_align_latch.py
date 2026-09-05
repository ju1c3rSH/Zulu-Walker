# -*- coding: utf-8 -*-
"""EodAlignProcessor latch semantics — the §3 edge rules as executable tests."""
import cv2
import numpy as np
import pytest

from modules.zw_opencv_module.processors.eod_align_processor import EodAlignProcessor
from modules.zw_uart_module.protocol import (
    DEV_FLAG_FOUND, DEV_FLAG_MATCH, DEV_FLAG_READY,
)

pytest.importorskip("cv2")

W, H = 320, 320


def red_frame(x=W // 2 - 30, y=200):
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[y:y + 60, x:x + 60] = (60, 60, 200)
    return frame


def blank_frame():
    return np.zeros((H, W, 3), dtype=np.uint8)


def make_processor(**cfg_overrides):
    """Processor with injected config + a manually-advanced fake clock."""
    cfg = {
        "colors": [
            {"id": 1, "name": "red", "lab": [0, 80, 40, 80, 10, 80]},
            {"id": 2, "name": "green", "lab": [0, 80, -120, -10, 0, 30]},
            {"id": 3, "name": "blue", "lab": [0, 80, 30, 100, -120, -60]},
        ],
        "hostage_colors": [],
        "bucket": {"id": 1, "name": "black", "lab": [0, 45, -128, 128, -128, 128]},
        "home_colors": [],
        "blob": {"min_area_px": 100, "max_area_px": 300000,
                 "edge_margin_px": 4, "morph_open": 3, "morph_close": 7},
        "zones": {"grab": [0.35, 0.45, 0.65, 0.85], "place": [0.35, 0.40, 0.65, 0.85]},
        "latch": {"frames": 3, "lost_hold_ms": 300},
        "shape": {"waist_ratio_max": 0.88, "cone_small_end_max": 0.70,
                  "cone_slope_min": 0.30, "cylinder_end_min": 0.82,
                  "min_confidence": 0.25},
        "target": {"ref_target_width_px": 200,
                   "ring_radii_px": {"r10": 220, "r9": 150, "r8": 85},
                   "hit_detect_enabled": False,
                   "lock_frames": 5, "hit_lost_frames": 4, "hit_area_ratio": 0.45},
    }
    for key, value in cfg_overrides.items():
        cfg[key].update(value)
    clock = {"t": 100.0}
    proc = EodAlignProcessor(name="eod_align", config=cfg, now_fn=lambda: clock["t"])
    return proc, clock


def test_latch_after_n_frames_and_hold_through_dropout():
    proc, clock = make_processor()
    proc.set_mode("BOMB", expected_id=1)

    for _ in range(2):   # below latch threshold
        result = proc.process(red_frame())
        assert not result.result_data["target_found"]
    result = proc.process(red_frame())   # 3rd frame -> latch
    assert result.result_data["target_found"]
    assert result.result_data["flags"] & DEV_FLAG_FOUND
    assert result.result_data["match"]

    # brief dropout: latch must hold (短暂丢失不掉锁), error frozen
    clock["t"] += 0.033
    result = proc.process(blank_frame())
    assert result.result_data["target_found"]
    assert result.result_data["raw_found"] is False
    assert result.result_data["flags"] & DEV_FLAG_FOUND


def test_long_dropout_clears_found_latch_and_zeroes_error():
    proc, clock = make_processor()
    proc.set_mode("BOMB", expected_id=1)
    for _ in range(5):
        proc.process(red_frame())
    clock["t"] += 0.4   # > lost_hold_ms
    result = proc.process(blank_frame())
    data = result.result_data
    assert not data["target_found"]
    assert not (data["flags"] & DEV_FLAG_FOUND)
    assert data["error_x"] == 0 and data["error_y"] == 0 and data["id"] == 0


def test_set_mode_resets_all_latches():
    proc, clock = make_processor()
    proc.set_mode("BOMB", expected_id=1)
    for _ in range(5):
        proc.process(red_frame())
    assert proc.process(red_frame()).result_data["flags"] & DEV_FLAG_FOUND

    proc.set_mode("BOMB", expected_id=1)   # new subscription cycle -> 清锁
    result = proc.process(red_frame())
    assert not result.result_data["target_found"]
    assert result.result_data["flags"] == 0


def test_expected_color_mismatch_blocks_match_flag():
    proc, clock = make_processor()
    proc.set_mode("BOMB", expected_id=2)   # expecting GREEN but seeing RED
    for _ in range(6):
        result = proc.process(red_frame())
    data = result.result_data
    assert data["target_found"]                       # found latch independent
    assert not (data["flags"] & DEV_FLAG_MATCH)
    assert data["id"] == 1                            # reports what it sees


def test_bomb_ready_requires_zone_and_match():
    proc, clock = make_processor()
    proc.set_mode("BOMB", expected_id=1)
    # blob bottom edge y=260/320=0.81 inside grab zone 0.45..0.85
    for _ in range(6):
        result = proc.process(red_frame())
    data = result.result_data
    assert data["flags"] & DEV_FLAG_READY


def test_hostage_mode_classifies_shape():
    proc, clock = make_processor()
    proc.set_mode("HOSTAGE", expected_id=1)   # cylinder expected
    # tall red rectangle = frontal cylinder
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[80:260, 130:190] = (60, 60, 200)
    for _ in range(6):
        result = proc.process(frame)
    data = result.result_data
    assert data["target_found"]
    assert data["shape_id"] == 1
    assert data["flags"] & DEV_FLAG_MATCH


def test_target_mode_ring_and_hit_disabled_by_default():
    proc, clock = make_processor()
    proc.set_mode("TARGET", expected_id=1)
    # target blob centered -> boreline distance small -> ring 10 by config radii
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[130:190, 130:190] = (60, 60, 200)   # 60px wide at center
    for _ in range(6):
        result = proc.process(frame)
    data = result.result_data
    assert data["target_found"]
    assert data["ring_id"] == 10
    assert not data["hit"], "hit heuristic must be OFF until enabled"
