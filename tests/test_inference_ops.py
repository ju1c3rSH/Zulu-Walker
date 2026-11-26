"""Unit tests for the generic inference primitives.

Run: python -m pytest tests/test_inference_ops.py -q
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from framework.inference import ops
from framework.inference.contract import GeometryTransform

REPO_ROOT = Path(__file__).resolve().parents[1]
OPS_PATH = REPO_ROOT / "framework" / "inference" / "ops.py"

LABELS = ["ball", "cube"]


def _letterbox_transform() -> GeometryTransform:
    """Geometry of a 1280x352 source letterboxed into a 640x352 input."""
    return GeometryTransform(
        mode="letterbox",
        scale_x=0.5,
        scale_y=0.5,
        offset_x=0,
        offset_y=88,
        src_w=1280,
        src_h=352,
    )


def _prediction(anchors: list[tuple[float, ...]], layout: str) -> np.ndarray:
    """Raw prediction of the given anchors in the requested layout."""
    data = np.asarray(anchors, dtype=np.float32)
    if layout == "channels_first":
        return data.T.reshape(1, data.shape[1], data.shape[0])
    return data.reshape(1, data.shape[0], data.shape[1])


ANCHORS = [
    (320.0, 100.0, 40.0, 20.0, 0.9, 0.1),
    (100.0, 60.0, 20.0, 20.0, 0.2, 0.05),
    (200.0, 200.0, 30.0, 30.0, 0.1, 0.8),
]


def test_letterbox_reports_the_scale_and_the_padding():
    source = np.zeros((352, 1280, 3), np.uint8)
    image, transform = ops.resize_frame(source, (640, 352), mode="letterbox", value=114)
    assert image.shape == (352, 640, 3)
    assert transform.mode == "letterbox"
    assert transform.scale_x == pytest.approx(0.5)
    assert transform.scale_y == pytest.approx(0.5)
    assert transform.offset_x == 0
    assert transform.offset_y == 88
    assert transform.src_w == 1280
    assert transform.src_h == 352


def test_letterbox_pads_with_the_configured_value():
    source = np.full((352, 1280, 3), 200, np.uint8)
    image, _ = ops.resize_frame(source, (640, 352), mode="letterbox", value=114)
    assert image[0, 0, 0] == 114
    assert image[200, 0, 0] == 200


def test_letterbox_box_round_trips_back_to_source_pixels():
    source = np.zeros((352, 1280, 3), np.uint8)
    _, transform = ops.resize_frame(source, (640, 352), mode="letterbox")
    src_x, src_y, src_w, src_h = 100.0, 200.0, 50.0, 30.0
    input_x = src_x * transform.scale_x + transform.offset_x
    input_y = src_y * transform.scale_y + transform.offset_y
    input_w = src_w * transform.scale_x
    input_h = src_h * transform.scale_y
    back_x = (input_x - transform.offset_x) / transform.scale_x
    back_y = (input_y - transform.offset_y) / transform.scale_y
    assert back_x == pytest.approx(src_x)
    assert back_y == pytest.approx(src_y)
    assert input_w / transform.scale_x == pytest.approx(src_w)
    assert input_h / transform.scale_y == pytest.approx(src_h)


def test_crop_anchors_take_different_parts_of_the_frame():
    source = np.zeros((352, 1280, 3), np.uint8)
    source[:, :100] = 255
    left, left_transform = ops.resize_frame(source, (640, 352), mode="crop", anchor="left")
    right, right_transform = ops.resize_frame(source, (640, 352), mode="crop", anchor="right")
    center, center_transform = ops.resize_frame(source, (640, 352), mode="crop", anchor="center")
    assert not np.array_equal(left, right)
    assert not np.array_equal(left, center)
    assert left_transform.offset_x == 0
    assert center_transform.offset_x == -320
    assert right_transform.offset_x == -640
    assert left[:, 0].max() == 255
    assert right[:, 0].max() == 0


def test_crop_offsets_map_back_to_the_source_frame():
    source = np.zeros((352, 1280, 3), np.uint8)
    _, transform = ops.resize_frame(source, (640, 352), mode="crop", anchor="right")
    input_x = 60.0
    src_x = (input_x - transform.offset_x) / transform.scale_x
    assert src_x == pytest.approx(700.0)


def test_stretch_scales_the_two_axes_independently():
    source = np.zeros((352, 1280, 3), np.uint8)
    source[:, 600:620] = 255
    image, transform = ops.resize_frame(source, (640, 352), mode="stretch")
    assert transform.mode == "stretch"
    assert transform.scale_x == pytest.approx(0.5)
    assert transform.scale_y == pytest.approx(1.0)
    assert transform.offset_x == 0
    assert transform.offset_y == 0
    assert image.shape == (352, 640, 3)
    assert image[0, 300, 0] == 255
    assert image[0, 100, 0] == 0


def test_an_unknown_resize_mode_is_rejected():
    source = np.zeros((352, 1280, 3), np.uint8)
    with pytest.raises(ValueError):
        ops.resize_frame(source, (640, 352), mode="rotate")


def test_normalize_yolo_matches_the_hand_computed_reference():
    image = np.array([[[10, 20, 30], [40, 50, 60]]], np.uint8)
    scale = 1.0 / 255.0
    mean = [0.5, 0.25, 0.0]
    std = [0.5, 0.25, 1.0]
    result = ops.normalize_yolo(image, scale, mean, std)
    expected = (
        image.astype(np.float32) * scale - np.asarray(mean, np.float32)
    ) / np.asarray(std, np.float32)
    assert result.dtype == np.float32
    assert np.allclose(result, expected)
    assert result[0, 0, 0] == pytest.approx((10 / 255.0 - 0.5) / 0.5)


def test_engine_norm_params_are_the_yolo_params_divided_by_scale():
    scale = 1.0 / 255.0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    mean_e, std_e = ops.convert_norm_params(scale, mean, std)
    assert np.allclose(mean_e, np.asarray(mean, np.float32) / scale)
    assert np.allclose(std_e, np.asarray(std, np.float32) / scale)


def test_the_engine_conversion_is_equivalent_to_the_yolo_formula():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    scale = 1.0 / 255.0
    mean = [0.5, 0.25, 0.0]
    std = [0.25, 0.25, 1.0]
    mean_e, std_e = ops.convert_norm_params(scale, mean, std)
    assert np.allclose(
        ops.normalize_engine(image, mean_e, std_e),
        ops.normalize_yolo(image, scale, mean, std),
    )


def test_iou_matrix_is_computed_over_the_box_pairs():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], np.float32)
    iou = ops.iou_matrix(a, b)
    assert iou.shape == (1, 2)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[0, 1] == pytest.approx(25.0 / 175.0)


def test_nms_suppresses_the_overlapping_box_and_keeps_the_disjoint_one():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = ops.nms(boxes, scores, 0.45)
    assert keep.tolist() == [0, 2]


def test_nms_is_class_aware_when_class_ids_are_given():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = ops.nms(boxes, scores, 0.45, class_ids=np.array([0, 1, 0]))
    assert keep.tolist() == [0, 1, 2]


def test_nms_writes_the_kept_indices_into_the_output_buffer():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    buffer = np.zeros(3, np.int64)
    keep = ops.nms(boxes, scores, 0.45, out=buffer)
    assert keep.tolist() == [0, 2]
    assert buffer[0] == 0
    assert buffer[1] == 2


def test_softmax_normalises_the_last_axis():
    scores = np.array([[1.0, 2.0, 3.0]], np.float32)
    result = ops.softmax(scores)
    assert result.sum() == pytest.approx(1.0)
    assert int(result.argmax()) == 2
    buffer = np.empty((1, 3), np.float32)
    assert ops.softmax(scores, out=buffer) is buffer


@pytest.mark.parametrize("layout", ["channels_first", "channels_last"])
def test_decode_yolo_detects_both_prediction_layouts(layout):
    pred = _prediction(ANCHORS, layout)
    detections = ops.decode_yolo(
        pred, LABELS, _letterbox_transform(), conf_th=0.3, iou_th=0.4
    )
    assert len(detections) == 2
    assert detections[0].class_id == 0
    assert detections[0].label == "ball"
    assert detections[1].class_id == 1
    assert detections[1].label == "cube"


def test_decode_yolo_filters_by_the_score_threshold():
    pred = _prediction(ANCHORS, "channels_first")
    detections = ops.decode_yolo(pred, LABELS, _letterbox_transform(), conf_th=0.3)
    assert all(detection.score >= 0.3 for detection in detections)
    assert [detection.label for detection in detections] == ["ball", "cube"]


def test_decode_yolo_maps_the_boxes_back_to_source_pixels():
    pred = _prediction(ANCHORS, "channels_first")
    detections = ops.decode_yolo(pred, LABELS, _letterbox_transform(), conf_th=0.3)
    ball = detections[0]
    assert (ball.x, ball.y, ball.w, ball.h) == (600, 4, 80, 40)


def test_decode_yolo_applies_the_sigmoid_only_when_asked():
    pred = _prediction([(320.0, 100.0, 40.0, 20.0, -2.0, 0.5)], "channels_first")
    plain = ops.decode_yolo(
        pred, LABELS, _letterbox_transform(), conf_th=0.3, score_activation="none"
    )
    activated = ops.decode_yolo(
        pred, LABELS, _letterbox_transform(), conf_th=0.3, score_activation="sigmoid"
    )
    assert plain[0].score == pytest.approx(0.5)
    assert activated[0].score == pytest.approx(1.0 / (1.0 + np.exp(-0.5)), abs=1e-6)


def test_decode_yolo_writes_the_source_pixels_into_the_output_buffer():
    pred = _prediction(ANCHORS, "channels_first")
    buffer = np.zeros((8, 6), np.float32)
    detections = ops.decode_yolo(
        pred, LABELS, _letterbox_transform(), conf_th=0.3, out=buffer
    )
    assert buffer[0, 0] == pytest.approx(detections[0].x)
    assert buffer[0, 1] == pytest.approx(detections[0].y)
    assert buffer[0, 2] - buffer[0, 0] == pytest.approx(detections[0].w, abs=1.0)
    assert buffer[0, 4] == pytest.approx(detections[0].score)
    assert buffer[0, 5] == detections[0].class_id


def test_resize_frame_writes_into_the_supplied_buffer():
    source = np.full((352, 1280, 3), 30, np.uint8)
    buffer = np.empty((352, 640, 3), np.uint8)
    image, _ = ops.resize_frame(source, (640, 352), out=buffer)
    assert image is buffer


def test_resize_frame_gives_the_same_result_with_and_without_scratch():
    source = (np.arange(352 * 1280 * 3, dtype=np.uint8).reshape(352, 1280, 3))
    plain, _ = ops.resize_frame(source, (640, 352))
    buffer = np.empty((352, 640, 3), np.uint8)
    scratch = np.empty((176, 640, 3), np.uint8)
    buffered, _ = ops.resize_frame(source, (640, 352), out=buffer, scratch=scratch)
    assert buffered is buffer
    assert np.array_equal(buffered, plain)


def test_normalize_yolo_writes_into_the_supplied_buffer():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    buffer = np.empty((2, 3, 3), np.float32)
    result = ops.normalize_yolo(image, 1.0 / 255.0, [0, 0, 0], [1, 1, 1], out=buffer)
    assert result is buffer
    assert np.allclose(result, ops.normalize_yolo(image, 1.0 / 255.0, [0, 0, 0], [1, 1, 1]))


def test_iou_matrix_writes_into_the_supplied_buffer():
    a = np.array([[0, 0, 10, 10]], np.float32)
    b = np.array([[0, 0, 10, 10]], np.float32)
    buffer = np.empty((1, 1), np.float32)
    result = ops.iou_matrix(a, b, out=buffer)
    assert result is buffer
    assert result[0, 0] == pytest.approx(1.0)


def test_the_numpy_fallback_agrees_with_the_opencv_path(monkeypatch):
    rng = np.random.default_rng(7)
    cases: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(50):
        count = int(rng.integers(1, 30))
        x = rng.random(count) * 100.0
        y = rng.random(count) * 100.0
        w = rng.random(count) * 40.0 + 2.0
        h = rng.random(count) * 40.0 + 2.0
        boxes = np.stack([x, y, x + w, y + h], axis=1).astype(np.float32)
        scores = rng.random(count).astype(np.float32)
        cases.append((boxes, scores))
    fast = [ops.nms(boxes, scores, 0.45) for boxes, scores in cases]
    monkeypatch.setattr(ops, "cv2", None)
    slow = [ops.nms(boxes, scores, 0.45) for boxes, scores in cases]
    for fast_keep, slow_keep in zip(fast, slow):
        assert set(fast_keep.tolist()) == set(slow_keep.tolist())


def test_ops_imports_no_platform_or_engine_library():
    tree = ast.parse(OPS_PATH.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    forbidden = {
        "maix",
        "serial",
        "ncnn",
        "onnxruntime",
        "rknn_lite",
        "rknn",
        "torch",
        "espdl",
        "llama_cpp",
    }
    assert not roots & forbidden
    assert "framework.inference.manifest" not in roots
