"""Generic vision primitives of the inference kernel.

This module is the single implementation of resize, normalisation, IoU, NMS
and the raw YOLO decode for the whole framework.  It knows nothing about
engines, wrappers, runtimes or manifests, and it carries no semantics about
what is being detected: the only task level type it produces is the
framework's own Detection.

Every operator that runs on the per-frame path accepts an output buffer
(out=) and writes into it, so the canonical stage can preallocate at load
time and never allocate during inference.  Operators whose result has a
variable length, or whose result is a list of Python objects, document that
they still allocate.

Only numpy and cv2 may be used here (docs/architecture/inference/
01-scope-and-boundaries.md S1.1): cv2 carries no platform knowledge and is
roughly an order of magnitude faster than the numpy equivalent for resize
and NMS (06-performance-and-equivalence.md S10.2).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

from framework.hal.interface.ai import Detection
from framework.inference.contract import GeometryTransform

__all__ = [
    "convert_norm_params",
    "decode_yolo",
    "iou_matrix",
    "nms",
    "normalize_engine",
    "normalize_yolo",
    "resize_frame",
    "softmax",
]

RESIZE_MODES = ("letterbox", "crop", "stretch")
SCORE_ACTIVATIONS = ("none", "sigmoid")

_LAYOUT_ALIASES = {
    "channels_first": "channels_first",
    "channel_first": "channels_first",
    "first": "channels_first",
    "cf": "channels_first",
    "channels_last": "channels_last",
    "channel_last": "channels_last",
    "last": "channels_last",
    "cl": "channels_last",
}


def _prepare_out(
    out: np.ndarray | None,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Return out when it matches shape and dtype, otherwise a fresh buffer."""
    if out is None:
        return np.empty(shape, dtype=dtype)
    if out.shape != shape:
        raise ValueError(f"output buffer has shape {out.shape}, expected {shape}")
    if out.dtype != dtype:
        raise ValueError(f"output buffer has dtype {out.dtype}, expected {dtype}")
    if not out.flags["C_CONTIGUOUS"]:
        raise ValueError("output buffer must be C contiguous")
    return out


def _channel_params(value: Any, channels: int) -> np.ndarray:
    """Broadcast a scalar or per-channel parameter to one value per channel."""
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.repeat(arr, channels)
    if arr.size != channels:
        raise ValueError(f"expected {channels} parameter values, got {arr.size}")
    return arr


def _resample(
    src: np.ndarray,
    width: int,
    height: int,
    dst: np.ndarray | None = None,
) -> np.ndarray:
    """Scale src to (width, height), writing into dst when one is given."""
    if cv2 is not None:
        shrinking = width < src.shape[1] or height < src.shape[0]
        interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
        if dst is None:
            return cv2.resize(src, (width, height), interpolation=interpolation)
        return cv2.resize(src, (width, height), dst=dst, interpolation=interpolation)
    src_h, src_w = src.shape[:2]
    rows = (np.arange(height) * (src_h / height)).astype(np.int32)
    cols = (np.arange(width) * (src_w / width)).astype(np.int32)
    return src[rows[:, None], cols[None, :]]


def _scratch_buffer(
    scratch: np.ndarray | None,
    height: int,
    width: int,
    src: np.ndarray,
) -> np.ndarray | None:
    """The scratch buffer for the scaled image, or None when none was given."""
    if scratch is None:
        return None
    expected = (height, width) if src.ndim == 2 else (height, width, src.shape[2])
    if scratch.shape != expected:
        raise ValueError(f"scratch buffer has shape {scratch.shape}, expected {expected}")
    if scratch.dtype != src.dtype:
        raise ValueError(f"scratch buffer has dtype {scratch.dtype}, expected {src.dtype}")
    return scratch


def _anchor_offset(anchor: str, overflow: int, axis: str) -> int:
    """Crop offset of the anchor along one axis; an anchor of the other axis centres."""
    if anchor in ("center", ""):
        return overflow // 2
    if axis == "x":
        if anchor == "left":
            return 0
        if anchor == "right":
            return overflow
    else:
        if anchor == "top":
            return 0
        if anchor == "bottom":
            return overflow
    return overflow // 2


def resize_frame(
    src: np.ndarray,
    size: tuple[int, int],
    mode: str = "letterbox",
    value: int | str = 114,
    anchor: str = "center",
    out: np.ndarray | None = None,
    scratch: np.ndarray | None = None,
) -> tuple[np.ndarray, GeometryTransform]:
    """Resize src to size = (width, height) and report the geometry.

    mode is "letterbox" (scale to fit, pad the remainder), "crop" (scale to
    cover, cut the overflow away at anchor) or "stretch" (scale both axes
    independently).  All three share this one implementation.

    The returned GeometryTransform maps source pixels to input pixels and is
    inverted by x_src = (x_input - offset_x) / scale_x.  The letterbox offsets
    are padding amounts, the crop offsets are signed crop amounts and the
    stretch scales differ per axis.

    out receives the resized image.  scratch is an optional buffer for the
    intermediate scaled image of the letterbox and crop modes; supply it
    (and out) to keep the operator free of allocations.  Without scratch the
    intermediate image is allocated per call.
    """
    if src.ndim not in (2, 3):
        raise ValueError(f"expected a 2D or 3D image, got shape {src.shape}")
    if mode not in RESIZE_MODES:
        raise ValueError(f"unknown resize mode {mode!r}, expected one of {RESIZE_MODES}")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError(f"target size must be positive, got {(width, height)}")
    src_h, src_w = src.shape[:2]
    channels = 1 if src.ndim == 2 else src.shape[2]
    target_shape = (height, width) if src.ndim == 2 else (height, width, channels)
    out = _prepare_out(out, target_shape, src.dtype)

    if mode == "letterbox":
        scale = min(width / src_w, height / src_h)
        new_w = min(width, max(1, int(round(src_w * scale))))
        new_h = min(height, max(1, int(round(src_h * scale))))
        scaled = _resample(
            src, new_w, new_h, _scratch_buffer(scratch, new_h, new_w, src)
        )
        pad_x = (width - new_w) // 2
        pad_y = (height - new_h) // 2
        out[...] = 0 if value == "transparent" else value
        out[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled
        transform = GeometryTransform(
            mode="letterbox",
            scale_x=scale,
            scale_y=scale,
            offset_x=pad_x,
            offset_y=pad_y,
            src_w=src_w,
            src_h=src_h,
        )
        return out, transform

    if mode == "crop":
        scale = max(width / src_w, height / src_h)
        new_w = max(width, int(round(src_w * scale)))
        new_h = max(height, int(round(src_h * scale)))
        scaled = _resample(
            src, new_w, new_h, _scratch_buffer(scratch, new_h, new_w, src)
        )
        off_x = _anchor_offset(anchor, new_w - width, "x")
        off_y = _anchor_offset(anchor, new_h - height, "y")
        out[...] = scaled[off_y:off_y + height, off_x:off_x + width]
        transform = GeometryTransform(
            mode="crop",
            scale_x=scale,
            scale_y=scale,
            offset_x=-off_x,
            offset_y=-off_y,
            src_w=src_w,
            src_h=src_h,
        )
        return out, transform

    scale_x = width / src_w
    scale_y = height / src_h
    out[...] = _resample(src, width, height, out)
    transform = GeometryTransform(
        mode="stretch",
        scale_x=scale_x,
        scale_y=scale_y,
        offset_x=0,
        offset_y=0,
        src_w=src_w,
        src_h=src_h,
    )
    return out, transform


def normalize_yolo(
    image: np.ndarray,
    scale: float,
    mean: Any,
    std: Any,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the yolo formula (x * scale - mean) / std in float32.

    This is the formula of the framework tier; it is declared per artifact
    and is not interpreted by any other tier.
    """
    channels = image.shape[2] if image.ndim == 3 else 1
    mean_arr = _channel_params(mean, channels)
    std_arr = _channel_params(std, channels)
    out = _prepare_out(out, image.shape, np.dtype(np.float32))
    np.copyto(out, image, casting="unsafe")
    out *= np.float32(scale)
    out -= mean_arr
    out /= std_arr
    return out


def normalize_engine(
    image: np.ndarray,
    mean: Any,
    std: Any,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the engine formula (x - mean) / std in float32.

    The parameters are the engine form ones, see convert_norm_params.  This
    is the reference implementation of the formula; an engine that normalises
    on device performs the same arithmetic there.
    """
    channels = image.shape[2] if image.ndim == 3 else 1
    mean_arr = _channel_params(mean, channels)
    std_arr = _channel_params(std, channels)
    out = _prepare_out(out, image.shape, np.dtype(np.float32))
    np.copyto(out, image, casting="unsafe")
    out -= mean_arr
    out /= std_arr
    return out


def convert_norm_params(
    scale: float,
    mean: Any,
    std: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert the yolo form parameters into the engine form ones.

    The conversion is mean_e = mean / scale and std_e = std / scale; the two
    formulas stay independent and neither interprets the other.
    """
    scale_f = float(scale)
    if scale_f == 0.0:
        raise ValueError("scale must not be zero")
    mean_e = np.atleast_1d(np.asarray(mean, dtype=np.float32) / np.float32(scale_f))
    std_e = np.atleast_1d(np.asarray(std, dtype=np.float32) / np.float32(scale_f))
    return mean_e, std_e


def iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Pairwise IoU of two sets of xyxy boxes; the result is (len(a), len(b)).

    The arithmetic is vectorised over the box pairs.  When out is given the
    result is written into it and returned; its shape must be (len(a),
    len(b)).
    """
    a = np.asarray(boxes_a, dtype=np.float32).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float32).reshape(-1, 4)
    shape = (a.shape[0], b.shape[0])
    if out is None:
        target = np.zeros(shape, dtype=np.float32)
    else:
        target = np.asarray(out)
        if target.shape != shape:
            raise ValueError(f"output buffer has shape {target.shape}, expected {shape}")
        if target.dtype != np.float32:
            raise ValueError(f"output buffer has dtype {target.dtype}, expected float32")
    if a.shape[0] == 0 or b.shape[0] == 0:
        target[...] = 0.0
        return target
    top_left = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom_right = np.minimum(a[:, None, 2:], b[None, :, 2:])
    extent = np.clip(bottom_right - top_left, 0.0, None)
    intersection = extent[..., 0] * extent[..., 1]
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    union = area_a + area_b - intersection
    np.divide(intersection, np.maximum(union, 1e-9), out=target)
    np.copyto(target, 0.0, where=union <= 0.0)
    return target


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> np.ndarray:
    """Greedy NMS with vectorised IoU; the fallback for hosts without cv2."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        iou = iou_matrix(boxes[current:current + 1], boxes[rest])[0]
        order = rest[iou <= iou_th]
    return np.asarray(keep, dtype=np.int64)


def _nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> np.ndarray:
    """Indices of the boxes that survive NMS, highest score first.

    The OpenCV path takes rectangles as (x, y, width, height), so the xyxy
    boxes are converted first; that conversion is the one temporary array
    the fast path allocates.
    """
    if cv2 is not None and (scores.size == 0 or float(scores.min()) > 0.0):
        rects = np.empty_like(boxes)
        rects[:, 0] = boxes[:, 0]
        rects[:, 1] = boxes[:, 1]
        rects[:, 2] = np.maximum(boxes[:, 2] - boxes[:, 0], 0.0)
        rects[:, 3] = np.maximum(boxes[:, 3] - boxes[:, 1], 0.0)
        picked = cv2.dnn.NMSBoxes(rects, scores, 0.0, iou_th)
        return np.asarray(picked, dtype=np.int64).reshape(-1)
    return _nms_numpy(boxes, scores, iou_th)


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_th: float,
    class_ids: np.ndarray | None = None,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Greedy non maximum suppression over xyxy boxes.

    class_ids makes the suppression class aware: boxes of different classes
    never suppress each other.  The returned array holds the indices of the
    surviving boxes, highest score first; its length is variable, so it is
    always allocated.  When out is given the indices are written into it and
    the returned array is a view of its first entries.
    """
    work = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    score_arr = np.asarray(scores, dtype=np.float32).reshape(-1)
    if work.shape[0] != score_arr.shape[0]:
        raise ValueError(f"got {work.shape[0]} boxes and {score_arr.shape[0]} scores")
    if work.shape[0] == 0:
        indices = np.zeros(0, dtype=np.int64)
    else:
        shifted = work
        if class_ids is not None:
            classes = np.asarray(class_ids).reshape(-1).astype(np.int64)
            if classes.shape[0] != work.shape[0]:
                raise ValueError(
                    f"got {work.shape[0]} boxes and {classes.shape[0]} class ids"
                )
            span = float(
                max(work[:, 2].max(), work[:, 3].max())
                - min(work[:, 0].min(), work[:, 1].min())
                + 1.0
            )
            shifted = work.copy()
            shift = classes.astype(np.float32) * np.float32(span)
            shifted[:, 0] += shift
            shifted[:, 2] += shift
            shifted[:, 1] += shift
            shifted[:, 3] += shift
        indices = _nms_indices(shifted, score_arr, float(iou_th))
    if out is None:
        return indices
    out_arr = np.asarray(out)
    if out_arr.size < indices.size:
        raise ValueError(
            f"output buffer holds {out_arr.size} indices, {indices.size} are needed"
        )
    out_arr[:indices.size] = indices
    return out_arr[:indices.size]


def softmax(scores: np.ndarray, axis: int = -1, out: np.ndarray | None = None) -> np.ndarray:
    """Numerically stable softmax of scores along axis."""
    arr = np.asarray(scores, dtype=np.float32)
    out = _prepare_out(out, arr.shape, np.dtype(np.float32))
    np.copyto(out, arr, casting="unsafe")
    out -= out.max(axis=axis, keepdims=True)
    np.exp(out, out=out)
    out /= out.sum(axis=axis, keepdims=True)
    return out


def _resolve_layout(shape: tuple[int, int], expected: int, output_layout: str) -> str:
    """Decide whether the prediction is (4+nc, N) or (N, 4+nc)."""
    rows, cols = shape
    requested = str(output_layout).strip().lower()
    if requested == "auto":
        if rows == expected and cols == expected:
            return "channels_first"
        if rows == expected:
            return "channels_first"
        if cols == expected:
            return "channels_last"
        raise ValueError(
            f"prediction of shape {shape} matches neither {expected} rows nor {expected} columns"
        )
    layout = _LAYOUT_ALIASES.get(requested)
    if layout is None:
        raise ValueError(f"unknown output layout {output_layout!r}")
    if layout == "channels_first" and rows != expected:
        raise ValueError(
            f"prediction of shape {shape} has no {expected} rows for the channels first layout"
        )
    if layout == "channels_last" and cols != expected:
        raise ValueError(
            f"prediction of shape {shape} has no {expected} columns for the channels last layout"
        )
    return layout


def decode_yolo(
    pred: np.ndarray,
    labels: Sequence[str],
    transform: GeometryTransform,
    conf_th: float = 0.3,
    iou_th: float = 0.4,
    output_layout: str = "auto",
    score_activation: str = "none",
    out: np.ndarray | None = None,
) -> list[Detection]:
    """Decode a raw YOLO prediction into source frame detections.

    The prediction is read as boxes in input image pixels followed by one
    score per label; output_layout decides whether the prediction is
(4+nc, N) or (N, 4+nc), "auto" deciding by shape.  score_activation applies
a sigmoid to the class scores when it says "sigmoid".

    Boxes are suppressed by NMS in the input image space, where the model
    predicted them, and the survivors are mapped back to source frame pixels
    through transform.

    The returned list of Detection objects is always allocated.  When out is
    given it receives one row per detection with
(x1, y1, x2, y2, score, class_id) in source frame pixels.
    """
    label_list = list(labels)
    class_count = len(label_list)
    if class_count == 0:
        raise ValueError("decode_yolo needs the label list to know the class count")
    arr = np.asarray(pred)
    if arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"expected a batch of one, got shape {arr.shape}")
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D prediction, got shape {arr.shape}")
    expected = 4 + class_count
    layout = _resolve_layout(arr.shape, expected, output_layout)
    data = arr.T if layout == "channels_first" else arr
    data = np.asarray(data, dtype=np.float32)

    coords = data[:, :4]
    scores = data[:, 4:]
    activation = str(score_activation).strip().lower()
    if activation not in SCORE_ACTIVATIONS:
        raise ValueError(f"unknown score activation {score_activation!r}")
    if activation == "sigmoid":
        scores = 1.0 / (1.0 + np.exp(-scores))

    class_ids = np.argmax(scores, axis=1).astype(np.int64)
    best = scores[np.arange(scores.shape[0]), class_ids].astype(np.float32)
    selected = np.nonzero(best >= conf_th)[0]
    if selected.size == 0:
        return []

    picked = coords[selected]
    cx = picked[:, 0]
    cy = picked[:, 1]
    box_w = picked[:, 2]
    box_h = picked[:, 3]
    input_boxes = np.stack(
        [cx - box_w / 2.0, cy - box_h / 2.0, cx + box_w / 2.0, cy + box_h / 2.0], axis=1
    ).astype(np.float32)
    selected_scores = best[selected]
    selected_classes = class_ids[selected]

    survivors = nms(input_boxes, selected_scores, iou_th, class_ids=selected_classes)
    input_boxes = input_boxes[survivors]
    selected_scores = selected_scores[survivors]
    selected_classes = selected_classes[survivors]

    src_x1 = (input_boxes[:, 0] - transform.offset_x) / transform.scale_x
    src_y1 = (input_boxes[:, 1] - transform.offset_y) / transform.scale_y
    src_x2 = (input_boxes[:, 2] - transform.offset_x) / transform.scale_x
    src_y2 = (input_boxes[:, 3] - transform.offset_y) / transform.scale_y

    if out is not None:
        out_arr = np.asarray(out)
        if out_arr.ndim != 2 or out_arr.shape[1] != 6:
            raise ValueError(f"output buffer must have shape (n, 6), got {out_arr.shape}")
        if out_arr.shape[0] < src_x1.size:
            raise ValueError(
                f"output buffer holds {out_arr.shape[0]} rows, {src_x1.size} are needed"
            )
        out_arr[:src_x1.size, 0] = src_x1
        out_arr[:src_x1.size, 1] = src_y1
        out_arr[:src_x1.size, 2] = src_x2
        out_arr[:src_x1.size, 3] = src_y2
        out_arr[:src_x1.size, 4] = selected_scores
        out_arr[:src_x1.size, 5] = selected_classes

    detections: list[Detection] = []
    for index in range(src_x1.size):
        x = int(round(float(src_x1[index])))
        y = int(round(float(src_y1[index])))
        width = max(0, int(round(float(src_x2[index] - src_x1[index]))))
        height = max(0, int(round(float(src_y2[index] - src_y1[index]))))
        class_id = int(selected_classes[index])
        detections.append(
            Detection(
                x=x,
                y=y,
                w=width,
                h=height,
                class_id=class_id,
                score=float(selected_scores[index]),
                label=label_list[class_id],
            )
        )
    return detections
