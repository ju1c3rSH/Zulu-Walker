#!/usr/bin/env python
"""Benchmark the ops the inference layer needs, numpy path vs OpenCV path.

The inference layer's rules (docs/architecture/inference_architecture.md) forbid
platform and engine libraries outside framework/inference/engines/.  Whether
OpenCV may be used by framework/inference/ops.py is a performance decision, not
a taste decision, so it is measured here instead of argued about.

Usage on a development host:
    python scripts/bench_inference_ops.py

Usage on the device (the numbers that matter):
    python scripts/bench_inference_ops.py --width 1280 --height 352 --target 640 352

Both paths compute the same thing:
  - letterbox: scale the source into the target box, then pad to the target size
  - normalize: (pixel - mean) / std, in float32
  - pack:      HWC uint8 -> NCHW float32
  - nms:       greedy NMS over (x1, y1, x2, y2, score)

Only the letterbox differs in interpolation (OpenCV bilinear vs numpy nearest);
that difference is reported on its own line so the accuracy cost of a
numpy-only rule is visible next to the speed cost.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Callable

import numpy as np

try:
    import cv2
except ImportError:  # cv2 is optional: the numpy path is the one we must keep
    cv2 = None


# --------------------------------------------------------------------------- #
#  numpy implementations
# --------------------------------------------------------------------------- #
def letterbox_numpy_nearest(
    src: np.ndarray,
    tw: int,
    th: int,
    pad: tuple[int, int, int, int],
    value: int = 114,
) -> np.ndarray:
    """Scale with nearest-neighbour sampling, then pad."""
    sh, sw = src.shape[:2]
    scale = min(tw / sw, th / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    # Nearest neighbour via index gather: this is the fastest numpy-only option.
    yi = (np.arange(nh) * (sh / nh)).astype(np.int32)
    xi = (np.arange(nw) * (sw / nw)).astype(np.int32)
    scaled = src[yi[:, None], xi[None, :]]
    top, left, bottom, right = pad
    out = np.full((th, tw, src.shape[2]), value, dtype=src.dtype)
    out[top:top + nh, left:left + nw] = scaled
    return out


def letterbox_numpy_bilinear(
    src: np.ndarray,
    tw: int,
    th: int,
    pad: tuple[int, int, int, int],
    value: int = 114,
) -> np.ndarray:
    """Scale with bilinear sampling, pure numpy (slow; reference for accuracy)."""
    sh, sw = src.shape[:2]
    scale = min(tw / sw, th / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    fx = (np.arange(nw) + 0.5) * (sw / nw) - 0.5
    fy = (np.arange(nh) + 0.5) * (sh / nh) - 0.5
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, sw - 1)
    y1 = np.clip(y0 + 1, 0, sh - 1)
    wx = (fx - x0).astype(np.float32)[None, :, None]
    wy = (fy - y0).astype(np.float32)[:, None, None]
    a = src[y0[:, None], x0[None, :]].astype(np.float32)
    b = src[y0[:, None], x1[None, :]].astype(np.float32)
    c = src[y1[:, None], x0[None, :]].astype(np.float32)
    d = src[y1[:, None], x1[None, :]].astype(np.float32)
    blended = (a * (1 - wx) + b * wx) * (1 - wy) + (c * (1 - wx) + d * wx) * wy
    scaled = np.clip(blended, 0, 255).astype(np.uint8)
    top_p, left_p, bottom_p, right_p = pad
    out = np.full((th, tw, src.shape[2]), value, dtype=np.uint8)
    out[top_p:top_p + nh, left_p:left_p + nw] = scaled
    return out


def nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_th: float) -> list[int]:
    """Greedy NMS. boxes is (N, 4) as x1, y1, x2, y2."""
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_th]
    return keep


# --------------------------------------------------------------------------- #
#  OpenCV implementations
# --------------------------------------------------------------------------- #
def letterbox_cv2(
    src: np.ndarray,
    tw: int,
    th: int,
    pad: tuple[int, int, int, int],
    value: int = 114,
) -> np.ndarray:
    sh, sw = src.shape[:2]
    scale = min(tw / sw, th / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    interp = cv2.INTER_LINEAR if scale < 1 else cv2.INTER_AREA
    scaled = cv2.resize(src, (nw, nh), interpolation=interp)
    top, left, bottom, right = pad
    return cv2.copyMakeBorder(
        scaled, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(value,) * 3)


# --------------------------------------------------------------------------- #
#  harness
# --------------------------------------------------------------------------- #
def timeit(fn: Callable[[], Any], iters: int) -> tuple[float, float]:
    fn()  # warm up
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    total_ms = (time.perf_counter() - t0) * 1000.0 / iters
    return total_ms, total_ms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=352)
    ap.add_argument("--target", type=int, nargs=2, default=[640, 352])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--boxes", type=int, default=300)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    sw, sh = args.width, args.height
    tw, th = args.target
    rng = np.random.default_rng(0)
    src = rng.integers(0, 256, (sh, sw, 3), dtype=np.uint8)

    scale = min(tw / sw, th / sh)
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    pad = ((th - nh) // 2, (tw - nw) // 2, th - nh - (th - nh) // 2, tw - nw - (tw - nw) // 2)

    results: list[dict[str, Any]] = []

    def bench(name: str, fn: Callable[[], Any], iters: int | None = None) -> None:
        ms, _ = timeit(fn, iters or args.iters)
        results.append({"op": name, "ms": round(ms, 3)})

    bench("letterbox_numpy_nearest", lambda: letterbox_numpy_nearest(src, tw, th, pad))
    bench("letterbox_numpy_bilinear", lambda: letterbox_numpy_bilinear(src, tw, th, pad), max(5, args.iters // 10))
    if cv2 is not None:
        bench("letterbox_cv2", lambda: letterbox_cv2(src, tw, th, pad))
    else:
        results.append({"op": "letterbox_cv2", "ms": None, "note": "cv2 not installed"})

    blob = letterbox_numpy_nearest(src, tw, th, pad)
    bench("normalize_numpy", lambda: (blob.astype(np.float32) - 0.0) / 255.0)
    if cv2 is not None:
        bench("normalize_cv2", lambda: cv2.subtract(blob, (0, 0, 0)).astype(np.float32) / 255.0)

    bench("pack_nchw_numpy", lambda: np.ascontiguousarray(blob.transpose(2, 0, 1), dtype=np.float32))
    if cv2 is not None:
        bench("pack_nchw_cv2", lambda: cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy())

    # Layout and dtype conversions: the cost of picking the wrong canonical form.
    conv_iters = max(3, args.iters // 4)
    hwcu = blob                       # uint8, HWC (what the camera actually produces)
    hwcf = blob.astype(np.float32)   # float32, HWC
    nchw = np.ascontiguousarray(hwcf.transpose(2, 0, 1))  # float32, C, H, W
    bench("cast_u8_hwc_to_f32", lambda: hwcu.astype(np.float32), conv_iters)
    bench("transpose_hwc_to_nchw", lambda: np.ascontiguousarray(hwcf.transpose(2, 0, 1)), conv_iters)
    bench("transpose_nchw_to_hwc", lambda: np.ascontiguousarray(nchw.transpose(1, 2, 0)), conv_iters)

    n = args.boxes
    boxes = np.column_stack([
        rng.integers(0, sw - 100, n),
        rng.integers(0, sh - 100, n),
        rng.integers(50, sw, n),
        rng.integers(50, sh, n),
    ]).astype(np.float32)
    scores = rng.random(n).astype(np.float32)
    bench("nms_numpy", lambda: nms_numpy(boxes, scores, 0.45))
    if cv2 is not None:
        bench("nms_cv2", lambda: cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), 0.45, 0.45))

    payload = {
        "input": [sw, sh],
        "target": [tw, th],
        "scale": round(scale, 4),
        "padded": [nw, nh],
        "iters": args.iters,
        "numpy": np.__version__,
        "cv2": getattr(cv2, "__version__", None),
        "results": results,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"source {sw}x{sh} -> target {tw}x{th} "
              f"(scale {scale:.3f}, resized {nw}x{nh}), {args.iters} iters")
        print(f"numpy {np.__version__}, cv2 {getattr(cv2, '__version__', None)}")
        for r in results:
            ms = r["ms"]
            print(f"  {r['op']:<28} {'n/a' if ms is None else f'{ms:8.3f} ms'}"
                  + (f"  ({r['note']})" if r.get("note") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
