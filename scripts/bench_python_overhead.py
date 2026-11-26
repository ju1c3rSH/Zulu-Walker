#!/usr/bin/env python
"""Measure the Python-language overhead the inference design adds per frame.

The design in docs/architecture/inference_architecture.md runs, per frame:
a dict for engine I/O, a wrapper closure, a decode pass, and one Detection
object per box.  Each of those is a language-level construct, and each has a
cost that does not show up in an NPU or CPU timing.  This script measures the
language constructs themselves so the budget is based on numbers.

Run on a development host:
    python scripts/bench_python_overhead.py

Run on the device (the numbers that matter):
    python scripts/bench_python_overhead.py --frames 1000
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, TypedDict, Unpack, runtime_checkable

import numpy as np


# --------------------------------------------------------------------------- #
#  The objects the design actually creates per frame
# --------------------------------------------------------------------------- #
@dataclass
class DetectionPlain:
    """Mirrors framework/hal/interface/ai.py:Detection (plain dataclass)."""
    x: int
    y: int
    w: int
    h: int
    class_id: int
    score: float
    label: str = ""
    angle: Optional[float] = None
    keypoints: list = field(default_factory=list)
    mask_index: int = -1
    seg_mask: Optional[np.ndarray] = None
    mask_stats: Optional[object] = None


@dataclass(slots=True)
class DetectionSlots:
    """Same fields, __slots__ enabled."""
    x: int
    y: int
    w: int
    h: int
    class_id: int
    score: float
    label: str = ""
    angle: Optional[float] = None
    keypoints: list = field(default_factory=list)
    mask_index: int = -1
    seg_mask: Optional[np.ndarray] = None
    mask_stats: Optional[object] = None


@dataclass(frozen=True)
class ArtifactRefFrozen:
    engine_id: str
    path: str
    wrapper: str
    wrapper_params: dict
    preprocess: Any
    io: Any


class DetectKwargs(TypedDict, total=False):
    conf_th: float
    iou_th: float


@runtime_checkable
class AIInference(Protocol):
    def detect(self, frame: np.ndarray, **kwargs: Unpack[DetectKwargs]) -> list:
        ...


class FakeInference:
    def detect(self, frame: np.ndarray, **kwargs: Unpack[DetectKwargs]) -> list:
        return []


# --------------------------------------------------------------------------- #
#  harness
# --------------------------------------------------------------------------- #
def bench(name: str, fn, iters: int) -> dict[str, Any]:
    fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    us = (time.perf_counter() - t0) * 1e6 / iters
    return {"name": name, "us": round(us, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--boxes", type=int, default=100)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    it = args.iters

    # A realistic YOLO11 raw head: 1 x 84 x 8400 float32 == 2.8 MB
    head = np.zeros((1, 84, 8400), dtype=np.float32)
    blob = np.zeros((1, 3, 640, 352), dtype=np.float32)
    boxes = np.column_stack([
        np.arange(args.boxes) % 600,
        np.arange(args.boxes) % 300,
        np.arange(args.boxes) % 600 + 40,
        np.arange(args.boxes) % 300 + 40,
    ]).astype(np.float32)
    scores = np.linspace(0.9, 0.1, args.boxes).astype(np.float32)

    rows: list[dict[str, Any]] = []

    # --- container / call overhead --------------------------------------- #
    rows.append(bench("dict_literal_2_items", lambda: {"image": blob}, it))
    rows.append(bench("dict_from_existing", lambda: dict(_D), it))
    rows.append(bench("func_call_no_args", lambda: _noop(), it))
    rows.append(bench("closure_call", lambda: _closure(), it))
    rows.append(bench("kwargs_pack_2", lambda: _kw(conf_th=0.4, iou_th=0.45), it))
    rows.append(bench("lock_acquire_release", _lock_cycle, it))
    rows.append(bench("isinstance_runtime_checkable",
                      lambda: isinstance(FakeInference(), AIInference), 2000))

    # --- object construction --------------------------------------------- #
    rows.append(bench("dataclass_plain_1", lambda: DetectionPlain(1, 2, 3, 4, 0, 0.5), it))
    rows.append(bench("dataclass_slots_1", lambda: DetectionSlots(1, 2, 3, 4, 0, 0.5), it))
    rows.append(bench("dataclass_frozen_1", lambda: ArtifactRefFrozen("e", "p", "w", {}, None, None), it))
    rows.append(bench(f"detection_list_{args.boxes}",
                      lambda: [DetectionPlain(i, i, 4, 4, 0, 0.5) for i in range(args.boxes)], it // 10))
    rows.append(bench(f"detection_slots_list_{args.boxes}",
                      lambda: [DetectionSlots(i, i, 4, 4, 0, 0.5) for i in range(args.boxes)], it // 10))

    # --- tensor handling ---------------------------------------------------- #
    rows.append(bench("tensor_view_2_8mb", lambda: head[0], it))
    rows.append(bench("tensor_copy_2_8mb", lambda: head[0].copy(), it // 20))
    rows.append(bench("np_array_copy_2_8mb", lambda: np.array(head[0]), it // 20))
    rows.append(bench("np_ascontiguous_NCHW_2_6mb",
                      lambda: np.ascontiguousarray(blob.transpose(0, 2, 3, 1)), it // 20))

    # --- decode style ------------------------------------------------------ #
    rows.append(bench("decode_vectorized_topk",
                      lambda: np.argpartition(scores, -min(100, len(scores)))[-min(100, len(scores)):], it // 10))
    rows.append(bench("decode_python_loop_100",
                      lambda: [_decode_one(boxes, scores, i) for i in range(min(100, args.boxes))], it // 10))

    # --- per-frame assembly ------------------------------------------------ #
    def per_frame() -> None:
        payload = {"image": blob}
        out = {"pred": head[0]}
        _ = payload["image"].shape
        _ = out["pred"].shape
    rows.append(bench("frame_dict_shuffle", per_frame, it))

    budget = args.frames  # frames per second target
    per_frame_us = sum(r["us"] for r in rows if r["name"] in {
        "dict_literal_2_items", "closure_call", "kwargs_pack_2",
        f"detection_list_{args.boxes}", "tensor_view_2_8mb",
        "decode_vectorized_topk"})

    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "rows": rows,
        "selected_per_frame_us": round(per_frame_us, 3),
        "frame_budget_us_at_60fps": 16666.7,
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"python {sys.version.split()[0]} on {platform.machine()} "
              f"({platform.system()}), numpy {np.__version__}")
        print(f"{'construct':<34}{'us/op':>10}")
        for r in rows:
            print(f"  {r['name']:<32}{r['us']:>10.3f}")
        print(f"selected language overhead per frame: {per_frame_us:.1f} us "
              f"({per_frame_us / 16666.7 * 100:.3f}% of a 60 fps budget)")
    return 0


_D = {"image": None}


def _noop() -> None:
    return None


def _closure() -> None:
    return None


def _kw(conf_th: float = 0.0, iou_th: float = 0.0) -> None:
    return None


def _decode_one(boxes: np.ndarray, scores: np.ndarray, i: int) -> tuple:
    return (float(boxes[i, 0]), float(boxes[i, 1]), float(scores[i]))


_lock = __import__("threading").Lock()


def _lock_cycle() -> None:
    _lock.acquire()
    _lock.release()


if __name__ == "__main__":
    raise SystemExit(main())
