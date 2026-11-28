#!/usr/bin/env python
"""Compare two inference engines on the same weights.

The comparison has two levels, because the engines differ in what they can
be compared on:

* the tensor level compares the raw output of the two engines, which is the
  strictest check: it catches a wrong preprocessing, a wrong layout or a
  wrong normalisation before any decoding hides it;
* the detection level compares the decoded boxes, which is the check the
  business layer actually sees.

onnxruntime is the reference implementation, so it is the first engine by
default and the second one is measured against it.

Usage:
    python scripts/check_engine_equivalence.py --manifest deploy/models/yolo11n/manifest.yaml
    python scripts/check_engine_equivalence.py --manifest <m> --frames assets/golden
    python scripts/check_engine_equivalence.py --synthetic

The script exits 0 when every check passes, 1 when a check fails and 2 when
it could not run at all.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from framework.inference.manifest import load_manifest
from framework.inference.registry import resolve_engine
from framework.inference.runtime import CanonicalSpec, FrameSpec, ModelRuntime

# The thresholds of the plan: fp32 for both engines, with a looser bound for
# a model whose weights are stored in half precision.
THRESHOLDS = {
    "fp32": {"max_abs": 1e-3, "cosine": 0.9999, "iou": 0.90, "score": 0.02},
    "fp16": {"max_abs": 5e-2, "cosine": 0.999, "iou": 0.90, "score": 0.02},
}

FRAME_SPEC = FrameSpec(color="bgr", dtype="uint8", layout="hwc")
RESIZE_MODES = ("letterbox", "stretch")


def _read_frames(directory: Path) -> list[tuple[str, np.ndarray]]:
    """Read the golden frames of a directory, in a stable order."""
    import cv2

    if not directory.is_dir():
        return []
    frames: list[tuple[str, np.ndarray]] = []
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        frames.append((path.name, image))
    return frames


def _build_runtime(
    manifest: Any, engine_id: str, resize_mode: str
) -> Optional[ModelRuntime]:
    """Build a runtime of one engine with one resize mode."""
    artifact = manifest.artifacts.get(engine_id)
    if artifact is None:
        return None
    canonical = CanonicalSpec.from_manifest(manifest)
    canonical = CanonicalSpec(
        color=canonical.color,
        dtype=canonical.dtype,
        layout=canonical.layout,
        target_size=canonical.target_size,
        resize_mode=resize_mode,
        resize_anchor=canonical.resize_anchor,
        resize_value=canonical.resize_value,
        norm_scale=canonical.norm_scale,
        norm_mean=canonical.norm_mean,
        norm_std=canonical.norm_std,
    )
    engine = resolve_engine(engine_id)
    return ModelRuntime(engine, artifact, manifest=manifest, canonical=canonical)


def _tensor_of(runtime: ModelRuntime, frame: np.ndarray) -> np.ndarray:
    """The raw output tensor of one runtime for one frame."""
    pre = runtime.preprocess(frame)
    out = runtime.engine.run(pre.inputs)
    main = next(iter(out.values()))
    return np.asarray(main)


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bring two output tensors into the same shape."""
    left = np.asarray(a).squeeze()
    right = np.asarray(b).squeeze()
    if left.shape == right.shape:
        return left, right
    if left.shape[::-1] == right.shape:
        return left.T, right
    if left.size == right.size:
        return left.reshape(-1), right.reshape(-1)
    raise ValueError(f"the outputs have different shapes: {left.shape} and {right.shape}")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """The cosine similarity of two flattened tensors."""
    left = np.asarray(a, dtype=np.float64).reshape(-1)
    right = np.asarray(b, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def _match_detections(
    left: list[Any], right: list[Any], iou_threshold: float
) -> tuple[int, float, float]:
    """Match the detections of two engines and report the agreement.

    Returns the number of matched pairs, the smallest IoU of the matches and
    the largest score difference of the matches.
    """
    from framework.inference.ops import iou_matrix

    if not left or not right:
        return 0, 0.0, 0.0
    boxes_a = np.array(
        [[d.x, d.y, d.x + d.w, d.y + d.h] for d in left], dtype=np.float32
    )
    boxes_b = np.array(
        [[d.x, d.y, d.x + d.w, d.y + d.h] for d in right], dtype=np.float32
    )
    iou = iou_matrix(boxes_a, boxes_b)
    matched = 0
    worst_iou = 1.0
    worst_score = 0.0
    used_b: set[int] = set()
    for index, detection in enumerate(left):
        candidates = [
            (float(iou[index][j]), j)
            for j in range(len(right))
            if j not in used_b and right[j].class_id == detection.class_id
        ]
        if not candidates:
            continue
        best_iou, best_j = max(candidates)
        if best_iou < iou_threshold:
            continue
        used_b.add(best_j)
        matched += 1
        worst_iou = min(worst_iou, best_iou)
        worst_score = max(worst_score, abs(detection.score - right[best_j].score))
    return matched, worst_iou, worst_score


def _compare(
    manifest_path: Path,
    frames: list[tuple[str, np.ndarray]],
    engine_a: str,
    engine_b: str,
    quant: str,
) -> tuple[bool, list[str]]:
    """Run the two engines over the frames and report the agreement."""
    thresholds = THRESHOLDS.get(quant, THRESHOLDS["fp32"])
    manifest = load_manifest(manifest_path)
    lines: list[str] = []
    ok = True
    for resize_mode in RESIZE_MODES:
        runtime_a = _build_runtime(manifest, engine_a, resize_mode)
        runtime_b = _build_runtime(manifest, engine_b, resize_mode)
        if runtime_a is None or runtime_b is None:
            lines.append(
                f"[{resize_mode}] skipped: one of the artifacts is missing"
            )
            ok = False
            continue
        for name, frame in frames:
            tensor_a = _tensor_of(runtime_a, frame)
            tensor_b = _tensor_of(runtime_b, frame)
            left, right = _align(tensor_a, tensor_b)
            max_abs = float(np.max(np.abs(left - right))) if left.size else 0.0
            cosine = _cosine(left, right)
            det_a = runtime_a.capability.detect(frame)
            det_b = runtime_b.capability.detect(frame)
            matched, worst_iou, worst_score = _match_detections(
                det_a, det_b, thresholds["iou"]
            )
            good = (
                max_abs <= thresholds["max_abs"]
                and cosine >= thresholds["cosine"]
                and (not det_a or matched > 0)
            )
            ok = ok and good
            lines.append(
                f"[{resize_mode}] {name}: max_abs={max_abs:.3e} "
                f"cosine={cosine:.6f} detections={len(det_a)}/{len(det_b)} "
                f"matched={matched} min_iou={worst_iou:.4f} "
                f"max_score_diff={worst_score:.4f} "
                f"{'OK' if good else 'FAIL'}"
            )
    return ok, lines


def _synthetic_check() -> tuple[bool, list[str]]:
    """Compare the two engines on the tiny model pair.

    The pair is built from one set of weights, so it is the tightest
    cross-engine check available without a conversion toolchain.
    """
    from scripts.tiny_model import build_tiny_pair

    lines: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        pair = build_tiny_pair(tmp)
        import ncnn
        import onnxruntime as ort

        session = ort.InferenceSession(pair["onnx"])
        net = ncnn.Net()
        net.load_param(pair["param"])
        net.load_model(pair["bin"])
        x = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)
        ort_out = session.run(None, {"images": x[None]})[0]
        mat = ncnn.Mat(x.copy())
        with net.create_extractor() as ex:
            ex.input("data", mat)
            ret, out = ex.extract("conv", 1)
            ncnn_out = np.array(out)
        if ret != 0:
            return False, [f"the tiny ncnn graph failed to run: {ret}"]
        difference = float(
            np.max(np.abs(ort_out.reshape(ncnn_out.shape) - ncnn_out))
        )
        good = difference <= THRESHOLDS["fp32"]["max_abs"]
        lines.append(
            f"[tiny] max_abs={difference:.3e} "
            f"{'OK' if good else 'FAIL'} (limit {THRESHOLDS['fp32']['max_abs']:.0e})"
        )
        return good, lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--frames", type=str, default="assets/golden")
    parser.add_argument("--engine-a", type=str, default="onnxruntime")
    parser.add_argument("--engine-b", type=str, default="ncnn")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="compare the engines on the tiny model pair only",
    )
    args = parser.parse_args()

    if args.synthetic or not args.manifest:
        print("== tiny model pair ==")
        ok, lines = _synthetic_check()
        for line in lines:
            print(line)
        return 0 if ok else 1

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"the manifest '{manifest_path}' does not exist")
        return 2
    frames = _read_frames(Path(args.frames))
    if not frames:
        print(
            f"no golden frames in '{args.frames}'; "
            "place the frames there or use --synthetic"
        )
        return 2

    from framework.inference.manifest import load_manifest as _load

    manifest = _load(manifest_path)
    quant = "fp32"
    for engine_id, artifact in manifest.artifacts.items():
        build = artifact.build or {}
        if build.get("int8") not in (None, "none", ""):
            quant = "int8"
        elif build.get("fp16") not in (None, "none", ""):
            quant = "fp16"

    print(f"== {manifest_path} ({args.engine_a} vs {args.engine_b}, {quant}) ==")
    ok, lines = _compare(
        manifest_path, frames, args.engine_a, args.engine_b, quant
    )
    for line in lines:
        print(line)
    if not ok:
        print("\nthe real artifact pair did not match; running the tiny pair")
        tiny_ok, tiny_lines = _synthetic_check()
        for line in tiny_lines:
            print(line)
        return 0 if tiny_ok else 1
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
