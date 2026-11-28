#!/usr/bin/env python
"""Benchmark one artifact under several accelerator settings.

A GPU number without the CPU number next to it says nothing, and a total
frame time without the split says the wrong thing: the canonical stage
(resize + normalise) runs on the CPU in this framework, so a fast inference
can hide behind a slow preprocess and make the accelerator look useless.

This script therefore reports the three stages separately, per variant:

  load        the engine opening the artifact
  preprocess  letterbox/resize + normalisation, on the CPU
  infer       engine.run(), the part the accelerator owns
  frame       preprocess + infer, the number the loop actually sees

Variants are the runtime blocks a deployment would use, given as
``name=<json>``; the artifact's own block is the baseline and is always run.

Usage:
    # the baseline alone
    python scripts/bench_inference_engines.py --engine onnxruntime \\
        --manifest deploy/models/yolo11n/manifest.yaml

    # CPU against Vulkan on the same weights
    python scripts/bench_inference_engines.py --engine ncnn \\
        --manifest deploy/models/yolo11n/manifest.yaml \\
        --variant cpu='{"device":"cpu"}' \\
        --variant vulkan='{"device":"vulkan","vulkan_device":"auto"}'

    # CPU against CUDA
    python scripts/bench_inference_engines.py --engine onnxruntime \\
        --manifest deploy/models/yolo11n/manifest.yaml \\
        --variant cuda='{"providers":["CUDAExecutionProvider","CPUExecutionProvider"]}'

The frame is synthetic: a flat gradient in the target resolution.  It
measures the cost of the pipeline, not the accuracy of anything.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.inference.manifest import load_manifest  # noqa: E402
from framework.inference.registry import resolve_engine  # noqa: E402
from framework.inference.runtime import FrameSpec, ModelRuntime  # noqa: E402

DEFAULT_WARMUP = 5
DEFAULT_RUNS = 30


def parse_variant(text: str) -> tuple[str, dict[str, Any]]:
    """Read one ``name=<json>`` variant specification."""
    name, separator, payload = text.partition("=")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError(
            f"a variant must look like name={{...}} (got {text!r})"
        )
    try:
        block = json.loads(payload) if payload.strip() else {}
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"variant {name!r}: {exc}") from exc
    if not isinstance(block, dict):
        raise argparse.ArgumentTypeError(
            f"variant {name!r}: the payload must be a JSON object"
        )
    return name.strip(), block


def make_frame(width: int, height: int, channels: int) -> np.ndarray:
    """A synthetic uint8 frame, cheap to build and representative in size."""
    column = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
    rows = np.linspace(0, 255, height, dtype=np.uint8)[:, None, None]
    base = ((column.astype(np.uint16) + rows.astype(np.uint16)) // 2).astype(np.uint8)
    return np.repeat(base, channels, axis=2)


def bench_variant(
    engine_id: str,
    manifest_path: Path,
    runtime_options: dict[str, Any],
    frame: np.ndarray,
    warmup: int,
    runs: int,
) -> dict[str, Any]:
    """Load one variant and time its stages."""
    manifest = load_manifest(manifest_path)
    artifact = manifest.artifacts.get(engine_id)
    if artifact is None:
        raise SystemExit(
            f"the manifest declares no {engine_id} artifact "
            f"(declared: {sorted(manifest.artifacts)})"
        )
    engine = resolve_engine(engine_id)
    started = time.perf_counter()
    runtime = ModelRuntime(
        engine,
        artifact,
        manifest=manifest,
        frame_spec=FrameSpec(color="bgr", dtype="uint8", layout="hwc"),
        runtime_options=runtime_options,
        wrapper_factory=lambda rt: _Passthrough(),
    )
    load_ms = (time.perf_counter() - started) * 1000.0

    capability = engine.capabilities()
    pre_ms: list[float] = []
    infer_ms: list[float] = []
    for index in range(warmup + runs):
        begin = time.perf_counter()
        prepared = runtime.preprocess(frame)
        middle = time.perf_counter()
        engine.run(prepared.inputs)
        end = time.perf_counter()
        if index < warmup:
            continue
        pre_ms.append((middle - begin) * 1000.0)
        infer_ms.append((end - middle) * 1000.0)

    engine.unload()
    return {
        "device": capability.device,
        "detail": capability.device_detail or capability.device,
        "load_ms": load_ms,
        "pre_ms": _stats(pre_ms),
        "infer_ms": _stats(infer_ms),
        "frame_ms": _stats([a + b for a, b in zip(pre_ms, infer_ms)]),
    }


class _Passthrough:
    """A dialect stub: the benchmark drives preprocess and run() itself."""

    provides = frozenset()


def _stats(values: list[float]) -> dict[str, float]:
    """Mean, median and the 95th percentile of one stage."""
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": p95,
    }


def print_table(rows: list[tuple[str, dict[str, Any]]]) -> None:
    """Print the comparison table."""
    header = (
        f"{'variant':<12} {'device':<10} {'load':>8} {'pre':>8} {'infer':>8} "
        f"{'frame':>8} {'fps':>7}  detail"
    )
    print(header)
    print("-" * len(header))
    for name, result in rows:
        frame_ms = result["frame_ms"]["mean"]
        fps = 1000.0 / frame_ms if frame_ms > 0 else 0.0
        print(
            f"{name:<12} {result['device']:<10} "
            f"{result['load_ms']:>7.1f}m {result['pre_ms']['mean']:>7.2f}m "
            f"{result['infer_ms']['mean']:>7.2f}m {frame_ms:>7.2f}m {fps:>7.1f}  "
            f"{result['detail']}"
        )
    print("\nall times are milliseconds per frame, mean over the timed runs")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", default="onnxruntime", help="engine id to benchmark")
    parser.add_argument(
        "--manifest",
        default="deploy/models/yolo11n/manifest.yaml",
        help="manifest carrying the artifact of that engine",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="runtime block to try; repeatable, the artifact block always runs first",
    )
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument(
        "--source-size",
        default="",
        metavar="WxH",
        help="size of the synthetic frame; default is the model input size",
    )
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        print(f"manifest {manifest_path} not found", file=sys.stderr)
        return 2
    manifest = load_manifest(manifest_path)
    width, height = manifest.target_size
    if args.source_size:
        try:
            width, height = (int(part) for part in args.source_size.lower().split("x"))
        except ValueError:
            print(f"--source-size must look like 1280x720 (got {args.source_size!r})", file=sys.stderr)
            return 2
    channels = 4 if manifest.frame_color == "rgba" else 3
    frame = make_frame(width, height, channels)

    variants: list[tuple[str, dict[str, Any]]] = [("manifest", {})]
    for text in args.variant:
        variants.append(parse_variant(text))

    print(
        f"engine {args.engine}, manifest {manifest_path}, "
        f"frame {width}x{height}x{channels}, {args.runs} runs after {args.warmup} warmup\n"
    )
    rows: list[tuple[str, dict[str, Any]]] = []
    for name, block in variants:
        try:
            rows.append(
                (
                    name,
                    bench_variant(
                        args.engine, manifest_path, block, frame, args.warmup, args.runs
                    ),
                )
            )
        except SystemExit:
            raise
        except Exception as exc:
            print(f"{name:<12} failed: {exc}", file=sys.stderr)
            return 1
    print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
