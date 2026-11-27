"""Helpers of the engine tests: a manifest writer and the tiny model pair."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NEWLINE = chr(10)


def write_manifest(
    directory: Path,
    engine_id: str,
    artifact: dict[str, Any],
    *,
    name: str = "tiny",
    family: str = "yolo11",
    labels: Sequence[str] = ("ball",),
    size: tuple[int, int] = (4, 4),
    color: str = "rgb",
    dtype: str = "uint8",
    resize_mode: str = "letterbox",
    norm: Optional[dict[str, Any]] = None,
    wrapper: str = "vision.det.raw",
    wrapper_params: Optional[dict[str, Any]] = None,
) -> Path:
    """Write a manifest that declares one artifact for one engine.

    The labels live next to the manifest, because the manifest resolves the
    label file relative to its own directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    labels_path = directory / "labels.txt"
    labels_path.write_text(NEWLINE.join(labels) + NEWLINE, encoding="utf-8")
    block = dict(artifact)
    block.setdefault("wrapper", wrapper)
    block.setdefault("wrapper_params", wrapper_params or {})
    document = {
        "schema_version": 3,
        "name": name,
        "family": family,
        "input": {
            "frame": {"color": color, "dtype": dtype, "layout": "hwc"},
            "size": list(size),
            "resize": {"mode": resize_mode, "value": 114},
            "norm": norm
            or {
                "scale": 1.0,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        },
        "labels_file": labels_path.name,
        "artifacts": {engine_id: block},
    }
    path = directory / "manifest.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def tiny_pair(directory: Path) -> dict:
    """Build the tiny onnx and ncnn artifacts in a directory."""
    from scripts.tiny_model import build_tiny_pair

    return build_tiny_pair(directory)


def onnx_artifact() -> dict[str, Any]:
    """The artifact block of the tiny onnx model."""
    return {
        "file": "tiny.onnx",
        "wrapper": "vision.det.raw",
        "wrapper_params": {"output_layout": "auto", "score_activation": "none"},
        "preprocess": {"resize": "framework", "norm": "framework", "formula": "yolo"},
        "io": {"inputs": {"image": "images"}, "outputs": {"pred": "output0"}},
    }


def ncnn_artifact() -> dict[str, Any]:
    """The artifact block of the tiny ncnn model."""
    return {
        "file": "tiny.param",
        "bin": "tiny.bin",
        "wrapper": "vision.det.raw",
        "wrapper_params": {"output_layout": "auto", "score_activation": "none"},
        "preprocess": {
            "resize": "framework",
            "norm": "engine",
            "formula": "substract_mean_normalize",
        },
        "io": {"inputs": {"image": "data"}, "outputs": {"pred": "conv"}},
    }
