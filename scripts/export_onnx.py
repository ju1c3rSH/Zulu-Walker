#!/usr/bin/env python
"""Export a trained checkpoint to the onnx artifact of the framework.

The toolchain is offline: it does not belong to the runtime framework and
its output does not belong to the repository.  The recipe travels in the
`build` block of the manifest, the artifact travels next to it.

The export must produce the raw head of the model - no NMS inside the
graph - because the framework decodes the tensor and suppresses the boxes
itself.

Usage:
    python scripts/export_onnx.py --weights yolo11n.pt --imgsz 640 --opset 17
    python scripts/export_onnx.py --weights yolo11n.pt --output deploy/models/yolo11n/yolo11n.onnx

ultralytics is an offline dependency and is not part of the framework
requirements.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True, help="the .pt checkpoint")
    parser.add_argument("--output", type=str, default="", help="the .onnx to write")
    parser.add_argument("--imgsz", type=int, default=640, help="the square input size")
    parser.add_argument("--opset", type=int, default=17, help="the opset of the export")
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="keep the graph unsimplified (the default)",
    )
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.is_file():
        print(f"the checkpoint '{weights}' does not exist")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics is not installed; it is an offline dependency of the "
            "export, not of the framework"
        )
        return 2

    model = YOLO(str(weights))
    produced = Path(
        model.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=not args.no_simplify,
            nms=False,
            dynamic=False,
        )
    )
    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(produced, target)
        print(f"wrote {target}")
        print(
            "check the tensor names of the artifact and declare them in the "
            "manifest:"
        )
        _print_io(target)
    else:
        print(f"wrote {produced}")
        _print_io(produced)
    return 0


def _print_io(path: Path) -> None:
    """Print the tensor names of an onnx artifact."""
    try:
        import onnxruntime as ort
    except ImportError:
        return
    session = ort.InferenceSession(str(path))
    print("  inputs :", [(i.name, i.shape) for i in session.get_inputs()])
    print("  outputs:", [(o.name, o.shape) for o in session.get_outputs()])


if __name__ == "__main__":
    raise SystemExit(main())
