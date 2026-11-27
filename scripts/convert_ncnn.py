#!/usr/bin/env python
"""Convert an onnx artifact into the ncnn artifact of the framework.

The converter is a C++ tool of the ncnn project, so it is not part of the
framework and not part of its requirements.  This script finds it, runs it
and then reports the blob names of the result: the ncnn weight file carries
neither the names nor the shapes of the tensors, so the manifest is the only
declaration of the io mapping and it must be written from what the tool
produced.

Usage:
    python scripts/convert_ncnn.py --onnx deploy/models/yolo11n/yolo11n.onnx
    python scripts/convert_ncnn.py --onnx <f.onnx> --converter C:/tools/onnx2ncnn.exe

The converter is looked up in this order:

1. the --converter argument
2. the NCNN_CONVERTER environment variable
3. onnx2ncnn and pnnx on the PATH
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONVERTER_NAMES = ("onnx2ncnn", "onnx2ncnn.exe", "pnnx", "pnnx.exe")


def _find_converter(explicit: str) -> Optional[Path]:
    """Locate the ncnn conversion tool."""
    if explicit:
        path = Path(explicit)
        return path if path.is_file() else None
    from_env = os.environ.get("NCNN_CONVERTER", "").strip()
    if from_env:
        path = Path(from_env)
        if path.is_file():
            return path
    for name in CONVERTER_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _run_converter(converter: Path, onnx_path: Path, out_dir: Path) -> bool:
    """Run the converter and report whether it produced the artifacts."""
    name = converter.name.lower()
    if name.startswith("onnx2ncnn"):
        param = out_dir / (onnx_path.stem + ".param")
        binary = out_dir / (onnx_path.stem + ".bin")
        command = [str(converter), str(onnx_path), str(param), str(binary)]
    else:
        param = out_dir / (onnx_path.stem + ".ncnn.param")
        binary = out_dir / (onnx_path.stem + ".ncnn.bin")
        command = [str(converter), str(onnx_path)]
    print("running:", " ".join(command))
    result = subprocess.run(command, cwd=str(out_dir))
    if result.returncode != 0:
        return False
    if not param.is_file() or not binary.is_file():
        print(f"the converter did not produce {param.name} and {binary.name}")
        return False
    return True


def _print_blob_names(param: Path) -> None:
    """Print the blob names of the ncnn graph, for the manifest."""
    try:
        import ncnn
    except ImportError:
        print("ncnn is not installed, so the blob names cannot be read back")
        return
    net = ncnn.Net()
    if net.load_param(str(param)) != 0:
        print("the generated graph does not load")
        return
    print("  input blobs :", list(net.input_names()))
    print("  output blobs:", list(net.output_names()))
    print("  declare them in the manifest:")
    print("    io: { inputs: { image: <input blob> }, outputs: { pred: <output blob> } }")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=str, required=True, help="the .onnx artifact")
    parser.add_argument("--converter", type=str, default="", help="the converter to use")
    parser.add_argument("--out-dir", type=str, default="", help="where to write the artifacts")
    args = parser.parse_args()

    onnx_path = Path(args.onnx)
    if not onnx_path.is_file():
        print(f"the onnx artifact '{onnx_path}' does not exist")
        return 2

    converter = _find_converter(args.converter)
    if converter is None:
        print(
            "no ncnn converter found.  Install the ncnn tools (onnx2ncnn or "
            "pnnx), or point NCNN_CONVERTER at the executable."
        )
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else onnx_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    if not _run_converter(converter, onnx_path, out_dir):
        return 1
    _print_blob_names(out_dir / (onnx_path.stem + ".param"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
