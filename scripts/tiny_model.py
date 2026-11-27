"""Build a tiny model pair that exists in both engine dialects.

The pair exists for one reason: two engines can only be compared when they
run the same weights.  A real artifact pair needs a C++ toolchain to build
(onnx2ncnn or pnnx), so this module builds the smallest possible network -
one convolution - in both dialects from one set of weights.

The two dialects agree by construction:

* the onnxruntime side is an NCHW graph, so its tensor memory is
  C planes of an H x W image;
* the ncnn side is a planar Mat, whose memory is C planes of an H x W
  image as well.

The weights share the onnx layout (out_c, in_c, kh, kw), which is also the
layout ncnn expects, so the same bytes feed both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

__all__ = ["TINY_CHANNELS", "TINY_HEIGHT", "TINY_WIDTH", "build_tiny_pair"]

TINY_CHANNELS = 3
TINY_HEIGHT = 4
TINY_WIDTH = 4
TINY_OUTPUTS = 2
TINY_KERNEL = 3

# The blob names of the ncnn graph, which the manifest must declare.
NCNN_INPUT_BLOB = "data"
NCNN_OUTPUT_BLOB = "conv"

# The tensor names of the onnx graph.
ONNX_INPUT_NAME = "images"
ONNX_OUTPUT_NAME = "output0"


def _weights() -> Tuple[np.ndarray, np.ndarray]:
    """The weights of the tiny convolution, in the onnx layout."""
    count = TINY_OUTPUTS * TINY_CHANNELS * TINY_KERNEL * TINY_KERNEL
    values = np.linspace(-0.5, 0.5, count, dtype=np.float32)
    weight = values.reshape(
        TINY_OUTPUTS, TINY_CHANNELS, TINY_KERNEL, TINY_KERNEL
    )
    bias = np.array([0.25, -0.25], dtype=np.float32)
    return weight, bias


def build_tiny_pair(directory: str | Path) -> dict:
    """Write the onnx and the ncnn artifacts of the tiny model.

    Returns the paths of the three files and the weights they share.
    """
    base = Path(directory)
    base.mkdir(parents=True, exist_ok=True)
    weight, bias = _weights()

    onnx_path = base / "tiny.onnx"
    param_path = base / "tiny.param"
    bin_path = base / "tiny.bin"

    _write_onnx(onnx_path, weight, bias)
    _write_ncnn(param_path, bin_path, weight, bias)

    return {
        "onnx": str(onnx_path),
        "param": str(param_path),
        "bin": str(bin_path),
        "weight": weight,
        "bias": bias,
        "input_shape": (1, TINY_CHANNELS, TINY_HEIGHT, TINY_WIDTH),
    }


def _write_onnx(path: Path, weight: np.ndarray, bias: np.ndarray) -> None:
    """Write the onnx graph of the tiny convolution."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    inputs = helper.make_tensor_value_info(
        ONNX_INPUT_NAME,
        TensorProto.FLOAT,
        [1, TINY_CHANNELS, TINY_HEIGHT, TINY_WIDTH],
    )
    outputs = helper.make_tensor_value_info(
        ONNX_OUTPUT_NAME,
        TensorProto.FLOAT,
        [1, TINY_OUTPUTS, TINY_HEIGHT, TINY_WIDTH],
    )
    node = helper.make_node(
        "Conv",
        [ONNX_INPUT_NAME, "conv_w", "conv_b"],
        [ONNX_OUTPUT_NAME],
        kernel_shape=[TINY_KERNEL, TINY_KERNEL],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
        dilations=[1, 1],
        group=1,
    )
    graph = helper.make_graph(
        [node],
        "tiny",
        [inputs],
        [outputs],
        initializer=[
            numpy_helper.from_array(weight.astype(np.float32), "conv_w"),
            numpy_helper.from_array(bias.astype(np.float32), "conv_b"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _write_ncnn(
    param_path: Path, bin_path: Path, weight: np.ndarray, bias: np.ndarray
) -> None:
    """Write the ncnn graph and the weights of the tiny convolution.

    The .param is the text form of the graph; the .bin is the raw weight
    data, preceded by the flag that says "float32 follows".
    """
    weight_data_size = int(weight.size)
    lines = [
        "7767517",
        "2 2",
        f"Input {NCNN_INPUT_BLOB} 0 1 {NCNN_INPUT_BLOB}",
        (
            f"Convolution {NCNN_OUTPUT_BLOB} 1 1 {NCNN_INPUT_BLOB} {NCNN_OUTPUT_BLOB} "
            f"0={TINY_OUTPUTS} 1={TINY_KERNEL} 11={TINY_KERNEL} "
            f"3=1 13=1 4=1 14=1 15=1 16=1 5=1 6={weight_data_size}"
        ),
    ]
    param_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = np.concatenate(
        [weight.reshape(-1).astype(np.float32), bias.reshape(-1).astype(np.float32)]
    )
    flag = np.zeros(1, dtype=np.int32)
    with open(bin_path, "wb") as handle:
        handle.write(flag.tobytes())
        handle.write(payload.tobytes())
