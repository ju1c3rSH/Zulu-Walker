"""Unit tests for the desktop AI backend.

Run: python -m pytest tests/test_desktop_ai_facade.py -q

The tests drive the backend over a fake engine, so no inference library and
no device are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.hal.interface import AIInference
from framework.hal.interface.ai import ManifestError
from framework.hal.platforms.linux.ai import LinuxAI
from framework.inference.contract import Capabilities
from helpers_inference import onnx_artifact, tiny_pair, write_manifest


class FakeEngine:
    """An engine that answers with one detection."""

    engine_id = "fake_engine"

    def __init__(self) -> None:
        self.loaded = False

    def load(self, artifact, **opts) -> None:
        self.loaded = True

    def input_specs(self):
        return []

    def output_specs(self):
        return []

    def capabilities(self) -> Capabilities:
        return Capabilities(
            device="cpu",
            quant="fp32",
            layouts=("nhwc",),
            dynamic_shape=False,
            input_accepts=("numpy_bgr",),
            io_binding="named",
            releases_gil=True,
            load_holds_gil=False,
        )

    def run(self, inputs, **options):
        return {
            "pred": np.array(
                [[10.0], [100.0], [20.0], [20.0], [0.9]], dtype=np.float32
            )
        }

    def unload(self) -> None:
        self.loaded = False


def _backend(tmp_path: Path) -> LinuxAI:
    """A desktop backend over the fake engine."""
    return LinuxAI("fake_engine", engine_factory=lambda: FakeEngine())


def test_the_backend_is_not_a_stub_anymore(tmp_path):
    """The desktop backend used to report a failure for every frame."""
    backend = _backend(tmp_path)
    assert backend.last_error_code == ""
    assert backend.error_count == 0
    assert backend.degraded is False
    assert isinstance(backend, AIInference)


def test_the_backend_needs_a_manifest_to_register_a_model(tmp_path):
    backend = _backend(tmp_path)
    with pytest.raises(ManifestError):
        backend.add("tiny", str(tmp_path / "tiny.onnx"))


def test_the_backend_registers_and_switches_a_model(tmp_path):
    backend = _backend(tmp_path)
    tiny_pair(tmp_path)
    manifest_path = write_manifest(tmp_path, "fake_engine", onnx_artifact())
    backend.add(
        "tiny",
        str(tmp_path / "tiny.onnx"),
        manifest_path=str(manifest_path),
    )
    assert backend.models == ["tiny"]
    assert backend.switch("tiny") is True
    assert backend.loaded is True
    assert backend.model_family == "yolo11"
    assert backend.labels == ["ball"]


def test_the_backend_detects_on_a_frame(tmp_path):
    backend = _backend(tmp_path)
    tiny_pair(tmp_path)
    manifest_path = write_manifest(tmp_path, "fake_engine", onnx_artifact())
    backend.add(
        "tiny",
        str(tmp_path / "tiny.onnx"),
        manifest_path=str(manifest_path),
    )
    backend.switch("tiny")

    detections = backend.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "ball"
    assert backend.last_error_code == ""


def test_the_backend_reports_the_frame_contract_it_declares(tmp_path):
    """The platform declares the shape of the frames it delivers."""
    backend = _backend(tmp_path)
    tiny_pair(tmp_path)
    manifest_path = write_manifest(tmp_path, "fake_engine", onnx_artifact())
    backend.add(
        "tiny",
        str(tmp_path / "tiny.onnx"),
        manifest_path=str(manifest_path),
    )
    backend.switch("tiny")
    assert backend.input_width == 4
    assert backend.input_height == 4
