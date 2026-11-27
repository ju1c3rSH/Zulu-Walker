"""Unit tests for the split of the AI backend contract.

Run: python -m pytest tests/test_inference_protocol_split.py -q

The base protocol carries the registry, the state and the detection task;
the tasks that not every model provides live in their own optional
protocols.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.hal.interface import (
    AIInference,
    AIInferenceClassify,
    AIInferenceMask,
    AIInferenceSegment,
)
from framework.hal.interface.ai import E_DIALECT_UNSUPPORTED
from framework.inference.contract import Capabilities
from framework.inference.facade import AIFacade
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


def _facade(tmp_path: Path) -> AIFacade:
    """A facade over the fake engine, with one registered model."""
    tiny_pair(tmp_path)
    manifest_path = write_manifest(tmp_path, "fake_engine", onnx_artifact())
    facade = AIFacade(
        "fake_engine", engine_factory=lambda: FakeEngine()
    )
    facade.add(
        "tiny",
        str(tmp_path / "tiny.onnx"),
        manifest_path=str(manifest_path),
    )
    facade.switch("tiny")
    return facade


def test_the_base_protocol_carries_the_detection_task(tmp_path):
    facade = _facade(tmp_path)
    assert isinstance(facade, AIInference)


def test_the_optional_protocols_are_separate(tmp_path):
    facade = _facade(tmp_path)
    assert isinstance(facade, AIInferenceSegment)
    assert isinstance(facade, AIInferenceClassify)
    assert isinstance(facade, AIInferenceMask)


def test_a_task_the_dialect_does_not_provide_is_reported(tmp_path):
    """A dialect without segmentation says so instead of returning nothing."""
    facade = _facade(tmp_path)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(Exception) as exc:
        facade.segment(frame)
    assert exc.value.code == E_DIALECT_UNSUPPORTED


def test_the_base_protocol_reports_the_family_of_the_active_model(tmp_path):
    facade = _facade(tmp_path)
    assert facade.model_family == "yolo11"
