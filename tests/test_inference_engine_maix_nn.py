"""Unit tests for the maix.nn engine adapter.

Run: python -m pytest tests/test_inference_engine_maix_nn.py -q

The firmware is not installed on a development machine, so the tests inject
a stand-in ``maix.nn`` package.  What is under test is the adapter's contract
behaviour: which options it forwards to the firmware constructor, which it
drops, and how it maps the objects the runtime hands back.

The stand-in constructor is deliberately strict about its keyword arguments,
matching the ``py::init`` binding of the firmware.  A wider signature would
hide exactly the defect these tests exist to catch.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.hal.interface.ai import (
    E_INVALID_INPUT,
    E_MODEL_NOT_LOADED,
    E_MODEL_TYPE_UNSUPPORTED,
    AIError,
    ModelConfigError,
)
from framework.inference.contract import ArtifactRef
from framework.inference.engines import maix_nn
from framework.inference.manifest import IOMapping, PreprocessSpec

ENGINE_ID = "maix_nn"
NEWLINE = chr(10)


class FakeObject:
    """One decoded object as maix.nn reports it."""

    def __init__(self, class_id: int, score: float) -> None:
        self.class_id = class_id
        self.score = score
        self.x = 1
        self.y = 2
        self.w = 3
        self.h = 4


class FakeYOLO11:
    """Stand-in for the firmware class, recording how it was constructed."""

    instances: list["FakeYOLO11"] = []

    def __init__(self, model: str, dual_buff: bool = False) -> None:
        self.model = model
        self.dual_buff = dual_buff
        self.detect_calls: list[dict[str, Any]] = []
        self.objects = [FakeObject(0, 0.9)]
        FakeYOLO11.instances.append(self)

    def detect(self, frame: Any, **options: Any) -> list[FakeObject]:
        self.detect_calls.append({"frame": frame, **options})
        return self.objects


@pytest.fixture()
def fake_firmware(monkeypatch):
    """Install a stand-in ``maix.nn`` package for one test."""
    FakeYOLO11.instances.clear()

    nn_module = types.ModuleType("maix.nn")
    nn_module.YOLO11 = FakeYOLO11

    image_module = types.ModuleType("maix.image")
    image_module.cv2image = lambda array, bgr=False, copy=True: ("native", array.shape)

    parent = types.ModuleType("maix")
    parent.nn = nn_module
    parent.image = image_module

    monkeypatch.setitem(sys.modules, "maix", parent)
    monkeypatch.setitem(sys.modules, "maix.nn", nn_module)
    monkeypatch.setitem(sys.modules, "maix.image", image_module)
    return nn_module


def _artifact(tmp_path: Path, family: str = "yolo11") -> ArtifactRef:
    """An artifact of one baked MUD file, as the runtime hands it over."""
    mud = tmp_path / "tiny.mud"
    mud.write_text(f"[extra]{NEWLINE}model_type={family}{NEWLINE}", encoding="utf-8")
    return ArtifactRef(
        engine_id=ENGINE_ID,
        path=str(mud),
        wrapper="vision.det.end2end",
        wrapper_params={},
        preprocess=PreprocessSpec(resize_owner="baked", norm_owner="baked"),
        io=IOMapping(inputs={"image": "image"}, outputs={"objects": "objects"}),
    )


def _loaded(tmp_path: Path, **opts: Any):
    """The engine after a successful load, and the model it constructed."""
    engine = maix_nn.create_engine()
    engine.load(_artifact(tmp_path), **opts)
    return engine, FakeYOLO11.instances[-1]


def test_the_engine_is_available_when_the_firmware_is_importable(fake_firmware):
    assert maix_nn.create_engine() is not None


def test_the_engine_is_unavailable_without_the_firmware(monkeypatch):
    monkeypatch.setitem(sys.modules, "maix", None)
    monkeypatch.setitem(sys.modules, "maix.nn", None)
    assert maix_nn.create_engine() is None


def test_load_drops_the_options_the_firmware_does_not_accept(fake_firmware, tmp_path):
    """The framework keys are accepted and ignored, never forwarded.

    The runtime hands every engine ``canonical`` and ``runtime``.  The
    firmware constructor takes neither, so forwarding them turns a healthy
    model into E_MODEL_LOAD_FAILED.
    """
    engine, model = _loaded(
        tmp_path, canonical=object(), runtime={"device": "npu"}
    )
    assert model.dual_buff is True
    assert engine.engine_id == ENGINE_ID


def test_load_forwards_the_options_the_firmware_does_accept(fake_firmware, tmp_path):
    engine, model = _loaded(
        tmp_path, canonical=object(), runtime={}, dual_buff=False
    )
    assert model.dual_buff is False


def test_load_refuses_an_unknown_family(fake_firmware, tmp_path):
    engine = maix_nn.create_engine()
    with pytest.raises(ModelConfigError) as caught:
        engine.load(_artifact(tmp_path, family="magic_net"))
    assert caught.value.code == E_MODEL_TYPE_UNSUPPORTED


def test_run_returns_the_objects_under_the_logical_output(fake_firmware, tmp_path):
    engine, model = _loaded(tmp_path)
    frame = object()

    out = engine.run({"image": frame}, conf_th=0.5, iou_th=0.45)

    assert list(out) == ["objects"]
    assert out["objects"] is model.objects
    assert model.detect_calls == [
        {"frame": frame, "conf_th": 0.5, "iou_th": 0.45}
    ]


def test_run_drops_the_options_the_firmware_does_not_know(fake_firmware, tmp_path):
    engine, model = _loaded(tmp_path)

    engine.run({"image": object()}, conf_th=0.4, bbox_norm="none")

    assert model.detect_calls[0] == {"frame": model.detect_calls[0]["frame"], "conf_th": 0.4}


def test_run_without_a_loaded_model_is_refused(fake_firmware):
    engine = maix_nn.create_engine()
    with pytest.raises(AIError) as caught:
        engine.run({"image": object()})
    assert caught.value.code == E_MODEL_NOT_LOADED


def test_run_without_the_logical_input_is_refused(fake_firmware, tmp_path):
    engine, _model = _loaded(tmp_path)
    with pytest.raises(ModelConfigError) as caught:
        engine.run({"frame": object()})
    assert caught.value.code == E_INVALID_INPUT


def test_a_bgr_frame_is_converted_to_the_native_image(fake_firmware, tmp_path):
    """The canonical HWC array goes through the firmware image converter."""
    engine, model = _loaded(tmp_path)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    engine.run({"image": frame})

    received = model.detect_calls[0]["frame"]
    assert received[0] == "native"
    assert received[1] == (4, 4, 3)
