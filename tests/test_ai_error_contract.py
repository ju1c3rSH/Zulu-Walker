"""Contract tests for the AI layer error handling.

A backend must never fail silently:

* a contract or configuration violation raises, because continuing would
  silently produce wrong boxes;
* a runtime failure is recorded with a stable error code and leaves the
  backend in the degraded state instead of looking like an empty scene.
"""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from framework.hal.interface.ai import (
    ContractError,
    Detection,
    E_INFER_FAILED,
    E_MODEL_TYPE_UNSUPPORTED,
    E_OUTPUT_LAYOUT_MISMATCH,
    ModelConfigError,
)

AI_PATH = (
    Path(__file__).resolve().parents[1]
    / "framework" / "hal" / "platforms" / "maixcam2" / "ai.py"
)


class _FakeRuntime:
    """Stand-in for a runtime class that decodes detections."""

    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs

    def detect(self, img, conf_th=0.3, iou_th=0.4, **kwargs):
        return []


def _load_platform_module(monkeypatch):
    """Load the maixcam2 AI module against a fake maix package."""
    maix = types.ModuleType("maix")
    nn = types.ModuleType("maix.nn")
    image = types.ModuleType("maix.image")

    class YOLO11(_FakeRuntime):
        pass

    class Classifier(_FakeRuntime):
        pass

    class HandLandmarks(_FakeRuntime):
        pass

    class NN(_FakeRuntime):
        pass

    nn.YOLO11 = YOLO11
    nn.Classifier = Classifier
    nn.HandLandmarks = HandLandmarks
    nn.NN = NN
    image.cv2image = lambda *a, **k: None
    image.image2cv = lambda *a, **k: None
    maix.nn = nn
    maix.image = image

    monkeypatch.setitem(sys.modules, "maix", maix)
    monkeypatch.setitem(sys.modules, "maix.nn", nn)
    monkeypatch.setitem(sys.modules, "maix.image", image)

    spec = importlib.util.spec_from_file_location("cam2_ai_under_test", AI_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_ai(mod, model, family="yolo11"):
    ai = mod.MaixCam2AI()
    ai._model = model
    ai._model_family = family
    ai._model_type = "auto"
    ai._active_name = "test"
    return ai


def _write_mud(path, model_type):
    path.write_text(
        "[basic]\n"
        "type = axmodel\n"
        "model = model.axmodel\n"
        "\n"
        "[extra]\n"
        f"model_type = {model_type}\n",
        encoding="utf-8",
    )
    return str(path)


# ---------- model type resolution ---------- #

def test_unknown_model_type_raises_instead_of_falling_back(monkeypatch):
    mod = _load_platform_module(monkeypatch)
    with pytest.raises(ModelConfigError) as exc:
        mod.MaixCam2AI._resolve_class("yolov99", "")
    assert exc.value.code == E_MODEL_TYPE_UNSUPPORTED


def test_auto_resolves_family_from_the_mud_declaration(monkeypatch, tmp_path):
    mod = _load_platform_module(monkeypatch)
    mud = _write_mud(tmp_path / "model.mud", "yolo11")
    slot_cls, family = mod.MaixCam2AI._resolve_class("auto", mud)
    assert family == "yolo11"
    assert slot_cls is not None


def test_auto_with_unknown_mud_family_raises(monkeypatch, tmp_path):
    mod = _load_platform_module(monkeypatch)
    mud = _write_mud(tmp_path / "model.mud", "yolov42")
    with pytest.raises(ModelConfigError):
        mod.MaixCam2AI._resolve_class("auto", mud)


def test_auto_without_readable_declaration_falls_back_to_the_default(monkeypatch, tmp_path):
    mod = _load_platform_module(monkeypatch)
    missing = str(tmp_path / "absent.mud")
    slot_cls, family = mod.MaixCam2AI._resolve_class("auto", missing)
    assert family == "yolo11"
    assert slot_cls is not None


# ---------- keypoint layout ---------- #

def test_keypoints_use_the_declared_stride(monkeypatch):
    mod = _load_platform_module(monkeypatch)
    obj = types.SimpleNamespace(points=[1, 2, 3, 4, 5, 6])
    kps = mod.MaixCam2AI._convert_keypoints(obj, "yolo11")
    assert len(kps) == 2
    assert (kps[0].x, kps[0].y, kps[0].score) == (1.0, 2.0, 3.0)


def test_ambiguous_keypoint_payload_follows_the_declaration(monkeypatch):
    mod = _load_platform_module(monkeypatch)
    obj = types.SimpleNamespace(points=[1, 2, 3, 4, 5, 6])
    kps = mod.MaixCam2AI._convert_keypoints(obj, "yolov5")
    assert len(kps) == 3


def test_unpackable_keypoint_payload_raises(monkeypatch):
    mod = _load_platform_module(monkeypatch)
    obj = types.SimpleNamespace(points=[1, 2, 3, 4, 5])
    with pytest.raises(ContractError) as exc:
        mod.MaixCam2AI._convert_keypoints(obj, "yolo11")
    assert exc.value.code == E_OUTPUT_LAYOUT_MISMATCH


def test_keypoints_without_a_declared_family_raise(monkeypatch):
    mod = _load_platform_module(monkeypatch)
    obj = types.SimpleNamespace(points=[1, 2, 3, 4])
    with pytest.raises(ContractError):
        mod.MaixCam2AI._convert_keypoints(obj, "mystery")


# ---------- failure classification ---------- #

def test_runtime_failure_is_recorded_instead_of_raised(monkeypatch):
    mod = _load_platform_module(monkeypatch)

    class Boom:
        def detect(self, *a, **k):
            raise RuntimeError("npu timeout")

    ai = _make_ai(mod, Boom())
    assert ai.detect(np.zeros((4, 4, 3), dtype=np.uint8)) == []
    assert ai.last_error_code == E_INFER_FAILED
    assert ai.error_count == 1
    assert ai.degraded is True


def test_model_without_detect_raises_a_contract_error(monkeypatch):
    mod = _load_platform_module(monkeypatch)
    ai = _make_ai(mod, object())
    with pytest.raises(ContractError) as exc:
        ai.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert exc.value.code == E_OUTPUT_LAYOUT_MISMATCH
    assert ai.degraded is True


def test_a_successful_frame_clears_the_error_state(monkeypatch):
    mod = _load_platform_module(monkeypatch)

    class Ok:
        def detect(self, *a, **k):
            return []

    ai = _make_ai(mod, Ok())
    ai.detect(np.zeros((4, 4, 3), dtype=np.uint8))
    assert ai.degraded is False
    assert ai.last_error_code == ""


# ---------- backends expose the same error surface ---------- #

def test_stub_backends_expose_the_error_state():
    from framework.hal.interface import AIInference
    from framework.hal.platforms.mock.ai import MockAI

    ai = MockAI()
    assert ai.last_error_code == ""
    assert ai.error_count == 0
    assert ai.degraded is False
    assert isinstance(ai, AIInference)


def test_the_desktop_backend_exposes_the_same_error_state():
    """The desktop backend is kernel driven now, not a stub."""
    from framework.hal.interface import AIInference
    from framework.hal.platforms.linux.ai import LinuxAI

    try:
        ai = LinuxAI()
    except Exception:
        pytest.skip("no inference engine is available on this machine")
    assert ai.last_error_code == ""
    assert ai.error_count == 0
    assert ai.degraded is False
    assert isinstance(ai, AIInference)


# ---------- processor mapping ---------- #

def _processor_module():
    pytest.importorskip("cv2")
    from modules.zw_opencv_module.processors.ai_inference_processor import (
        AIInferenceProcessor,
    )
    return AIInferenceProcessor


def test_processor_maps_a_contract_error_to_an_error_code():
    AIInferenceProcessor = _processor_module()

    class FakeAI:
        loaded = True
        model_type = "auto"

        def detect(self, *a, **k):
            raise ModelConfigError("bad model type", E_MODEL_TYPE_UNSUPPORTED)

    processor = AIInferenceProcessor("ai")
    processor.set_ai(FakeAI())
    result = processor.process(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result.success is False
    assert result.error_code == E_MODEL_TYPE_UNSUPPORTED


def test_processor_reports_filter_reasons():
    AIInferenceProcessor = _processor_module()

    class FakeAI:
        loaded = True
        model_type = "auto"

        def detect(self, *a, **k):
            # 5x5 px box: below the area threshold, so it gets filtered.
            return [Detection(x=0, y=0, w=5, h=5, class_id=0, score=0.9)]

    processor = AIInferenceProcessor("ai")
    processor.set_ai(FakeAI())
    result = processor.process(np.zeros((64, 64, 3), dtype=np.uint8))
    assert result.success is True
    assert result.result_data["discard_reasons"]["area_too_small"] == 1
