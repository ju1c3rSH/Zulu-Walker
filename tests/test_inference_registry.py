"""Tests for the inference engine registry and the maix.nn adapter.

Run: python -m pytest tests/test_inference_registry.py -q
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from framework.hal.interface.ai import (
    E_ENGINE_UNAVAILABLE,
    E_MODEL_TYPE_UNSUPPORTED,
    ModelConfigError,
)
from framework.inference.contract import ArtifactRef
from framework.inference.engines import (
    REGISTRY,
    available_engines,
    engine_factory,
    register_engine,
)
from framework.inference.engines.maix_nn import (
    ENGINE_ID,
    MaixNnEngine,
    model_class_name,
    model_family_from_mud,
)
from framework.inference.registry import engine_availability, resolve_engine

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWN_ENGINE_IDS = ("maix_nn", "ncnn", "onnxruntime", "rknn_lite")
ENGINE_LIBRARIES = ("maix", "ncnn", "onnxruntime", "rknn_lite")
RUNTIME_LIBRARY_OF = {
    "maix_nn": "maix.nn",
    "ncnn": "ncnn",
    "onnxruntime": "onnxruntime",
    "rknn_lite": "rknn_lite",
}


@dataclass
class _FakeIO:
    outputs: dict[str, str] = field(default_factory=lambda: {"objects": "objects"})


@dataclass
class _FakeArtifact:
    io: _FakeIO


class _FakeModel:
    def __init__(self, objects):
        self._objects = objects
        self.frames = []
        self.calls = []

    def detect(self, frame, **kwargs):
        self.frames.append(frame)
        self.calls.append(dict(kwargs))
        return self._objects


def _module_present(module_name: str) -> bool:
    """True when the module can be located, without executing it."""
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def _mud(tmp_path: Path, model_type: str) -> str:
    mud = tmp_path / "model.mud"
    mud.write_text(f"[extra]\nmodel_type = {model_type}\n", encoding="utf-8")
    return str(mud)


def _artifact(path: str, outputs: dict[str, str] | None = None) -> ArtifactRef:
    return ArtifactRef(
        engine_id=ENGINE_ID,
        path=path,
        wrapper="vision.det.end2end",
        wrapper_params={},
        preprocess=object(),
        io=_FakeIO(outputs=outputs or {"objects": "objects"}),
    )


def test_registry_lists_every_known_engine():
    assert set(REGISTRY) == set(KNOWN_ENGINE_IDS)


def test_engine_factory_returns_the_binding_of_a_known_id():
    assert engine_factory("maix_nn") is REGISTRY["maix_nn"]


def test_engine_factory_of_an_unknown_id_is_none():
    assert engine_factory("no_such_engine") is None


def test_register_engine_binds_a_factory_to_an_id():
    def factory():
        return None

    register_engine("temporary_engine", factory)
    try:
        assert engine_factory("temporary_engine") is factory
        assert "temporary_engine" in REGISTRY
    finally:
        REGISTRY.pop("temporary_engine")


def test_unknown_engine_id_raises_engine_unavailable():
    with pytest.raises(ModelConfigError) as excinfo:
        resolve_engine("no_such_engine")
    assert excinfo.value.code == E_ENGINE_UNAVAILABLE


def test_engine_without_its_runtime_library_is_reported_unavailable():
    def factory():
        return None

    register_engine("unavailable_engine", factory)
    try:
        assert engine_factory("unavailable_engine")() is None
        assert "unavailable_engine" not in available_engines()
        with pytest.raises(ModelConfigError) as excinfo:
            resolve_engine("unavailable_engine")
        assert excinfo.value.code == E_ENGINE_UNAVAILABLE
    finally:
        REGISTRY.pop("unavailable_engine")


def test_a_failing_factory_is_reported_as_unavailable_instead_of_raising():
    def factory():
        raise RuntimeError("library is broken")

    register_engine("broken_engine", factory)
    try:
        with pytest.raises(ModelConfigError) as excinfo:
            resolve_engine("broken_engine")
        assert excinfo.value.code == E_ENGINE_UNAVAILABLE
    finally:
        REGISTRY.pop("broken_engine")


def test_resolve_engine_returns_the_engine_of_the_factory():
    sentinel = object()

    def factory():
        return sentinel

    register_engine("working_engine", factory)
    try:
        assert resolve_engine("working_engine") is sentinel
    finally:
        REGISTRY.pop("working_engine")


def test_importing_the_engines_package_pulls_in_no_engine_library():
    code = (
        "import sys\n"
        "import framework.inference.engines\n"
        "print([name for name in "
        f"{list(ENGINE_LIBRARIES)!r} if name in sys.modules])\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"


def test_importing_the_maix_nn_adapter_pulls_in_no_maix():
    code = (
        "import sys\n"
        "import framework.inference.engines.maix_nn\n"
        "print('maix' in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False"


def test_model_family_is_read_from_the_extra_section_of_the_mud(tmp_path):
    assert model_family_from_mud(_mud(tmp_path, "yolo11")) == "yolo11"


def test_model_family_is_empty_for_a_missing_or_broken_mud(tmp_path):
    assert model_family_from_mud(str(tmp_path / "absent.mud")) == ""
    broken = tmp_path / "broken.mud"
    broken.write_text("this is not an ini file\n", encoding="utf-8")
    assert model_family_from_mud(str(broken)) == ""


def test_model_family_maps_to_its_runtime_class_name():
    assert model_class_name("yolo11") == "YOLO11"
    assert model_class_name("yolov8") == "YOLOv8"
    assert model_class_name("yolov5") == "YOLOv5"
    assert model_class_name("classifier") == "Classifier"
    assert model_class_name("hand_landmarks") == "HandLandmarks"


def test_unknown_model_family_raises_model_type_unsupported():
    with pytest.raises(ModelConfigError) as excinfo:
        model_class_name("yolo42")
    assert excinfo.value.code == E_MODEL_TYPE_UNSUPPORTED


def test_load_rejects_an_unknown_model_family(tmp_path):
    engine = MaixNnEngine()
    with pytest.raises(ModelConfigError) as excinfo:
        engine.load(_artifact(_mud(tmp_path, "yolo42")))
    assert excinfo.value.code == E_MODEL_TYPE_UNSUPPORTED


@pytest.mark.skipif(
    _module_present("maix.nn"), reason="the maix.nn runtime is installed here"
)
def test_load_without_the_maix_runtime_reports_engine_unavailable(tmp_path):
    engine = MaixNnEngine()
    with pytest.raises(ModelConfigError) as excinfo:
        engine.load(_artifact(_mud(tmp_path, "yolo11")))
    assert excinfo.value.code == E_ENGINE_UNAVAILABLE


def test_run_returns_the_decoded_objects_under_the_logical_output_name(tmp_path):
    engine = MaixNnEngine()
    objects = [object(), object()]
    model = _FakeModel(objects)
    engine._model = model
    engine._artifact = _artifact(str(tmp_path / "model.mud"))

    result = engine.run({"image": "frame"})

    assert result == {"objects": objects}
    assert model.frames == ["frame"]


def test_run_forwards_the_thresholds_and_drops_the_unknown_options(tmp_path):
    engine = MaixNnEngine()
    model = _FakeModel([])
    engine._model = model
    engine._artifact = _artifact(str(tmp_path / "model.mud"))

    engine.run({"image": "frame"}, conf_th=0.3, iou_th=0.4, unknown=1)

    assert model.calls == [{"conf_th": 0.3, "iou_th": 0.4}]


def test_run_without_options_leaves_the_runtime_defaults_alone(tmp_path):
    engine = MaixNnEngine()
    model = _FakeModel([])
    engine._model = model
    engine._artifact = _artifact(str(tmp_path / "model.mud"))

    engine.run({"image": "frame"})

    assert model.calls == [{}]


def test_run_uses_the_first_logical_output_of_the_artifact(tmp_path):
    engine = MaixNnEngine()
    engine._model = _FakeModel([])
    engine._artifact = _artifact(
        str(tmp_path / "model.mud"), outputs={"pred": "objects", "extra": "more"}
    )

    assert engine.run({"image": "frame"}) == {"pred": []}


def test_run_without_a_loaded_model_fails():
    engine = MaixNnEngine()
    with pytest.raises(Exception):
        engine.run({"image": "frame"})


def test_maix_nn_declares_npu_int8_and_the_held_gil():
    capabilities = MaixNnEngine().capabilities()
    assert capabilities.device == "npu"
    assert capabilities.quant == "int8"
    assert capabilities.input_accepts == ("native_image", "numpy_bgr")
    assert capabilities.io_binding == "positional"
    assert capabilities.layouts == ("nhwc",)
    assert capabilities.releases_gil is False
    assert capabilities.load_holds_gil is True


def test_maix_nn_declares_no_tensor_specs():
    engine = MaixNnEngine()
    assert engine.input_specs() == []
    assert engine.output_specs() == []


def test_availability_probe_reports_every_registered_engine():
    availability = engine_availability()
    assert set(availability) == set(REGISTRY)
    assert all(isinstance(value, bool) for value in availability.values())


def test_an_engine_without_its_runtime_library_is_never_available():
    availability = engine_availability()
    for engine_id, library in RUNTIME_LIBRARY_OF.items():
        if not _module_present(library):
            assert availability[engine_id] is False


def test_available_engines_agrees_with_the_availability_probe():
    availability = engine_availability()
    expected = {engine_id for engine_id, ok in availability.items() if ok}
    assert set(available_engines()) == expected
