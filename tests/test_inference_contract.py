"""Contract tests for the inference kernel core data structures.

Run: python -m pytest tests/test_inference_contract.py -q
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from framework.inference.contract import (
    ArtifactRef,
    Capabilities,
    DetectionAPI,
    GeometryTransform,
    InferenceEngine,
    PreprocessResult,
    TensorSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO_ROOT / "framework" / "inference" / "contract.py"


def _artifact(**overrides):
    kwargs = {
        "engine_id": "ncnn",
        "path": "/root/models/steelball_1280x352/steelball_1280x352.param",
        "wrapper": "vision.det.raw",
        "wrapper_params": {},
        "preprocess": object(),
        "io": object(),
    }
    kwargs.update(overrides)
    return ArtifactRef(**kwargs)


def _field_names(cls) -> list[str]:
    return [f.name for f in dataclasses.fields(cls)]


def _import_roots(node) -> set[str]:
    roots: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Import):
            for alias in child.names:
                roots.add(alias.name.split(".")[0])
        elif isinstance(child, ast.ImportFrom) and not child.level and child.module:
            roots.add(child.module.split(".")[0])
    return roots


def test_tensor_spec_declares_name_shape_dtype():
    assert _field_names(TensorSpec) == ["name", "shape", "dtype"]
    spec = TensorSpec(name="image", shape=(1, 3, -1, -1), dtype="float32")
    assert spec.shape == (1, 3, -1, -1)


def test_tensor_spec_is_frozen():
    spec = TensorSpec(name="image", shape=(1, 3, 640, 640), dtype="float32")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.dtype = "uint8"


def test_capabilities_declares_exactly_the_documented_fields():
    assert _field_names(Capabilities) == [
        "device",
        "quant",
        "layouts",
        "dynamic_shape",
        "input_accepts",
        "io_binding",
        "releases_gil",
        "load_holds_gil",
        "memory_hint",
        "load_cost_hint",
    ]


def test_capabilities_optional_hints_default_to_none():
    caps = Capabilities(
        device="axera",
        quant="int8",
        layouts=("nhwc",),
        dynamic_shape=False,
        input_accepts=("native_image", "numpy_bgr"),
        io_binding="named",
        releases_gil=False,
        load_holds_gil=True,
    )
    assert caps.memory_hint is None
    assert caps.load_cost_hint is None
    assert caps.input_accepts == ("native_image", "numpy_bgr")


def test_capabilities_has_no_builtin_nms_field():
    assert "builtin_nms" not in _field_names(Capabilities)


def test_capabilities_is_frozen():
    caps = Capabilities(
        device="cpu",
        quant="fp32",
        layouts=("nchw",),
        dynamic_shape=True,
        input_accepts=("numpy_bgr",),
        io_binding="named",
        releases_gil=True,
        load_holds_gil=True,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.device = "npu"


def test_artifact_ref_declares_exactly_the_documented_fields():
    assert _field_names(ArtifactRef) == [
        "engine_id",
        "path",
        "wrapper",
        "wrapper_params",
        "preprocess",
        "io",
        "runtime",
        "build",
        "aux_paths",
    ]


def test_artifact_ref_uses_manifest_owned_types():
    fields = {f.name: f for f in dataclasses.fields(ArtifactRef)}
    assert fields["preprocess"].type == "PreprocessSpec"
    assert fields["io"].type == "IOMapping"
    assert fields["aux_paths"].type == "dict[str, str]"


def test_artifact_ref_is_frozen():
    art = _artifact()
    with pytest.raises(dataclasses.FrozenInstanceError):
        art.path = "/other/model.param"


def test_artifact_ref_aux_paths_default_is_empty_and_not_shared():
    first = _artifact()
    second = _artifact()
    assert first.aux_paths == {}
    assert second.aux_paths == {}
    first.aux_paths["bin"] = "/root/models/steelball_1280x352/steelball_1280x352.bin"
    assert second.aux_paths == {}


def test_artifact_ref_other_defaults_are_not_shared():
    first = _artifact()
    second = _artifact()
    first.runtime["pkg"] = "ncnn"
    first.build["recipe"] = "x"
    assert second.runtime == {}
    assert second.build == {}


def test_artifact_ref_accepts_a_companion_file():
    art = _artifact(
        aux_paths={"bin": "/root/models/steelball_1280x352/steelball_1280x352.bin"},
    )
    assert art.path.endswith(".param")
    assert art.aux_paths["bin"].endswith(".bin")


def test_geometry_transform_declares_the_inverse_mapping_fields():
    assert _field_names(GeometryTransform) == [
        "mode",
        "scale_x",
        "scale_y",
        "offset_x",
        "offset_y",
        "src_w",
        "src_h",
    ]


def test_geometry_transform_maps_input_pixels_back_to_the_source_frame():
    transform = GeometryTransform(
        mode="letterbox",
        scale_x=0.5,
        scale_y=0.5,
        offset_x=114,
        offset_y=0,
        src_w=1280,
        src_h=704,
    )
    x_input = 314.0
    assert (x_input - transform.offset_x) / transform.scale_x == 400.0
    assert transform.src_w == 1280


def test_preprocess_result_carries_inputs_and_transform():
    assert _field_names(PreprocessResult) == ["inputs", "transform"]
    transform = GeometryTransform(
        mode="stretch",
        scale_x=1.0,
        scale_y=1.0,
        offset_x=0,
        offset_y=0,
        src_w=640,
        src_h=640,
    )
    result = PreprocessResult(inputs={"image": object()}, transform=transform)
    assert set(result.inputs) == {"image"}
    assert result.transform is transform


def test_inference_engine_declares_the_documented_protocol_surface():
    assert InferenceEngine.__annotations__["engine_id"] == "str"
    for name in (
        "load",
        "input_specs",
        "output_specs",
        "capabilities",
        "run",
        "unload",
    ):
        assert hasattr(InferenceEngine, name), name


def test_inference_engine_method_signatures():
    load = inspect.signature(InferenceEngine.load)
    assert list(load.parameters) == ["self", "artifact", "opts"]
    assert load.parameters["opts"].kind is inspect.Parameter.VAR_KEYWORD
    run = inspect.signature(InferenceEngine.run)
    assert list(run.parameters) == ["self", "inputs"]
    assert run.return_annotation == "dict[str, Any]"
    assert inspect.signature(InferenceEngine.input_specs).return_annotation == "list[TensorSpec]"
    assert inspect.signature(InferenceEngine.output_specs).return_annotation == "list[TensorSpec]"
    assert inspect.signature(InferenceEngine.capabilities).return_annotation == "Capabilities"


def test_detection_api_exposes_detect():
    assert hasattr(DetectionAPI, "detect")
    detect = inspect.signature(DetectionAPI.detect)
    assert list(detect.parameters) == ["self", "frame", "kw"]
    assert detect.parameters["kw"].kind is inspect.Parameter.VAR_KEYWORD


def test_contract_does_not_import_manifest_at_runtime():
    code = (
        "import framework.inference.contract as c; "
        "import sys; "
        "assert 'framework.inference.manifest' not in sys.modules; "
        "print('OK')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_contract_runtime_imports_are_stdlib_only():
    tree = ast.parse(CONTRACT_PATH.read_text(encoding="utf-8"))
    runtime_roots: set[str] = set()
    guarded_roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            guarded_roots |= _import_roots(node)
            continue
        runtime_roots |= _import_roots(node)
    assert runtime_roots <= {"__future__", "dataclasses", "typing"}
    assert "framework" in guarded_roots
    for forbidden in ("numpy", "cv2", "maix", "ncnn", "onnxruntime", "rknn_lite"):
        assert forbidden not in runtime_roots
        assert forbidden not in guarded_roots


def test_contract_exports_its_public_names():
    import framework.inference.contract as contract

    assert set(contract.__all__) == {
        "ArtifactRef",
        "Capabilities",
        "DetectionAPI",
        "GeometryTransform",
        "InferenceEngine",
        "PreprocessResult",
        "TensorSpec",
    }
