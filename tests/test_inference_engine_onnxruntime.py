"""Unit tests for the onnxruntime engine adapter.

Run: python -m pytest tests/test_inference_engine_onnxruntime.py -q

The tests drive the adapter against a tiny model that is generated in the
test, so no model file and no device are needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    E_ENGINE_UNAVAILABLE,
    E_MODEL_LOAD_FAILED,
    E_MODEL_NOT_LOADED,
    E_OUTPUT_LAYOUT_MISMATCH,
    ModelConfigError,
    ModelLoadError,
)
from framework.inference.contract import ArtifactRef
from framework.inference.engines.onnxruntime import ENGINE_ID, create_engine
from framework.inference.manifest import load_manifest
from helpers_inference import onnx_artifact, tiny_pair, write_manifest

pytest.importorskip("onnxruntime")

import onnxruntime as ort


def _artifact_of(tmp_path: Path, **overrides) -> ArtifactRef:
    """The artifact of the tiny onnx model, ready to be loaded."""
    pair = tiny_pair(tmp_path)
    block = onnx_artifact()
    block.update(overrides)
    manifest_path = write_manifest(tmp_path, ENGINE_ID, block)
    manifest = load_manifest(manifest_path)
    return manifest.artifacts[ENGINE_ID]


def test_the_engine_is_registered_and_reported_as_available():
    assert create_engine() is not None


def test_the_engine_loads_the_artifact_and_binds_the_io_names(tmp_path):
    artifact = _artifact_of(tmp_path)
    engine = create_engine()
    engine.load(artifact)

    assert engine.engine_id == ENGINE_ID
    assert [spec.name for spec in engine.input_specs()] == ["images"]
    assert [spec.name for spec in engine.output_specs()] == ["output0"]
    capabilities = engine.capabilities()
    assert capabilities.layouts == ("nchw",)
    assert capabilities.io_binding == "named"
    assert capabilities.releases_gil is True
    assert capabilities.load_holds_gil is True
    assert capabilities.input_accepts == ("numpy_bgr",)


def test_the_engine_transposes_the_canonical_frame_into_nchw(tmp_path):
    artifact = _artifact_of(tmp_path)
    engine = create_engine()
    engine.load(artifact)
    frame = np.arange(4 * 4 * 3, dtype=np.float32).reshape(4, 4, 3)

    outputs = engine.run({"image": frame})

    session = ort.InferenceSession(artifact.path)
    expected = session.run(
        None, {"images": np.transpose(frame, (2, 0, 1))[None]}
    )[0]
    assert list(outputs) == ["pred"]
    np.testing.assert_allclose(outputs["pred"], expected, rtol=1e-5, atol=1e-5)


def test_the_engine_reports_a_missing_artifact_as_a_load_failure(tmp_path):
    artifact = _artifact_of(tmp_path)
    broken = ArtifactRef(
        engine_id=artifact.engine_id,
        path=str(tmp_path / "does_not_exist.onnx"),
        wrapper=artifact.wrapper,
        wrapper_params=artifact.wrapper_params,
        preprocess=artifact.preprocess,
        io=artifact.io,
        runtime=artifact.runtime,
        build=artifact.build,
        aux_paths=artifact.aux_paths,
    )
    engine = create_engine()
    with pytest.raises(ModelLoadError) as exc:
        engine.load(broken)
    assert exc.value.code == E_MODEL_LOAD_FAILED


def test_the_engine_refuses_an_input_name_the_model_does_not_have(tmp_path):
    artifact = _artifact_of(tmp_path)
    wrong = ArtifactRef(
        engine_id=artifact.engine_id,
        path=artifact.path,
        wrapper=artifact.wrapper,
        wrapper_params=artifact.wrapper_params,
        preprocess=artifact.preprocess,
        io=type(artifact.io)(
            inputs={"image": "not_a_tensor"},
            outputs={"pred": "output0"},
        ),
        runtime=artifact.runtime,
        build=artifact.build,
        aux_paths=artifact.aux_paths,
    )
    engine = create_engine()
    with pytest.raises(ModelConfigError) as exc:
        engine.load(wrong)
    assert exc.value.code == E_CONTRACT_MISSING


def test_the_engine_refuses_an_output_name_the_model_does_not_have(tmp_path):
    artifact = _artifact_of(tmp_path)
    wrong = ArtifactRef(
        engine_id=artifact.engine_id,
        path=artifact.path,
        wrapper=artifact.wrapper,
        wrapper_params=artifact.wrapper_params,
        preprocess=artifact.preprocess,
        io=type(artifact.io)(
            inputs={"image": "images"},
            outputs={"pred": "not_a_tensor"},
        ),
        runtime=artifact.runtime,
        build=artifact.build,
        aux_paths=artifact.aux_paths,
    )
    engine = create_engine()
    with pytest.raises(Exception) as exc:
        engine.load(wrong)
    assert exc.value.code == E_OUTPUT_LAYOUT_MISMATCH


def test_the_engine_reports_a_provider_that_is_not_in_use(tmp_path):
    """A provider that is not installed must not pass for a working one."""
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    runtime["fallback"] = "allow"
    engine = create_engine()
    engine.load(artifact, runtime=runtime)

    capabilities = engine.capabilities()
    if "cuda" in capabilities.device:
        pytest.skip("this machine has a CUDA provider")
    assert capabilities.device == "cpu"
    assert "CPUExecutionProvider" in capabilities.device_detail


def test_a_requested_accelerator_that_is_not_active_is_a_startup_failure(tmp_path):
    """The default policy: an accelerator request that lands on the CPU fails."""
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    engine = create_engine()

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("this machine has a CUDA provider")

    with pytest.raises(ModelConfigError) as caught:
        engine.load(artifact, runtime=runtime)

    assert caught.value.code == E_ENGINE_UNAVAILABLE
    assert "onnxruntime-gpu" in str(caught.value)


def test_the_failed_load_leaves_the_engine_unloaded(tmp_path):
    """A refused accelerator must not leave a half-ready engine behind."""
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    engine = create_engine()

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("this machine has a CUDA provider")

    with pytest.raises(ModelConfigError):
        engine.load(artifact, runtime=runtime)

    with pytest.raises(Exception) as caught:
        engine.run({"image": np.zeros((4, 4, 3), dtype=np.float32)})
    assert caught.value.code == E_MODEL_NOT_LOADED


def test_an_explicit_allow_accepts_the_cpu_downgrade(tmp_path, caplog):
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    runtime["fallback"] = "allow"
    engine = create_engine()

    if "CUDAExecutionProvider" in ort.get_available_providers():
        pytest.skip("this machine has a CUDA provider")

    with caplog.at_level("WARNING"):
        engine.load(artifact, runtime=runtime)

    assert engine.capabilities().device == "cpu"
    assert any("CUDAExecutionProvider" in record.message for record in caplog.records)


def test_a_cpu_only_request_never_fails(tmp_path):
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CPUExecutionProvider"]
    runtime["fallback"] = "forbid"
    engine = create_engine()
    engine.load(artifact, runtime=runtime)

    assert engine.capabilities().device == "cpu"


def test_an_unknown_provider_name_is_refused_before_the_session_is_built(tmp_path):
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CudaExecutionProvider"]
    engine = create_engine()

    with pytest.raises(ModelConfigError) as caught:
        engine.load(artifact, runtime=runtime)

    assert caught.value.code == E_CONTRACT_MISSING
    assert "CUDAExecutionProvider" in str(caught.value)


def test_provider_options_reach_the_session_in_dict_form(tmp_path, monkeypatch):
    """device_id and friends are unreachable through a plain name list."""
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CPUExecutionProvider"]
    runtime["provider_options"] = {"CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}}
    spy = _spy_ort(ort)
    monkeypatch.setattr(
        "framework.inference.engines.onnxruntime._import_onnxruntime", lambda: spy
    )
    engine = create_engine()
    engine.load(artifact, runtime=runtime)

    assert spy.providers == [
        {"CPUExecutionProvider": {"arena_extend_strategy": "kSameAsRequested"}}
    ]


def test_the_device_is_derived_from_the_provider_the_session_activated(tmp_path, monkeypatch):
    """A CUDA session must report cuda even when the host has no GPU to prove it."""
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    spy = _spy_ort(ort, active=["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(
        "framework.inference.engines.onnxruntime._import_onnxruntime", lambda: spy
    )
    engine = create_engine()
    engine.load(artifact, runtime=runtime)

    capabilities = engine.capabilities()
    assert capabilities.device == "cuda"
    assert "CUDAExecutionProvider" in capabilities.device_detail


def test_a_missing_accelerator_does_not_fail_when_the_session_reports_it_active(
    tmp_path, monkeypatch
):
    artifact = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["providers"] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    spy = _spy_ort(ort, active=["CUDAExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(
        "framework.inference.engines.onnxruntime._import_onnxruntime", lambda: spy
    )
    engine = create_engine()
    engine.load(artifact, runtime=runtime)

    assert engine.capabilities().device == "cuda"


class _Session:
    """A session that reports the providers the test wants to see active."""

    def __init__(self, inner, active):
        self._inner = inner
        self._active = active

    def get_providers(self):
        return list(self._active)

    def get_inputs(self):
        return self._inner.get_inputs()

    def get_outputs(self):
        return self._inner.get_outputs()

    def run(self, *args, **kwargs):
        return self._inner.run(*args, **kwargs)


class _SpyOrt:
    """A stand-in onnxruntime module that records how the session was built."""

    def __init__(self, real, active):
        self.__version__ = real.__version__
        self.SessionOptions = real.SessionOptions
        self.ExecutionMode = real.ExecutionMode
        self.GraphOptimizationLevel = real.GraphOptimizationLevel
        self._real = real
        self._active = active
        self.providers = None

    def get_available_providers(self):
        """Report the providers the test believes are installed."""
        if self._active is None:
            return self._real.get_available_providers()
        return list(self._active)

    def InferenceSession(self, path, sess_options=None, providers=None):
        self.providers = providers
        inner = self._real.InferenceSession(
            path, sess_options=sess_options, providers=providers
        )
        if self._active is None:
            return inner
        return _Session(inner, self._active)


def _spy_ort(real, active=None):
    return _SpyOrt(real, active)


def test_the_engine_refuses_to_run_without_a_model():
    engine = create_engine()
    with pytest.raises(Exception) as exc:
        engine.run({"image": np.zeros((4, 4, 3), dtype=np.float32)})
    assert exc.value.code == E_MODEL_NOT_LOADED
