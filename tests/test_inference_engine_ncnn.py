"""Unit tests for the ncnn engine adapter.

Run: python -m pytest tests/test_inference_engine_ncnn.py -q

The tests drive the adapter against a tiny model that is generated in the
test, so no model file and no device are needed.  They are skipped when the
ncnn library is not installed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.hal.interface.ai import (
    E_ENGINE_UNAVAILABLE,
    E_MODEL_LOAD_FAILED,
    E_MODEL_NOT_LOADED,
    E_OUTPUT_LAYOUT_MISMATCH,
    ModelLoadError,
)
from framework.inference.contract import ArtifactRef
from framework.inference.engines.ncnn import ENGINE_ID, create_engine
from framework.inference.manifest import load_manifest
from framework.inference.runtime import CanonicalSpec
from helpers_inference import ncnn_artifact, tiny_pair, write_manifest

pytest.importorskip("ncnn")

import ncnn


def _artifact_of(tmp_path: Path, **overrides) -> tuple[ArtifactRef, Any]:
    """The artifact of the tiny ncnn model and its canonical contract.

    The canonical contract travels with the load, exactly as the runtime
    hands it over.
    """
    tiny_pair(tmp_path)
    block = ncnn_artifact()
    block.update(overrides)
    manifest_path = write_manifest(tmp_path, ENGINE_ID, block)
    manifest = load_manifest(manifest_path)
    canonical = CanonicalSpec.from_manifest(manifest)
    return manifest.artifacts[ENGINE_ID], canonical


def _reference(frame: np.ndarray) -> np.ndarray:
    """The output of the same weights in onnxruntime."""
    import onnxruntime as ort

    from scripts.tiny_model import build_tiny_pair

    with tempfile.TemporaryDirectory() as tmp:
        pair = build_tiny_pair(tmp)
        session = ort.InferenceSession(pair["onnx"])
        planar = np.transpose(frame.astype(np.float32), (2, 0, 1))
        return session.run(None, {"images": planar[None]})[0]


def _has_vulkan() -> bool:
    """Whether this machine has a Vulkan device for ncnn."""
    getter = getattr(ncnn, "get_gpu_count", None)
    if getter is None:
        return False
    try:
        return int(getter()) > 0
    except Exception:
        return False


def test_the_engine_is_registered_and_reported_as_available():
    assert create_engine() is not None


def test_the_engine_loads_the_graph_and_binds_the_blob_names(tmp_path):
    artifact, canonical = _artifact_of(tmp_path)
    engine = create_engine()
    engine.load(artifact, canonical=canonical)

    assert engine.engine_id == ENGINE_ID
    capabilities = engine.capabilities()
    assert capabilities.layouts == ("nhwc",)
    assert capabilities.io_binding == "named"
    assert capabilities.input_accepts == ("numpy_pixels", "numpy_bgr")
    assert engine.input_specs() == []
    assert engine.output_specs() == []


def test_the_engine_matches_the_reference_engine(tmp_path):
    """The same weights must give the same numbers in both engines."""
    artifact, canonical = _artifact_of(tmp_path)
    engine = create_engine()
    engine.load(artifact, canonical=canonical)
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)

    outputs = engine.run({"image": frame})

    assert list(outputs) == ["pred"]
    expected = _reference(frame)
    np.testing.assert_allclose(
        outputs["pred"].reshape(expected.shape), expected, rtol=1e-4, atol=1e-4
    )


def test_the_engine_applies_the_normalisation_of_the_artifact(tmp_path):
    """The engine tier normalises on the device, with the engine formula."""
    artifact, canonical = _artifact_of(tmp_path)
    engine = create_engine()
    engine.load(artifact, canonical=canonical)
    frame = np.full((4, 4, 3), 128, dtype=np.uint8)

    outputs = engine.run({"image": frame})

    expected = _reference(frame)
    np.testing.assert_allclose(
        outputs["pred"].reshape(expected.shape), expected, rtol=1e-4, atol=1e-4
    )


def test_the_engine_reports_a_missing_weight_file_as_a_load_failure(tmp_path):
    artifact, canonical = _artifact_of(tmp_path)
    broken = ArtifactRef(
        engine_id=artifact.engine_id,
        path=artifact.path,
        wrapper=artifact.wrapper,
        wrapper_params=artifact.wrapper_params,
        preprocess=artifact.preprocess,
        io=artifact.io,
        runtime=artifact.runtime,
        build=artifact.build,
        aux_paths={"bin": str(tmp_path / "does_not_exist.bin")},
    )
    engine = create_engine()
    with pytest.raises(ModelLoadError) as exc:
        engine.load(broken, canonical=canonical)
    assert exc.value.code == E_MODEL_LOAD_FAILED


def test_the_engine_refuses_a_blob_name_the_graph_does_not_have(tmp_path):
    artifact, canonical = _artifact_of(tmp_path)
    wrong = ArtifactRef(
        engine_id=artifact.engine_id,
        path=artifact.path,
        wrapper=artifact.wrapper,
        wrapper_params=artifact.wrapper_params,
        preprocess=artifact.preprocess,
        io=type(artifact.io)(
            inputs={"image": "not_a_blob"},
            outputs={"pred": "conv"},
        ),
        runtime=artifact.runtime,
        build=artifact.build,
        aux_paths=artifact.aux_paths,
    )
    engine = create_engine()
    with pytest.raises(Exception) as exc:
        engine.load(wrong, canonical=canonical)
    assert exc.value.code == E_OUTPUT_LAYOUT_MISMATCH


def test_the_engine_refuses_to_run_without_a_model():
    engine = create_engine()
    with pytest.raises(Exception) as exc:
        engine.run({"image": np.zeros((4, 4, 3), dtype=np.uint8)})
    assert exc.value.code == E_MODEL_NOT_LOADED


def test_the_engine_fails_loudly_when_vulkan_is_asked_for_and_absent(tmp_path):
    """A Vulkan request that cannot be honoured must not fall back to the CPU."""
    artifact, canonical = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["device"] = "vulkan"
    engine = create_engine()
    if _has_vulkan():
        pytest.skip("this machine has a Vulkan device")
    with pytest.raises(Exception) as exc:
        engine.load(artifact, canonical=canonical, runtime=runtime)
    assert exc.value.code == E_ENGINE_UNAVAILABLE


def test_the_engine_runs_on_vulkan_when_the_device_is_there(tmp_path):
    """The Vulkan path must work when the device is present."""
    if not _has_vulkan():
        pytest.skip("this machine has no Vulkan device")
    artifact, canonical = _artifact_of(tmp_path)
    runtime = dict(artifact.runtime or {})
    runtime["device"] = "vulkan"
    engine = create_engine()
    engine.load(artifact, canonical=canonical, runtime=runtime)

    assert engine.capabilities().device == "vulkan"
    frame = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    outputs = engine.run({"image": frame})
    expected = _reference(frame)
    np.testing.assert_allclose(
        outputs["pred"].reshape(expected.shape), expected, rtol=1e-2, atol=1e-2
    )
