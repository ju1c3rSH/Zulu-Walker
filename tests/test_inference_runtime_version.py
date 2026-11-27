"""Unit tests for the pinned runtime version contract.

Run: python -m pytest tests/test_inference_runtime_version.py -q

The manifest pins the runtime version an artifact was validated against.  The
pin is compared when the engine loads the artifact, so the tests cover both
the comparison itself and the two adapters that call it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    E_MODEL_LOAD_FAILED,
    ModelConfigError,
    ModelLoadError,
)
from framework.inference.engines.version import (
    check_runtime_version,
    version_components,
)
from helpers_inference import ncnn_artifact, onnx_artifact, tiny_pair, write_manifest


def test_the_components_of_a_version_are_read_in_order():
    assert version_components("1.26.0") == (1, 26, 0)
    assert version_components("v1.26.0") == (1, 26, 0)
    assert version_components("1.0.20261020") == (1, 0, 20261020)
    assert version_components("1.26.x") == (1, 26)
    assert version_components("1.26rc1") == (1, 26)


def test_an_exact_pin_accepts_the_same_version():
    check_runtime_version("1.26.0", "1.26.0", "onnxruntime")


def test_a_prefix_pin_accepts_a_newer_patch():
    check_runtime_version("1.26", "1.26.3", "onnxruntime")


def test_a_pin_that_does_not_match_is_a_load_failure():
    with pytest.raises(ModelLoadError) as exc:
        check_runtime_version("1.26", "1.27.0", "onnxruntime")
    assert exc.value.code == E_MODEL_LOAD_FAILED


def test_a_pin_that_cannot_be_read_is_a_contract_error():
    with pytest.raises(ModelConfigError) as exc:
        check_runtime_version("nightly", "1.26.0", "onnxruntime")
    assert exc.value.code == E_CONTRACT_MISSING


def test_a_library_without_a_version_cannot_verify_the_pin():
    with pytest.raises(ModelLoadError) as exc:
        check_runtime_version("1.26", "", "onnxruntime")
    assert exc.value.code == E_MODEL_LOAD_FAILED


def test_a_missing_pin_is_not_compared():
    check_runtime_version(None, "1.26.0", "onnxruntime")
    check_runtime_version("", "1.26.0", "onnxruntime")


def _artifact_of(tmp_path: Path, engine_id: str, block: dict, runtime: dict):
    """The artifact of the tiny model, with a runtime block.

    The canonical contract travels with the load, exactly as the runtime
    hands it over, because the ncnn artifact normalises on the device.
    """
    tiny_pair(tmp_path)
    block = dict(block)
    block["runtime"] = runtime
    manifest_path = write_manifest(tmp_path, engine_id, block)
    from framework.inference.manifest import load_manifest
    from framework.inference.runtime import CanonicalSpec

    manifest = load_manifest(manifest_path)
    canonical = CanonicalSpec.from_manifest(manifest)
    return manifest.artifacts[engine_id], canonical


def test_the_onnxruntime_adapter_accepts_a_pin_that_matches(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from framework.inference.engines.onnxruntime import create_engine

    artifact, _ = _artifact_of(
        tmp_path, "onnxruntime", onnx_artifact(), {"version": ort.__version__}
    )
    engine = create_engine()
    engine.load(artifact)
    assert engine.engine_id == "onnxruntime"


def test_the_onnxruntime_adapter_refuses_a_pin_that_does_not_match(tmp_path):
    pytest.importorskip("onnxruntime")
    from framework.inference.engines.onnxruntime import create_engine

    artifact, _ = _artifact_of(
        tmp_path, "onnxruntime", onnx_artifact(), {"version": "0.0.1"}
    )
    engine = create_engine()
    with pytest.raises(ModelLoadError) as exc:
        engine.load(artifact)
    assert exc.value.code == E_MODEL_LOAD_FAILED


def test_the_ncnn_adapter_refuses_a_pin_that_does_not_match(tmp_path):
    pytest.importorskip("ncnn")
    from framework.inference.engines.ncnn import create_engine

    artifact, canonical = _artifact_of(
        tmp_path, "ncnn", ncnn_artifact(), {"version": "0.0.1"}
    )
    engine = create_engine()
    with pytest.raises(ModelLoadError) as exc:
        engine.load(artifact, canonical=canonical)
    assert exc.value.code == E_MODEL_LOAD_FAILED


def test_the_ncnn_adapter_accepts_a_pin_that_matches(tmp_path):
    ncnn = pytest.importorskip("ncnn")
    from framework.inference.engines.ncnn import create_engine

    artifact, canonical = _artifact_of(
        tmp_path, "ncnn", ncnn_artifact(), {"version": ncnn.__version__}
    )
    engine = create_engine()
    engine.load(artifact, canonical=canonical)
    assert engine.engine_id == "ncnn"
