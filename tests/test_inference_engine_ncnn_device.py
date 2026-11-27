"""Tests of the ncnn Vulkan wiring: device choice and instance lifetime.

Run: python -m pytest tests/test_inference_engine_ncnn_device.py -q

The policy (which index a spec resolves to) is covered by
tests/test_inference_devices.py without touching any library.  This file
covers the wiring: that the adapter asks the library for its ranking, hands
the index to the net before the graph is loaded, and releases the GPU
instance it created.

Two kinds of test live here.  The lifecycle tests use a stand-in ncnn module
so they can count calls; the selection tests use the real library and skip
when this machine has no Vulkan device.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, Tuple

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    E_ENGINE_UNAVAILABLE,
    ModelConfigError,
)
from framework.inference.contract import ArtifactRef
from framework.inference.engines import ncnn as ncnn_engine
from framework.inference.manifest import load_manifest
from framework.inference.runtime import CanonicalSpec

ncnn = pytest.importorskip("ncnn")

from helpers_inference import ncnn_artifact, tiny_pair, write_manifest  # noqa: E402


def _artifact_of(tmp_path: Path, **runtime: Any) -> Tuple[Any, CanonicalSpec]:
    """The tiny ncnn artifact and its canonical contract, with runtime overrides."""
    tiny_pair(tmp_path)
    block = ncnn_artifact()
    block["runtime"] = {**block.get("runtime", {}), **runtime}
    manifest = load_manifest(write_manifest(tmp_path, "ncnn", block))
    return manifest.artifacts["ncnn"], CanonicalSpec.from_manifest(manifest)


def _has_vulkan() -> bool:
    """Whether this machine exposes a Vulkan device to ncnn."""
    try:
        return int(ncnn.get_gpu_count()) > 0
    except Exception:
        return False


def _devices() -> Tuple[Tuple[int, str], ...]:
    """The Vulkan devices this machine reports, as (index, name) pairs."""
    devices = []
    for index in range(int(ncnn.get_gpu_count())):
        devices.append((index, str(ncnn.get_gpu_info(index).device_name())))
    return tuple(devices)


vulkan_only = pytest.mark.skipif(not _has_vulkan(), reason="no Vulkan device")


# --------------------------------------------------------------------- #
# Selection against the real library
# --------------------------------------------------------------------- #


@vulkan_only
def test_auto_follows_the_library_ranking_rather_than_index_zero(tmp_path):
    """The library ranks devices; index 0 is often the integrated one."""
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="auto")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    assert engine.device_index == int(ncnn.get_default_gpu_index())


@vulkan_only
def test_the_chosen_device_is_named_in_the_capabilities(tmp_path):
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="auto")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    name = str(ncnn.get_gpu_info(engine.device_index).device_name())
    assert name in engine.capabilities().device_detail


@vulkan_only
def test_an_absent_vulkan_device_setting_means_auto(tmp_path):
    artifact, canonical = _artifact_of(tmp_path, device="vulkan")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    assert engine.device_index == int(ncnn.get_default_gpu_index())


@vulkan_only
def test_a_vulkan_device_can_be_selected_by_name(tmp_path):
    devices = _devices()
    unique = [
        (index, name)
        for index, name in devices
        if sum(name == other for _, other in devices) == 1
    ]
    if not unique:
        pytest.skip("every Vulkan device of this machine shares its name")
    index, name = unique[0]
    artifact, canonical = _artifact_of(
        tmp_path, device="vulkan", vulkan_device=name.lower()
    )
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    assert engine.device_index == index


@vulkan_only
def test_an_unknown_vulkan_device_name_fails_loudly(tmp_path):
    artifact, canonical = _artifact_of(
        tmp_path, device="vulkan", vulkan_device="no_such_gpu_vendor_xyz"
    )
    engine = ncnn_engine.create_engine()
    with pytest.raises(ModelConfigError) as caught:
        engine.load(artifact, canonical=canonical)

    assert caught.value.code == E_ENGINE_UNAVAILABLE


def test_the_cpu_path_reports_no_device_index(tmp_path):
    artifact, canonical = _artifact_of(tmp_path, device="cpu")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    assert engine.device_index is None
    assert engine.capabilities().device_detail == ""


# --------------------------------------------------------------------- #
# Lifetime, counted on a stand-in module
# --------------------------------------------------------------------- #


class _FakeOption:
    """The subset of ncnn.Option the adapter touches."""

    def __init__(self) -> None:
        self.use_vulkan_compute = False
        self.num_threads = 1
        self.use_fp16_packed = True
        self.use_fp16_storage = True
        self.use_fp16_arithmetic = True


class _FakeNet:
    """A net that records what the adapter asks of it, in order."""

    def __init__(self, log: list) -> None:
        self.opt = _FakeOption()
        self._log = log

    def __del__(self) -> None:
        try:
            self._log.append(("net_gone", None))
        except Exception:
            pass

    def set_vulkan_device(self, index: int) -> None:
        self._log.append(("device", index))

    def load_param(self, path: str) -> int:
        self._log.append(("param", self.opt.use_vulkan_compute))
        return 0

    def load_model(self, path: str) -> int:
        self._log.append(("model", self.opt.use_vulkan_compute))
        return 0

    def input_names(self):
        return ["data"]

    def output_names(self):
        return ["conv"]


class _FakeGpuInfo:
    def __init__(self, name: str) -> None:
        self._name = name

    def device_name(self) -> str:
        return self._name


_FAKE_NAMES = (
    "Intel(R) RaptorLake-S Mobile Graphics Controller",
    "NVIDIA GeForce RTX 5060 Laptop GPU",
)


def _fake_module(log: list, default_index: int = 1) -> types.ModuleType:
    """A stand-in ncnn module with two devices and a default of its own."""
    module = types.ModuleType("ncnn")
    module.__version__ = "1.0.20261020"
    module.Net = lambda: _FakeNet(log)
    module.create_gpu_instance = lambda: log.append(("create", None))
    module.destroy_gpu_instance = lambda: log.append(("destroy", None))
    module.get_gpu_count = lambda: len(_FAKE_NAMES)
    module.get_default_gpu_index = lambda: default_index
    module.get_gpu_info = lambda index: _FakeGpuInfo(_FAKE_NAMES[index])
    return module


@pytest.fixture()
def fake_ncnn(monkeypatch: pytest.MonkeyPatch) -> list:
    log: list = []
    monkeypatch.setattr(ncnn_engine, "_import_ncnn", lambda: _fake_module(log))
    return log


def test_the_gpu_instance_is_created_on_load_and_destroyed_on_unload(
    tmp_path, fake_ncnn
):
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="auto")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)
    engine.unload()

    assert fake_ncnn.count(("create", None)) == 1
    assert fake_ncnn.count(("destroy", None)) == 1
    assert fake_ncnn[-1] == ("destroy", None)


def test_the_net_is_dropped_before_the_gpu_instance_is_destroyed(tmp_path, fake_ncnn):
    """A loaded Vulkan net keeps device pointers; the instance must outlive it."""
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="auto")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)
    engine.unload()

    order = [name for name, _ in fake_ncnn]
    assert "net_gone" in order
    assert order.index("net_gone") < order.index("destroy")


def test_every_load_creates_one_instance_and_every_unload_releases_one(
    tmp_path, fake_ncnn
):
    """The pairing is what keeps the reference count of the library balanced."""
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="auto")
    engine_a = ncnn_engine.create_engine()
    engine_a.load(artifact, canonical=canonical)
    engine_b = ncnn_engine.create_engine()
    engine_b.load(artifact, canonical=canonical)

    assert fake_ncnn.count(("create", None)) == 2
    assert fake_ncnn.count(("destroy", None)) == 0

    engine_a.unload()
    engine_b.unload()

    assert fake_ncnn.count(("create", None)) == fake_ncnn.count(("destroy", None))


def test_the_device_is_bound_before_the_graph_is_loaded(tmp_path, fake_ncnn):
    """Weight packing for Vulkan happens at load, so the device must be set first."""
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="auto")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    order = [name for name, _ in fake_ncnn]
    assert order.index("device") < order.index("param")
    assert order.index("param") < order.index("model")
    assert fake_ncnn[order.index("param")][1] is True
    assert fake_ncnn[order.index("model")][1] is True


def test_the_binding_uses_the_default_index_of_the_library(tmp_path, fake_ncnn):
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="auto")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    assert ("device", 1) in fake_ncnn


def test_an_explicit_index_wins_over_the_library_default(tmp_path, fake_ncnn):
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device=0)
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    assert ("device", 0) in fake_ncnn


def test_a_relative_name_selects_the_matching_device(tmp_path, fake_ncnn):
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="nvidia")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    assert ("device", 1) in fake_ncnn


def test_no_gpu_instance_is_created_for_the_cpu_path(tmp_path, fake_ncnn):
    artifact, canonical = _artifact_of(tmp_path, device="cpu")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)
    engine.unload()

    assert ("create", None) not in fake_ncnn
    assert ("destroy", None) not in fake_ncnn


def test_fp16_off_clears_the_packing_flags(tmp_path, fake_ncnn):
    artifact, canonical = _artifact_of(tmp_path, device="cpu", fp16="off")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    option = engine._net.opt
    assert option.use_fp16_packed is False
    assert option.use_fp16_storage is False
    assert option.use_fp16_arithmetic is False


def test_fp16_on_sets_the_packing_flags(tmp_path, fake_ncnn):
    artifact, canonical = _artifact_of(tmp_path, device="cpu", fp16="on")
    engine = ncnn_engine.create_engine()
    engine.load(artifact, canonical=canonical)

    option = engine._net.opt
    assert option.use_fp16_packed is True
    assert option.use_fp16_storage is True
    assert option.use_fp16_arithmetic is True


def test_a_contract_error_after_the_load_releases_the_instance(tmp_path, fake_ncnn):
    """A blob the graph does not declare must not leave the Vulkan instance up."""
    artifact, canonical = _artifact_of(tmp_path, device="vulkan", vulkan_device="auto")
    broken = ArtifactRef(
        engine_id=artifact.engine_id,
        path=artifact.path,
        wrapper=artifact.wrapper,
        wrapper_params=artifact.wrapper_params,
        preprocess=artifact.preprocess,
        io=type(artifact.io)(
            inputs={"image": "data"},
            outputs={"pred": "not_a_blob"},
        ),
        runtime=artifact.runtime,
        build=artifact.build,
        aux_paths=artifact.aux_paths,
    )
    engine = ncnn_engine.create_engine()
    with pytest.raises(Exception):
        engine.load(broken, canonical=canonical)

    assert fake_ncnn.count(("create", None)) == fake_ncnn.count(("destroy", None))


def test_an_unknown_fp16_mode_is_a_contract_error(tmp_path, fake_ncnn):
    artifact, canonical = _artifact_of(tmp_path, device="cpu", fp16="maybe")
    engine = ncnn_engine.create_engine()
    with pytest.raises(ModelConfigError) as caught:
        engine.load(artifact, canonical=canonical)

    assert caught.value.code == E_CONTRACT_MISSING
