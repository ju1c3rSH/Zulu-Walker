"""Tests of the deployment level accelerator override.

Run: python -m pytest tests/test_inference_runtime_options.py -q

The artifact declares the runtime it was validated against; the deployment
declares the accelerator it wants.  The second one is passed down as
``runtime`` options and merged over the artifact block by the engine, so one
manifest serves a CPU laptop, a DirectML desktop and a CUDA server.

These tests pin the plumbing only: that the options travel from the
configuration to ``engine.load()`` unchanged, and that a per-model override
wins over the facade wide one.  What the engines do with them is covered by
the per-engine test files.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from framework.inference.contract import ArtifactRef
from framework.inference.manifest import load_manifest
from framework.inference.runtime import ModelRuntime
from helpers_inference import onnx_artifact, tiny_pair, write_manifest

OVERRIDE = {"providers": ["CUDAExecutionProvider", "CPUExecutionProvider"]}


class _Engine:
    """An engine that records what the runtime handed it."""

    engine_id = "onnxruntime"

    def __init__(self) -> None:
        self.loaded: Optional[dict[str, Any]] = None

    def load(self, artifact: ArtifactRef, **opts: Any) -> None:
        self.loaded = opts

    def input_specs(self):
        return []

    def output_specs(self):
        return []

    def capabilities(self):
        from framework.inference.contract import Capabilities

        return Capabilities(
            device="cuda",
            quant="fp32",
            layouts=("nchw",),
            dynamic_shape=False,
            input_accepts=("numpy_bgr",),
            io_binding="named",
            releases_gil=True,
            load_holds_gil=True,
        )

    def run(self, inputs, **options):
        return {}

    def unload(self) -> None:
        pass


class _Capability:
    provides = frozenset({"detect"})

    def detect(self, frame, **kwargs):
        return []


def _artifact_and_manifest(tmp_path: Path):
    tiny_pair(tmp_path)
    manifest = load_manifest(write_manifest(tmp_path, "onnxruntime", onnx_artifact()))
    return manifest.artifacts["onnxruntime"], manifest


def test_the_runtime_hands_the_options_to_the_engine(tmp_path):
    artifact, manifest = _artifact_and_manifest(tmp_path)
    engine = _Engine()

    ModelRuntime(
        engine,
        artifact,
        manifest=manifest,
        runtime_options=OVERRIDE,
        wrapper_factory=lambda runtime: _Capability(),
    )

    assert engine.loaded["runtime"] == OVERRIDE


def test_an_absent_override_hands_an_empty_block(tmp_path):
    artifact, manifest = _artifact_and_manifest(tmp_path)
    engine = _Engine()

    ModelRuntime(
        engine,
        artifact,
        manifest=manifest,
        wrapper_factory=lambda runtime: _Capability(),
    )

    assert engine.loaded["runtime"] == {}


def test_the_canonical_contract_still_travels_with_the_options(tmp_path):
    artifact, manifest = _artifact_and_manifest(tmp_path)
    engine = _Engine()

    ModelRuntime(
        engine,
        artifact,
        manifest=manifest,
        runtime_options=OVERRIDE,
        wrapper_factory=lambda runtime: _Capability(),
    )

    assert engine.loaded["canonical"] is not None
    assert engine.loaded["canonical"].target_size == (4, 4)


def test_the_override_does_not_mutate_the_artifact_block(tmp_path):
    artifact, manifest = _artifact_and_manifest(tmp_path)
    before = dict(artifact.runtime)
    engine = _Engine()

    ModelRuntime(
        engine,
        artifact,
        manifest=manifest,
        runtime_options=OVERRIDE,
        wrapper_factory=lambda runtime: _Capability(),
    )

    assert artifact.runtime == before


# --------------------------------------------------------------------- #
# Facade: the deployment block is the default, a model entry overrides it
# --------------------------------------------------------------------- #


def _facade(engine: _Engine, **kwargs):
    from framework.inference.facade import AIFacade

    return AIFacade("onnxruntime", engine_factory=lambda: engine, **kwargs)


def test_the_facade_hands_its_options_to_the_engine(tmp_path):
    artifact, _ = _artifact_and_manifest(tmp_path)
    engine = _Engine()
    facade = _facade(engine, runtime_options=OVERRIDE)
    facade.add("tiny", artifact.path, manifest_path=str(tmp_path / "manifest.yaml"))
    facade.switch("tiny")

    assert engine.loaded["runtime"] == OVERRIDE


def test_a_model_entry_overrides_the_facade_default(tmp_path):
    artifact, _ = _artifact_and_manifest(tmp_path)
    engine = _Engine()
    facade = _facade(engine, runtime_options={"providers": ["CPUExecutionProvider"]})
    facade.add(
        "tiny",
        artifact.path,
        manifest_path=str(tmp_path / "manifest.yaml"),
        runtime_options=OVERRIDE,
    )
    facade.switch("tiny")

    assert engine.loaded["runtime"] == OVERRIDE


def test_a_model_entry_without_options_keeps_the_facade_default(tmp_path):
    artifact, _ = _artifact_and_manifest(tmp_path)
    engine = _Engine()
    facade = _facade(engine, runtime_options=OVERRIDE)
    facade.add("tiny", artifact.path, manifest_path=str(tmp_path / "manifest.yaml"))
    facade.switch("tiny")

    assert engine.loaded["runtime"] == OVERRIDE


def test_linux_ai_forwards_the_options(tmp_path):
    from framework.hal.platforms.linux import create_ai

    engine = _Engine()
    ai = create_ai(
        "onnxruntime",
        engine_factory=lambda: engine,
        runtime_options=OVERRIDE,
    )
    artifact, _ = _artifact_and_manifest(tmp_path)
    ai.add("tiny", artifact.path, manifest_path=str(tmp_path / "manifest.yaml"))
    ai.switch("tiny")

    assert engine.loaded["runtime"] == OVERRIDE
