"""Unit tests for the model runtime and the inference facade.

Run: python -m pytest tests/test_inference_runtime.py -q

The tests use a fake engine and a fake dialect, so no inference library and
no device are needed.  The fakes drive the same contract the real ones do:
the engine takes named tensors, the dialect declares the task capabilities
it provides, and the runtime owns the canonical preprocess stage.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pytest
import yaml

from framework.hal.interface.ai import (
    AIError,
    Detection,
    E_DIALECT_UNSUPPORTED,
    E_ENGINE_UNAVAILABLE,
    E_MODEL_LOAD_FAILED,
    E_SWITCH_TIMEOUT,
    AIInference,
    ManifestError,
    ModelConfigError,
    ModelLoadError,
)
from framework.inference import ops
from framework.inference.contract import Capabilities
from framework.inference.facade import AIFacade
from framework.inference.manifest import load_manifest
from framework.inference.runtime import EngineStats, FrameSpec, ModelRuntime

ENGINE_ID = "fake_engine"
WRAPPER_MODULE_NAME = "zw_fake_inference_wrapper"
INPUT_BOX = (10.0, 100.0, 50.0, 140.0)


class Gate:
    """A latch the fake dialect can block on."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def wait(self, timeout: float = 10.0) -> None:
        self.started.set()
        self.release.wait(timeout)


GATES: dict[str, Gate] = {}


class FakeDetectionAPI:
    """The capability object of the fake dialect: detection only."""

    provides = frozenset({"detect"})

    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime
        self.payloads: list[Any] = []

    def detect(self, frame: Any, **kwargs: Any) -> list[Detection]:
        self.payloads.append(frame)
        gate = GATES.get(str(self._runtime.params.get("gate", "")))
        if gate is not None:
            gate.wait()
        prepared = self._runtime.preprocess(frame)
        outputs = self._runtime.engine.run(prepared.inputs)
        prediction = outputs[self._runtime.artifact.io.main_output]
        assert prediction is not None
        transform = prepared.transform
        x1, y1, x2, y2 = INPUT_BOX
        source_x = (x1 - transform.offset_x) / transform.scale_x
        source_y = (y1 - transform.offset_y) / transform.scale_y
        source_right = (x2 - transform.offset_x) / transform.scale_x
        source_bottom = (y2 - transform.offset_y) / transform.scale_y
        labels = self._runtime.labels
        return [
            Detection(
                x=int(round(source_x)),
                y=int(round(source_y)),
                w=int(round(source_right - source_x)),
                h=int(round(source_bottom - source_y)),
                class_id=0,
                score=0.9,
                label=labels[0] if labels else "",
            )
        ]


def build_fake_capability(runtime: ModelRuntime) -> FakeDetectionAPI:
    """The factory of the fake dialect."""
    return FakeDetectionAPI(runtime)


_wrapper_module = types.ModuleType(WRAPPER_MODULE_NAME)
_wrapper_module.build_fake_capability = build_fake_capability
sys.modules[WRAPPER_MODULE_NAME] = _wrapper_module
WRAPPER_REF = f"{WRAPPER_MODULE_NAME}:build_fake_capability"


class FakeEngine:
    """An engine that records what it was given."""

    engine_id = ENGINE_ID

    def __init__(self, input_accepts: tuple[str, ...] = ("numpy_bgr",)) -> None:
        self._input_accepts = input_accepts
        self.load_count = 0
        self.unload_count = 0
        self.received: list[dict[str, Any]] = []

    def load(self, artifact: Any, **opts: Any) -> None:
        self.load_count += 1
        self.artifact = artifact

    def input_specs(self) -> list:
        return []

    def output_specs(self) -> list:
        return []

    def capabilities(self) -> Capabilities:
        return Capabilities(
            device="cpu",
            quant="fp32",
            layouts=("hwc",),
            dynamic_shape=False,
            input_accepts=self._input_accepts,
            io_binding="named",
            releases_gil=True,
            load_holds_gil=False,
        )

    def run(self, inputs: dict[str, Any], **options: Any) -> dict[str, Any]:
        self.received.append(inputs)
        return {"pred": np.zeros((1, 6, 1), dtype=np.float32)}

    def unload(self) -> None:
        self.unload_count += 1


def _engine_factory(
    engines: list[FakeEngine],
    input_accepts: tuple[str, ...] = ("numpy_bgr",),
):
    """A factory that hands out fake engines and keeps them for inspection."""

    def factory() -> FakeEngine:
        engine = FakeEngine(input_accepts=input_accepts)
        engines.append(engine)
        return engine

    return factory


def _write_manifest(
    tmp_path: Path,
    *,
    name: str = "steelball",
    family: str = "yolo11",
    color: str = "bgr",
    size: tuple[int, int] = (640, 352),
    resize_mode: str = "letterbox",
    resize_owner: str = "framework",
    norm_owner: str = "framework",
    formula: Optional[str] = "yolo",
    model_file: str = "model.bin",
    wrapper_params: Optional[dict[str, Any]] = None,
    create_model: bool = True,
    companion: Optional[str] = None,
    create_companion: bool = True,
) -> tuple[Path, Path]:
    """Write a manifest and the artifact file it declares."""
    if create_model:
        (tmp_path / model_file).write_bytes(b"fake artifact")
    preprocess: dict[str, Any] = {"resize": resize_owner, "norm": norm_owner}
    if formula is not None and norm_owner != "baked":
        preprocess["formula"] = formula
    artifact: dict[str, Any] = {
        "file": model_file,
        "wrapper": WRAPPER_REF,
        "preprocess": preprocess,
        "io": {"inputs": {"image": "data"}, "outputs": {"pred": "output"}},
    }
    if wrapper_params is not None:
        artifact["wrapper_params"] = wrapper_params
    if companion is not None:
        artifact["bin"] = companion
        if create_companion:
            (tmp_path / companion).write_bytes(b"fake companion")
    document = {
        "schema_version": 3,
        "name": name,
        "family": family,
        "input": {
            "frame": {"color": color, "dtype": "uint8", "layout": "hwc"},
            "size": list(size),
            "resize": {"mode": resize_mode, "value": 114},
            "norm": {
                "scale": 1.0 / 255.0,
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
        },
        "labels": ["ball", "cube"],
        "artifacts": {ENGINE_ID: artifact},
    }
    manifest_path = tmp_path / f"{Path(model_file).stem}.yaml"
    manifest_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return manifest_path, tmp_path / model_file


def _facade(
    tmp_path: Path,
    *,
    switch_timeout_ms: int = 2000,
    input_accepts: tuple[str, ...] = ("numpy_bgr",),
    frame_spec: Optional[FrameSpec] = None,
) -> tuple[AIFacade, list[FakeEngine]]:
    """A facade over fake engines."""
    engines: list[FakeEngine] = []
    facade = AIFacade(
        ENGINE_ID,
        switch_timeout_ms=switch_timeout_ms,
        frame_spec=frame_spec,
        engine_factory=_engine_factory(engines, input_accepts),
    )
    return facade, engines


def _frame(height: int = 352, width: int = 1280) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------- registration and loading ---------- #


def test_add_registers_the_model_without_loading_it(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)

    assert facade.add("steelball", str(model_path), manifest_path=str(manifest_path)) is True

    assert facade.models == ["steelball"]
    assert facade.loaded is False
    assert all(engine.load_count == 0 for engine in engines)


def test_add_rejects_a_model_file_the_manifest_does_not_declare(tmp_path):
    facade, _ = _facade(tmp_path)
    manifest_path, _ = _write_manifest(tmp_path)
    other = tmp_path / "other.bin"
    other.write_bytes(b"other")

    with pytest.raises(ManifestError):
        facade.add("steelball", str(other), manifest_path=str(manifest_path))


def test_add_without_a_manifest_is_a_contract_error(tmp_path):
    facade, _ = _facade(tmp_path)
    with pytest.raises(ManifestError):
        facade.add("steelball", str(tmp_path / "model.bin"))


def test_switch_loads_a_model_that_is_not_loaded_yet(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)
    facade.add("steelball", str(model_path), manifest_path=str(manifest_path))

    assert facade.switch("steelball") is True

    assert facade.loaded is True
    assert facade.active_model == "steelball"
    assert engines[-1].load_count == 1
    assert facade.stats_by_engine()[ENGINE_ID].load_ms >= 0.0


def test_switch_to_a_loaded_model_swaps_the_pointer(tmp_path):
    facade, engines = _facade(tmp_path)
    first_manifest, first_model = _write_manifest(tmp_path, name="first")
    second_manifest, second_model = _write_manifest(
        tmp_path, name="second", model_file="second.bin"
    )
    facade.add("first", str(first_model), manifest_path=str(first_manifest))
    facade.add("second", str(second_model), manifest_path=str(second_manifest))
    facade.switch("first")
    first_engine = engines[-1]

    facade.switch("second")
    second_engine = engines[-1]
    facade.switch("first")

    assert facade.active_model == "first"
    assert first_engine.load_count == 1
    assert second_engine.load_count == 1


def test_switch_to_an_unknown_model_is_rejected(tmp_path):
    facade, _ = _facade(tmp_path)
    with pytest.raises(ModelConfigError) as exc:
        facade.switch("absent")
    assert exc.value.code == "E_MODEL_NOT_LOADED"


def test_an_unknown_engine_is_unavailable():
    with pytest.raises(ModelConfigError) as exc:
        AIFacade("no_such_engine")
    assert exc.value.code == E_ENGINE_UNAVAILABLE


def test_a_switch_that_waits_for_an_inference_times_out(tmp_path):
    gate = Gate()
    GATES["slow"] = gate
    try:
        facade, _ = _facade(tmp_path, switch_timeout_ms=50)
        first_manifest, first_model = _write_manifest(
            tmp_path, name="first", wrapper_params={"gate": "slow"}
        )
        second_manifest, second_model = _write_manifest(
            tmp_path, name="second", model_file="second.bin"
        )
        facade.add("first", str(first_model), manifest_path=str(first_manifest))
        facade.add("second", str(second_model), manifest_path=str(second_manifest))
        facade.switch("first")

        failures: list[BaseException] = []

        def run_inference() -> None:
            try:
                facade.detect(_frame())
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=run_inference)
        worker.start()
        assert gate.started.wait(2.0)

        with pytest.raises(AIError) as exc:
            facade.switch("second")
        assert exc.value.code == E_SWITCH_TIMEOUT

        gate.release.set()
        worker.join(5.0)
        assert not failures
    finally:
        GATES.pop("slow", None)


# ---------- load time file validation ---------- #


def test_loading_fails_when_the_artifact_file_is_missing(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(
        tmp_path, model_file="missing.bin", create_model=False
    )
    facade.add("missing", str(model_path), manifest_path=str(manifest_path))

    with pytest.raises(ModelLoadError) as exc:
        facade.switch("missing")

    assert exc.value.code == E_MODEL_LOAD_FAILED
    assert "missing.bin" in exc.value.message
    assert all(engine.load_count == 0 for engine in engines)


def test_loading_fails_when_a_companion_file_is_missing(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(
        tmp_path, companion="weights.bin", create_companion=False
    )
    facade.add("missing", str(model_path), manifest_path=str(manifest_path))

    with pytest.raises(ModelLoadError) as exc:
        facade.switch("missing")

    assert exc.value.code == E_MODEL_LOAD_FAILED
    assert "weights.bin" in exc.value.message
    assert all(engine.load_count == 0 for engine in engines)


def test_a_runtime_whose_files_exist_loads(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path, companion="weights.bin")
    facade.add("present", str(model_path), manifest_path=str(manifest_path))

    assert facade.switch("present") is True

    assert facade.loaded is True
    assert engines[-1].load_count == 1


# ---------- the canonical stage ---------- #


def test_the_canonical_stage_is_skipped_for_a_baked_artifact(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(
        tmp_path, resize_owner="baked", norm_owner="baked", formula=None
    )
    facade.add("baked", str(model_path), manifest_path=str(manifest_path))
    facade.switch("baked")
    frame = _frame()

    facade.detect(frame)

    delivered = engines[-1].received[-1]["image"]
    assert delivered is frame


def test_the_canonical_stage_resizes_and_normalises_a_normal_artifact(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)
    facade.add("normal", str(model_path), manifest_path=str(manifest_path))
    facade.switch("normal")

    facade.detect(_frame())

    delivered = engines[-1].received[-1]["image"]
    assert delivered.shape == (352, 640, 3)
    assert delivered.dtype == np.float32


def test_the_transform_maps_coordinates_back_to_the_source_frame(tmp_path):
    facade, _ = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)
    facade.add("normal", str(model_path), manifest_path=str(manifest_path))
    facade.switch("normal")

    detections = facade.detect(_frame(height=352, width=1280))

    assert len(detections) == 1
    detection = detections[0]
    assert detection.x == 20
    assert detection.y == 24
    assert detection.w == 80
    assert detection.h == 80


def test_the_scratch_buffer_does_not_change_the_resize_result(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(
        tmp_path, norm_owner="engine", formula="substract_mean_normalize"
    )
    facade.add("engine_norm", str(model_path), manifest_path=str(manifest_path))
    facade.switch("engine_norm")
    frame = _frame()

    facade.detect(frame)

    delivered = engines[-1].received[-1]["image"]
    plain, plain_transform = ops.resize_frame(
        frame, (640, 352), mode="letterbox", value=114
    )
    assert np.array_equal(delivered, plain)
    assert plain_transform.mode == "letterbox"


def test_the_frame_colour_is_converted_to_the_manifest_contract(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(
        tmp_path,
        color="rgb",
        size=(640, 352),
        norm_owner="engine",
        formula="substract_mean_normalize",
    )
    facade.add("rgb", str(model_path), manifest_path=str(manifest_path))
    facade.switch("rgb")
    frame = np.zeros((352, 640, 3), dtype=np.uint8)
    frame[..., 0] = 10
    frame[..., 1] = 20
    frame[..., 2] = 30

    facade.detect(frame)

    delivered = engines[-1].received[-1]["image"]
    assert delivered[0, 0, 0] == 30
    assert delivered[0, 0, 1] == 20
    assert delivered[0, 0, 2] == 10


def test_the_per_frame_path_reuses_its_working_buffers(tmp_path):
    facade, engines = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)
    facade.add("normal", str(model_path), manifest_path=str(manifest_path))
    facade.switch("normal")
    frame = _frame()

    facade.detect(frame)
    first = engines[-1].received[-1]["image"]
    facade.detect(frame)
    second = engines[-1].received[-1]["image"]

    assert first is second


# ---------- frame adaptation ---------- #


def test_the_native_frame_is_used_when_the_engine_declares_it(tmp_path):
    facade, engines = _facade(tmp_path, input_accepts=("native_image", "numpy_bgr"))
    manifest_path, model_path = _write_manifest(
        tmp_path, resize_owner="baked", norm_owner="baked", formula=None
    )
    facade.add("native", str(model_path), manifest_path=str(manifest_path))
    facade.switch("native")
    native = object()
    camera = types.SimpleNamespace(last_raw=native)

    facade.detect(_frame(), camera=camera)

    assert engines[-1].received[-1]["image"] is native


def test_the_numpy_frame_is_used_when_the_engine_declares_no_native_frame(tmp_path):
    facade, engines = _facade(tmp_path, input_accepts=("numpy_bgr",))
    manifest_path, model_path = _write_manifest(
        tmp_path, resize_owner="baked", norm_owner="baked", formula=None
    )
    facade.add("numpy", str(model_path), manifest_path=str(manifest_path))
    facade.switch("numpy")
    frame = _frame()
    camera = types.SimpleNamespace(last_raw=object())

    facade.detect(frame, camera=camera)

    assert engines[-1].received[-1]["image"] is frame


def test_the_numpy_frame_is_used_when_the_camera_has_no_native_frame(tmp_path):
    facade, engines = _facade(tmp_path, input_accepts=("native_image", "numpy_bgr"))
    manifest_path, model_path = _write_manifest(
        tmp_path, resize_owner="baked", norm_owner="baked", formula=None
    )
    facade.add("native", str(model_path), manifest_path=str(manifest_path))
    facade.switch("native")
    frame = _frame()
    camera = types.SimpleNamespace(last_raw=None)

    facade.detect(frame, camera=camera)

    assert engines[-1].received[-1]["image"] is frame


# ---------- capability proxying ---------- #


def test_a_capability_the_dialect_does_not_provide_is_refused(tmp_path):
    facade, _ = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)
    facade.add("normal", str(model_path), manifest_path=str(manifest_path))
    facade.switch("normal")

    with pytest.raises(ModelConfigError) as exc:
        facade.segment(_frame())
    assert exc.value.code == E_DIALECT_UNSUPPORTED

    with pytest.raises(ModelConfigError) as exc:
        facade.classify(_frame())
    assert exc.value.code == E_DIALECT_UNSUPPORTED

    with pytest.raises(ModelConfigError) as exc:
        facade.get_mask(0)
    assert exc.value.code == E_DIALECT_UNSUPPORTED

    assert facade.last_error_code == E_DIALECT_UNSUPPORTED
    assert facade.degraded is True


def test_detect_without_a_model_is_recorded_and_not_raised(tmp_path):
    facade, _ = _facade(tmp_path)
    assert facade.detect(_frame()) == []
    assert facade.last_error_code == "E_MODEL_NOT_LOADED"
    assert facade.error_count == 1
    assert facade.degraded is True


# ---------- statistics ---------- #


def test_stats_by_engine_reflects_the_recorded_inference_counts(tmp_path):
    facade, _ = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)
    facade.add("normal", str(model_path), manifest_path=str(manifest_path))
    facade.switch("normal")

    for _ in range(3):
        facade.detect(_frame())

    stats = facade.stats_by_engine()
    assert stats[ENGINE_ID].infer_count == 3
    assert stats[ENGINE_ID].infer_avg_ms >= 0.0
    assert stats[ENGINE_ID].error_count == 0


def test_a_failed_inference_is_counted_and_degrades_the_facade(tmp_path):
    facade, _ = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)
    facade.add("normal", str(model_path), manifest_path=str(manifest_path))
    facade.switch("normal")

    def failing_detect(payload: Any, **kwargs: Any) -> list[Detection]:
        raise RuntimeError("the engine exploded")

    facade._active.capability.detect = failing_detect
    assert facade.detect(_frame()) == []
    assert facade.last_error_code == "E_INFER_FAILED"
    assert facade.degraded is True
    assert facade.stats_by_engine()[ENGINE_ID].error_count == 1


# ---------- the runtime surface ---------- #


def test_the_runtime_exposes_the_engine_the_artifact_the_params_and_the_stats(tmp_path):
    manifest_path, model_path = _write_manifest(tmp_path, wrapper_params={"conf_th": 0.4})
    manifest = load_manifest(manifest_path)
    artifact = manifest.artifacts[ENGINE_ID]
    engine = FakeEngine()

    runtime = ModelRuntime(engine, artifact, manifest=manifest)

    assert runtime.engine is engine
    assert runtime.artifact is artifact
    assert runtime.params == {"conf_th": 0.4}
    assert isinstance(runtime.stats, EngineStats)
    assert runtime.stats.engine_id == ENGINE_ID
    assert runtime.stats.model_name == "steelball"
    assert callable(runtime.preprocess)


def test_the_runtime_exposes_the_labels_and_the_family_it_was_built_with(tmp_path):
    manifest_path, model_path = _write_manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    artifact = manifest.artifacts[ENGINE_ID]

    runtime = ModelRuntime(engine=FakeEngine(), artifact=artifact, manifest=manifest)

    assert runtime.labels == ("ball", "cube")
    assert runtime.family == "yolo11"


def test_a_runtime_can_be_built_with_the_labels_and_the_family_explicitly(tmp_path):
    manifest_path, model_path = _write_manifest(tmp_path)
    manifest = load_manifest(manifest_path)
    artifact = manifest.artifacts[ENGINE_ID]

    runtime = ModelRuntime(
        FakeEngine(),
        artifact,
        labels=["one", "two"],
        family="yolov5",
    )

    assert runtime.labels == ("one", "two")
    assert runtime.family == "yolov5"


def test_the_facade_satisfies_the_ai_inference_protocol(tmp_path):
    facade, _ = _facade(tmp_path)
    assert isinstance(facade, AIInference)


def test_the_facade_reports_the_contract_of_the_active_model(tmp_path):
    facade, _ = _facade(tmp_path)
    manifest_path, model_path = _write_manifest(tmp_path)
    facade.add("normal", str(model_path), manifest_path=str(manifest_path))
    facade.switch("normal")

    assert facade.input_width == 640
    assert facade.input_height == 352
    assert facade.labels == ["ball", "cube"]
    assert facade.model_path == str(model_path)
    assert facade.model_family == "yolo11"
