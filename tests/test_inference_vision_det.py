"""Tests for the built-in detection dialects of the inference kernel.

The fakes below stand in for the runtime, the engine and the artifact, so
the dialects are exercised without an engine library and without a device.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from framework.hal.interface.ai import Detection, Keypoint
from framework.inference import ops
from framework.inference.contract import GeometryTransform, PreprocessResult
from framework.inference.wrappers import resolve_wrapper_factory
from framework.inference.wrappers.vision_det import build_end2end, build_raw

LABELS = ["ball", "nut", "bolt"]


class FakeEngine:
    """Engine double: returns the prepared outputs and records its inputs."""

    def __init__(self, outputs: dict[str, Any]) -> None:
        self._outputs = outputs
        self.runs: list[dict[str, Any]] = []

    def run(self, inputs: dict[str, Any]) -> dict[str, Any]:
        self.runs.append(dict(inputs))
        return self._outputs


@dataclass
class FakeIOMapping:
    inputs: dict[str, str] = field(default_factory=lambda: {"image": "data"})
    outputs: dict[str, str] = field(default_factory=lambda: {"prediction": "output0"})


@dataclass
class FakeArtifact:
    io: FakeIOMapping = field(default_factory=FakeIOMapping)


class FakeRuntime:
    """Runtime double exposing the surface a wrapper is allowed to use."""

    def __init__(
        self,
        engine: FakeEngine,
        artifact: FakeArtifact,
        labels: list[str],
        transform: GeometryTransform,
        params: dict[str, Any] | None = None,
    ) -> None:
        self.engine = engine
        self.artifact = artifact
        self.labels = list(labels)
        self.transform = transform
        self.params = dict(params or {})
        self.preprocess_calls = 0

    def preprocess(self, frame: Any) -> PreprocessResult:
        self.preprocess_calls += 1
        return PreprocessResult(inputs={"image": frame}, transform=self.transform)


@dataclass
class FakeObject:
    x: int
    y: int
    w: int
    h: int
    score: float
    class_id: int
    angle: float | None = None
    keypoints: list[Any] | None = None


def letterbox_transform(scale: float = 0.5) -> GeometryTransform:
    return GeometryTransform(
        mode="letterbox",
        scale_x=scale,
        scale_y=scale,
        offset_x=0,
        offset_y=0,
        src_w=200,
        src_h=100,
    )


def raw_runtime(
    prediction: np.ndarray,
    outputs: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    labels: list[str] | None = None,
) -> FakeRuntime:
    io = FakeIOMapping(outputs=outputs or {"prediction": "output0"})
    engine = FakeEngine({name: prediction for name in io.outputs})
    return FakeRuntime(
        engine=engine,
        artifact=FakeArtifact(io=io),
        labels=labels if labels is not None else LABELS,
        transform=letterbox_transform(),
        params=params,
    )


def detection_rows(rows: list[list[float]]) -> np.ndarray:
    return np.asarray(rows, dtype=np.float32)


def test_raw_dialect_returns_detections_in_source_frame_pixels():
    prediction = detection_rows([[50.0, 40.0, 20.0, 10.0, 0.9, 0.05, 0.02]])
    runtime = raw_runtime(prediction)

    detections = build_raw(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert (detections[0].x, detections[0].y) == (80, 70)
    assert (detections[0].w, detections[0].h) == (40, 20)
    assert detections[0].score == pytest.approx(0.9)
    assert detections[0].class_id == 0


def test_raw_dialect_attaches_the_label_text():
    prediction = detection_rows([[50.0, 40.0, 20.0, 10.0, 0.1, 0.8, 0.05]])
    runtime = raw_runtime(prediction)

    detections = build_raw(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert detections[0].class_id == 1
    assert detections[0].label == "nut"


def test_raw_dialect_uses_the_runtime_preprocess_result():
    prediction = detection_rows([[50.0, 40.0, 20.0, 10.0, 0.9, 0.05, 0.02]])
    runtime = raw_runtime(prediction)
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    build_raw(runtime).detect(frame)

    assert runtime.preprocess_calls == 1
    assert runtime.engine.runs[0] == {"image": frame}


def test_raw_dialect_respects_the_production_conf_threshold():
    prediction = detection_rows(
        [
            [50.0, 40.0, 20.0, 10.0, 0.9, 0.05, 0.02],
            [10.0, 10.0, 4.0, 4.0, 0.25, 0.02, 0.01],
        ]
    )
    runtime = raw_runtime(prediction)

    detections = build_raw(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 1


def test_raw_dialect_accepts_a_per_call_conf_threshold():
    prediction = detection_rows(
        [
            [50.0, 40.0, 20.0, 10.0, 0.9, 0.05, 0.02],
            [10.0, 10.0, 4.0, 4.0, 0.25, 0.02, 0.01],
        ]
    )
    runtime = raw_runtime(prediction)

    detections = build_raw(runtime).detect(
        np.zeros((100, 200, 3), dtype=np.uint8), conf_th=0.2
    )

    assert len(detections) == 2


def test_raw_dialect_takes_the_main_output_from_the_artifact_io_map():
    prediction = detection_rows([[50.0, 40.0, 20.0, 10.0, 0.9, 0.05, 0.02]])
    runtime = raw_runtime(
        prediction, outputs={"raw_tensor": 0, "auxiliary": 1}
    )

    detections = build_raw(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert list(runtime.artifact.io.outputs)[0] == "raw_tensor"


def test_raw_dialect_forwards_output_layout_and_score_activation(monkeypatch):
    captured: dict[str, Any] = {}

    def fake_decode(
        pred: Any,
        labels: Any,
        transform: Any,
        conf_th: float = 0.3,
        iou_th: float = 0.4,
        output_layout: str = "auto",
        score_activation: str = "none",
        out: Any = None,
    ) -> list[Detection]:
        captured.update(
            labels=list(labels),
            conf_th=conf_th,
            iou_th=iou_th,
            output_layout=output_layout,
            score_activation=score_activation,
        )
        return []

    monkeypatch.setattr(ops, "decode_yolo", fake_decode)
    runtime = raw_runtime(
        detection_rows([[50.0, 40.0, 20.0, 10.0, 0.9, 0.05, 0.02]]),
        params={"output_layout": "channels_first", "score_activation": "sigmoid"},
    )

    build_raw(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert captured["output_layout"] == "channels_first"
    assert captured["score_activation"] == "sigmoid"
    assert captured["labels"] == LABELS
    assert captured["conf_th"] == 0.3
    assert captured["iou_th"] == 0.4


def test_raw_dialect_decodes_a_channels_first_prediction():
    prediction = np.asarray(
        [[[50.0], [40.0], [20.0], [10.0], [0.9], [0.05], [0.02]]],
        dtype=np.float32,
    ).reshape(1, 7, 1)
    runtime = raw_runtime(prediction, params={"output_layout": "channels_first"})

    detections = build_raw(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].label == "ball"


def test_raw_dialect_applies_the_sigmoid_score_activation():
    prediction = detection_rows([[50.0, 40.0, 20.0, 10.0, 0.0, -5.0, -5.0]])
    runtime = raw_runtime(prediction, params={"score_activation": "sigmoid"})

    detections = build_raw(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 1
    assert detections[0].score == pytest.approx(0.5, abs=1e-3)


def test_end2end_dialect_maps_engine_objects_into_detections():
    objects = [FakeObject(x=10, y=20, w=30, h=40, score=0.75, class_id=2)]
    io = FakeIOMapping(outputs={"objects": "objects"})
    runtime = FakeRuntime(
        engine=FakeEngine({"objects": objects}),
        artifact=FakeArtifact(io=io),
        labels=LABELS,
        transform=letterbox_transform(),
    )
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    detections = build_end2end(runtime).detect(frame)

    assert len(detections) == 1
    assert isinstance(detections[0], Detection)
    assert (detections[0].x, detections[0].y) == (10, 20)
    assert (detections[0].w, detections[0].h) == (30, 40)
    assert detections[0].score == pytest.approx(0.75)
    assert detections[0].class_id == 2
    assert detections[0].label == "bolt"
    assert runtime.engine.runs[0] == {"image": frame}
    assert runtime.preprocess_calls == 0


def test_end2end_dialect_does_not_suppress_overlapping_boxes():
    objects = [
        FakeObject(x=0, y=0, w=100, h=100, score=0.9, class_id=0),
        FakeObject(x=5, y=5, w=100, h=100, score=0.8, class_id=0),
    ]
    io = FakeIOMapping(outputs={"objects": "objects"})
    runtime = FakeRuntime(
        engine=FakeEngine({"objects": objects}),
        artifact=FakeArtifact(io=io),
        labels=LABELS,
        transform=letterbox_transform(),
    )

    detections = build_end2end(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 2
    assert [det.score for det in detections] == pytest.approx([0.9, 0.8])


def test_end2end_dialect_reads_the_main_output_from_the_artifact_io_map():
    objects = [FakeObject(x=1, y=2, w=3, h=4, score=0.5, class_id=0)]
    io = FakeIOMapping(outputs={"decoded": "decoded"})
    runtime = FakeRuntime(
        engine=FakeEngine({"decoded": objects}),
        artifact=FakeArtifact(io=io),
        labels=LABELS,
        transform=letterbox_transform(),
    )

    detections = build_end2end(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert len(detections) == 1


def test_end2end_dialect_tolerates_missing_optional_attributes():
    objects = [FakeObject(x=1, y=2, w=3, h=4, score=0.5, class_id=0)]
    io = FakeIOMapping(outputs={"objects": "objects"})
    runtime = FakeRuntime(
        engine=FakeEngine({"objects": objects}),
        artifact=FakeArtifact(io=io),
        labels=LABELS,
        transform=letterbox_transform(),
    )

    detections = build_end2end(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert detections[0].angle is None
    assert detections[0].keypoints == []


def test_end2end_dialect_maps_angle_and_keypoints():
    points = [Keypoint(x=1.0, y=2.0, score=0.5)]
    objects = [
        FakeObject(
            x=1, y=2, w=3, h=4, score=0.5, class_id=0, angle=45.0, keypoints=points
        )
    ]
    io = FakeIOMapping(outputs={"objects": "objects"})
    runtime = FakeRuntime(
        engine=FakeEngine({"objects": objects}),
        artifact=FakeArtifact(io=io),
        labels=LABELS,
        transform=letterbox_transform(),
    )

    detections = build_end2end(runtime).detect(np.zeros((100, 200, 3), dtype=np.uint8))

    assert detections[0].angle == pytest.approx(45.0)
    assert detections[0].keypoints == points


def test_capability_objects_declare_the_detect_capability():
    runtime = raw_runtime(detection_rows([[50.0, 40.0, 20.0, 10.0, 0.9, 0.05, 0.02]]))

    assert build_raw(runtime).provides == frozenset({"detect"})
    assert build_end2end(runtime).provides == frozenset({"detect"})


def test_factory_callables_build_capability_objects():
    runtime = raw_runtime(detection_rows([[50.0, 40.0, 20.0, 10.0, 0.9, 0.05, 0.02]]))

    raw_api = build_raw(runtime)
    end2end_api = build_end2end(runtime)

    assert callable(raw_api.detect)
    assert callable(end2end_api.detect)
    assert raw_api.provides == frozenset({"detect"})
    assert end2end_api.provides == frozenset({"detect"})


def test_the_registry_resolves_the_two_dialects_to_these_factories():
    assert resolve_wrapper_factory("vision.det.raw") is build_raw
    assert resolve_wrapper_factory("vision.det.end2end") is build_end2end
