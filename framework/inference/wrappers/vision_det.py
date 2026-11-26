"""Built-in detection dialects of the inference kernel.

Two dialects expose the same task interface, `detect(frame) -> list[Detection]`:

* the raw dialect decodes a raw prediction tensor and applies NMS itself
* the end-to-end dialect maps the objects an end-to-end engine already decoded

Both declare the task capabilities they provide, so the facade can proxy them
without guessing.  The module knows no engine, no manifest and no runtime
implementation: it only uses the runtime's public surface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    Detection,
    Keypoint,
    ModelConfigError,
)
from framework.inference import ops

if TYPE_CHECKING:
    from framework.inference.runtime import ModelRuntime

__all__ = [
    "DEFAULT_CONF_TH",
    "DEFAULT_IOU_TH",
    "DEFAULT_OUTPUT_LAYOUT",
    "DEFAULT_SCORE_ACTIVATION",
    "LOGICAL_INPUT_NAME",
    "End2endDetAPI",
    "RawDetAPI",
    "build_end2end",
    "build_raw",
]

DEFAULT_CONF_TH = 0.3
DEFAULT_IOU_TH = 0.4
DEFAULT_OUTPUT_LAYOUT = "auto"
DEFAULT_SCORE_ACTIVATION = "none"
LOGICAL_INPUT_NAME = "image"


class RawDetAPI:
    """Raw dialect: decode the prediction tensor and suppress boxes.

    Preprocessing belongs to the runtime, decoding and NMS to ops, so this
    class only routes the three steps and applies the thresholds.
    """

    provides = frozenset({"detect"})

    def __init__(self, rt: ModelRuntime) -> None:
        self._rt = rt
        self._main_out = _main_output_name(rt)
        params = _params(rt)
        self._conf_th = float(params.get("conf_th", DEFAULT_CONF_TH))
        self._iou_th = float(params.get("iou_th", DEFAULT_IOU_TH))
        self._output_layout = str(params.get("output_layout", DEFAULT_OUTPUT_LAYOUT))
        self._score_activation = str(
            params.get("score_activation", DEFAULT_SCORE_ACTIVATION)
        )

    def detect(self, frame: Any, **kwargs: Any) -> list[Detection]:
        """Run one detection on a frame and return source frame detections."""
        conf_th = float(kwargs.pop("conf_th", self._conf_th))
        iou_th = float(kwargs.pop("iou_th", self._iou_th))
        output_layout = str(kwargs.pop("output_layout", self._output_layout))
        score_activation = str(kwargs.pop("score_activation", self._score_activation))
        labels = _label_list(self._rt)
        if not labels:
            raise ModelConfigError(
                "The logical model has no label list, the class count is unknown",
                E_CONTRACT_MISSING,
            )
        pre = self._rt.preprocess(frame)
        out = self._rt.engine.run(pre.inputs)
        pred = out[self._main_out]
        return ops.decode_yolo(
            pred,
            labels,
            pre.transform,
            conf_th=conf_th,
            iou_th=iou_th,
            output_layout=output_layout,
            score_activation=score_activation,
        )


class End2endDetAPI:
    """End-to-end dialect: map the objects the engine already decoded.

    The engine has done resize, normalisation, inference, decoding and NMS,
    so this class must not decode or suppress the boxes again.  The
    thresholds are handed to the engine, which filters with them, and the
    score filter is repeated afterwards because a runtime may not honour the
    threshold it was given.
    """

    provides = frozenset({"detect"})

    def __init__(self, rt: ModelRuntime) -> None:
        self._rt = rt
        self._main_out = _main_output_name(rt)
        params = _params(rt)
        self._conf_th = float(params.get("conf_th", DEFAULT_CONF_TH))
        self._iou_th = float(params.get("iou_th", DEFAULT_IOU_TH))

    def detect(self, frame: Any, **kwargs: Any) -> list[Detection]:
        """Run one detection on a frame and return source frame detections."""
        conf_th = float(kwargs.pop("conf_th", self._conf_th))
        iou_th = float(kwargs.pop("iou_th", self._iou_th))
        objects = self._rt.engine.run(
            {LOGICAL_INPUT_NAME: frame}, conf_th=conf_th, iou_th=iou_th
        )[self._main_out]
        labels = _label_list(self._rt)
        return [
            _to_detection(obj, labels)
            for obj in objects
            if _score_of(obj) >= conf_th
        ]


def build_raw(rt: ModelRuntime) -> RawDetAPI:
    """Build the capability object of the raw detection dialect."""
    return RawDetAPI(rt)


def build_end2end(rt: ModelRuntime) -> End2endDetAPI:
    """Build the capability object of the end-to-end detection dialect."""
    return End2endDetAPI(rt)


def _params(rt: Any) -> dict[str, Any]:
    params = getattr(rt, "params", None)
    return dict(params) if params else {}


def _label_list(rt: Any) -> list[str]:
    labels = getattr(rt, "labels", None)
    if not labels:
        return []
    return [str(label) for label in labels]


def _main_output_name(rt: Any) -> str:
    """Return the logical name of the main output declared by the artifact."""
    outputs = getattr(rt.artifact.io, "outputs", None) or {}
    if not outputs:
        raise ModelConfigError(
            "The artifact declares no logical output", E_CONTRACT_MISSING
        )
    return next(iter(outputs))


def _to_detection(obj: Any, labels: Sequence[str]) -> Detection:
    """Map one engine object into a framework detection."""
    class_id = _int_attr(obj, "class_id", -1)
    return Detection(
        x=_int_attr(obj, "x", 0),
        y=_int_attr(obj, "y", 0),
        w=_int_attr(obj, "w", 0),
        h=_int_attr(obj, "h", 0),
        class_id=class_id,
        score=_float_attr(obj, "score", 0.0),
        label=_label_for(class_id, labels),
        angle=_optional_float_attr(obj, "angle"),
        keypoints=_to_keypoints(getattr(obj, "keypoints", None)),
    )


def _score_of(obj: Any) -> float:
    """Return the score the engine reports for one object."""
    value = getattr(obj, "score", None)
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _label_for(class_id: int, labels: Sequence[str]) -> str:
    if 0 <= class_id < len(labels):
        return labels[class_id]
    return ""


def _int_attr(obj: Any, name: str, default: int) -> int:
    value = getattr(obj, name, None)
    if value is None:
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _float_attr(obj: Any, name: str, default: float) -> float:
    value = getattr(obj, name, None)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_float_attr(obj: Any, name: str) -> float | None:
    value = getattr(obj, name, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_keypoints(value: Any) -> list[Keypoint]:
    """Map whatever shape the engine reports keypoints in into Keypoint objects."""
    if not value:
        return []
    points: list[Keypoint] = []
    for item in value:
        if isinstance(item, Keypoint):
            points.append(item)
            continue
        x = getattr(item, "x", None)
        y = getattr(item, "y", None)
        score = getattr(item, "score", None)
        if isinstance(item, dict):
            x = item.get("x", x)
            y = item.get("y", y)
            score = item.get("score", score)
        elif isinstance(item, (tuple, list)):
            if len(item) < 2:
                continue
            x = item[0] if x is None else x
            y = item[1] if y is None else y
            if len(item) > 2 and score is None:
                score = item[2]
        if x is None or y is None:
            continue
        points.append(
            Keypoint(
                x=float(x),
                y=float(y),
                score=float(score) if score is not None else 0.0,
            )
        )
    return points
