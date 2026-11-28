"""Engine adapter for maix.nn, the NPU runtime of the MaixCAM devices.

The adapter is end-to-end: maix.nn returns already decoded objects, so the
engine never interprets them and never post-processes them; the wrapper
turns them into framework detections.

The runtime class is derived from the model family declared in the MUD file.
A class that does not match the family decodes the output with the wrong
post-processing and silently produces wrong boxes, so an unknown family is
an explicit failure instead of a fallback to a generic net class.

Every maix import is lazy and local, so the module can be imported and
unit-tested on a development machine without the firmware installed.
"""

from __future__ import annotations

import configparser
import importlib
import logging
from typing import Any, Optional

import numpy as np

from framework.hal.interface.ai import (
    E_CONTRACT_MISSING,
    E_ENGINE_UNAVAILABLE,
    E_INFER_FAILED,
    E_INVALID_INPUT,
    E_MODEL_LOAD_FAILED,
    E_MODEL_NOT_LOADED,
    E_MODEL_TYPE_UNSUPPORTED,
    AIError,
    ModelConfigError,
    ModelLoadError,
)
from framework.inference.contract import ArtifactRef, Capabilities, TensorSpec

logger = logging.getLogger(__name__)

__all__ = [
    "ENGINE_ID",
    "MODEL_FAMILY_CLASS_NAMES",
    "MaixNnEngine",
    "create_engine",
    "model_class_name",
    "model_family_from_mud",
]

ENGINE_ID = "maix_nn"
LOGICAL_INPUT_NAME = "image"
MUD_SECTION = "extra"
MUD_MODEL_TYPE_KEY = "model_type"

# The keys the runtime hands every engine regardless of its dialect.  This
# adapter consumes neither: the MUD file bakes the preprocessing, and the
# firmware constructor exposes no accelerator options.  They are accepted
# and dropped here rather than rejected, so an engine never has to know
# which keys the framework added after it was written.
FRAMEWORK_OPTS = frozenset({"canonical", "runtime"})

MODEL_FAMILY_CLASS_NAMES: dict[str, str] = {
    "yolo11": "YOLO11",
    "yolov8": "YOLOv8",
    "yolov5": "YOLOv5",
    "classifier": "Classifier",
    "hand_landmarks": "HandLandmarks",
}


def model_family_from_mud(model_path: str) -> str:
    """Read the model family from the [extra] model_type entry of a MUD file.

    The MUD file is the declaration of record for a model, so the family
    taken from it keeps the runtime class and the model's own
    post-processing in sync.  An empty string is returned when the file is
    missing or cannot be parsed.
    """
    if not model_path:
        return ""
    try:
        parser = configparser.ConfigParser(interpolation=None)
        if not parser.read(model_path):
            return ""
        return parser.get(MUD_SECTION, MUD_MODEL_TYPE_KEY, fallback="").strip()
    except Exception as exc:
        logger.debug("Cannot parse MUD '%s': %s", model_path, exc)
        return ""


def model_class_name(model_family: str) -> str:
    """Map a model family to the name of its maix.nn runtime class.

    Raises ModelConfigError with E_MODEL_TYPE_UNSUPPORTED for a family the
    framework does not know: falling back to a generic class would decode
    the output with the wrong post-processing and silently yield wrong boxes.
    """
    try:
        return MODEL_FAMILY_CLASS_NAMES[model_family]
    except KeyError:
        raise ModelConfigError(
            f"Unsupported model family '{model_family}'",
            E_MODEL_TYPE_UNSUPPORTED,
        ) from None


def _import_maix_nn() -> Any:
    """Import the maix.nn runtime, or report it as an unavailable engine."""
    try:
        return importlib.import_module("maix.nn")
    except Exception as exc:
        raise ModelConfigError(
            f"The maix.nn runtime library is not importable: {exc}",
            E_ENGINE_UNAVAILABLE,
        ) from exc


def _import_maix_image() -> Any:
    try:
        return importlib.import_module("maix.image")
    except Exception as exc:
        raise ModelConfigError(
            f"The maix.image module is not importable: {exc}",
            E_ENGINE_UNAVAILABLE,
        ) from exc


class MaixNnEngine:
    """InferenceEngine implementation backed by maix.nn.

    The engine owns one loaded model.  Its io addressing is implicit: the
    single model takes the logical input "image" and returns the decoded
    objects under the first logical output name declared by the artifact.
    """

    engine_id = ENGINE_ID

    def __init__(self) -> None:
        self._model: Any = None
        self._artifact: Optional[ArtifactRef] = None
        self._model_family: str = ""

    def load(self, artifact: ArtifactRef, **opts: Any) -> None:
        """Load the artifact into a maix.nn runtime class matching its family.

        The family is resolved from the MUD file first, so an unsupported
        family is reported as a contract problem instead of a missing
        runtime.  A runtime that refuses the file raises ModelLoadError
        with E_MODEL_LOAD_FAILED; nothing is degraded silently.

        The framework keys travel in ``opts`` like they do for every other
        engine.  They are dropped before the firmware constructor is called,
        because that constructor takes the model path and ``dual_buff``
        only; anything else is a TypeError from the binding.
        """
        model_family = model_family_from_mud(artifact.path)
        if not model_family:
            raise ModelConfigError(
                f"Cannot read the model family from MUD '{artifact.path}'",
                E_MODEL_TYPE_UNSUPPORTED,
            )
        class_name = model_class_name(model_family)
        maix_nn = _import_maix_nn()
        slot_cls = getattr(maix_nn, class_name, None)
        if slot_cls is None:
            raise ModelConfigError(
                f"Firmware provides no model class '{class_name}' "
                f"for family '{model_family}'",
                E_MODEL_TYPE_UNSUPPORTED,
            )
        kwargs: dict[str, Any] = {"dual_buff": True}
        kwargs.update(
            {key: value for key, value in opts.items() if key not in FRAMEWORK_OPTS}
        )
        try:
            self._model = slot_cls(artifact.path, **kwargs)
        except Exception as exc:
            self._model = None
            raise ModelLoadError(
                f"Failed to load model '{artifact.path}': {exc}",
                E_MODEL_LOAD_FAILED,
            ) from exc
        self._artifact = artifact
        self._model_family = model_family

    def input_specs(self) -> list[TensorSpec]:
        """Empty: maix.nn exposes no tensor-level specs.

        The load-time cross-check against the manifest io declaration must
        skip engines that declare no specs.
        """
        return []

    def output_specs(self) -> list[TensorSpec]:
        """Empty: maix.nn exposes no tensor-level specs."""
        return []

    def capabilities(self) -> Capabilities:
        """Describe the maix.nn runtime: NPU, int8, implicit single-model io.

        The GIL flags follow the pybind11 default of the maix bindings,
        which hold the GIL during both loading and inference.
        """
        return Capabilities(
            device="npu",
            quant="int8",
            layouts=("nhwc",),
            dynamic_shape=False,
            input_accepts=("native_image", "numpy_bgr"),
            io_binding="positional",
            releases_gil=False,
            load_holds_gil=True,
            load_cost_hint="200-500ms",
        )

    def run(self, inputs: dict[str, Any], **options: Any) -> dict[str, Any]:
        """Run one inference step and return the decoded objects.

        The objects are returned untouched under the logical output name
        declared by the artifact; interpreting them is the wrapper's job.

        The run options are forwarded to the runtime's own detect call, so a
        dialect can set the thresholds the runtime filters with instead of
        relying on the runtime defaults.
        """
        if self._model is None:
            raise AIError(
                f"Engine '{ENGINE_ID}' has no model loaded", E_MODEL_NOT_LOADED
            )
        if LOGICAL_INPUT_NAME not in inputs:
            raise ModelConfigError(
                f"engine.run() requires the logical input '{LOGICAL_INPUT_NAME}'",
                E_INVALID_INPUT,
            )
        frame = self._to_native_frame(inputs[LOGICAL_INPUT_NAME])
        try:
            objects = self._model.detect(frame, **self._detect_options(options))
        except Exception as exc:
            raise AIError(f"maix.nn detect() failed: {exc}", E_INFER_FAILED) from exc
        return {self._output_name(): objects}

    @staticmethod
    def _detect_options(options: dict[str, Any]) -> dict[str, float]:
        """Keep the options the runtime understands, drop the rest."""
        return {
            key: float(options[key])
            for key in ("conf_th", "iou_th")
            if key in options and options[key] is not None
        }

    def unload(self) -> None:
        """Release the model reference."""
        self._model = None
        self._artifact = None
        self._model_family = ""

    def _to_native_frame(self, frame: Any) -> Any:
        """Convert a BGR frame into the native image the runtime consumes."""
        if not isinstance(frame, np.ndarray):
            return frame
        maix_image = _import_maix_image()
        return maix_image.cv2image(frame[:, :, ::-1], bgr=False, copy=True)

    def _output_name(self) -> str:
        """Return the logical name of the primary output declared by the artifact."""
        if self._artifact is None:
            raise AIError(
                f"Engine '{ENGINE_ID}' has no artifact bound", E_MODEL_NOT_LOADED
            )
        outputs = self._artifact.io.outputs
        if not outputs:
            raise AIError(
                "The artifact declares no logical output", E_CONTRACT_MISSING
            )
        return next(iter(outputs))


def create_engine() -> Optional[MaixNnEngine]:
    """Build an unloaded maix.nn engine, or None when the runtime is absent."""
    try:
        importlib.import_module("maix.nn")
    except Exception as exc:
        logger.info("maix.nn is not importable: %s", exc)
        return None
    return MaixNnEngine()
