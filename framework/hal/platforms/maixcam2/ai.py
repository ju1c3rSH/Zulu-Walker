from __future__ import annotations

import logging
import time
from typing import Optional, Union

import numpy as np

import maix.image
import maix.nn

from framework.hal.interface.ai import (
    ContractError,
    Detection,
    E_INFER_FAILED,
    E_INVALID_INPUT,
    E_MODEL_LOAD_FAILED,
    E_MODEL_NOT_LOADED,
    E_MODEL_TYPE_UNSUPPORTED,
    E_OUTPUT_LAYOUT_MISMATCH,
    Keypoint,
    MaskStats,
    ModelConfigError,
    ModelLoadError,
    SegmentResult,
)

logger = logging.getLogger(__name__)

# Runtime class per explicitly configured model type.
_SLOT_CLASSES = {
    "yolo": maix.nn.YOLO11,
    "classifier": maix.nn.Classifier,
    "hand_landmarks": maix.nn.HandLandmarks,
    "nn": maix.nn.NN,
}

# MUD [extra] model_type -> runtime class.  The MUD file is the declaration
# of record for a model, so resolving "auto" from it keeps the runtime class
# and the model's own post-processing in sync.  A family the running firmware
# does not provide resolves to None and is reported as an error instead of
# being replaced by a generic net class.
_MUD_MODEL_TYPES = {
    "yolo11": lambda: maix.nn.YOLO11,
    "yolov8": lambda: getattr(maix.nn, "YOLOv8", None),
    "yolov5": lambda: getattr(maix.nn, "YOLOv5", None),
    "classifier": lambda: maix.nn.Classifier,
    "hand_landmarks": lambda: maix.nn.HandLandmarks,
}

# Keypoint packing per model family: (first index, stride).
# stride 3 packs (x, y, score), stride 2 packs (x, y).  hand_landmarks
# carries 8 leading values before the point list.
_KEYPOINT_LAYOUTS = {
    "hand_landmarks": (8, 3),
    "yolo11": (0, 3),
    "yolov8": (0, 3),
    "yolov5": (0, 2),
}


class MaixCam2AI:
    """MaixCAM2 NPU inference backend.

    Failures are handled on two levels:

    * Contract and configuration problems (unresolvable model type, a
      runtime class that cannot decode the output, a keypoint payload that
      does not match the declared layout) raise, because continuing would
      silently produce wrong boxes.
    * Runtime problems (frame conversion, a failing NPU call) are recorded
      and surfaced through `last_error_code` and `degraded`, so one bad
      frame does not take the vision pipeline down.
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict] = {}  # nick_name -> {path, type, kwargs}
        self._active_name: str = ""
        self._model: Optional[Union[maix.nn.YOLO11, maix.nn.Classifier, maix.nn.HandLandmarks, maix.nn.NN]] = None
        self._model_type: str = ""
        self._model_family: str = ""
        self._model_path: str = ""

        self._last_error_code: str = ""
        self._error_count: int = 0
        self._degraded: bool = False

        self._infer_count: int = 0
        self._infer_total_us: int = 0
        self._infer_window_start: float = time.monotonic()
        self._infer_fps: float = 0.0
        self._infer_avg_ms: float = 0.0
        self._infer_timestamp_ns: int = 0

    # ------------------------------------------------------------------ #
    #  Registry API
    # ------------------------------------------------------------------ #

    @property
    def models(self) -> list[str]:
        return list(self._registry.keys())

    @property
    def active_model(self) -> str:
        return self._active_name

    @property
    def model_type(self) -> str:
        return self._model_type

    @property
    def model_family(self) -> str:
        """Model family resolved from the model declaration."""
        return self._model_family

    def add(self, nick_name: str, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        if nick_name in self._registry:
            logger.warning("Model '%s' already registered, overwriting", nick_name)
        self._registry[nick_name] = {
            "path": model_path,
            "type": model_type,
            "kwargs": kwargs,
        }
        logger.info("Registered model '%s' -> %s (type=%s)", nick_name, model_path, model_type)
        return True

    def remove(self, nick_name: str) -> None:
        if nick_name not in self._registry:
            logger.warning("Attempt to remove unknown model '%s'", nick_name)
            return
        if self._active_name == nick_name:
            self.unload()
        del self._registry[nick_name]
        logger.info("Removed model '%s'", nick_name)

    # TODO: preload every model into the NPU and turn switch() into a plain
    # pointer swap (O(1)).  Today each switch() reloads the .mud from flash
    # (~200-500ms).  Two models cost 30-40MB of CMM, which the 256MB pool
    # easily absorbs.
    def switch(self, nick_name: str) -> bool:
        """Activate a registered model.

        Raises ModelConfigError when the model type cannot be resolved and
        ModelLoadError when the runtime class refuses the model file.
        """
        if nick_name not in self._registry:
            self._report_error(
                E_MODEL_NOT_LOADED, f"Cannot switch to unknown model '{nick_name}'")
            return False
        if self._active_name == nick_name and self._model is not None:
            return True  # already active

        self.unload()
        info = self._registry[nick_name]
        path = info["path"]
        model_type = info["type"]
        kwargs = info["kwargs"]

        slot_cls, family = self._resolve_class(model_type, path)

        try:
            self._model = slot_cls(path, dual_buff=True, **kwargs)
        except Exception as e:
            self._model = None
            self._model_path = ""
            self._active_name = ""
            message = f"Failed to load model '{nick_name}' from {path}: {e}"
            self._report_error(E_MODEL_LOAD_FAILED, message)
            raise ModelLoadError(message, E_MODEL_LOAD_FAILED) from e

        self._model_path = path
        self._model_type = model_type
        self._model_family = family
        self._active_name = nick_name
        self._clear_error()
        logger.info("Switched to model '%s' (%s, family=%s)", nick_name, path, family)
        return True

    # ------------------------------------------------------------------ #
    #  Convenience aliases (single-model usage)
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        return self.add("default", model_path, model_type=model_type, **kwargs) and self.switch("default")

    def unload(self) -> None:
        if self._active_name:
            logger.info("Unloading model '%s'", self._active_name)
        self._model = None
        self._model_type = ""
        self._model_family = ""
        self._model_path = ""
        self._active_name = ""

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def input_width(self) -> int:
        if self._model is None:
            return 0
        try:
            return self._model.input_width()
        except Exception as e:
            logger.warning("Failed to get input_width: %s", e)
            return 0

    @property
    def input_height(self) -> int:
        if self._model is None:
            return 0
        try:
            return self._model.input_height()
        except Exception as e:
            logger.warning("Failed to get input_height: %s", e)
            return 0

    @property
    def labels(self) -> list[str]:
        if self._model is None:
            return []
        try:
            return self._model.labels
        except Exception as e:
            logger.warning("Failed to get labels: %s", e)
            return []

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def last_error_code(self) -> str:
        return self._last_error_code

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def degraded(self) -> bool:
        return self._degraded

    @property
    def infer_fps(self) -> float:
        return self._infer_fps

    @property
    def infer_avg_ms(self) -> float:
        return self._infer_avg_ms

    @property
    def infer_timestamp_ns(self) -> int:
        return self._infer_timestamp_ns

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def detect(
        self, frame,
        **kwargs
    ) -> list[Detection]:
        """Run object detection on frame.

        *frame* can be:
        - a BGR numpy array (standard pipeline path)
        - a `maix.image.Image` (zero-copy path, pass `_raw=True`)

        Returns a list of Detection dataclasses.  Runtime failures return an
        empty list and are recorded in `last_error_code`; a contract
        violation raises ContractError.
        """
        conf_th = kwargs.pop("conf_th", 0.3)
        iou_th = kwargs.pop("iou_th", 0.4)

        _is_raw = kwargs.pop("_raw", False)

        if self._model is None:
            self._report_error(
                E_MODEL_NOT_LOADED, "detect() called but no model is loaded")
            return []

        if not hasattr(self._model, "detect"):
            # The runtime class cannot decode this model's output.  Refuse
            # instead of returning an empty list that reads as "no target".
            self._fail_contract(
                f"Model class {type(self._model).__name__} provides no detect(); "
                f"family '{self._model_family}' does not match the output layout"
            )

        try:
            if _is_raw:
                img = frame
            else:
                frame_rgb = frame[:, :, ::-1]
                img = maix.image.cv2image(frame_rgb, bgr=False, copy=True)
        except Exception as e:
            self._report_error(E_INVALID_INPUT, f"cv2image conversion failed: {e}")
            return []

        try:
            t0 = time.perf_counter_ns()
            objects = self._model.detect(img, conf_th=conf_th, iou_th=iou_th, **kwargs)
            t_ns = time.perf_counter_ns() - t0
        except Exception as e:
            self._report_error(E_INFER_FAILED, f"Model detect() failed: {e}")
            return []

        t_us = t_ns // 1000
        self._infer_timestamp_ns = time.perf_counter_ns()
        self._infer_count += 1
        self._infer_total_us += t_us
        elapsed = time.monotonic() - self._infer_window_start
        if elapsed >= 1.0:
            self._infer_fps = self._infer_count / elapsed
            self._infer_avg_ms = (self._infer_total_us / self._infer_count) / 1000.0
            self._infer_count = 0
            self._infer_total_us = 0
            self._infer_window_start = time.monotonic()

        # Post-filter: some models don't fully honor conf_th internally
        objects = [o for o in objects if o.score >= conf_th]

        if not objects:
            logger.debug("detect() returned 0 objects")
        else:
            logger.debug("detect() returned %d objects", len(objects))

        results: list = []
        for obj in objects:
            kps = self._convert_keypoints(obj, self._model_family)
            angle: Optional[float]
            if hasattr(obj, "angle") and obj.angle != -9999:
                angle = float(obj.angle)
            else:
                angle = None

            seg_mask_np: Optional[np.ndarray] = None
            mask_stats: Optional[MaskStats] = None

            if hasattr(obj, "seg_mask") and obj.seg_mask is not None:
                seg_mask_np = self._extract_seg_mask(obj.seg_mask)
                if seg_mask_np is not None:
                    ys, xs = np.nonzero(seg_mask_np > 127)
                    if xs.size:
                        # The mask is box local, so the box origin is added
                        # to turn the mask centroid into frame coordinates.
                        mask_stats = MaskStats(
                            center_x=float(obj.x + xs.mean()),
                            center_y=float(obj.y + ys.mean()),
                            area_px=int(xs.size),
                        )

            det = Detection(
                x=obj.x,
                y=obj.y,
                w=obj.w,
                h=obj.h,
                class_id=obj.class_id,
                score=obj.score,
                label="",
                angle=angle,
                keypoints=kps,
                mask_index=-1,
                seg_mask=seg_mask_np,
                mask_stats=mask_stats,
            )
            results.append(det)

        if _is_raw and self._model is not None:
            for obj in objects:
                if hasattr(obj, "seg_mask") and obj.seg_mask is not None:
                    try:
                        self._model.draw_seg_mask(img, obj.x, obj.y, obj.seg_mask, threshold=127)
                    except Exception:
                        pass

        self._clear_error()
        return results

    def segment(
        self, frame: np.ndarray, **kwargs
    ) -> list[SegmentResult]:
        """Run segmentation inference and return structured mask data.

        Derives the results from detect(); a ContractError raised there
        propagates unchanged.
        """
        detections = self.detect(frame, **kwargs)
        results: list[SegmentResult] = []
        for d in detections:
            if d.mask_stats is not None and d.mask_stats.area_px > 0:
                results.append(SegmentResult(
                    class_id=d.class_id,
                    center_x=d.mask_stats.center_x,
                    center_y=d.mask_stats.center_y,
                    area_px=d.mask_stats.area_px,
                    bbox_x=d.x,
                    bbox_y=d.y,
                    bbox_w=d.w,
                    bbox_h=d.h,
                    score=d.score,
                    detection=d,
                ))
        return results

    def classify(self, frame: np.ndarray, **kwargs) -> list[tuple[int, float]]:
        """Run classification on frame.

        Returns list of (class_id, score) tuples.  Runtime failures return an
        empty list and are recorded in `last_error_code`.
        """
        top_k = kwargs.get("top_k", 1)

        if self._model is None:
            self._report_error(
                E_MODEL_NOT_LOADED, "classify() called but no model is loaded")
            return []

        if not hasattr(self._model, "classify"):
            self._fail_contract(
                f"Model class {type(self._model).__name__} provides no classify(); "
                f"family '{self._model_family}' does not match the output layout"
            )

        try:
            img = maix.image.cv2image(frame, bgr=True, copy=True)
        except Exception as e:
            self._report_error(E_INVALID_INPUT, f"cv2image conversion failed: {e}")
            return []

        try:
            results = self._model.classify(img)
        except Exception as e:
            self._report_error(E_INFER_FAILED, f"Model classify() failed: {e}")
            return []

        self._clear_error()
        out: list[tuple[int, float]] = []
        for item in results[:top_k]:
            if isinstance(item, (list, tuple)):
                out.append((int(item[0]), float(item[1])))
            else:
                out.append((int(item), 0.0))
        return out

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        """Retrieve a segmentation mask by index.

        Not implemented: the mask travels inside each Detection as
        `seg_mask`, so callers do not need this accessor.
        """
        return None

    # ------------------------------------------------------------------ #
    #  Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _report_error(self, code: str, message: str) -> None:
        """Record a recoverable failure so it stays visible to the caller."""
        self._last_error_code = code
        self._error_count += 1
        self._degraded = True
        logger.error("[%s] %s", code, message)

    def _clear_error(self) -> None:
        """Mark the backend healthy again after a successful inference."""
        self._last_error_code = ""
        self._degraded = False

    def _fail_contract(self, message: str) -> None:
        """Record a contract violation and raise it.

        The violation is never downgraded to an empty result: an empty list
        is indistinguishable from "the scene contains no target".
        """
        self._last_error_code = E_OUTPUT_LAYOUT_MISMATCH
        self._error_count += 1
        self._degraded = True
        logger.error("[%s] %s", E_OUTPUT_LAYOUT_MISMATCH, message)
        raise ContractError(message, E_OUTPUT_LAYOUT_MISMATCH)

    @staticmethod
    def _resolve_class(model_type: Union[str, type], model_path: str = ""):
        """Resolve *model_type* to a runtime class and a model family name.

        Returns `(slot_cls, family)`.  Raises ModelConfigError when the
        type cannot be resolved: falling back to a generic net class would
        run the wrong post-processing on the output and silently yield
        wrong boxes.
        """
        if isinstance(model_type, type):
            family = ""
            name = model_type.__name__.lower().replace("_", "")
            for candidate in _KEYPOINT_LAYOUTS:
                if candidate.replace("_", "") in name:
                    family = candidate
                    break
            return model_type, family

        if model_type == "auto":
            family = MaixCam2AI._read_mud_model_type(model_path)
            if not family:
                logger.warning(
                    "Cannot read the model family from '%s'; assuming 'yolo11'. "
                    "Set model_type explicitly to make the contract explicit.",
                    model_path,
                )
                family = "yolo11"
        else:
            family = model_type

        if family in _MUD_MODEL_TYPES:
            slot_cls = _MUD_MODEL_TYPES[family]()
            if slot_cls is None:
                raise ModelConfigError(
                    f"Firmware provides no model class for family '{family}'",
                    E_MODEL_TYPE_UNSUPPORTED,
                )
            return slot_cls, family

        if family in _SLOT_CLASSES:
            return _SLOT_CLASSES[family], family

        raise ModelConfigError(
            f"Unsupported model type '{model_type}'", E_MODEL_TYPE_UNSUPPORTED)

    @staticmethod
    def _read_mud_model_type(model_path: str) -> str:
        """Read the model family from a MUD ini file ([extra] model_type).

        Returns "" when the file is missing or cannot be parsed.
        """
        if not model_path:
            return ""
        try:
            import configparser
            parser = configparser.ConfigParser(interpolation=None)
            if not parser.read(model_path):
                return ""
            return parser.get("extra", "model_type", fallback="").strip()
        except Exception as e:
            logger.debug("Cannot parse MUD '%s': %s", model_path, e)
            return ""

    @staticmethod
    def _convert_keypoints(obj, family: str = "") -> list[Keypoint]:
        """Convert a MaixPy flat point list into Keypoint records.

        The packing comes from the declared model family, never from the
        length of the list: a wrong stride silently shifts every landmark.
        A payload that fits exactly one stride uses that stride; a payload
        that fits several is resolved by the declared family.
        """
        pts = getattr(obj, "points", None)
        if not pts:
            return []

        layout = _KEYPOINT_LAYOUTS.get(family)
        if layout is None:
            raise ContractError(
                f"No keypoint layout declared for model family '{family}'",
                E_OUTPUT_LAYOUT_MISMATCH,
            )
        offset, declared_step = layout

        payload = len(pts) - offset
        if payload <= 0:
            return []

        candidates = [s for s in (2, 3) if payload % s == 0]
        if not candidates:
            raise ContractError(
                f"Keypoint payload of {len(pts)} values does not match any "
                f"declared layout (offset={offset})",
                E_OUTPUT_LAYOUT_MISMATCH,
            )
        step = declared_step if declared_step in candidates else candidates[0]
        if step != declared_step:
            logger.warning(
                "Keypoint payload of %d values does not fit the declared "
                "stride %d of model family '%s'; using stride %d",
                len(pts), declared_step, family, step,
            )

        kps: list[Keypoint] = []
        for i in range(offset, len(pts), step):
            score = float(pts[i + 2]) if step == 3 else 0.0
            kps.append(
                Keypoint(
                    x=float(pts[i]), y=float(pts[i + 1]),
                    score=score, id=(i - offset) // step,
                )
            )
        return kps

    # ------------------------------------------------------------------ #
    #  Mask analysis
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_seg_mask(mask_img) -> Optional[np.ndarray]:
        try:
            mask_np = maix.image.image2cv(mask_img, ensure_bgr=False, copy=True)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            return mask_np
        except Exception as e:
            logger.warning("Failed to convert seg_mask: %s", e)
            return None
