from __future__ import annotations

import logging
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
    E_OUTPUT_LAYOUT_MISMATCH,
    Keypoint,
    MaskStats,
    ModelConfigError,
    SegmentResult,
)

logger = logging.getLogger(__name__)

# Model classes available on MaixCAM (cv181x); the current deployment uses
# YOLOv5 with a CVITEK .cvimodel. Missing types stay None so that
# _resolve_class and switch fail explicitly. Never substitute a different
# class: decoding YOLOv8/YOLO11 output with YOLOv5 post-processing would yield
# wrong coordinates without raising, which is worse than a visible load error.
_SLOT_CLASSES = {
    "yolov5": maix.nn.YOLOv5,
    "yolov8": getattr(maix.nn, "YOLOv8", None),
    "yolo11": getattr(maix.nn, "YOLO11", None),
    "classifier": getattr(maix.nn, "Classifier", None),
    "nn": getattr(maix.nn, "NN", None),
}
# On CAM1 "auto" resolves to yolov5, currently the only real model type.
_DEFAULT_TYPE = "yolov5"


class MaixCam1AI:
    """NPU inference backend for MaixCAM and MaixCAM-Pro.

    Differences from MaixCam2AI:

    1. Model format: CAM1 accepts CVITEK ``.cvimodel`` files (mud key
       ``model``), not Axera ``.axmodel`` (``model_npu``). The current
       deployment uses ``model_239147.cvimodel`` (YOLOv5, three classes:
       yao_gu, yuan_zhu, yuan_tai).
    2. Model class: ``nn.YOLOv5`` rather than ``nn.YOLO11``. Older firmware may
       lack YOLOv8/YOLO11; missing types fail explicitly instead of falling
       back (see the _SLOT_CLASSES comment).
    3. Constructor: YOLOv5 takes ``model=<mud path>``. ``dual_buff`` is
       supported and defaults to True on CAM1, but it makes ``detect()``
       return the previous call's image, adding a one-frame delay (~33 ms at
       30 fps) to tasks ordered by first appearance. The code disables it in
       switch().
    4. Performance: CAM1 has far less NPU compute and memory than CAM2, so the
       existing low-resolution input configuration is intentional.
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict] = {}   # nick_name -> {path, type, kwargs}
        self._active_name: str = ""
        self._model = None
        self._model_type: str = ""
        self._model_path: str = ""

        self._last_error_code: str = ""
        self._error_count: int = 0
        self._degraded: bool = False

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
    def last_error_code(self) -> str:
        return self._last_error_code

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def degraded(self) -> bool:
        return self._degraded

    def add(self, nick_name: str, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        if nick_name in self._registry:
            logger.warning("Model '%s' already registered, overwriting", nick_name)
        self._registry[nick_name] = {
            "path": model_path,
            "type": model_type,
            "kwargs": kwargs,
        }
        # Use warning rather than info for key state: app/main.py pins the
        # root logger at WARNING via basicConfig, so info records never reach
        # the device console and registration/switch status cannot be grepped.
        logger.warning("[CAM1] Registered model '%s' -> %s (type=%s)",
                       nick_name, model_path, model_type)
        return True

    def remove(self, nick_name: str) -> None:
        if nick_name not in self._registry:
            logger.warning("Attempt to remove unknown model '%s'", nick_name)
            return
        if self._active_name == nick_name:
            self.unload()
        del self._registry[nick_name]
        logger.info("Removed model '%s'", nick_name)

    def switch(self, nick_name: str) -> bool:
        """Load and activate a registered model.

        Each switch reloads the .cvimodel from flash into the NPU (~200-500 ms,
        similar to CAM2). Only one model is deployed now, so this is fine; if a
        second model is added, note that the first switch during data
        subscription blocks one streaming tick.
        """
        if nick_name not in self._registry:
            logger.error("Cannot switch to unknown model '%s'", nick_name)
            return False
        if self._active_name == nick_name and self._model is not None:
            return True  # already active

        self.unload()
        info = self._registry[nick_name]
        path = info["path"]
        model_type = info["type"]
        kwargs = dict(info["kwargs"])

        slot_cls = self._resolve_class(model_type)
        if slot_cls is None:
            logger.error("Unsupported model type '%s' for '%s'", model_type, nick_name)
            return False

        # CAM1: YOLOv5(model=<mud>, dual_buff=False).
        # Set dual_buff explicitly: CAM1 supports it and defaults to True,
        # which makes detect() return the previous call's image and adds a
        # one-frame delay (~33 ms at 30 fps) to first-appearance ordering and
        # pixel-crop decisions. Disable it for real-time behavior.
        # On firmware that rejects the kwarg, inspect the TypeError message to
        # distinguish an incompatible signature from a broken model, avoiding
        # a pointless NPU reload that would hide the real error.
        kwargs.setdefault("dual_buff", False)
        try:
            self._model = slot_cls(model=path, **kwargs)
        except TypeError as e:
            msg = str(e)
            if "dual_buff" in msg:
                # Older firmware has no dual_buff: drop it and retry once.
                kwargs.pop("dual_buff", None)
                try:
                    self._model = slot_cls(model=path, **kwargs)
                except Exception as e2:
                    self._report_error(
                        E_MODEL_LOAD_FAILED,
                        f"Failed to load model '{nick_name}' from {path}: {e2}")
                    self._model = None
                    return False
            else:
                self._report_error(
                    E_MODEL_LOAD_FAILED,
                    f"Failed to load model '{nick_name}' from {path}: {e}")
                self._model = None
                return False
        except Exception as e:
            self._report_error(
                E_MODEL_LOAD_FAILED,
                f"Failed to load model '{nick_name}' from {path}: {e}")
            self._model = None
            return False

        self._model_path = path
        self._model_type = model_type
        self._active_name = nick_name
        # As above: warning level is needed to pass the root logger's threshold.
        logger.warning("[CAM1] Switched to model '%s' (%s) labels=%s",
                       nick_name, path, self.labels)
        return True

    # ------------------------------------------------------------------ #
    #  Convenience aliases
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        return self.add("default", model_path, model_type=model_type, **kwargs) and self.switch("default")

    def unload(self) -> None:
        if self._active_name:
            logger.info("Unloading model '%s'", self._active_name)
        self._model = None
        self._model_type = ""
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

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def detect(self, frame, **kwargs) -> list[Detection]:
        """Run object detection on a frame.

        *frame* may be a BGR numpy array (the normal pipeline path) or a
        ``maix.image.Image`` for the zero-copy path (pass ``_raw=True``).
        Current tasks use the numpy path.

        Return an empty list when no model is loaded or inference fails, so
        the caller can keep emitting zero-valued heartbeats. Raise on contract
        errors (post-processing not matching the output layout); see
        E_OUTPUT_LAYOUT_MISMATCH.
        """
        conf_th = kwargs.pop("conf_th", 0.48)
        iou_th = kwargs.pop("iou_th", 0.5)
        _is_raw = kwargs.pop("_raw", False)

        if self._model is None:
            self._report_error(
                E_MODEL_NOT_LOADED, "detect() called but no model is loaded")
            return []

        if not hasattr(self._model, "detect"):
            # The runtime class cannot decode this model's output: refuse
            # instead of returning an empty list that reads as "no target".
            self._fail_contract(
                f"Model class {type(self._model).__name__} provides no detect(); "
                f"type '{self._model_type}' does not match the output layout"
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
            objects = self._model.detect(img, conf_th=conf_th, iou_th=iou_th, **kwargs)
        except Exception as e:
            # Do not retry without iou_th: CAM1's official nn_yolov5.py and
            # yolov5.md both pass it, so a TypeError most likely originates
            # inside detect(). Swallowing it and retrying would hide the real
            # error and silently change detection results.
            self._report_error(E_INFER_FAILED, f"Model detect() failed: {e}")
            return []

        # Apply ``or []`` outside the try: some firmware returns None instead
        # of an empty list when there is no target, and ``for o in None`` would
        # raise TypeError out of this function into the caller (which has no
        # try wrapper), breaking frame processing and violating the promise to
        # return an empty list for the caller to handle.
        objects = [o for o in (objects or []) if o.score >= conf_th]
        if not objects:
            logger.debug("detect() returned 0 objects")
        else:
            logger.debug("detect() returned %d objects", len(objects))

        results: list[Detection] = []
        for obj in objects:
            kps = self._convert_keypoints(obj, self._model_type)
            angle: Optional[float]
            if hasattr(obj, "angle") and obj.angle != -9999:
                angle = float(obj.angle)
            else:
                angle = None

            # CAM1 YOLOv5 has no segmentation model; keep the seg_mask branch
            # as a placeholder to align with the interface.
            seg_mask_np: Optional[np.ndarray] = None
            mask_stats: Optional[MaskStats] = None
            if hasattr(obj, "seg_mask") and obj.seg_mask is not None:
                seg_mask_np = self._extract_seg_mask(obj.seg_mask)
                if seg_mask_np is not None:
                    ys, xs = np.nonzero(seg_mask_np > 127)
                    if xs.size:
                        mask_stats = MaskStats(
                            center_x=float(obj.x + xs.mean()),
                            center_y=float(obj.y + ys.mean()),
                            area_px=int(xs.size),
                        )

            results.append(Detection(
                x=obj.x, y=obj.y, w=obj.w, h=obj.h,
                class_id=obj.class_id, score=obj.score, label="",
                angle=angle, keypoints=kps, mask_index=-1,
                seg_mask=seg_mask_np, mask_stats=mask_stats,
            ))

        return results

    def segment(self, frame: np.ndarray, **kwargs) -> list[SegmentResult]:
        """Return segmentation results derived from detection masks.

        CAM1 has no dedicated segmentation model; results are built from the
        mask statistics that ``detect()`` attaches to detections.
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
                    bbox_x=d.x, bbox_y=d.y, bbox_w=d.w, bbox_h=d.h,
                    score=d.score, detection=d,
                ))
        return results

    def classify(self, frame: np.ndarray, **kwargs) -> list[tuple[int, float]]:
        if self._model is None:
            self._report_error(
                E_MODEL_NOT_LOADED, "classify() called but no model is loaded")
            return []
        if not hasattr(self._model, "classify"):
            self._fail_contract(
                f"Model class {type(self._model).__name__} provides no classify(); "
                f"type '{self._model_type}' does not match the output layout"
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
        top_k = kwargs.get("top_k", 1)
        out: list[tuple[int, float]] = []
        for item in list(results)[:top_k]:
            if isinstance(item, (list, tuple)):
                out.append((int(item[0]), float(item[1])))
            else:
                out.append((int(item), 0.0))
        return out

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
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
    def _resolve_class(model_type):
        """Return the MaixPy NN class for CAM1; None marks an unsupported type.

        "auto" resolves to yolov5 on CAM1, currently the only real model type.
        """
        if model_type == "auto":
            return _SLOT_CLASSES[_DEFAULT_TYPE]
        if model_type in _SLOT_CLASSES:
            return _SLOT_CLASSES[model_type]
        if isinstance(model_type, type):
            return model_type
        # Unknown type: fail explicitly.  Falling back to yolov5 would run
        # the wrong post-processing on the output and silently yield wrong
        # boxes, which is worse than a visible failure.
        raise ModelConfigError(
            f"Unsupported model type '{model_type}'", E_MODEL_TYPE_UNSUPPORTED)

    @staticmethod
    def _convert_keypoints(obj, model_type: str = "") -> list[Keypoint]:
        kps: list[Keypoint] = []
        if not hasattr(obj, "points"):
            return kps
        pts = obj.points
        if not pts:
            return kps
        offset = 0
        step = 2
        if model_type == "hand_landmarks":
            offset = 8
            step = 3
        elif len(pts) % 3 == 0:
            step = 3
        for i in range(offset, len(pts), step):
            score = float(pts[i + 2]) if step == 3 else 0.0
            kps.append(Keypoint(
                x=float(pts[i]), y=float(pts[i + 1]),
                score=score, id=(i - offset) // step,
            ))
        return kps

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
