from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

import maix.image
import maix.nn

from framework.hal.interface.ai import Detection, Keypoint, MaskStats, SegmentResult

logger = logging.getLogger(__name__)

_SLOT_CLASSES = {
    "yolo": maix.nn.YOLO11,
    "classifier": maix.nn.Classifier,
    "hand_landmarks": maix.nn.HandLandmarks,
    "nn": maix.nn.NN,
}


class MaixCam2AI:
    def __init__(self) -> None:
        self._registry: dict[str, dict] = {}  # nick_name -> {path, type, kwargs}
        self._active_name: str = ""
        self._model: Optional[Union[maix.nn.YOLO11, maix.nn.Classifier, maix.nn.HandLandmarks, maix.nn.NN]] = None
        self._model_type: str = ""
        self._model_path: str = ""

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

    # TODO: 预加载所有模型到 NPU，switch() 改为纯指针交换 (O(1), ~1µs)
    # 当前每次 switch() 重新从 flash 加载 .mud 到 NPU (~200-500ms)
    # 双模型 CMM 约 30-40MB，256MB CMM 完全可接受
    def switch(self, nick_name: str) -> bool:
        if nick_name not in self._registry:
            logger.error("Cannot switch to unknown model '%s'", nick_name)
            return False
        if self._active_name == nick_name and self._model is not None:
            return True  # already active

        self.unload()
        info = self._registry[nick_name]
        path = info["path"]
        model_type = info["type"]
        kwargs = info["kwargs"]

        slot_cls = self._resolve_class(model_type)
        if slot_cls is None:
            logger.error("Unsupported model type '%s' for '%s'", model_type, nick_name)
            return False

        try:
            self._model = slot_cls(path, dual_buff=True, **kwargs)
            self._model_path = path
            self._model_type = model_type
            self._active_name = nick_name
            logger.info("Switched to model '%s' (%s)", nick_name, path)
            return True
        except Exception as e:
            logger.error("Failed to load model '%s' from %s: %s", nick_name, path, e)
            self._model = None
            self._model_path = ""
            self._active_name = ""
            return False

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

    def detect(
        self, frame,
        **kwargs
    ) -> list[Detection]:
        """Run object detection on frame.

        *frame* can be:
        - a BGR numpy array (standard pipeline path)
        - a ``maix.image.Image`` (zero-copy path, pass ``_raw=True``)

        Returns a list of Detection dataclasses.  Returns an empty list
        when no model is loaded or inference fails.
        """
        conf_th = kwargs.pop("conf_th", 0.5)
        iou_th = kwargs.pop("iou_th", 0.45)

        _is_raw = kwargs.pop("_raw", False)

        if self._model is None:
            logger.warning("detect() called but no model is loaded")
            return []

        if not hasattr(self._model, "detect"):
            logger.warning(
                "detect() requires a model with detect() method, "
                "current model type=%s (nick_name=%s)",
                type(self._model).__name__, self._active_name or "N/A",
            )
            return []

        try:
            if _is_raw:
                img = frame
            else:
                frame_rgb = frame[:, :, ::-1]
                img = maix.image.cv2image(frame_rgb, bgr=False, copy=True)
        except Exception as e:
            logger.error("cv2image conversion failed: %s", e)
            return []

        try:
            objects = self._model.detect(img, conf_th=conf_th, iou_th=iou_th, **kwargs)
        except Exception as e:
            logger.error("Model detect() failed: %s", e)
            return []

        # Post-filter: some models don't fully honor conf_th internally
        objects = [o for o in objects if o.score >= conf_th]

        if not objects:
            logger.debug("detect() returned 0 objects")
        else:
            logger.debug("detect() returned %d objects", len(objects))

        results: list = []
        for obj in objects:
            kps = self._convert_keypoints(obj, self._model_type)
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

        return results

    def segment(
        self, frame: np.ndarray, **kwargs
    ) -> list[SegmentResult]:
        """Run segmentation inference and return structured mask data.

        Calls ``detect()`` internally, then extracts ``mask_stats``
        from each returned ``Detection``.  Returns an empty list when
        no model is loaded or the model does not produce segmentation
        masks.
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

        Returns list of (class_id, score) tuples.  Returns empty list
        when no model is loaded or inference fails.
        """
        top_k = kwargs.get("top_k", 1)

        if self._model is None:
            logger.warning("classify() called but no model is loaded")
            return []

        if not hasattr(self._model, "classify"):
            logger.warning(
                "classify() requires a model with classify() method, "
                "current model type=%s (nick_name=%s)",
                type(self._model).__name__, self._active_name or "N/A",
            )
            return []

        try:
            img = maix.image.cv2image(frame, bgr=True, copy=True)
        except Exception as e:
            logger.error("cv2image conversion failed: %s", e)
            return []

        try:
            results = self._model.classify(img)
        except Exception as e:
            logger.error("Model classify() failed: %s", e)
            return []

        out: list[tuple[int, float]] = []
        for item in results[:top_k]:
            if isinstance(item, (list, tuple)):
                out.append((int(item[0]), float(item[1])))
            else:
                out.append((int(item), 0.0))
        return out

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        """Retrieve segmentation mask for the last detection.

        Currently a placeholder — returns None.
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

    @staticmethod
    def _resolve_class(model_type: Union[str, type]):
        """Return the MaixPy NN class for *model_type*.

        Supports ``"auto"`` (defaults to ``maix.nn.YOLO11``), explicit
        type strings, or direct class references.
        Note: "auto" does NOT use file-extension heuristic — it
        unconditionally returns ``maix.nn.YOLO11``.
        """
        if model_type == "auto":
            return maix.nn.YOLO11  # most common default
        if model_type in _SLOT_CLASSES:
            return _SLOT_CLASSES[model_type]
        # Allow passing a class directly
        if isinstance(model_type, type):
            return model_type
        logger.warning("Unknown model_type '%s', falling back to maix.nn.NN", model_type)
        return maix.nn.NN

    @staticmethod
    def _convert_keypoints(obj, model_type: str = "") -> list[Keypoint]:
        """Convert a MaixPy flat *points* list to `list[Keypoint]`."""
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
