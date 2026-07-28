from __future__ import annotations

import cv2
import numpy as np

from framework.hal.interface import Detection

from .base import AbstractModelHandler
from .registry import ModelHandlerRegistry


@ModelHandlerRegistry.register("yolo_seg")
class SegmentationHandler(AbstractModelHandler):
    def draw(
        self, frame: np.ndarray, detections: list[Detection]
    ) -> np.ndarray:
        labels = self._ai.labels if hasattr(self._ai, "labels") else []

        mask_overlay = np.zeros_like(frame, dtype=np.uint8)
        mask_alpha = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

        for det in detections:
            x1, y1 = det.x, det.y
            x2, y2 = det.x + det.w, det.y + det.h

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = (
                labels[det.class_id]
                if det.class_id < len(labels)
                else str(det.class_id)
            )
            cv2.putText(
                frame, f"{label_text}:{det.score:.2f}", (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
            )

            if det.mask_stats is not None and det.mask_stats.area_px > 0:
                area_text = f"area:{det.mask_stats.area_px}px"
                cv2.putText(
                    frame, area_text, (x1, y2 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1,
                )

                cx = int(det.mask_stats.center_x)
                cy = int(det.mask_stats.center_y)
                cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                cv2.putText(
                    frame, "center", (cx + 5, cy - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1,
                )

            if det.seg_mask is not None:
                mask_h, mask_w = det.seg_mask.shape[:2]
                if mask_h > 0 and mask_w > 0:
                    x1_clip = max(0, x1)
                    y1_clip = max(0, y1)
                    x2_clip = min(x1 + mask_w, frame.shape[1])
                    y2_clip = min(y1 + mask_h, frame.shape[0])
                    crop_w = x2_clip - x1_clip
                    crop_h = y2_clip - y1_clip
                    if crop_w > 0 and crop_h > 0:
                        mask_offset_x = x1_clip - x1
                        mask_offset_y = y1_clip - y1
                        mask_crop = det.seg_mask[
                            mask_offset_y:mask_offset_y + crop_h,
                            mask_offset_x:mask_offset_x + crop_w,
                        ]
                        mask_bin = mask_crop > 127
                        row_region = slice(y1_clip, y1_clip + crop_h)
                        col_region = slice(x1_clip, x1_clip + crop_w)
                        mask_alpha[row_region, col_region] += mask_bin.astype(np.float32) * 0.35
                        mask_overlay[row_region, col_region][mask_bin] = (0, 255, 128)

        if mask_alpha.any():
            alpha_clip = np.clip(mask_alpha, 0.0, 1.0)[..., None]
            frame[:] = (frame * (1.0 - alpha_clip) + mask_overlay * alpha_clip).astype(np.uint8)

        return frame
