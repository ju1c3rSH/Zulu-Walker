# -*- coding: utf-8 -*-
"""
EOD2026 color blob detector (LAB threshold based).

Pure cv2/numpy — runs on device AND on PC (pytest). Stateless per call:
all latching/debouncing lives in the processor layer (single writer rule).

Error convention (must match panzer docs/design.md §2.5):
  error_x > 0 : target is RIGHT of image center
  error_y > 0 : target is BELOW image center
  normalized to ±5000 at frame half-width/half-height, 0 = centered.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class EodBlobResult:
    """Per-frame detection result for one subscription mode."""
    target_found: bool = False
    error_x: int = 0
    error_y: int = 0
    dev_id: int = 0            # color_id / ring_id / shape_id source value
    bbox: Optional[Tuple[int, int, int, int]] = None   # x, y, w, h
    center: Optional[Tuple[float, float]] = None       # cx, cy
    area_px: int = 0
    ready: bool = False        # center inside the configured zone
    in_zone_pos: Optional[Tuple[float, float]] = None  # anchor point used for zone check
    mask: Optional[np.ndarray] = None                  # binary mask of winning blob
    contour: Optional[np.ndarray] = None               # winning contour (for shape classification)
    all_candidates: List[dict] = field(default_factory=list)  # per-color summaries


def normalize_error(pos: float, center: float, half: float, scale: int = 5000) -> int:
    """Normalize a pixel offset to ±scale across the half-frame."""
    if half <= 0:
        return 0
    return int(max(-scale, min(scale, (pos - center) / half * scale)))


class EodColorDetector:
    """LAB-threshold blob detector with expected-color priority.

    candidates: list of dicts:
        {"id": 1, "name": "red", "lab": [l_min, l_max, a_min, a_max, b_min, b_max]}
    expected_id: task-code id from CMD_REQUEST reserved byte (0 = no preference).
    """

    def __init__(
        self,
        candidates: List[dict],
        min_area_px: int = 400,
        max_area_px: int = 300000,
        edge_margin_px: int = 4,
        grab_zone: Tuple[float, float, float, float] = (0.35, 0.45, 0.65, 0.85),
        morph_open: int = 3,
        morph_close: int = 7,
    ) -> None:
        self._candidates = candidates
        self._min_area = min_area_px
        self._max_area = max_area_px
        self._edge_margin = edge_margin_px
        self._grab_zone = grab_zone   # normalized (x0, y0, x1, y1), anchor point inside => ready
        self._kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_open, morph_open))
        self._kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (morph_close, morph_close))
        self._last_lab: Optional[np.ndarray] = None
        self._last_shape: Tuple[int, int] = (0, 0)

    # ---- internals ----

    def _lab_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self._last_lab is None or self._last_shape != (w, h):
            self._last_shape = (w, h)
        self._last_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        return self._last_lab

    def _blobs_for_color(self, lab: np.ndarray, thresholds: List[int]):
        """inRange with OpenMV-style LAB thresholds converted to OpenCV encoding.

        Configured thresholds follow the MaixPy/OpenMV convention (L 0..100,
        A/B signed -128..127); cv2 LAB is uint8 with L 0..255 and A/B offset
        by +128. Convert before inRange so threshold values port 1:1 from the
        MaixPy threshold editor / docs.
        """
        lo = np.array([
            max(0, min(255, round(thresholds[0] * 2.55))),
            max(0, min(255, thresholds[2] + 128)),
            max(0, min(255, thresholds[4] + 128)),
        ], dtype=np.uint8)
        hi = np.array([
            max(0, min(255, round(thresholds[1] * 2.55))),
            max(0, min(255, thresholds[3] + 128)),
            max(0, min(255, thresholds[5] + 128)),
        ], dtype=np.uint8)
        mask = cv2.inRange(lab, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours, mask

    def _rect_ok(self, x: int, y: int, w: int, h: int, frame_w: int, frame_h: int) -> bool:
        """Edge-margin + area window check for a bbox (shared by both paths)."""
        if w * h < self._min_area or w * h > self._max_area:
            return False
        m = self._edge_margin
        if x <= m or y <= m or x + w >= frame_w - m or y + h >= frame_h - m:
            return False
        return True

    def _contour_ok(self, contour, w: int, h: int) -> bool:
        area = cv2.contourArea(contour)
        if area < self._min_area or area > self._max_area:
            return False
        x, y, bw, bh = cv2.boundingRect(contour)
        return self._rect_ok(x, y, bw, bh, w, h)

    # ---- MaixPy fast path (device) ----

    def _detect_maix(self, frame_bgr, img, expected_id, zone_anchor,
                     cands, need_contour) -> Optional[EodBlobResult]:
        """find_blobs on the raw maix Image (MaixCV C++) — no numpy in the
        selection loop. Returns None to signal 'fall back to cv2'."""
        from .maix_accel import find_blobs_maix

        try:
            w, h = img.width(), img.height()
        except Exception:
            return None

        result = EodBlobResult()
        best = None   # (key, cand, blob_dict)
        for cand in cands:
            blobs = find_blobs_maix(
                img, cand["lab"],
                area_threshold=self._min_area,
                pixels_threshold=max(1, self._min_area // 4))
            if blobs is None:
                return None   # maix failed -> cv2 fallback
            for b in blobs:
                if not self._rect_ok(b["x"], b["y"], b["w"], b["h"], w, h):
                    continue
                result.all_candidates.append({
                    "id": cand.get("id", 0),
                    "name": cand.get("name", "?"),
                    "area_px": int(b["pixels"]),
                })
                is_expected = (expected_id != 0 and cand.get("id") == expected_id)
                key = (1, b["pixels"]) if is_expected else (0, b["pixels"])
                if best is None or key > best[0]:
                    best = (key, cand, b)

        if best is None:
            return result   # authoritative "no target this frame"

        _, cand, b = best
        x, y, bw, bh = b["x"], b["y"], b["w"], b["h"]
        cx, cy = b["cx"], b["cy"]

        result.target_found = True
        result.dev_id = cand.get("id", 0)
        result.bbox = (x, y, bw, bh)
        result.center = (cx, cy)
        result.area_px = int(b["pixels"])
        result.error_x = normalize_error(cx, w / 2.0, w / 2.0)
        result.error_y = normalize_error(cy, h / 2.0, h / 2.0)

        if need_contour and frame_bgr is not None:
            result.contour = self._contour_in_crop(frame_bgr, cand["lab"],
                                                   result.bbox)

        zx0, zy0, zx1, zy1 = self._grab_zone
        ax, ay = (cx, y + bh) if zone_anchor == "bottom" else (cx, cy)
        result.in_zone_pos = (ax, ay)
        result.ready = (zx0 * w <= ax <= zx1 * w) and (zy0 * h <= ay <= zy1 * h)
        return result

    def _contour_in_crop(self, frame_bgr, thresholds, bbox) -> Optional[np.ndarray]:
        """Refine a fast-path bbox into a contour via a small ROI-crop cv2 pass
        (used for hostage shape classification; full-frame cost avoided)."""
        fh, fw = frame_bgr.shape[:2]
        x, y, bw, bh = bbox
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(fw, x + bw + pad), min(fh, y + bh + pad)
        crop = frame_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        contours, _ = self._blobs_for_color(lab, thresholds)
        if not contours:
            return None
        target_area = float(bw * bh)
        best = min(contours,
                   key=lambda c: abs(cv2.contourArea(c) - target_area))
        shifted = best.astype(np.int32) + np.array([x0, y0], dtype=np.int32)
        return shifted.reshape(-1, 1, 2)

    # ---- public API ----

    def detect(
        self,
        frame_bgr: np.ndarray,
        expected_id: int = 0,
        zone_anchor: str = "bottom",   # "bottom" (bomb in gripper reach) | "center"
        candidates: Optional[List[dict]] = None,
        maix_img=None,
        need_contour: bool = False,
    ) -> EodBlobResult:
        """Detect the best blob among candidate colors.

        Priority: expected_id color (if configured and found) > largest area.
        candidates overrides the configured candidate set for this call.
        maix_img: raw maix Image (device fast path via C++ find_blobs);
        need_contour: refine the winning blob into a contour (hostage shape
        classification) — on the fast path this runs a cheap ROI-crop cv2
        pass instead of a full-frame one.
        Falls back to the full cv2 path whenever the fast path is
        unavailable or fails.
        """
        cands = candidates if candidates is not None else self._candidates
        if maix_img is not None:
            result = self._detect_maix(frame_bgr, maix_img, expected_id,
                                       zone_anchor, cands, need_contour)
            if result is not None:
                return result
            # maix path unavailable/failed -> cv2 fallback below
        result = EodBlobResult()
        if frame_bgr is None:
            return result
        h, w = frame_bgr.shape[:2]
        lab = self._lab_frame(frame_bgr)

        best = None   # (score_key, cand, contour)
        for cand in cands:
            contours, mask = self._blobs_for_color(lab, cand["lab"])
            for contour in contours:
                if not self._contour_ok(contour, w, h):
                    continue
                area = cv2.contourArea(contour)
                is_expected = (expected_id != 0 and cand.get("id") == expected_id)
                result.all_candidates.append({
                    "id": cand.get("id", 0),
                    "name": cand.get("name", "?"),
                    "area_px": int(area),
                })
                # expected color wins outright; otherwise rank by area
                key = (1, area) if is_expected else (0, area)
                if best is None or key > best[0]:
                    best = (key, cand, contour, mask)

        if best is None:
            return result

        _, cand, contour, mask = best
        x, y, bw, bh = cv2.boundingRect(contour)
        cx = x + bw / 2.0
        cy = y + bh / 2.0

        result.target_found = True
        result.dev_id = cand.get("id", 0)
        result.bbox = (x, y, bw, bh)
        result.center = (cx, cy)
        result.area_px = int(cv2.contourArea(contour))
        result.error_x = normalize_error(cx, w / 2.0, w / 2.0)
        result.error_y = normalize_error(cy, h / 2.0, h / 2.0)
        result.contour = contour
        result.mask = (mask == 255) & self._contour_mask(contour, w, h) \
            if mask is not None else None

        # zone check (READY_TO_PICK / READY_TO_PLACE anchor)
        zx0, zy0, zx1, zy1 = self._grab_zone
        ax, ay = (cx, y + bh) if zone_anchor == "bottom" else (cx, cy)
        result.in_zone_pos = (ax, ay)
        result.ready = (zx0 * w <= ax <= zx1 * w) and (zy0 * h <= ay <= zy1 * h)
        return result

    @staticmethod
    def _contour_mask(contour, w: int, h: int) -> np.ndarray:
        m = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(m, [contour], -1, 255, thickness=cv2.FILLED)
        return m.astype(bool)
