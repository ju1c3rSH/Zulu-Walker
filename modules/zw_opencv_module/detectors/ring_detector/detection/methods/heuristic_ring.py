from typing import Optional
import cv2
import numpy as np

from .base import BaseRingDetectionMethod, TARGET_W
from ..target_creation import create_ring_target
from modules.zw_opencv_module.models.color import Color
from modules.zw_opencv_module.models.ring import RingTarget
from modules.zw_opencv_module.detectors._shared.kalman_utils import kalman_update
from utils.log_util import log_print



class HeuristicRingMethod(BaseRingDetectionMethod):
    """Concentric ring detection via HoughCircles on color mask + concentric validation.

    Pipeline:
        1. Build color mask from HSV ranges (dashed ring → scattered color dots)
        2. Morph close to bridge dash gaps into arcs
        3. HoughCircles on morphed mask → candidate centers + radii
        4. Group by center proximity → identify concentric groups
        5. Ring-band color density verification per group
        6. Multi-ring confirmation → high confidence when ≥2 rings verified

    Performance tiers:
        - Discovery (no tracking): 320×240 + Hough dp=1.5                       ~3-5 ms
        - ROI tracking (last_center known): 150×150 cropped Hough                ~1-2 ms
        - Kalman prediction (lost): pure math, no image ops                      ~0.01 ms
    """

    MORPH_KERNEL = 5
    MORPH_CLOSE_ITER = 2
    _LOG_INTERVAL = 60

    # --- concentric ring parameters ---------------------------------
    RING_GAP_PX_DEFAULT = 10       # expected pixel gap between concentric rings (2cm → px)
    CONCENTRIC_MIN_RINGS = 2       # minimum concentric rings required to confirm
    RING_BAND_COLOR_THRESHOLD = 0.12  # minimum color pixel density on ring band
    RING_BAND_WIDTH = 3            # half-width of ring band for density measurement
    HOUGH_DP = 1.5                 # accumulator resolution (1.5 = half-res, faster)
    HOUGH_PARAM2 = 12             # accumulator threshold (higher = fewer false circles)
    HOUGH_MIN_DIST = 10           # minimum distance between circle centers

    def __init__(self, detector=None):
        super().__init__(name="heuristic_ring", detector=detector)

    # ================================================================
    #  public entry
    # ================================================================

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[RingTarget]:
        if self.detector is None:
            return None

        ts = self.detector._get_tracking(target_color)

        discovery_w = 320 if ts.last_center is None else TARGET_W
        small, scale, small_hw = self._scale_frame(frame, discovery_w)

        bk = getattr(self.detector, "blur_kernel", 3)
        if bk % 2 == 0:
            bk += 1
        bs = getattr(self.detector, "blur_sigma", 1.5)
        small = cv2.GaussianBlur(small, (bk, bk), bs)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = self._build_color_mask(hsv, target_color)
        if mask is None:
            self._info(f"[HeuristicRing] {target_color.name}: no color ranges configured")
            return self._fallback_predict(ts, target_color)

        mask_px = cv2.countNonZero(mask)
        if mask_px < 10:
            self._info(f"[HeuristicRing] target={target_color.name} mask={mask_px}px (too sparse)")
            return self._fallback_predict(ts, target_color)

        morphed = self._morph_mask(mask)
        self.detector._last_mask = mask
        self.detector._last_morphed = morphed

        # Fast verify: when tracking, check ring-band color at predicted center
        if ts.last_center is not None:
            result = self._fast_roi_verify(morphed, ts, target_color, scale, small_hw)
            if result is not None:
                return result

        result = self._try_roi(small, morphed, ts, target_color, scale, small_hw)
        if result is not None:
            return result

        result = self._try_global(morphed, ts, target_color, scale)
        if result is not None:
            return result

        return self._fallback_predict(ts, target_color)

    # ================================================================
    #  color mask
    # ================================================================

    def _build_color_mask(self, hsv: np.ndarray, color: Color) -> Optional[np.ndarray]:
        ranges = self.detector.color_ranges.get(color, [])
        if not ranges:
            return None
        mask = None
        for lower, upper in ranges:
            chunk = cv2.inRange(hsv, lower, upper)
            mask = chunk if mask is None else cv2.bitwise_or(mask, chunk)
        return mask

    def _morph_mask(self, mask: np.ndarray) -> np.ndarray:
        k = self.MORPH_KERNEL
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel,
                                   iterations=self.MORPH_CLOSE_ITER)
        return closed

    # ================================================================
    #  fast ROI verify (tracking path)
    # ================================================================

    def _fast_roi_verify(self, morphed, ts, target_color, scale, small_hw):
        """Tracking fast path: verify the ring is still at the predicted center."""
        roi = self._decide_roi(ts, scale, small_hw)
        if roi is None:
            return None

        x1, y1, x2, y2 = roi
        roi_morphed = morphed[y1:y2, x1:x2]

        # Use stored outer radius, or estimate from ROI size
        r = getattr(ts, "_ring_outer_radius", None)
        if r is None:
            r = min(roi_morphed.shape) // 3

        roi_h, roi_w = roi_morphed.shape
        cx_roi = roi_w // 2
        cy_roi = roi_h // 2
        r_scaled = r * scale
        density = self._ring_band_color_density(
            (cx_roi, cy_roi), r_scaled, roi_morphed, band_width=self.RING_BAND_WIDTH + 2,
        )

        if density > self.RING_BAND_COLOR_THRESHOLD:
            center = ((cx_roi + x1) / scale, (cy_roi + y1) / scale)
            ts.roi_miss_count = 0
            ts._ring_outer_radius = r  # keep current radius alive
            self._log_detection("ROI_fast", target_color, center, r, 100)
            return self._finalize(center, 100.0, ts, target_color, scale)

        # Fast verify failed → fall through to full Hough in _try_roi
        return None

    # ================================================================
    #  ROI / global dispatch
    # ================================================================

    def _try_roi(self, frame, morphed, ts, target_color, scale, small_hw):
        roi = self._decide_roi(ts, scale, small_hw)
        if roi is None:
            max_miss = getattr(self.detector, "max_roi_miss", 5)
            self._info(f"[HeuristicRing] ROI skipped: miss={ts.roi_miss_count}/{max_miss} "
                       f"center={'none' if ts.last_center is None else 'known'}")
            ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            return None

        x1, y1, x2, y2 = roi
        roi_morphed = morphed[y1:y2, x1:x2]

        # HoughCircles on ROI mask — fast (~1-2 ms on 150×150)
        return self._detect_and_finalize(roi_morphed, ts, target_color, scale,
                                          offset=(x1, y1), context="ROI")

    def _try_global(self, morphed, ts, target_color, scale):
        return self._detect_and_finalize(morphed, ts, target_color, scale,
                                          offset=(0, 0), context="global")

    # ================================================================
    #  core: HoughCircles + concentric validation
    # ================================================================

    def _detect_and_finalize(self, mask, ts, target_color, scale,
                              offset, context):
        h, w = mask.shape[:2]
        max_radius = max(min(w, h) * 9 // 20, 80)
        dp = self.HOUGH_DP
        param2 = self.HOUGH_PARAM2
        if context == "ROI":
            dp = 1.2
            param2 = 10

        circles = cv2.HoughCircles(
            mask, cv2.HOUGH_GRADIENT, dp=dp,
            minDist=self.HOUGH_MIN_DIST, param1=100, param2=param2,
            minRadius=6, maxRadius=max_radius,
        )
        if circles is None:
            self._info(f"[HeuristicRing] {context}: target={target_color.name} "
                       f"hough: no circles found")
            if context == "ROI":
                ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            if getattr(ts, "roi_miss_count", 0) >= 3:
                ts._ring_outer_radius = None
            return None

        circles = np.round(circles[0]).astype(int)

        groups = self._group_concentric(circles)
        if not groups:
            self._info(f"[HeuristicRing] {context}: target={target_color.name} "
                       f"hough={len(circles)} circles, 0 concentric groups")
            if context == "ROI":
                ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            if getattr(ts, "roi_miss_count", 0) >= 3:
                ts._ring_outer_radius = None
            return None

        best_center, best_conf, best_outer_r = None, 0, 0
        ox, oy = offset
        gap_px = getattr(self.detector, "ring_gap_px", self.RING_GAP_PX_DEFAULT)

        for group_center_crop, radii in groups:
            conf = self._evaluate_ring_group(
                group_center_crop, radii, mask, gap_px,
            )
            if conf > best_conf:
                best_conf = conf
                best_center = (
                    (group_center_crop[0] + ox) / scale,
                    (group_center_crop[1] + oy) / scale,
                )
                best_outer_r = max(radii) / scale

        if best_center is None:
            self._info(f"[HeuristicRing] {context}: target={target_color.name} "
                       f"hough={len(circles)} circles groups={len(groups)} "
                       f"all failed verification")
            if context == "ROI":
                ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            if getattr(ts, "roi_miss_count", 0) >= 3:
                ts._ring_outer_radius = None
            return None

        ts._ring_outer_radius = best_outer_r
        self._log_detection(context, target_color, best_center, best_outer_r, best_conf)

        if context == "ROI":
            ts.roi_miss_count = 0

        return self._finalize(best_center, best_conf, ts, target_color, scale)

    # ================================================================
    #  concentric grouping
    # ================================================================

    def _group_concentric(self, circles):
        """Group Hough circles by center proximity. Returns [(avg_center, radii), ...]."""
        if len(circles) < self.CONCENTRIC_MIN_RINGS:
            return []

        used = [False] * len(circles)
        raw_groups = []
        for i in range(len(circles)):
            if used[i]:
                continue
            cx1, cy1, r1 = circles[i]
            group = [(float(cx1), float(cy1), float(r1))]
            used[i] = True
            for j in range(len(circles)):
                if used[j]:
                    continue
                cx2, cy2, r2 = circles[j]
                if np.hypot(cx1 - cx2, cy1 - cy2) < 15:
                    group.append((float(cx2), float(cy2), float(r2)))
                    used[j] = True
            raw_groups.append(group)

        groups = []
        for group in raw_groups:
            if len(group) < self.CONCENTRIC_MIN_RINGS:
                continue
            avg_cx = sum(c[0] for c in group) / len(group)
            avg_cy = sum(c[1] for c in group) / len(group)
            radii = sorted(c[2] for c in group)
            groups.append(((avg_cx, avg_cy), radii))

        return groups

    # ================================================================
    #  ring band color density
    # ================================================================

    def _ring_band_color_density(self, center, r, mask, band_width=None):
        """Color pixel density on an annulus of [r - band_width, r + band_width].

        Returns fraction of annulus pixels that are >0 in mask.
        """
        if band_width is None:
            band_width = self.RING_BAND_WIDTH
        h, w = mask.shape
        cx, cy = int(center[0]), int(center[1])
        r_outer = max(int(r + band_width), 2)
        r_inner = max(int(r - band_width), 1)
        if r_inner >= r_outer:
            return 0.0

        annulus = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(annulus, (cx, cy), r_outer, 255, -1)
        cv2.circle(annulus, (cx, cy), r_inner, 0, -1)

        annulus_px = cv2.countNonZero(annulus)
        if annulus_px == 0:
            return 0.0
        intersection = cv2.bitwise_and(mask, mask, mask=annulus)
        color_px = cv2.countNonZero(intersection)
        return color_px / annulus_px

    def _evaluate_ring_group(self, center, radii, mask, gap_px):
        """Score a concentric circle group by verifying each ring band.
        """
        if len(radii) < self.CONCENTRIC_MIN_RINGS:
            return 0

        confirmed = 0
        total = len(radii)
        for r in radii:
            density = self._ring_band_color_density(
                center, r, mask, band_width=self.RING_BAND_WIDTH,
            )
            if density > self.RING_BAND_COLOR_THRESHOLD:
                confirmed += 1

        if confirmed < self.CONCENTRIC_MIN_RINGS:
            return 0

        # gap consistency bonus: expected gap between adjacent rings
        gap_ok = 0
        radii_sorted = sorted(radii)
        for i in range(len(radii_sorted) - 1):
            gap = radii_sorted[i + 1] - radii_sorted[i]
            if gap > 0 and abs(gap - gap_px) < gap_px * 0.5:
                gap_ok += 1

        ratio = confirmed / total
        gap_bonus = 0.0
        if gap_ok >= self.CONCENTRIC_MIN_RINGS - 1:
            gap_bonus = 10.0
        return min(60.0 + ratio * 30.0 + gap_bonus, 100.0)

    # ================================================================
    #  finalize / kalman / fallback
    # ================================================================

    def _finalize(self, center, conf, ts, target_color, scale):
        cx, cy = center
        smoothed = self._kalman_predict(ts, (cx, cy))
        ts.last_center = (cx, cy)
        ts.roi_miss_count = 0
        ts._center_history.append((cx, cy))
        final = smoothed if smoothed is not None else (cx, cy)
        return create_ring_target(final, target_color, conf)

    def _kalman_predict(self, ts, center):
        cx, cy = center
        result, ts.tracking_initialized, ts.lost_frames, ts._last_predict_time = kalman_update(
            ts.kf, (cx, cy),
            ts.tracking_initialized, ts.lost_frames,
            getattr(self.detector, "max_lost_frames", 30),
            getattr(self.detector, "q_base", 0.1),
            getattr(self.detector, "q_vel_base", 0.01),
            ts._last_predict_time,
        )
        return result

    def _fallback_predict(self, ts, target_color):
        result, ts.tracking_initialized, ts.lost_frames, ts._last_predict_time = kalman_update(
            ts.kf, None,
            ts.tracking_initialized, ts.lost_frames,
            getattr(self.detector, "max_lost_frames", 30),
            getattr(self.detector, "q_base", 0.1),
            getattr(self.detector, "q_vel_base", 0.01),
            ts._last_predict_time,
        )
        if result is not None:
            cx, cy = result
            ts.last_center = (cx, cy)
            ts._center_history.append((cx, cy))
            self._info("[HeuristicRing] fallback=kalman")
            return create_ring_target(result, target_color, 60.0)

        if ts._center_history:
            cxs = [p[0] for p in ts._center_history]
            cys = [p[1] for p in ts._center_history]
            center = (sum(cxs) / len(cxs), sum(cys) / len(cys))
            self._info("[HeuristicRing] fallback=history_avg")
            return create_ring_target(center, target_color, 40.0)

        return None

    # ================================================================
    #  logging
    # ================================================================

    def _tick(self) -> int:
        if not hasattr(self.detector, '_ring_log_frame'):
            self.detector._ring_log_frame = 0
        self.detector._ring_log_frame += 1
        return self.detector._ring_log_frame

    def _info(self, msg: str, force: bool = False) -> None:
        frame = self._tick()
        if force or frame % self._LOG_INTERVAL == 0:
            log_print(msg)

    def _log_detection(self, context, target_color, center, outer_r, conf):
        msg = (f"[HeuristicRing] {context}: "
               f"target={target_color.name} "
               f"center=({center[0]:.0f},{center[1]:.0f}) "
               f"outer_r={outer_r:.0f} conf={conf:.0f}")
        self._info(msg, force=False)
