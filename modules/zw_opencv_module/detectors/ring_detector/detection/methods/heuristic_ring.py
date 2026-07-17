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
    """Contour-driven ring detection via EdgeDrawing + color mask verification.

    Pipeline:
        1. EdgeDrawing on grayscale (CLAHE → blur → detectEdges → morph close)
           produces continuous arcs from the fine red line connecting dashed segments.
        2. Color mask (raw, no morph) from HSV inRange for ring-band density checks.
        3. findContours on edge image → fitEllipse → candidate center + outer radius.
        4. Ring-band color density verification on raw color mask at outer radius.
        5. HoughCircles concentric verification around candidate center (dp=2.0).
        6. Confidence: base from fitEllipse, +20 per confirmed concentric ring.

    Performance:
        - Discovery: EdgeDrawing + contours + Hough verify  ~4-6 ms
        - ROI tracking: fast ring-band density verify        ~0.3 ms
        - Kalman prediction: pure math                       ~0.01 ms
    """

    _LOG_INTERVAL = 60

    AXIS_RATIO_MAX = 2.0
    RING_GAP_PX_DEFAULT = 10
    RING_BAND_COLOR_THRESHOLD = 0.05
    RING_BAND_WIDTH = 8
    INNER_RADIUS_RATIO = 0.55
    HOUGH_DP = 2.0
    HOUGH_PARAM2 = 8
    HOUGH_MIN_DIST = 8

    def __init__(self, detector=None):
        super().__init__(name="heuristic_ring", detector=detector)

    # ================================================================
    #  public entry
    # ================================================================

    def detect(self, frame: np.ndarray, target_color: Color) -> Optional[RingTarget]:
        if self.detector is None or self.detector.ed is None:
            return None

        ts = self.detector._get_tracking(target_color)

        discovery_w = 320 if ts.last_center is None else TARGET_W
        small, scale, small_hw = self._scale_frame(frame, discovery_w)

        # ── color mask (raw, no morph) ──
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        color_mask = self._build_color_mask(hsv, target_color)
        if color_mask is None:
            self._info(f"[HeuristicRing] {target_color.name}: no color ranges configured")
            return self._fallback_predict(ts, target_color)

        # ── EdgeDrawing edge image ──
        edges = self._build_edge_image(small)

        self.detector._last_mask = color_mask
        self.detector._last_edge_preview = edges

        mask_px = cv2.countNonZero(color_mask)
        if mask_px < 5:
            self._info(f"[HeuristicRing] target={target_color.name} color_mask={mask_px}px (too sparse)")
            return self._fallback_predict(ts, target_color)

        if not getattr(self.detector, "force_global", False):
            if ts.last_center is not None:
                result = self._fast_roi_verify(color_mask, ts, target_color, scale, small_hw)
                if result is not None:
                    return result

            result = self._try_roi(edges, color_mask, ts, target_color, scale, small_hw)
            if result is not None:
                return result

        result = self._try_global(edges, color_mask, ts, target_color, scale)
        if result is not None:
            return result

        return self._fallback_predict(ts, target_color)

    def _build_edge_image(self, small: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = self.detector._clahe.apply(gray)

        bk = getattr(self.detector, "blur_kernel", 3)
        if bk % 2 == 0:
            bk += 1
        bs = getattr(self.detector, "blur_sigma", 1.5)
        blurred = cv2.GaussianBlur(gray, (bk, bk), bs)

        self.detector.ed.detectEdges(blurred)
        edges = self.detector.ed.getEdgeImage()

        ek = getattr(self.detector, "edge_morph_kernel", 3)
        if ek % 2 == 0:
            ek += 1
        ei = getattr(self.detector, "edge_morph_iterations", 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ek, ek))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=ei)
        return edges

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

    # ================================================================
    #  fast ROI verify (tracking path, no Hough, no EdgeDrawing)
    # ================================================================

    def _fast_roi_verify(self, color_mask, ts, target_color, scale, small_hw):
        roi = self._decide_roi(ts, scale, small_hw)
        if roi is None:
            return None

        x1, y1, x2, y2 = roi
        roi_mask = color_mask[y1:y2, x1:x2]

        r = getattr(ts, "_ring_outer_radius", None)
        if r is not None and r < 25:
            r = None
        if r is None:
            r = min(roi_mask.shape) // 3

        roi_h, roi_w = roi_mask.shape
        cx_roi = roi_w // 2
        cy_roi = roi_h // 2
        r_scaled = r * scale
        density_outer = self._ring_band_color_density(
            (cx_roi, cy_roi), r_scaled, roi_mask,
            band_width=self.RING_BAND_WIDTH + 2,
        )
        density_inner = self._ring_band_color_density(
            (cx_roi, cy_roi), r_scaled * self.INNER_RADIUS_RATIO, roi_mask,
            band_width=self.RING_BAND_WIDTH + 2,
        )

        if max(density_outer, density_inner) > self.RING_BAND_COLOR_THRESHOLD:
            # refine center using color-mask moments (thin ring band)
            ref_band = max(int(r_scaled * 0.3), self.RING_BAND_WIDTH)
            if r_scaled + ref_band > min(roi_h, roi_w) // 2:
                return None  # ring too large for ROI — fall through to full detection
            annulus = np.zeros_like(roi_mask)
            cv2.circle(annulus, (roi_w // 2, roi_h // 2), int(r_scaled + ref_band), 255, -1)
            cv2.circle(annulus, (roi_w // 2, roi_h // 2),
                       max(int(r_scaled * self.INNER_RADIUS_RATIO - ref_band), 1), 0, -1)
            intersect = cv2.bitwise_and(roi_mask, annulus)
            M = cv2.moments(intersect)
            if M["m00"] > 0:
                cx_roi = int(M["m10"] / M["m00"])
                cy_roi = int(M["m01"] / M["m00"])
            center = ((cx_roi + x1) / scale, (cy_roi + y1) / scale)
            ts.roi_miss_count = 0
            ts._ring_outer_radius = r
            self._log_detection("ROI_fast", target_color, center, r, 100)
            return self._finalize(center, 100.0, ts, target_color, scale)

        return None

    # ================================================================
    #  ROI / global dispatch
    # ================================================================

    def _try_roi(self, edges, color_mask, ts, target_color, scale, small_hw):
        roi = self._decide_roi(ts, scale, small_hw)
        if roi is None:
            max_miss = getattr(self.detector, "max_roi_miss", 5)
            self._info(f"[HeuristicRing] ROI skipped: miss={ts.roi_miss_count}/{max_miss} "
                       f"center={'none' if ts.last_center is None else 'known'}")
            ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            return None

        x1, y1, x2, y2 = roi
        roi_edges = edges[y1:y2, x1:x2]
        roi_mask = color_mask[y1:y2, x1:x2]
        return self._detect_and_finalize(roi_edges, roi_mask, ts, target_color, scale,
                                          offset=(x1, y1), context="ROI")

    def _try_global(self, edges, color_mask, ts, target_color, scale):
        return self._detect_and_finalize(edges, color_mask, ts, target_color, scale,
                                          offset=(0, 0), context="global")

    # ================================================================
    #  core: contours from edge image + color verification
    # ================================================================

    def _detect_and_finalize(self, edges, color_mask, ts, target_color, scale,
                              offset, context):
        min_area = getattr(self.detector, "min_area", 80)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._info(f"[HeuristicRing] {context}: target={target_color.name} "
                       f"no contours")
            if context == "ROI":
                ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            if getattr(ts, "roi_miss_count", 0) >= 3:
                ts._ring_outer_radius = None
            return None

        best_center, best_score, best_outer_r = None, 0, 0
        gap_px = getattr(self.detector, "ring_gap_px", self.RING_GAP_PX_DEFAULT)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            ecx, ecy, outer_r, base_conf = None, None, None, 0
            try:
                (ecx, ecy), (ea, eb), _ = cv2.fitEllipse(cnt)
                axis_ratio = max(ea, eb) / max(min(ea, eb), 1)
                if axis_ratio <= self.AXIS_RATIO_MAX:
                    outer_r = (ea + eb) / 2.0
                    base_conf = 60
            except cv2.error:
                pass

            if base_conf == 0:
                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue
                ecx = M["m10"] / M["m00"]
                ecy = M["m01"] / M["m00"]
                outer_r = np.sqrt(area / np.pi)
                base_conf = 40

            density_outer = self._ring_band_color_density(
                (ecx, ecy), outer_r, color_mask, band_width=self.RING_BAND_WIDTH,
            )
            density_inner = self._ring_band_color_density(
                (ecx, ecy), outer_r * self.INNER_RADIUS_RATIO, color_mask,
                band_width=self.RING_BAND_WIDTH,
            )
            best_density = max(density_outer, density_inner)
            if best_density < self.RING_BAND_COLOR_THRESHOLD:
                continue

            circ_score = 1.0 / max(axis_ratio, 1.0)
            score = circ_score + best_density * 0.05
            if score > best_score:
                best_score = score
                best_center = (ecx, ecy)
                best_outer_r = outer_r
                # track best axis_ratio for logging
                best_axis = axis_ratio if base_conf >= 60 else 999

        if best_center is None:
            self._info(f"[HeuristicRing] {context}: target={target_color.name} "
                       f"contours={len(contours)} no candidate passed color verification")
            if context == "ROI":
                ts.roi_miss_count = getattr(ts, "roi_miss_count", 0) + 1
            if getattr(ts, "roi_miss_count", 0) >= 3:
                ts._ring_outer_radius = None
            return None

        ecx, ecy = best_center
        outer_r = best_outer_r

        ref_band = max(int(outer_r * 0.3), self.RING_BAND_WIDTH)
        cm_h, cm_w = color_mask.shape
        if outer_r + ref_band <= min(cm_h, cm_w) // 2:
            annulus = np.zeros_like(color_mask)
            cv2.circle(annulus, (int(ecx), int(ecy)), int(outer_r + ref_band), 255, -1)
            cv2.circle(annulus, (int(ecx), int(ecy)),
                       max(int(outer_r * self.INNER_RADIUS_RATIO - ref_band), 1), 0, -1)
            intersect = cv2.bitwise_and(color_mask, annulus)
            M = cv2.moments(intersect)
            if M["m00"] > 0:
                ecx = M["m10"] / M["m00"]
                ecy = M["m01"] / M["m00"]

        hough_rings = self._verify_concentric(edges, (ecx, ecy), outer_r)
        conf = min(60 + hough_rings * 20, 100)

        ox, oy = offset
        center = ((ecx + ox) / scale, (ecy + oy) / scale)
        outer_r_orig = outer_r / scale

        if context == "ROI":
            ts.roi_miss_count = 0
        ts._ring_outer_radius = outer_r_orig

        self._log_detection(context, target_color, center, outer_r_orig, conf)
        return self._finalize(center, conf, ts, target_color, scale)

    def _verify_concentric(self, edges, center, outer_r):
        """Run HoughCircles on edge image around candidate to count concentric rings."""
        cx, cy = int(center[0]), int(center[1])
        roi_r = int(outer_r * 2.5)
        x1 = max(0, cx - roi_r)
        y1 = max(0, cy - roi_r)
        x2 = min(edges.shape[1], cx + roi_r)
        y2 = min(edges.shape[0], cy + roi_r)
        if x2 <= x1 or y2 <= y1:
            return 0

        roi = edges[y1:y2, x1:x2]
        local_cx = cx - x1
        local_cy = cy - y1
        max_r = min(roi.shape) // 2

        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT, dp=self.HOUGH_DP,
            minDist=self.HOUGH_MIN_DIST, param1=100, param2=self.HOUGH_PARAM2,
            minRadius=4, maxRadius=int(max_r),
        )
        if circles is None:
            return 0

        circles = circles[0]
        center_dist_max = outer_r * 0.35
        rings = 0
        for hcx, hcy, _hr in circles:
            if np.hypot(hcx - local_cx, hcy - local_cy) < center_dist_max:
                rings += 1

        return min(rings, 3)

    # ================================================================
    #  ring band color density
    # ================================================================

    def _ring_band_color_density(self, center, r, mask, band_width=None):
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
        return cv2.countNonZero(intersection) / annulus_px

    # ================================================================
    #  finalize / kalman / fallback
    # ================================================================

    def _finalize(self, center, conf, ts, target_color, scale):
        cx, cy = center
        if getattr(self.detector, "kalman_enabled", True):
            smoothed = self._kalman_predict(ts, (cx, cy))
        else:
            smoothed = (cx, cy)
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
        if not getattr(self.detector, "kalman_enabled", True):
            return None

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
            return create_ring_target(result, target_color, 60.0, is_predicted=True)

        if ts._center_history:
            cxs = [p[0] for p in ts._center_history]
            cys = [p[1] for p in ts._center_history]
            center = (sum(cxs) / len(cxs), sum(cys) / len(cys))
            self._info("[HeuristicRing] fallback=history_avg")
            return create_ring_target(center, target_color, 40.0, is_predicted=True)

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
