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
    RING_BAND_NOISE_FLOOR = 0.005
    MOMENTS_MIN_DENSITY = 0.10
    COLOR_BLOB_MIN_AREA = 80
    RING_BAND_WIDTH = 8
    INNER_RADIUS_RATIO = 0.55
    HOUGH_DP = 2.0
    HOUGH_PARAM2 = 8
    HOUGH_MIN_DIST = 8

    HOUGH_ALT_DP = 1.5
    HOUGH_ALT_PARAM2 = 0.95
    HOUGH_ALT_MIN_RADIUS = 10
    HOUGH_ALT_PARAM1 = 100
    HOUGH_ALT_MIN_DIST = 30

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

        self._alt_img = self._build_alt_processed(small)

        self.detector._last_mask = color_mask
        self.detector._last_edge_preview = edges

        mask_px = cv2.countNonZero(color_mask)
        if mask_px < 5:
            self._info(f"[HeuristicRing] target={target_color.name} color_mask={mask_px}px (too sparse)")
            return self._fallback_predict(ts, target_color)

        result = self._try_global(edges, color_mask, ts, target_color, scale)
        if result is not None:
            return result

        result = self._try_alt_hough(color_mask, ts, target_color, scale)
        if result is not None:
            return result

        result = self._try_color_blob(color_mask, ts, target_color, scale, small_hw)
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

    def _build_alt_processed(self, small: np.ndarray) -> np.ndarray:
        eroded = cv2.erode(small, None, iterations=2)
        kernel_7 = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(eroded, kernel_7, iterations=1)

        gray = cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY)

        kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel_5)

        result = cv2.GaussianBlur(gradient, (7, 7), 3, 3)
        result = cv2.convertScaleAbs(result, alpha=4, beta=0)
        result = cv2.GaussianBlur(result, (7, 7), 3, 3)

        _, thresh = cv2.threshold(result, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.GaussianBlur(thresh, (9, 9), 3, 3)
        return thresh

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
    #  global dispatch
    # ================================================================

    def _try_global(self, edges, color_mask, ts, target_color, scale):
        return self._detect_and_finalize(edges, color_mask, ts, target_color, scale,
                                          offset=(0, 0), context="global")

    def _try_alt_hough(self, color_mask, ts, target_color, scale):
        alt = getattr(self, "_alt_img", None)
        if alt is None:
            return None

        h, w = alt.shape
        max_radius = int(min(h, w) * 0.35)

        circles = cv2.HoughCircles(
            alt, cv2.HOUGH_GRADIENT_ALT, dp=self.HOUGH_ALT_DP,
            minDist=self.HOUGH_ALT_MIN_DIST, param1=self.HOUGH_ALT_PARAM1,
            param2=self.HOUGH_ALT_PARAM2,
            minRadius=self.HOUGH_ALT_MIN_RADIUS, maxRadius=max_radius,
        )
        if circles is None:
            return None

        circles_coords = circles[0]
        best_center, best_score, best_r = None, 0, 0

        for hcx, hcy, hr in circles_coords:
            density_outer = self._ring_band_color_density(
                (hcx, hcy), hr, color_mask, band_width=self.RING_BAND_WIDTH,
            )
            density_inner = self._ring_band_color_density(
                (hcx, hcy), hr * self.INNER_RADIUS_RATIO, color_mask,
                band_width=self.RING_BAND_WIDTH,
            )
            best_density = max(density_outer, density_inner)
            if best_density <= self.RING_BAND_NOISE_FLOOR:
                continue

            score = best_density
            if score > best_score:
                best_score = score
                best_center = (hcx, hcy)
                best_r = hr

        if best_center is None:
            return None

        hcx, hcy = best_center
        center = (hcx / scale, hcy / scale)
        outer_r_orig = best_r / scale
        conf = min(60 + int(best_score * 100), 100)

        ts._ring_outer_radius = outer_r_orig

        self._store_ring_meta(target_color, center, outer_r_orig, None, None, 0.0, method="alt_hough")

        self._log_detection("alt_hough", target_color, center, outer_r_orig, conf)
        return self._finalize(center, conf, ts, target_color, scale)

    def _try_color_blob(self, color_mask, ts, target_color, scale, small_hw):
        contours, _ = cv2.findContours(
            color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(best)
        if area < self.COLOR_BLOB_MIN_AREA:
            return None

        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        center = (cx / scale, cy / scale)

        self._store_ring_meta(target_color, center, 0, area, None, 0.0, method="color_blob")
        self._info(f"[HeuristicRing] color_blob: target={target_color.name} "
                   f"center=({center[0]:.0f},{center[1]:.0f}) area={area:.0f}")
        return create_ring_target(center, target_color, 30.0)

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
        best_axes, best_angle, best_area = None, 0.0, 0.0
        best_density_value = 0.0
        gap_px = getattr(self.detector, "ring_gap_px", self.RING_GAP_PX_DEFAULT)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            ecx, ecy, outer_r, base_conf = None, None, None, 0
            axis_ratio = 999.0
            try:
                (ecx, ecy), (ea, eb), angle = cv2.fitEllipse(cnt)
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
            if best_density <= self.RING_BAND_NOISE_FLOOR:
                continue

            circ_score = 1.0 / max(axis_ratio, 1.0)
            score = circ_score + best_density * 0.05
            if score > best_score:
                best_score = score
                best_center = (ecx, ecy)
                best_outer_r = outer_r
                best_area = area
                best_density_value = best_density
                if base_conf >= 60:
                    best_axes = (ea, eb)
                    best_angle = angle
                else:
                    best_axes = None
                    best_angle = 0.0
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

        if best_density_value >= self.MOMENTS_MIN_DENSITY:
            ref_band = max(int(outer_r * 0.3), self.RING_BAND_WIDTH)
            cm_h, cm_w = color_mask.shape
            ref_outer_r = max(outer_r, min(cm_h, cm_w) * 0.08)
            if ref_outer_r + ref_band <= min(cm_h, cm_w) // 2:
                annulus = np.zeros_like(color_mask)
                for _ in range(3):
                    annulus.fill(0)
                    cv2.circle(annulus, (int(ecx), int(ecy)), int(ref_outer_r + ref_band), 255, -1)
                    cv2.circle(annulus, (int(ecx), int(ecy)),
                               max(int(ref_outer_r * self.INNER_RADIUS_RATIO - ref_band), 1), 0, -1)
                    intersect = cv2.bitwise_and(color_mask, annulus)
                    M = cv2.moments(intersect)
                    if M["m00"] == 0:
                        break
                    new_ecx = M["m10"] / M["m00"]
                    new_ecy = M["m01"] / M["m00"]
                    if abs(new_ecx - ecx) < 0.5 and abs(new_ecy - ecy) < 0.5:
                        ecx, ecy = new_ecx, new_ecy
                        break
                    ecx, ecy = new_ecx, new_ecy

        hough_rings = self._verify_concentric(edges, (ecx, ecy), outer_r)
        conf = min(60 + hough_rings * 20, 100)

        ox, oy = offset
        center = ((ecx + ox) / scale, (ecy + oy) / scale)
        outer_r_orig = outer_r / scale

        if context == "ROI":
            ts.roi_miss_count = 0
        ts._ring_outer_radius = outer_r_orig

        self._store_ring_meta(target_color, center, outer_r_orig, best_area / (scale * scale),
                              (best_axes[0] / scale, best_axes[1] / scale) if best_axes else None,
                              best_angle, method=context)

        self._log_detection(context, target_color, center, outer_r_orig, conf)
        return self._finalize(center, conf, ts, target_color, scale)

    def _verify_concentric(self, edges, center, outer_r):
        cx, cy = int(center[0]), int(center[1])
        roi_r = int(outer_r * 2.5)
        x1 = max(0, cx - roi_r)
        y1 = max(0, cy - roi_r)
        x2 = min(edges.shape[1], cx + roi_r)
        y2 = min(edges.shape[0], cy + roi_r)
        if x2 <= x1 or y2 <= y1:
            return 0

        alt = getattr(self, "_alt_img", None)
        if alt is not None:
            roi = alt[y1:y2, x1:x2]
        else:
            roi = edges[y1:y2, x1:x2]

        local_cx = cx - x1
        local_cy = cy - y1
        max_r = min(roi.shape) // 2

        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT_ALT, dp=self.HOUGH_ALT_DP,
            minDist=self.HOUGH_ALT_MIN_RADIUS, param1=self.HOUGH_ALT_PARAM1,
            param2=self.HOUGH_ALT_PARAM2,
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

        self.detector._last_ring_meta.pop(target_color, None)

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

    def _store_ring_meta(self, target_color, center, outer_radius, area, axes, angle, method=""):
        self.detector._last_ring_meta[target_color] = {
            'center': center,
            'outer_radius': outer_radius,
            'area': area,
            'axes': axes,
            'angle': angle,
            'method': method,
        }

    def _log_detection(self, context, target_color, center, outer_r, conf):
        msg = (f"[HeuristicRing] {context}: "
               f"target={target_color.name} "
               f"center=({center[0]:.0f},{center[1]:.0f}) "
               f"outer_r={outer_r:.0f} conf={conf:.0f}")
        self._info(msg, force=False)
