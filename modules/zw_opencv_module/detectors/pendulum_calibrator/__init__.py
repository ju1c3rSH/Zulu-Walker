"""
Pendulum rail calibrator.
Detects the pendulum rail axis from a camera frame using OpenCV,
providing a calibrated reference line for steel ball position projection.
"""
import cv2
import numpy as np
import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class RailCalibration:
    origin_x: float = 0.0
    origin_y: float = 0.0
    angle_rad: float = 0.0
    calibrated: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'dir_cos', math.cos(self.angle_rad))
        object.__setattr__(self, 'dir_sin', math.sin(self.angle_rad))

    def project(self, px: float, py: float) -> float:
        return (px - self.origin_x) * self.dir_cos + (py - self.origin_y) * self.dir_sin

    def replace_origin(self, origin_x: float, origin_y: float) -> 'RailCalibration':
        return RailCalibration(
            origin_x=origin_x,
            origin_y=origin_y,
            angle_rad=self.angle_rad,
            calibrated=True,
        )

    def to_dict(self) -> dict:
        return {
            'origin_x': self.origin_x,
            'origin_y': self.origin_y,
            'angle_rad': self.angle_rad,
            'calibrated': self.calibrated,
            'dir_cos': self.dir_cos,
            'dir_sin': self.dir_sin,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'RailCalibration':
        return cls(
            origin_x=d['origin_x'],
            origin_y=d['origin_y'],
            angle_rad=d['angle_rad'],
            calibrated=d.get('calibrated', False),
        )


class PendulumCalibrator:
    def __init__(self, frame_w: int = 640, frame_h: int = 640,
                 binary_threshold: int = 127,
                 min_contour_area_ratio: float = 0.04,
                 min_aspect_ratio: float = 1.0,
                 canny_low: int = 50, canny_high: int = 150,
                 hough_threshold: int = 50,
                 hough_min_line_len: int = 150,
                 edge_angle_max_deg: float = 15,
                 column_threshold: int = 180,
                 max_contour_area_ratio: float = 0.55):
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._binary_threshold = binary_threshold
        self._min_contour_area_ratio = min_contour_area_ratio
        self._max_contour_area_ratio = max_contour_area_ratio
        self._min_aspect_ratio = min_aspect_ratio
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._hough_threshold = hough_threshold
        self._hough_min_line_len = hough_min_line_len
        self._edge_angle_max_deg = edge_angle_max_deg
        self._column_threshold = column_threshold
        self._debug_binary = None
        self._debug_contour = None
        self._debug_rect = None
        self._debug_column_binary = None
        self._empty_debug = None
        self._diag = {}

    @property
    def last_contour(self):
        return self._debug_contour

    @property
    def last_rect(self):
        return self._debug_rect

    @property
    def last_binary(self):
        return self._debug_binary

    @property
    def last_column_binary(self):
        return self._debug_column_binary

    def get_last_diagnostics(self) -> dict:
        return dict(self._diag)

    def calibrate(self, frame_bgr: np.ndarray, ball_px: tuple = None) -> RailCalibration:
        self._diag = {}
        self._debug_binary = None
        self._debug_contour = None
        self._debug_rect = None
        self._debug_column_binary = None

        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            self._diag['fail_reason'] = 'gray_conversion'
            return RailCalibration(calibrated=False)

        self._diag['gray_mean'] = float(np.mean(gray))
        self._diag['gray_std'] = float(np.std(gray))

        # Primary: per-column centreline fit. Robust when the rail spans (or
        # exceeds) the full frame width and when the background is washed out
        # into a frame-filling blob that breaks minAreaRect (contour fallback).
        rail_info = self._detect_rail_column_centroid(gray)
        if rail_info is not None:
            self._diag['method'] = 'column_centroid'
        else:
            rail_info = self._detect_rail_contour(gray)
            if rail_info is not None:
                self._diag['method'] = 'contour'
                self._diag['used_fallback'] = False
            else:
                self._diag['used_fallback'] = True
                rail_info = self._detect_rail_edges(gray)
                if rail_info is not None:
                    self._diag['method'] = 'edge'

        if rail_info is None:
            if 'fail_reason' not in self._diag:
                self._diag['fail_reason'] = 'no_detection'
            self._diag['method'] = self._diag.get('method', 'none')
            return RailCalibration(calibrated=False)

        # A failed primary stage leaves a stale fail_reason; clear it so the
        # diagnostics reflect the stage that actually succeeded.
        self._diag.pop('fail_reason', None)

        center_x, center_y, angle_rad = rail_info

        if ball_px is not None and len(ball_px) >= 2:
            origin_x, origin_y = float(ball_px[0]), float(ball_px[1])
            self._diag['origin_source'] = 'ball_px'
        else:
            origin_x, origin_y = center_x, center_y
            self._diag['origin_source'] = 'contour midpoint (no ball_px)'

        self._diag['rect_center'] = (round(center_x, 1), round(center_y, 1))
        self._diag['angle_rad'] = angle_rad
        self._diag['origin'] = (origin_x, origin_y)
        self._diag['dir'] = (math.cos(angle_rad), math.sin(angle_rad))
        self._diag['calibrated'] = True

        return RailCalibration(
            origin_x=origin_x,
            origin_y=origin_y,
            angle_rad=angle_rad,
            calibrated=True,
        )

    def _detect_rail_column_centroid(self, gray: np.ndarray):
        """Per-column centreline fit (primary method).

        Binarise with a threshold that keeps only the bright PPR pipe
        (excluding a washed-out background), then for each column take the
        midpoint between the topmost/bottommost white pixel as a rail
        centreline sample.  The threshold is tried in descending order
        (``column_threshold`` first, then lower steps) so a dimmer pipe still
        separates from the washed-out background.  The vertical white extent
        is filtered adaptively against the per-frame median span rather than a
        fixed window, so a curved pipe whose bright arc is only a thin band
        (and ball occlusion / specular columns) is handled without tuning.
        A least-squares line fit over the surviving samples yields the rail
        axis, which stays correct even when the rail spans or exceeds the
        frame width (a case where minAreaRect degrades to the frame rect).
        """
        fw = gray.shape[1]
        fh = gray.shape[0]
        base_threshold = self._column_threshold
        # Descending threshold ladder: keep the configured value as the first
        # (brightest-isolation) attempt, then relax so a dim PPR pipe still
        # yields a usable bright band above the washed-out background (~140).
        ladder = [base_threshold]
        for t in (150, 120):
            if t < base_threshold and t not in ladder:
                ladder.append(t)

        xs = None
        ys = None
        used_threshold = None
        binary = None
        for threshold in ladder:
            try:
                _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            except Exception:
                self._diag['column_fail_reason'] = 'column_threshold_error'
                return None

            xs, ys = self._sample_column_centroids(binary, fw, fh)
            n_pts = len(xs) if xs is not None else 0
            if n_pts >= fw * 0.5:
                used_threshold = threshold
                break

        self._debug_column_binary = binary
        self._debug_binary = binary
        self._diag['column_threshold'] = used_threshold

        if xs is None or len(xs) < fw * 0.5:
            self._diag['column_points'] = len(xs) if xs is not None else 0
            # Preserve the specific guardrail reason set by the sampler (e.g.
            # column_median_height / column_off_band) under its own key so it
            # survives the caller's generic overwrite and is visible on-device.
            self._diag['column_fail_reason'] = self._diag.get('column_fail_reason', 'column_insufficient')
            self._diag['fail_reason'] = 'column_insufficient'
            return None

        # Success: a higher threshold in the ladder may have set a guardrail
        # reason before this (lower) threshold succeeded; clear it.
        self._diag.pop('column_fail_reason', None)

        n_pts = len(xs)
        self._diag['column_points'] = n_pts

        pts = np.array([(x, y) for x, y in zip(xs, ys)], dtype=np.float32)
        try:
            vx, vy, cx, cy = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
        except Exception:
            self._diag['column_fail_reason'] = 'column_fit_error'
            self._diag['fail_reason'] = 'column_insufficient'
            return None

        vx = float(vx[0])
        vy = float(vy[0])
        cx = float(cx[0])
        cy = float(cy[0])

        # cv2.fitLine returns an unoriented direction; normalise so the
        # horizontal rail reads as angle_rad ~ 0 (dir_cos=1, dir_sin=0).
        if vx < 0:
            vx = -vx
            vy = -vy
        angle_rad = math.atan2(vy, vx)

        self._diag['rect_center'] = (round(cx, 1), round(cy, 1))
        self._diag['angle_rad'] = angle_rad

        return (cx, cy, angle_rad)

    def _sample_column_centroids(self, binary: np.ndarray, fw: int, fh: int):
        """Per-column midpoint samples with an adaptive median span window.

        Collects the topmost/bottommost white pixel per column, then accepts
        columns whose white span sits within ``[0.6, 1.6] x median`` of the
        per-frame median span.  Guardrails: the median span must be at least
        5% of the frame height (rejects a frame-filling background blob), and
        the fitted centroid must lie in the central band.
        """
        spans = []
        tops = []
        bottoms = []
        xs_all = []
        for x in range(fw):
            col = binary[:, x]
            idx = np.where(col > 0)[0]
            if len(idx) == 0:
                continue
            top = int(idx.min())
            bottom = int(idx.max())
            xs_all.append(x)
            tops.append(top)
            bottoms.append(bottom)
            spans.append(bottom - top + 1)

        if len(spans) < fw * 0.5:
            return None, None

        med = float(statistics.median(spans))
        self._diag['column_median_h'] = round(med, 1)

        # Reject a frame-filling background blob: a real rail occupies a
        # minority of the frame height (curved pipe bright arc << fh).
        if med < fh * 0.05 or med > fh * 0.9:
            self._diag['column_fail_reason'] = 'column_median_height'
            return None, None

        lo = med * 0.6
        hi = med * 1.6
        xs = []
        ys = []
        for x, top, bottom, span in zip(xs_all, tops, bottoms, spans):
            if lo <= span <= hi:
                xs.append(x)
                ys.append((top + bottom) / 2.0)

        if len(xs) < fw * 0.5:
            self._diag['column_fail_reason'] = 'column_insufficient'
            return None, None

        mean_y = float(np.mean(ys))
        # The rail is expected near the vertical centre of the frame; a fit
        # pulled far off-band indicates the bright region is not the rail.
        if not (fh * 0.2 <= mean_y <= fh * 0.8):
            self._diag['column_fail_reason'] = 'column_off_band'
            return None, None

        return xs, ys

    def _detect_rail_contour(self, gray: np.ndarray):
        try:
            if np.std(gray) < 30:
                _, binary = cv2.threshold(gray, self._binary_threshold, 255, cv2.THRESH_BINARY)
                self._diag['threshold_method'] = 'fixed'
                self._diag['threshold_ret'] = self._binary_threshold
            else:
                ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                self._diag['threshold_method'] = 'otsu'
                self._diag['threshold_ret'] = int(ret)
        except Exception:
            self._diag['fail_reason'] = 'threshold_error'
            return None

        white_px = np.sum(binary == 255) / binary.size * 100
        self._diag['white_px_pct'] = round(white_px, 1)
        self._debug_binary = binary

        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            self._diag['fail_reason'] = 'find_contours_error'
            return None

        self._diag['contours_found'] = len(contours)

        if not contours:
            self._diag['contours_max_area'] = None
            self._diag['fail_reason'] = 'no_contours'
            return None

        max_area = max(cv2.contourArea(c) for c in contours)
        self._diag['contours_max_area'] = int(max_area)

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        frame_area = self._frame_w * self._frame_h
        min_area = frame_area * self._min_contour_area_ratio
        max_allowed_area = frame_area * self._max_contour_area_ratio

        self._diag['contour_area'] = int(area)
        self._diag['min_area_limit'] = int(min_area)
        self._diag['contour_area_ok'] = area >= min_area

        if area < min_area:
            self._diag['fail_reason'] = 'min_area'
            return None

        # Reject a frame-filling blob (the original angle==0 bug): a washed-out
        # background merging with the rail yields one contour covering ~60% of
        # the frame; a real rail is at most ~1/3 of the frame height.
        if area > max_allowed_area:
            self._diag['contour_area'] = int(area)
            self._diag['max_area_limit'] = int(max_allowed_area)
            self._diag['fail_reason'] = 'max_area'
            return None

        try:
            rect = cv2.minAreaRect(largest_contour)
        except Exception:
            self._diag['fail_reason'] = 'min_area_rect_error'
            return None

        (center_x, center_y), (w, h), angle = rect

        self._diag['rect_center'] = (round(center_x, 1), round(center_y, 1))
        self._diag['rect_angle'] = round(angle, 1)

        if w < 1 or h < 1:
            self._diag['fail_reason'] = 'degenerate_rect'
            return None

        ratio = max(w, h) / min(w, h)

        self._diag['contour_aspect'] = round(ratio, 1)
        self._diag['min_aspect_limit'] = self._min_aspect_ratio
        self._diag['contour_aspect_ok'] = ratio >= self._min_aspect_ratio

        # min_aspect_ratio=1.0 is a deliberate neutral floor:
        # at 1280x352 the rail may appear near-square due to perspective;
        # min_contour_area_ratio + center-bounds are sufficient filters here.
        if ratio < self._min_aspect_ratio:
            self._diag['fail_reason'] = 'aspect_ratio'
            return None

        fw, fh = gray.shape[1], gray.shape[0]
        cx_ok = fw * 0.1 <= center_x <= fw * 0.9
        cy_ok = fh * 0.1 <= center_y <= fh * 0.9
        self._diag['contour_center_ok'] = cx_ok and cy_ok

        if not (cx_ok and cy_ok):
            self._diag['fail_reason'] = 'center_out_of_bounds'
            return None

        self._debug_contour = largest_contour
        self._debug_rect = rect

        if angle < -45:
            angle = angle + 90

        angle_rad = math.radians(angle)

        return (center_x, center_y, angle_rad)

    def _detect_rail_edges(self, gray: np.ndarray):
        try:
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            canny = cv2.Canny(blurred, self._canny_low, self._canny_high)
        except Exception:
            self._diag['fail_reason'] = 'canny_error'
            return None

        self._diag['canny_low'] = self._canny_low
        self._diag['canny_high'] = self._canny_high
        self._debug_binary = canny

        if np.sum(canny > 0) == 0:
            self._diag['fail_reason'] = 'no_edges'
            return None

        try:
            lines = cv2.HoughLinesP(canny, rho=1, theta=math.pi / 180,
                                    threshold=self._hough_threshold,
                                    minLineLength=self._hough_min_line_len,
                                    maxLineGap=50)
        except Exception:
            self._diag['fail_reason'] = 'hough_error'
            return None

        if lines is None:
            self._diag['edge_lines_raw'] = 0
            self._diag['fail_reason'] = 'hough_no_lines'
            return None

        self._diag['edge_lines_raw'] = len(lines)

        midpoints_x = []
        midpoints_y = []
        angles = []
        max_angle_rad = math.radians(self._edge_angle_max_deg)

        for line in lines:
            pts = line[0] if line.ndim == 2 and line.shape[0] == 1 else line
            x1, y1, x2, y2 = pts
            line_angle = math.atan2(y2 - y1, x2 - x1)
            if abs(line_angle) < max_angle_rad:
                mx = (x1 + x2) / 2.0
                my = (y1 + y2) / 2.0
                midpoints_x.append(mx)
                midpoints_y.append(my)
                angles.append(line_angle)

        self._diag['edge_lines_filtered'] = len(midpoints_x)
        self._diag['edge_angle_max_deg'] = self._edge_angle_max_deg

        if len(midpoints_x) < 3:
            self._diag['fail_reason'] = 'hough_insufficient_lines'
            return None

        center_x = float(np.mean(midpoints_x))
        center_y = float(np.mean(midpoints_y))
        angle_rad = float(np.mean(angles))

        self._diag['rect_center'] = (round(center_x, 1), round(center_y, 1))
        self._diag['angle_rad'] = angle_rad

        return (center_x, center_y, angle_rad)

    def get_debug_frame(self) -> np.ndarray:
        if self._debug_binary is not None:
            debug = cv2.cvtColor(self._debug_binary, cv2.COLOR_GRAY2BGR)
        else:
            if self._empty_debug is None:
                self._empty_debug = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
            debug = self._empty_debug.copy()

        try:
            if self._debug_rect is not None:
                box = cv2.boxPoints(self._debug_rect)
                box = np.int32(box)
                cv2.drawContours(debug, [box], 0, (0, 0, 255), 2)
                center_x, center_y = self._debug_rect[0]
                cv2.circle(debug, (int(center_x), int(center_y)), 5, (255, 0, 0), -1)

            if self._debug_contour is not None:
                cv2.drawContours(debug, [self._debug_contour], -1, (0, 255, 0), 2)
        except Exception:
            pass

        return debug
