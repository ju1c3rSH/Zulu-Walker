"""
Pendulum rail calibrator.
Detects the pendulum rail axis from a camera frame using OpenCV,
providing a calibrated reference line for steel ball position projection.
"""
import cv2
import numpy as np
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class RailCalibration:
    origin_x: float
    origin_y: float
    angle_rad: float
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
                 edge_angle_max_deg: float = 15):
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._binary_threshold = binary_threshold
        self._min_contour_area_ratio = min_contour_area_ratio
        self._min_aspect_ratio = min_aspect_ratio
        self._canny_low = canny_low
        self._canny_high = canny_high
        self._hough_threshold = hough_threshold
        self._hough_min_line_len = hough_min_line_len
        self._edge_angle_max_deg = edge_angle_max_deg
        self._debug_binary = None
        self._debug_contour = None
        self._debug_rect = None
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

    def get_last_diagnostics(self) -> dict:
        return dict(self._diag)

    def calibrate(self, frame_bgr: np.ndarray, ball_px: tuple = None) -> RailCalibration:
        self._diag = {}
        self._debug_binary = None
        self._debug_contour = None
        self._debug_rect = None

        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            self._diag['fail_reason'] = 'gray_conversion'
            return RailCalibration(calibrated=False)

        self._diag['gray_mean'] = float(np.mean(gray))
        self._diag['gray_std'] = float(np.std(gray))

        rail_info = self._detect_rail_contour(gray)
        if rail_info is None:
            self._diag['used_fallback'] = True
            rail_info = self._detect_rail_edges(gray)
        else:
            self._diag['used_fallback'] = False

        if rail_info is None:
            if 'fail_reason' not in self._diag:
                self._diag['fail_reason'] = 'no_detection'
            return RailCalibration(calibrated=False)

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
        min_area = self._frame_w * self._frame_h * self._min_contour_area_ratio

        self._diag['contour_area'] = int(area)
        self._diag['min_area_limit'] = int(min_area)
        self._diag['contour_area_ok'] = area >= min_area

        if area < min_area:
            self._diag['fail_reason'] = 'min_area'
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
