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
    def __init__(self, frame_w: int = 640, frame_h: int = 640):
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._debug_binary = None
        self._debug_contour = None
        self._debug_rect = None
        self._empty_debug = None

    def calibrate(self, frame_bgr: np.ndarray, ball_px: tuple = None) -> RailCalibration:
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        except Exception:
            return RailCalibration(calibrated=False)

        rail_info = self._detect_rail_contour(gray)
        if rail_info is None:
            rail_info = self._detect_rail_edges(gray)

        if rail_info is None:
            return RailCalibration(calibrated=False)

        center_x, center_y, angle_rad = rail_info

        if ball_px is not None and len(ball_px) >= 2:
            origin_x, origin_y = float(ball_px[0]), float(ball_px[1])
        else:
            origin_x, origin_y = center_x, center_y

        return RailCalibration(
            origin_x=origin_x,
            origin_y=origin_y,
            angle_rad=angle_rad,
            calibrated=True,
        )

    def _detect_rail_contour(self, gray: np.ndarray):
        try:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            if np.std(gray) < 30:
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        except Exception:
            return None

        self._debug_binary = binary

        try:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        except Exception:
            return None

        if not contours:
            return None

        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)

        min_area = self._frame_w * self._frame_h * 0.04
        if area < min_area:
            return None

        try:
            rect = cv2.minAreaRect(largest_contour)
        except Exception:
            return None

        (center_x, center_y), (w, h), angle = rect

        if w < 1 or h < 1:
            return None
        ratio = max(w, h) / min(w, h)
        if ratio < 2.5:
            return None

        fw = gray.shape[1]
        fh = gray.shape[0]
        if center_x < fw * 0.1 or center_x > fw * 0.9:
            return None
        if center_y < fh * 0.1 or center_y > fh * 0.9:
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
            canny = cv2.Canny(blurred, 50, 150)
        except Exception:
            return None

        try:
            lines = cv2.HoughLinesP(canny, rho=1, theta=math.pi / 180,
                                    threshold=50, minLineLength=150,
                                    maxLineGap=50)
        except Exception:
            return None

        if lines is None or len(lines) == 0:
            return None

        midpoints_x = []
        midpoints_y = []
        angles = []
        max_angle_rad = math.radians(15)

        for line in lines:
            x1, y1, x2, y2 = line[0]
            line_angle = math.atan2(y2 - y1, x2 - x1)
            if abs(line_angle) < max_angle_rad:
                mx = (x1 + x2) / 2.0
                my = (y1 + y2) / 2.0
                midpoints_x.append(mx)
                midpoints_y.append(my)
                angles.append(line_angle)

        if len(midpoints_x) < 3:
            return None

        center_x = float(np.mean(midpoints_x))
        center_y = float(np.mean(midpoints_y))
        angle_rad = float(np.mean(angles))

        self._debug_binary = canny

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
