
import cv2
import numpy as np
from typing import Optional, Tuple
from collections import deque

from ...models.color import Color


class CargoDetector:
    def __init__(self, name: str = "cargo_detect"):
        self.name = name
        
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.1
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 3.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self._kf_q_base = 0.3
        self._kf_q_vel_base = 0.3

        self.tracking_initialized = False
        self.lost_frames = 0
        self.max_lost_frames = 10
        self._last_predict_time = None

        self.roi_size = 150
        self.max_roi_miss = 5
        self.min_circularity = 0.75
        self.min_area = 100
        self.smooth_window = 5
        self.kernel_open = 5
        self.kernel_close = 7

        self.last_center = None
        self.roi_miss_count = 0
        self._last_mask = None
        self._last_morphed = None

        self._center_history = deque(maxlen=self.smooth_window)
        self._radius_history = deque(maxlen=self.smooth_window)

        self.color_ranges = {
            Color.RED: [
                (np.array([0, 30, 0]), np.array([10, 255, 255])),
                (np.array([170, 30, 0]), np.array([180, 255, 255])),
            ],
            Color.GREEN: [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            Color.BLUE: [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
        }