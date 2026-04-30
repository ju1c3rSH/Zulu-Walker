# -*- coding: utf-8 -*-
"""卡尔曼滤波跟踪模块"""

from typing import Optional, Tuple
import cv2
import numpy as np


class KalmanTracker:
    """卡尔曼滤波跟踪器"""

    def __init__(self):
        # 主目标卡尔曼滤波器
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        # UV 点卡尔曼滤波器
        self.uv_kf = cv2.KalmanFilter(4, 2)
        self.uv_kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32
        )
        self.uv_kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.uv_kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.01
        self.uv_kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        # 跟踪状态
        self.tracking_initialized: bool = False
        self.lost_frames: int = 0
        self.max_lost_frames: int = 10

    def update(
        self, measurement: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[float, float]]:
        """
        使用卡尔曼滤波更新位置

        Args:
            measurement: 测量到的位置 (x, y)，None 表示检测失败

        Returns:
            平滑后的位置 (x, y)，未初始化时返回 None
        """
        if measurement is not None:
            x, y = measurement
            measurement_array = np.array([[x], [y]], dtype=np.float32)

            if not self.tracking_initialized:
                self.kf.statePost = np.array(
                    [[x], [y], [0], [0]], dtype=np.float32
                )
                self.kf.errorCovPost = np.eye(4, dtype=np.float32)
                self.tracking_initialized = True
                self.lost_frames = 0
            else:
                self.kf.correct(measurement_array)
                self.lost_frames = 0
        else:
            if self.tracking_initialized:
                self.lost_frames += 1

                if self.lost_frames > self.max_lost_frames:
                    self.tracking_initialized = False
                    self.lost_frames = 0
                    return None

        if self.tracking_initialized:
            prediction = self.kf.predict()
            return (float(prediction[0, 0]), float(prediction[1, 0]))

        return None

    def uv_update(
        self, uv_center: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[int, int]]:
        """
        UV 点卡尔曼滤波平滑

        Args:
            uv_center: 检测到的 UV 点坐标，None 表示丢失

        Returns:
            平滑后的 UV 点坐标
        """
        if uv_center is not None:
            measurement = np.array(
                [[np.float32(uv_center[0])], [np.float32(uv_center[1])]]
            )
            self.uv_kf.correct(measurement)

        prediction = self.uv_kf.predict()
        return (int(prediction[0]), int(prediction[1]))

    def reset(self) -> None:
        """重置跟踪器"""
        self.tracking_initialized = False
        self.lost_frames = 0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
