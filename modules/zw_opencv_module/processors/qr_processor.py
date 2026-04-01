# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import Optional
from .base import Processor, VisionResult


class QRProcessor(Processor):
    """QR码检测处理器"""

    def __init__(self, name: str = "qr_detect"):
        super().__init__(name)
        self._init_qr_detector()

    def _init_qr_detector(self):
        """初始化QR码检测器"""
        try:
            self.qr_detector = cv2.QRCodeDetector()
            self.qr_supported = True
        except AttributeError:
            print("Warning: cv2.QRCodeDetector not available. Install opencv-contrib-python.")
            self.qr_detector = None
            self.qr_supported = False

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        """
        检测QR码

        Args:
            frame: 输入图像帧
            context: 上下文（未使用）

        Returns:
            VisionResult: QR码检测结果
        """
        if not self.qr_supported or self.qr_detector is None:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message="QR code detector not available",
            )

        if frame is None:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message="Empty frame provided",
            )

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data, bbox, _ = self.qr_detector.detectAndDecode(gray)

            if data and bbox is not None and len(bbox) > 0:
                return VisionResult(
                    task_name=self.name,
                    result_data={"data": data, "bbox": bbox},
                    success=True,
                )
            else:
                return VisionResult(
                    task_name=self.name,
                    success=False,
                    error_message="QR code not found",
                )

        except Exception as e:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message=f"Error processing frame: {str(e)}",
            )

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """
        在帧上绘制QR码检测结果

        Args:
            frame: 输入图像帧
            result: QR码检测结果

        Returns:
            np.ndarray: 绘制后的帧
        """
        if frame is None:
            return frame

        if result.success and result.result_data:
            bbox = result.result_data.get("bbox")
            data = result.result_data.get("data")
            if bbox is not None and data:
                self._draw_qr_code(frame, bbox, data)
        else:
            self._draw_qr_not_found(frame)

        return frame

    def _draw_qr_code(self, frame: np.ndarray, bbox, data: str):
        """在帧上绘制QR码"""
        try:
            bbox = bbox.astype(int)

            if bbox.ndim == 3:
                bbox = bbox.reshape(-1, 2)

            n_points = len(bbox)
            for i in range(n_points):
                start_point = tuple(bbox[i])
                end_point = tuple(bbox[(i + 1) % n_points])
                cv2.line(frame, start_point, end_point, (0, 255, 0), 3)

            if n_points > 0:
                text_position = (bbox[0][0], bbox[0][1] - 10)
                text_size = cv2.getTextSize(data, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    frame,
                    (text_position[0] - 5, text_position[1] - text_size[1] - 5),
                    (text_position[0] + text_size[0] + 5, text_position[1] + 5),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    data,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            cv2.rectangle(frame, (5, 5), (400, 50), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"QR: {data[:30]}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        except Exception as e:
            print(f"Warning: Error drawing QR code: {e}")
            cv2.rectangle(frame, (5, 5), (400, 50), (0, 0, 0), -1)
            cv2.putText(
                frame,
                f"QR: {data[:30]}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    def _draw_qr_not_found(self, frame: np.ndarray):
        """在帧上绘制未找到QR码的提示"""
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame,
            "QR code not found",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
