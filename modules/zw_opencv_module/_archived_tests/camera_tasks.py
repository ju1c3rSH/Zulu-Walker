# -*- coding: utf-8 -*-
from typing import Any
from ..camera_stream import CameraStream
from ..task_sequence import TaskSequence

import cv2


class VisionResult:
    def __init__(
        self,
        task_name: str,
        result_data: Any = None,
        success: bool = False,
        error_message: str = "",
    ):
        self.task_name = task_name
        self.result_data = result_data
        self.success = success
        self.error_message = error_message


class CameraTasks:
    def __init__(self, task_sequence: TaskSequence, camera_stream: CameraStream):
        self.task_sequence = task_sequence
        self.current_task = None
        self.camera_stream = camera_stream
        self.current_frame = None

        # 检查QRCodeDetector是否可用
        try:
            self.qr_detector = cv2.QRCodeDetector()
            self.qr_supported = True
        except AttributeError:
            print(
                "Warning: cv2.QRCodeDetector not available. Install opencv-contrib-python."
            )
            self.qr_detector = None
            self.qr_supported = False

    def _get_frame(self):
        frame = self.camera_stream.read_frame()
        if frame is not None:
            self.current_frame = frame
            return frame
        else:
            raise RuntimeError("Failed to read frame from camera stream.")

    def _get_current_frame(self):
        if self.current_frame is not None:
            return self.current_frame
        else:
            raise RuntimeError("No current frame available.")

    def _draw_qr_code(self, frame, bbox, data):
        """在帧上绘制QR码检测结果"""
        try:
            # 确保bbox是整数坐标并处理形状
            bbox = bbox.astype(int)

            # 处理不同的bbox形状：可能是 (n, 1, 2) 或 (n, 2)
            if bbox.ndim == 3:
                # 形状为 (n, 1, 2)，压缩到 (n, 2)
                bbox = bbox.reshape(-1, 2)

            # 绘制QR码边框（四边形）
            n_points = len(bbox)
            for i in range(n_points):
                start_point = tuple(bbox[i])
                end_point = tuple(bbox[(i + 1) % n_points])
                cv2.line(frame, start_point, end_point, (0, 255, 0), 3)

            # 计算文字位置（在QR码上方）
            if n_points > 0:
                text_position = (bbox[0][0], bbox[0][1] - 10)

                # 添加文字背景（提高可读性）
                text_size = cv2.getTextSize(data, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(
                    frame,
                    (text_position[0] - 5, text_position[1] - text_size[1] - 5),
                    (text_position[0] + text_size[0] + 5, text_position[1] + 5),
                    (0, 0, 0),
                    -1,
                )

                # 绘制文字
                cv2.putText(
                    frame,
                    data,
                    text_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            # 在左上角也显示QR码内容
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
            # 仍然在左上角显示QR码内容
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

    def _draw_qr_not_found(self, frame):
        """在帧上绘制未找到QR码的提示"""
        # 添加半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (350, 55), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 绘制文字
        cv2.putText(
            frame,
            "QR code not found",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    def execute_read_qr(self) -> VisionResult:
        if not self.qr_supported or self.qr_detector is None:
            return VisionResult(
                task_name="READ_QR",
                success=False,
                error_message="QR code detector not available",
            )

        try:
            frame = self._get_frame()
            if frame is None:
                return VisionResult(
                    task_name="READ_QR",
                    success=False,
                    error_message="Camera read failed",
                )

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data, bbox, _ = self.qr_detector.detectAndDecode(gray)

            if data and bbox is not None:
                # 确保bbox是正确形状
                if len(bbox) > 0:
                    self._draw_qr_code(frame, bbox, data)
                return VisionResult(task_name="READ_QR", result_data=data, success=True)
            else:
                self._draw_qr_not_found(frame)
                return VisionResult(
                    task_name="READ_QR",
                    success=False,
                    error_message="QR code not found",
                )

        except RuntimeError as e:
            # 来自_get_frame或_get_current_frame的特定错误
            return VisionResult(
                task_name="READ_QR",
                success=False,
                error_message=f"Camera error: {str(e)}",
            )
        except Exception as e:
            return VisionResult(
                task_name="READ_QR",
                success=False,
                error_message=f"Unexpected error: {str(e)}",
            )

    def process_frame_for_qr(self, frame) -> VisionResult:
        """
        处理外部传入的帧进行QR码检测

        Args:
            frame: 要处理的图像帧

        Returns:
            VisionResult: QR检测结果
        """
        if not self.qr_supported or self.qr_detector is None:
            return VisionResult(
                task_name="READ_QR",
                success=False,
                error_message="QR code detector not available",
            )

        if frame is None:
            return VisionResult(
                task_name="READ_QR", success=False, error_message="Empty frame provided"
            )

        try:
            # 更新当前帧

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            data, bbox, _ = self.qr_detector.detectAndDecode(gray)

            """做好收尾工作，并且return数据"""
            if data and bbox is not None:
                if len(bbox) > 0:
                    self._draw_qr_code(frame, bbox, data)
                self.current_frame = frame
                return VisionResult(task_name="READ_QR", result_data=data, success=True)
            else:
                self._draw_qr_not_found(frame)
                self.current_frame = frame
                return VisionResult(
                    task_name="READ_QR",
                    success=False,
                    error_message="QR code not found",
                )

        except Exception as e:
            return VisionResult(
                task_name="READ_QR",
                success=False,
                error_message=f"Error processing frame: {str(e)}",
            )

    def remove_shadow(frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        v_corrected = clahe.apply(v)
        hsv_corrected = cv2.merge((h, s, v_corrected))
        return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)


