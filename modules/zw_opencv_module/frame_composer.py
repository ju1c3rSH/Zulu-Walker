# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import List, Tuple, Optional


class FrameComposer:
    """画面融合器，将多个帧融合为一个画面"""

    def __init__(
        self,
        layout: str = "grid",
        output_size: Tuple[int, int] = (1280, 720),
        labels: List[str] = None,
    ):
        """
        初始化画面融合器

        Args:
            layout: 布局模式 ("grid", "horizontal", "vertical")
            output_size: 输出画面尺寸 (width, height)
            labels: 相机标签列表
        """
        self.layout = layout
        self.output_size = output_size
        self.labels = labels or []

    def compose(
        self, frames: List[np.ndarray], camera_ids: List[str] = None
    ) -> np.ndarray:
        """
        将多个帧融合为一个画面

        Args:
            frames: 帧列表
            camera_ids: 相机ID列表（用于标签）

        Returns:
            np.ndarray: 融合后的画面
        """
        if not frames:
            return self._create_empty_frame()

        if len(frames) == 1:
            return self._resize_single_frame(frames[0])

        labels = camera_ids or self.labels

        if self.layout == "grid":
            return self._grid_layout(frames, labels)
        elif self.layout == "horizontal":
            return self._horizontal_layout(frames, labels)
        elif self.layout == "vertical":
            return self._vertical_layout(frames, labels)
        else:
            return self._grid_layout(frames, labels)

    def _create_empty_frame(self) -> np.ndarray:
        """创建空白帧"""
        frame = np.zeros((self.output_size[1], self.output_size[0], 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "No Camera",
            (self.output_size[0] // 2 - 50, self.output_size[1] // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        return frame

    def _resize_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """调整单个帧的尺寸"""
        return cv2.resize(frame, self.output_size)

    def _grid_layout(
        self, frames: List[np.ndarray], labels: List[str] = None
    ) -> np.ndarray:
        """网格布局"""
        n = len(frames)

        # 计算网格尺寸
        if n <= 1:
            cols, rows = 1, 1
        elif n <= 2:
            cols, rows = 2, 1
        elif n <= 4:
            cols, rows = 2, 2
        elif n <= 6:
            cols, rows = 3, 2
        elif n <= 9:
            cols, rows = 3, 3
        else:
            cols, rows = 4, 3

        # 计算每个单元格的尺寸
        cell_width = self.output_size[0] // cols
        cell_height = self.output_size[1] // rows

        # 创建输出画面
        output = np.zeros(
            (self.output_size[1], self.output_size[0], 3), dtype=np.uint8
        )

        for i, frame in enumerate(frames):
            if i >= cols * rows:
                break

            row = i // cols
            col = i % cols

            # 调整帧尺寸
            resized = cv2.resize(frame, (cell_width, cell_height))

            # 放置到输出画面
            y1 = row * cell_height
            y2 = y1 + cell_height
            x1 = col * cell_width
            x2 = x1 + cell_width
            output[y1:y2, x1:x2] = resized

            # 添加标签
            if labels and i < len(labels):
                cv2.rectangle(output, (x1 + 5, y1 + 5), (x1 + 100, y1 + 30), (0, 0, 0), -1)
                cv2.putText(
                    output,
                    labels[i],
                    (x1 + 10, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

        return output

    def _horizontal_layout(
        self, frames: List[np.ndarray], labels: List[str] = None
    ) -> np.ndarray:
        """水平布局"""
        n = len(frames)
        cell_width = self.output_size[0] // n
        cell_height = self.output_size[1]

        output = np.zeros(
            (self.output_size[1], self.output_size[0], 3), dtype=np.uint8
        )

        for i, frame in enumerate(frames):
            x1 = i * cell_width
            x2 = x1 + cell_width

            resized = cv2.resize(frame, (cell_width, cell_height))
            output[:, x1:x2] = resized

            if labels and i < len(labels):
                cv2.rectangle(output, (x1 + 5, 5), (x1 + 100, 30), (0, 0, 0), -1)
                cv2.putText(
                    output,
                    labels[i],
                    (x1 + 10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

        return output

    def _vertical_layout(
        self, frames: List[np.ndarray], labels: List[str] = None
    ) -> np.ndarray:
        """垂直布局"""
        n = len(frames)
        cell_width = self.output_size[0]
        cell_height = self.output_size[1] // n

        output = np.zeros(
            (self.output_size[1], self.output_size[0], 3), dtype=np.uint8
        )

        for i, frame in enumerate(frames):
            y1 = i * cell_height
            y2 = y1 + cell_height

            resized = cv2.resize(frame, (cell_width, cell_height))
            output[y1:y2, :] = resized

            if labels and i < len(labels):
                cv2.rectangle(output, (5, y1 + 5), (100, y1 + 30), (0, 0, 0), -1)
                cv2.putText(
                    output,
                    labels[i],
                    (10, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

        return output
