# -*- coding: utf-8 -*-
import cv2
import numpy as np
from typing import List, Tuple, Optional


class FrameComposer:
    """Compose multiple frames into a single output image."""

    def __init__(
        self,
        layout: str = "grid",
        output_size: Tuple[int, int] = (1280, 720),
        labels: List[str] = None,
    ):
        """Initialize the composer.

        Args:
            layout: One of "grid", "horizontal", or "vertical".
            output_size: Output image size as (width, height).
            labels: Camera labels drawn on the composed image.
        """
        self.layout = layout
        self.output_size = output_size
        self.labels = labels or []

    def compose(
        self, frames: List[np.ndarray], camera_ids: List[str] = None,
        fps_list: List[float] = None,
    ) -> np.ndarray:
        """Compose frames using the configured layout.

        Args:
            frames: Frames to arrange.
            camera_ids: IDs used as labels, falling back to the configured
                labels when this is empty.
            fps_list: Per-camera FPS values drawn on the output.

        Returns:
            The composed image.
        """
        if not frames:
            return self._create_empty_frame()

        labels = camera_ids or self.labels

        if self.layout == "grid":
            return self._grid_layout(frames, labels, fps_list)
        elif self.layout == "horizontal":
            return self._horizontal_layout(frames, labels, fps_list)
        elif self.layout == "vertical":
            return self._vertical_layout(frames, labels, fps_list)
        else:
            return self._grid_layout(frames, labels, fps_list)

    def _create_empty_frame(self) -> np.ndarray:
        """Create a black frame with a "No Camera" message."""
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

    @staticmethod
    def _draw_fps_in_rect(
        frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, fps: float
    ) -> None:
        if fps <= 0:
            return
        text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        pad = 3
        fx = x1 + pad
        fy = y2 - pad
        cv2.rectangle(
            frame,
            (fx, fy - th - pad),
            (fx + tw + pad, fy + baseline),
            (0, 0, 0), -1,
        )
        cv2.putText(frame, text, (fx, fy), font, scale, (255, 255, 255), thickness)

    def _resize_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize a frame to the configured output size."""
        return cv2.resize(frame, self.output_size)

    def _grid_layout(
        self, frames: List[np.ndarray], labels: List[str] = None,
        fps_list: List[float] = None,
    ) -> np.ndarray:
        """Arrange frames in a grid."""
        n = len(frames)

        # Choose grid dimensions from the frame count.
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

        # Compute each cell's dimensions.
        cell_width = self.output_size[0] // cols
        cell_height = self.output_size[1] // rows

        # Allocate the output canvas.
        output = np.zeros(
            (self.output_size[1], self.output_size[0], 3), dtype=np.uint8
        )

        for i, frame in enumerate(frames):
            if i >= cols * rows:
                break

            row = i // cols
            col = i % cols

            # Skip resizing when the frame already fits the cell.
            if frame.shape[0] == cell_height and frame.shape[1] == cell_width:
                resized = frame
            else:
                resized = cv2.resize(frame, (cell_width, cell_height))

            # Place the frame in its cell.
            y1 = row * cell_height
            y2 = y1 + cell_height
            x1 = col * cell_width
            x2 = x1 + cell_width
            output[y1:y2, x1:x2] = resized

            # Draw the camera label.
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

            # Draw the FPS overlay.
            if fps_list and i < len(fps_list) and fps_list[i] > 0:
                self._draw_fps_in_rect(output, x1, y1, x2, y2, fps_list[i])

        return output

    def _horizontal_layout(
        self, frames: List[np.ndarray], labels: List[str] = None,
        fps_list: List[float] = None,
    ) -> np.ndarray:
        """Arrange frames side by side."""
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

            if fps_list and i < len(fps_list) and fps_list[i] > 0:
                self._draw_fps_in_rect(output, x1, 0, x2, cell_height, fps_list[i])

        return output

    def _vertical_layout(
        self, frames: List[np.ndarray], labels: List[str] = None,
        fps_list: List[float] = None,
    ) -> np.ndarray:
        """Arrange frames top to bottom."""
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

            if fps_list and i < len(fps_list) and fps_list[i] > 0:
                self._draw_fps_in_rect(output, 0, y1, cell_width, y2, fps_list[i])

        return output
