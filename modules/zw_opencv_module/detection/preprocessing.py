# -*- coding: utf-8 -*-
"""图像预处理模块"""

from typing import Optional, Tuple
import cv2
import numpy as np


class Preprocessor:
    """图像预处理器"""

    def __init__(self):
        # EdgeDrawing 检测器
        self.ed = cv2.ximgproc.createEdgeDrawing()

        # 形态学核缓存
        self._morph_kernel_cache: Optional[np.ndarray] = None
        self._last_morph_kernel_size: int = 0

        # 边缘预览缓存
        self._last_edge_preview: Optional[np.ndarray] = None

        # 参数（从外部设置）
        self.ed_min_path_length: int = 164
        self.ed_gradient_threshold: int = 90
        self.ed_nfa_validation: bool = True
        self.morph_type: int = 4
        self.morph_kernel: int = 3
        self.morph_iterations: int = 1
        self.blur_kernel: int = 5
        self.blur_sigma: float = 38.0

        # 初始化 EdgeDrawing 参数
        self._update_ed_params()

    def _update_ed_params(self) -> None:
        """更新 EdgeDrawing 参数"""
        ed_params = self.ed.Params()
        ed_params.MinPathLength = self.ed_min_path_length
        ed_params.GradientThresholdValue = self.ed_gradient_threshold
        ed_params.NFAValidation = self.ed_nfa_validation
        self.ed.setParams(ed_params)

    def set_params(
        self,
        ed_min_path_length: int = None,
        ed_gradient_threshold: int = None,
        ed_nfa_validation: bool = None,
        morph_type: int = None,
        morph_kernel: int = None,
        morph_iterations: int = None,
        blur_kernel: int = None,
        blur_sigma: float = None,
    ) -> None:
        """设置预处理参数"""
        if ed_min_path_length is not None:
            self.ed_min_path_length = ed_min_path_length
        if ed_gradient_threshold is not None:
            self.ed_gradient_threshold = ed_gradient_threshold
        if ed_nfa_validation is not None:
            self.ed_nfa_validation = ed_nfa_validation
        if morph_type is not None:
            self.morph_type = morph_type
        if morph_kernel is not None:
            self.morph_kernel = morph_kernel
        if morph_iterations is not None:
            self.morph_iterations = morph_iterations
        if blur_kernel is not None:
            self.blur_kernel = blur_kernel
        if blur_sigma is not None:
            self.blur_sigma = blur_sigma

        self._update_ed_params()

    def resize_frame(
        self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 480)
    ) -> Tuple[np.ndarray, float]:
        """
        缩放图像

        Args:
            frame: 输入图像
            target_size: 目标尺寸 (width, height)

        Returns:
            (缩放后的图像, 缩放比例)
        """
        h, w = frame.shape[:2]
        target_w, target_h = target_size
        scale = min(target_h / h, target_w / w, 1.0)

        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        return small, scale

    def to_gray(self, frame: np.ndarray) -> np.ndarray:
        """转换为灰度图"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def to_hsv(self, frame: np.ndarray) -> np.ndarray:
        """转换为 HSV"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def gaussian_blur(self, gray: np.ndarray) -> np.ndarray:
        """高斯模糊"""
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保为奇数
        return cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.blur_sigma)

    def detect_edges(self, blurred: np.ndarray) -> Optional[np.ndarray]:
        """
        EdgeDrawing 边缘检测

        Args:
            blurred: 模糊后的灰度图

        Returns:
            边缘图像或 None
        """
        try:
            self.ed.detectEdges(blurred)
            edges = self.ed.getEdgeImage()
            return edges
        except cv2.error:
            return None

    def apply_morphology(self, edges: np.ndarray) -> np.ndarray:
        """
        应用形态学操作

        Args:
            edges: 边缘图像

        Returns:
            处理后的图像
        """
        if self.morph_type == 0:  # none
            return edges

        # 使用缓存的 kernel
        if (
            self._morph_kernel_cache is None
            or self._last_morph_kernel_size != self.morph_kernel
        ):
            self._morph_kernel_cache = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
            )
            self._last_morph_kernel_size = self.morph_kernel

        kernel = self._morph_kernel_cache

        if self.morph_type == 1:  # dilate
            return cv2.dilate(edges, kernel, iterations=self.morph_iterations)
        elif self.morph_type == 2:  # erode
            return cv2.erode(edges, kernel, iterations=self.morph_iterations)
        elif self.morph_type == 3:  # open
            return cv2.morphologyEx(
                edges, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations
            )
        elif self.morph_type == 4:  # close
            return cv2.morphologyEx(
                edges, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations
            )

        return edges

    def preprocess(
        self, frame: np.ndarray, target_size: Tuple[int, int] = (640, 480)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """
        完整预处理流程

        Args:
            frame: BGR 输入图像
            target_size: 目标尺寸

        Returns:
            (small, gray, hsv, scale)
        """
        small, scale = self.resize_frame(frame, target_size)
        gray = self.to_gray(small)
        hsv = self.to_hsv(small)
        return small, gray, hsv, scale

    def get_edge_preview(self, frame: np.ndarray = None) -> Optional[np.ndarray]:
        """
        获取边缘预览图像

        优先返回缓存的边缘图像，无缓存时重新计算。

        Args:
            frame: BGR 输入图像（仅在无缓存时使用）

        Returns:
            边缘图像或 None
        """
        if self._last_edge_preview is not None:
            preview = self._last_edge_preview
            self._last_edge_preview = None
            return preview

        if frame is None:
            return None

        # 降采样
        h, w = frame.shape[:2]
        scale = min(320 / h, 320 / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        gray = self.to_gray(small)
        blurred = self.gaussian_blur(gray)

        try:
            edges = self.detect_edges(blurred)
            if edges is not None:
                return edges
        except cv2.error:
            pass

        return np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)

    def set_edge_preview_cache(self, edges: np.ndarray) -> None:
        """缓存边缘预览图像"""
        self._last_edge_preview = edges
