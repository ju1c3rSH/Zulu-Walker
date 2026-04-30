# -*- coding: utf-8 -*-
"""椭圆检测模块"""

from typing import Optional, Tuple
import cv2
import numpy as np


class EllipseDetector:
    """椭圆检测器"""

    def __init__(self):
        # 参数
        self.min_area_threshold: int = 100
        self.min_contour_points: int = 15
        self.max_aspect_ratio: float = 2.0
        self.min_circularity: float = 0.4

    def fit_ellipse(
        self, contour: np.ndarray
    ) -> Optional[Tuple[cv2.ellipse, np.ndarray, float]]:
        """
        拟合椭圆

        Args:
            contour: 轮廓点

        Returns:
            (ellipse, contour, score) 或 None
        """
        if len(contour) < self.min_contour_points:
            return None

        contour_area = cv2.contourArea(contour)
        if contour_area < self.min_area_threshold:
            return None

        try:
            ellipse = cv2.fitEllipse(contour)
        except cv2.error:
            return None

        axes = ellipse[1]
        major_axis = max(axes)
        minor_axis = min(axes)

        if minor_axis <= 0 or major_axis <= 0:
            return None

        # 长短轴比检查
        aspect_ratio = major_axis / minor_axis
        if aspect_ratio > self.max_aspect_ratio:
            return None

        # 圆度检查
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
            if circularity < self.min_circularity:
                return None

        # 计算相似度分数
        area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
        ellipse_contour = cv2.ellipse2Poly(
            (int(ellipse[0][0]), int(ellipse[0][1])),
            (int(major_axis / 2), int(minor_axis / 2)),
            int(ellipse[2]),
            0, 360, 5
        )

        similarity = cv2.matchShapes(contour, ellipse_contour, cv2.CONTOURS_MATCH_I1, 0)
        score = area / (1 + similarity)

        return ellipse, contour, score

    def fit_ellipse_in_quad(
        self, edges: np.ndarray, quad: np.ndarray, img_shape: tuple
    ) -> Optional[Tuple[cv2.ellipse, np.ndarray, float]]:
        """
        在四边形内部寻找最佳椭圆

        Args:
            edges: 边缘图像
            quad: 四边形轮廓
            img_shape: 图像形状 (h, w)

        Returns:
            (ellipse, contour, score) 或 None
        """
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [quad], 255)

        masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
        inner_contours, _ = cv2.findContours(
            masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        best_ellipse = None
        best_contour = None
        best_score = 0

        for cnt in inner_contours:
            result = self.fit_ellipse(cnt)
            if result is None:
                continue

            ellipse, contour, score = result
            if score > best_score:
                best_score = score
                best_ellipse = ellipse
                best_contour = contour

        if best_ellipse is None:
            return None

        return best_ellipse, best_contour, best_score

    def is_center_aligned(
        self,
        quad_center: Tuple[float, float],
        ellipse_center: Tuple[float, float],
        quad: np.ndarray,
        max_offset_ratio: float = 0.15,
    ) -> bool:
        """
        判断椭圆中心是否位于四边形中心附近

        Args:
            quad_center: 四边形中心
            ellipse_center: 椭圆中心
            quad: 四边形顶点
            max_offset_ratio: 最大偏移比例

        Returns:
            是否对齐
        """
        # 计算四边形平均宽度
        points = quad.reshape(4, 2)
        w_top = np.linalg.norm(points[1] - points[0])
        w_bottom = np.linalg.norm(points[2] - points[3])
        avg_width = (w_top + w_bottom) / 2.0

        dist = np.hypot(
            ellipse_center[0] - quad_center[0],
            ellipse_center[1] - quad_center[1]
        )
        return dist <= avg_width * max_offset_ratio
