# -*- coding: utf-8 -*-
"""四边形检测模块"""

from typing import Optional, Tuple, List
import cv2
import numpy as np


class QuadDetector:
    """四边形检测器"""

    def __init__(self):
        # 参数
        self.min_area_threshold: int = 1680
        self.quad_aspect_ratio: float = 1.45
        self.enable_color_filter: bool = True

    def find_quadrilaterals(
        self,
        contours: List[np.ndarray],
        scale: float,
        hsv: np.ndarray = None,
        color_detector=None,
    ) -> List[np.ndarray]:
        """
        从轮廓中筛选四边形

        Args:
            contours: 轮廓列表
            scale: 图像缩放比例
            hsv: HSV 格式图像（用于颜色过滤）
            color_detector: ColorDetector 实例

        Returns:
            符合条件的四边形列表
        """
        quadrilaterals = []
        min_area_scaled = self.min_area_threshold * (scale * scale)

        for contour in contours:
            epsilon = cv2.arcLength(contour, True) * 0.02
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) != 4:
                continue

            area = cv2.contourArea(approx)
            if area <= min_area_scaled:
                continue

            # 凸包检测
            if not cv2.isContourConvex(approx):
                continue

            # 颜色过滤（可选）
            if self.enable_color_filter and hsv is not None and color_detector is not None:
                color = color_detector.detect_contour_color(hsv, approx, hsv.shape)
                if color != "Black":
                    continue

            # 角度约束
            if not self._check_angle_constraint(approx):
                continue

            # 长宽比检查
            if self._check_quad_aspect_ratio(approx, self.quad_aspect_ratio):
                quadrilaterals.append(approx)

        return quadrilaterals

    def _check_angle_constraint(
        self,
        approx: np.ndarray,
        angle_threshold: float = 15.0,
        right_angle_tolerance: float = 10.0,
    ) -> bool:
        """
        检查四边形的角度约束

        如果有两个角约等于90°，而另外有一个角大于90°+阈值，则排除。
        这可以过滤掉正对摄像头拍摄的非矩形四边形。

        Args:
            approx: 四边形的四个顶点
            angle_threshold: 大于90°的容差阈值
            right_angle_tolerance: 判定为直角的容差

        Returns:
            True: 通过检查，False: 不通过
        """
        angles = []
        pts = approx.reshape(4, 2)

        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            p3 = pts[(i + 2) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.degrees(np.arccos(cos_angle))
            angles.append(angle)

        # 统计接近90°的角
        right_angle_count = sum(1 for a in angles if abs(a - 90) <= right_angle_tolerance)

        # 统计明显大于90°的角
        obtuse_count = sum(1 for a in angles if a > 90 + angle_threshold)

        # 如果有两个角接近90°，且至少有一个角明显大于90°，则排除
        if right_angle_count >= 2 and obtuse_count >= 1:
            return False

        return True

    def _check_quad_aspect_ratio(
        self, quad: np.ndarray, expected_ratio: float, tolerance: float = 0.28
    ) -> bool:
        """
        检查透视变换后的四边形是否符合预期的原始矩形宽高比

        使用对边平均长度来估算原始矩形的宽高比，
        这种方法对透视变形有较好的鲁棒性。

        Args:
            quad: 四边形顶点 (4, 1, 2) 或 (4, 2)
            expected_ratio: 预期的原始矩形长宽比 (宽/高)
            tolerance: 相对容差

        Returns:
            是否符合预期长宽比
        """
        ordered = self._order_quad_points(quad)
        points = ordered.reshape(4, 2).astype(np.float32)

        tl, tr, br, bl = points

        # 宽度：上下两边的平均长度
        top_width = np.linalg.norm(tr - tl)
        bottom_width = np.linalg.norm(br - bl)
        estimated_width = (top_width + bottom_width) / 2

        # 高度：左右两边的平均长度
        left_height = np.linalg.norm(bl - tl)
        right_height = np.linalg.norm(br - tr)
        estimated_height = (left_height + right_height) / 2

        if estimated_height <= 0 or estimated_width <= 0:
            return False

        measured_ratio = estimated_width / estimated_height

        if expected_ratio == 1.0:
            return abs(measured_ratio - 1.0) <= tolerance

        # 检查正常方向
        diff_normal = abs(measured_ratio - expected_ratio) / expected_ratio
        if diff_normal <= tolerance:
            return True

        # 检查90°旋转的情况
        inv_expected_ratio = 1.0 / expected_ratio
        diff_rotated = abs(measured_ratio - inv_expected_ratio) / inv_expected_ratio
        return diff_rotated <= tolerance

    def _order_quad_points(self, quad: np.ndarray) -> np.ndarray:
        """
        将四边形的四个顶点按顺时针顺序排列：左上、右上、右下、左下

        Args:
            quad: 四边形顶点，形状为 (4, 1, 2) 或 (4, 2)

        Returns:
            有序的四边形顶点，形状为 (4, 1, 2)
        """
        points = quad.reshape(4, 2).astype(np.float32)
        center = np.mean(points, axis=0)

        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        # 找到左上角点（x+y最小）
        sums = sorted_points[:, 0] + sorted_points[:, 1]
        top_left_idx = np.argmin(sums)

        ordered = np.roll(sorted_points, -top_left_idx, axis=0)
        return ordered.reshape(4, 1, 2).astype(np.float32)

    def get_quad_center_perspective(
        self, quad: np.ndarray, quad_aspect_ratio: float = 1.45
    ) -> Optional[Tuple[float, float]]:
        """
        使用透视变换计算四边形的真实几何中心

        Args:
            quad: 四边形顶点
            quad_aspect_ratio: 长宽比

        Returns:
            校正后的中心坐标 (x, y)，或 None
        """
        if quad is None or len(quad) != 4:
            return None

        ordered_quad = self._order_quad_points(quad)
        src_points = ordered_quad.reshape(4, 2).astype(np.float32)

        # 计算平均边长
        edges = []
        for i in range(4):
            p1 = src_points[i]
            p2 = src_points[(i + 1) % 4]
            edges.append(np.linalg.norm(p2 - p1))

        avg_edge = np.mean(edges)

        # 根据长宽比计算目标矩形尺寸
        aspect_ratio = quad_aspect_ratio
        if aspect_ratio >= 1.0:
            target_width = int(avg_edge)
            target_height = int(avg_edge / aspect_ratio)
        else:
            target_width = int(avg_edge * aspect_ratio)
            target_height = int(avg_edge)

        target_width = max(50, target_width)
        target_height = max(50, target_height)

        dst_points = np.array(
            [
                [0, 0],
                [target_width, 0],
                [target_width, target_height],
                [0, target_height],
            ],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

        corrected_center = np.array(
            [[target_width / 2, target_height / 2]], dtype=np.float32
        )

        src_center = cv2.perspectiveTransform(
            corrected_center.reshape(1, 1, 2), M_inv
        )[0, 0]

        return (float(src_center[0]), float(src_center[1]))

    def compute_quad_center(self, quad: np.ndarray) -> Tuple[float, float]:
        """计算四边形中心（简单平均）"""
        points = quad.reshape(4, 2)
        center = np.mean(points, axis=0)
        return (float(center[0]), float(center[1]))