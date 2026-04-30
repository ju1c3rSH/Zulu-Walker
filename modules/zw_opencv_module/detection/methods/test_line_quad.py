# -*- coding: utf-8 -*-
"""测试方法：线段 + 四边形 + 透视变换 + 霍夫圆"""

from typing import Optional, Tuple
import cv2
import numpy as np

from ...circle import CircleTargets, CircleTargetItem, ShapeType
from ..detect_method import DetectMethod
from ..preprocessing import Preprocessor
from ..color_detection import ColorDetector
from ..quad_detection import QuadDetector
from ..ellipse_detection import EllipseDetector
from ..tracking import KalmanTracker
from .base import DetectionMethodBase


class TestLineQuadMethod(DetectionMethodBase):
    """测试方法：线段 + 四边形 + 透视变换 + 霍夫圆"""

    def __init__(
        self,
        preprocessor: Preprocessor,
        color_detector: ColorDetector,
        quad_detector: QuadDetector,
        ellipse_detector: EllipseDetector,
        tracker: KalmanTracker,
        params,
    ):
        super().__init__("test_line_quad")
        self.preprocessor = preprocessor
        self.color_detector = color_detector
        self.quad_detector = quad_detector
        self.ellipse_detector = ellipse_detector
        self.tracker = tracker
        self.params = params
        self.circle_target = CircleTargets()
        self._last_edge_preview: Optional[np.ndarray] = None

        # LSD 线段检测器
        self.lsd = cv2.createLineSegmentDetector(0)

    @property
    def method(self) -> DetectMethod:
        return DetectMethod.TEST_LINE_QUAD

    def detect(
        self,
        frame: np.ndarray,
        target_color: Optional[str] = None,
    ) -> CircleTargets:
        self.circle_target.clear()

        h, w = frame.shape[:2]
        scale = min(320 / h, 320 / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        blurred = self.preprocessor.gaussian_blur(gray)

        result = None

        try:
            # Step A: 提取线段
            lines = self._extract_lines_ed_lsd(blurred)

            # Step B: 从线段寻找四边形
            if lines is not None and len(lines) >= 4 and len(lines) <= 200:
                quad = self._find_quad_from_lines(lines, small.shape)

                if quad is not None:
                    # Step C: 透视变换 + 霍夫圆检测
                    circle_result = self._detect_circle_in_quad(blurred, quad)

                    if circle_result is not None:
                        circle_center, radius = circle_result

                        # Step D: 中心对齐验证
                        quad_center = self.quad_detector.compute_quad_center(quad)
                        max_offset = 20.0 * scale
                        if self._validate_center_alignment(quad_center, circle_center, max_offset):
                            result = (circle_center, radius, quad)

            # 回退到快速方法
            if result is None:
                fallback_result = self._fallback_ellipse_detection(small, gray, scale, hsv)
                if fallback_result is not None:
                    ellipse, contour, quad = fallback_result
                    center = ellipse[0]
                    radius = max(ellipse[1]) / 2
                    result = (center, radius, quad)

        except Exception as e:
            print(f"[TEST_LINE_QUAD] Error: {e}")
            try:
                fallback_result = self._fallback_ellipse_detection(small, gray, scale, hsv)
                if fallback_result is not None:
                    ellipse, contour, quad = fallback_result
                    center = ellipse[0]
                    radius = max(ellipse[1]) / 2
                    result = (center, radius, quad)
            except Exception as e2:
                print(f"[TEST_LINE_QUAD] Fallback error: {e2}")

        # 卡尔曼滤波平滑
        final_center = None
        final_radius = None

        if result is not None:
            center, radius, quad = result
            final_radius = radius
            smoothed_center = self.tracker.update(center)
            final_center = smoothed_center if smoothed_center is not None else center
        else:
            predicted = self.tracker.update(None)
            if predicted is not None:
                final_center = predicted

        # 生成结果
        if final_center is not None and final_radius is not None:
            color_name = target_color if target_color else 'Red'
            target_item = self._create_target_item_from_center(
                final_center, final_radius, scale, color_name
            )
            self.circle_target.add_target(target_item)

        # 缓存边缘图像
        self._cache_edge_preview(gray, small.shape)

        return self.circle_target

    def get_edge_preview(self) -> Optional[np.ndarray]:
        preview = self._last_edge_preview
        self._last_edge_preview = None
        return preview

    def _extract_lines_ed_lsd(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """使用 EdgeDrawing + LSD 提取线段"""
        try:
            edges = self.preprocessor.detect_edges(gray)
            if edges is None or np.count_nonzero(edges) < 100:
                return np.array([])

            lines = self.lsd.detect(edges)[0]
            if lines is None or len(lines) == 0:
                return np.array([])

            return lines.reshape(-1, 4)
        except cv2.error:
            return np.array([])

    def _find_quad_from_lines(
        self, lines: np.ndarray, img_shape: tuple
    ) -> Optional[np.ndarray]:
        """从线段组合中寻找最佳四边形"""
        if lines is None or len(lines) < 4:
            return None

        h, w = img_shape[:2]
        min_line_length = 30

        # 过滤短线段
        lengths = np.sqrt(
            (lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2
        )
        valid_mask = lengths >= min_line_length
        valid_lines = lines[valid_mask]

        if len(valid_lines) < 4:
            return None

        # 计算线段角度
        angles = np.arctan2(
            valid_lines[:, 3] - valid_lines[:, 1],
            valid_lines[:, 2] - valid_lines[:, 0]
        )
        angles_deg = np.degrees(angles) % 180

        # 直方图聚类找主方向
        hist, bin_edges = np.histogram(angles_deg, bins=36, range=(0, 180))
        peak_indices = np.argsort(hist)[-4:]
        peak_angles = (bin_edges[peak_indices] + bin_edges[peak_indices + 1]) / 2

        # 寻找相差约90度的两个方向
        best_pair = None
        best_score = 0

        for i in range(len(peak_angles)):
            for j in range(i + 1, len(peak_angles)):
                diff = abs(peak_angles[i] - peak_angles[j])
                diff = min(diff, 180 - diff)
                if 70 < diff < 110:
                    score = hist[peak_indices[i]] + hist[peak_indices[j]]
                    if score > best_score:
                        best_score = score
                        best_pair = (peak_angles[i], peak_angles[j])

        if best_pair is None:
            return None

        angle1, angle2 = best_pair

        # 按角度分组线段
        group1, group2 = [], []
        for i, line in enumerate(valid_lines):
            angle = angles_deg[i]
            diff1 = min(abs(angle - angle1), 180 - abs(angle - angle1))
            diff2 = min(abs(angle - angle2), 180 - abs(angle - angle2))
            if diff1 < diff2:
                group1.append(line)
            else:
                group2.append(line)

        if len(group1) < 2 or len(group2) < 2:
            return None

        # 找每组中距离最远的两条平行线
        line1a, line1b = self._find_extreme_parallel(group1, angle1)
        line2a, line2b = self._find_extreme_parallel(group2, angle2)

        if None in (line1a, line1b, line2a, line2b):
            return None

        # 计算四条线的交点
        corners = []
        for l1 in [line1a, line1b]:
            for l2 in [line2a, line2b]:
                pt = self._line_intersection(l1, l2)
                if pt is not None:
                    corners.append(pt)

        if len(corners) != 4:
            return None

        corners = np.array(corners, dtype=np.float32)

        # 检查点是否在图像范围内
        if np.any(corners[:, 0] < -w * 0.1) or np.any(corners[:, 0] > w * 1.1):
            return None
        if np.any(corners[:, 1] < -h * 0.1) or np.any(corners[:, 1] > h * 1.1):
            return None

        # 计算凸包面积
        hull = cv2.convexHull(corners.astype(np.float32))
        area = cv2.contourArea(hull)

        min_area_scaled = self.params.min_area_threshold_quad * 0.5
        if area < min_area_scaled:
            return None

        # 检查长宽比
        rect = cv2.minAreaRect(hull)
        width, height = rect[1]
        if width > 0 and height > 0:
            aspect = max(width, height) / min(width, height)
            if aspect > 5.0:
                return None

        return self.quad_detector._order_quad_points(corners)

    def _find_extreme_parallel(
        self, lines_group, ref_angle
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """找距离最远的两条平行线"""
        if len(lines_group) < 2:
            return None, None

        distances = []
        for line in lines_group:
            x1, y1, x2, y2 = line
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            perp_angle = ref_angle + 90
            dist = mid_x * np.cos(np.radians(perp_angle)) + mid_y * np.sin(np.radians(perp_angle))
            distances.append(dist)

        distances = np.array(distances)
        min_idx = np.argmin(distances)
        max_idx = np.argmax(distances)

        return lines_group[min_idx], lines_group[max_idx]

    def _line_intersection(
        self, line1: np.ndarray, line2: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """计算两条线段的交点"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)

        return (x, y)

    def _detect_circle_in_quad(
        self, gray: np.ndarray, quad: np.ndarray
    ) -> Optional[Tuple[Tuple[float, float], float]]:
        """在四边形内使用透视变换和霍夫圆检测"""
        ordered_quad = self.quad_detector._order_quad_points(quad)
        src_points = ordered_quad.reshape(4, 2).astype(np.float32)

        quad_area = cv2.contourArea(src_points)
        target_size = int(np.sqrt(quad_area) * 0.8)
        target_size = max(100, min(400, target_size))

        dst_points = np.array([
            [0, 0],
            [target_size, 0],
            [target_size, target_size],
            [0, target_size]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

        warped = cv2.warpPerspective(gray, M, (target_size, target_size))

        min_radius = int(target_size * 0.15)
        max_radius = int(target_size * 0.4)

        circles = cv2.HoughCircles(
            warped,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=target_size // 2,
            param1=50,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius
        )

        if circles is None or len(circles) == 0:
            return None

        center_target = np.array([target_size / 2, target_size / 2])
        best_circle = None
        best_dist = float('inf')

        for circle in circles[0]:
            cx, cy, r = circle
            dist = np.sqrt((cx - center_target[0])**2 + (cy - center_target[1])**2)
            if dist < best_dist:
                best_dist = dist
                best_circle = (cx, cy, r)

        if best_circle is None:
            return None

        cx, cy, r = best_circle

        src_center = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float32), M_inv
        )[0, 0]

        scale_factor = np.sqrt(quad_area) / target_size
        src_radius = r * scale_factor

        return ((float(src_center[0]), float(src_center[1])), float(src_radius))

    def _validate_center_alignment(
        self,
        quad_center: Tuple[float, float],
        circle_center: Tuple[float, float],
        max_offset: float = 20.0,
    ) -> bool:
        """验证四边形中心和圆心是否对齐"""
        dist = np.sqrt(
            (quad_center[0] - circle_center[0])**2 +
            (quad_center[1] - circle_center[1])**2
        )
        return dist <= max_offset

    def _fallback_ellipse_detection(
        self, small: np.ndarray, gray: np.ndarray, scale: float, hsv: np.ndarray
    ) -> Optional[Tuple]:
        """回退检测方法"""
        blurred = self.preprocessor.gaussian_blur(gray)

        edges = self.preprocessor.detect_edges(blurred)
        if edges is None or np.count_nonzero(edges) < 100:
            return None

        morphed = self.preprocessor.apply_morphology(edges)

        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        quadrilaterals = self.quad_detector.find_quadrilaterals(
            contours, scale, hsv, self.color_detector
        )
        if not quadrilaterals:
            return None

        largest_quads = sorted(quadrilaterals, key=cv2.contourArea, reverse=True)[:5]

        for quad in largest_quads:
            result = self.ellipse_detector.fit_ellipse_in_quad(morphed, quad, small.shape)
            if result is not None:
                ellipse, contour, score = result
                return (ellipse, contour, quad)

        return None

    def _cache_edge_preview(self, gray: np.ndarray, shape: tuple) -> None:
        """缓存边缘预览"""
        try:
            edges = self.preprocessor.detect_edges(gray)
            if edges is not None:
                self._last_edge_preview = edges
            else:
                self._last_edge_preview = np.zeros((shape[0], shape[1]), dtype=np.uint8)
        except Exception:
            self._last_edge_preview = np.zeros((shape[0], shape[1]), dtype=np.uint8)

    def _create_target_item_from_center(
        self,
        center: Tuple[float, float],
        radius: float,
        scale: float,
        color: str,
        quad: np.ndarray = None,
    ) -> CircleTargetItem:
        """从圆心和半径创建目标项"""
        center_orig = (int(center[0] / scale), int(center[1] / scale))
        radius_orig = radius / scale
        area = np.pi * radius_orig * radius_orig

        contour_points = cv2.ellipse2Poly(
            center_orig,
            (int(radius_orig), int(radius_orig)),
            0, 0, 360, 5
        )

        quad_orig = None
        if quad is not None:
            quad_orig = (quad.astype(np.float32) / scale).astype(np.int32)

        return CircleTargetItem(
            index=0,
            center_coordinates=center_orig,
            radius=radius_orig,
            area=area,
            shape_type=ShapeType.CIRCLE,
            contour_points=contour_points,
            bounding_box=(
                int(center_orig[0] - radius_orig),
                int(center_orig[1] - radius_orig),
                int(2 * radius_orig),
                int(2 * radius_orig),
            ),
            color=color,
            major_axis=2 * radius_orig,
            minor_axis=2 * radius_orig,
            quad_points=quad_orig,
        )