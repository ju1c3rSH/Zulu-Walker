from typing import Optional
from .circle import CircleTargets, CircleTargetItem, ShapeType
import cv2
import numpy as np
from enum import Enum
from utils.point import Point


class DetectMethod(Enum):
    CONTOUR_ELLIPSE = "contour_ellipse"  # 轮廓检测 + 椭圆拟合
    EDGE_CONTOUR_ELLIPSE = (
        "edge_contour_ellipse"  # 边缘检测 + 椭圆拟合,无需颜色二值化
    )


class CircleTargetDetector:
    def __init__(self, name: str = "circle_target"):
        self.name = name
        self.color_ranges = {
            "Red": [
                # 亮红到暗红（包含黑色调的红色）
                (np.array([0, 30, 0]), np.array([10, 255, 255])),
                (np.array([170, 30, 0]), np.array([180, 255, 255])),
            ],
            "Green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
        }
        self.circle_target = CircleTargets()
        self.min_area_threshold = 150  # 最小面积阈值
        self.min_contour_points = 15  # 椭圆拟合最少需要20个点

        self.detect_method = DetectMethod.EDGE_CONTOUR_ELLIPSE

        # 边缘检测参数
        self.edge_canny_threshold1 = 50
        self.edge_canny_threshold2 = 150

        # 形态学操作参数
        self.morph_type = 1  # 0=none, 1=dilate, 2=erode, 3=open, 4=close
        self.morph_kernel = 3
        self.morph_iterations = 1
        self._morph_kernel_cache = None  # 缓存形态学核

        # 高斯模糊参数
        self.blur_kernel = 5
        self.blur_sigma = 1.0

        # 缓存边缘预览图像
        self._last_canny_preview: Optional[np.ndarray] = None

        self.color_h_ranges = {
            "Red": (0, 15, 165, 180),  # 红色跨越 0 度：(0-15) 和 (165-180)
            "Green": (40, 80, 0, 0),  # 绿色：(40-80）
            "Blue": (100, 130, 0, 0),  # 蓝色：(100-130)
        }
        self.color_s_min = 40  # 饱和度最小值
        self.color_v_min = 50  # 明度最小值
        self.debug_color = False  # 调试模式：打印检测到的 HSV 值

    def set_detect_method(self, method: DetectMethod):
        """设置检测方法"""
        self.detect_method = method

    def get_detect_method(self) -> DetectMethod:
        """获取当前检测方法"""
        return self.detect_method

    @staticmethod
    def get_supported_methods() -> list:
        """返回支持的方法列表"""
        return list(DetectMethod)

    def get_method_params(self, method: DetectMethod) -> dict:
        """
        获取指定检测方法的参数

        Args:
            method: 检测方法

        Returns:
            参数字典
        """
        method_name = method.value
        params = {}

        # 通用参数
        params["min_area_threshold"] = self.min_area_threshold
        params["min_contour_points"] = self.min_contour_points

        if method_name == "edge_contour_ellipse":
            params["edge_canny_threshold1"] = self.edge_canny_threshold1
            params["edge_canny_threshold2"] = self.edge_canny_threshold2
            params["morph_type"] = self.morph_type
            params["morph_kernel"] = self.morph_kernel
            params["morph_iterations"] = self.morph_iterations
            params["blur_kernel"] = self.blur_kernel
            params["blur_sigma"] = self.blur_sigma

        return params

    def set_method_params(self, method: DetectMethod, params: dict):
        """
        设置指定检测方法的参数

        Args:
            method: 检测方法
            params: 参数字典
        """
        # 通用参数
        if "min_area_threshold" in params:
            self.min_area_threshold = params["min_area_threshold"]
        if "min_contour_points" in params:
            self.min_contour_points = params["min_contour_points"]

        method_name = method.value

        if method_name == "edge_contour_ellipse":
            if "edge_canny_threshold1" in params:
                self.edge_canny_threshold1 = params["edge_canny_threshold1"]
            if "edge_canny_threshold2" in params:
                self.edge_canny_threshold2 = params["edge_canny_threshold2"]
            if "morph_type" in params:
                self.morph_type = params["morph_type"]
            if "morph_kernel" in params:
                self.morph_kernel = params["morph_kernel"]
            if "morph_iterations" in params:
                self.morph_iterations = params["morph_iterations"]
            if "blur_kernel" in params:
                self.blur_kernel = params["blur_kernel"]
            if "blur_sigma" in params:
                self.blur_sigma = params["blur_sigma"]

    def update_params(self, params: dict):
        """
        更新边缘检测参数（从调试窗口调用，保持向后兼容）

        Args:
            params: 参数字典
        """
        if "canny_threshold1" in params:
            self.edge_canny_threshold1 = params["canny_threshold1"]
        if "canny_threshold2" in params:
            self.edge_canny_threshold2 = params["canny_threshold2"]
        if "morph_type" in params:
            self.morph_type = params["morph_type"]
        if "morph_kernel" in params:
            self.morph_kernel = params["morph_kernel"]
        if "morph_iterations" in params:
            self.morph_iterations = params["morph_iterations"]
        if "min_area" in params:
            self.min_area_threshold = params["min_area"]
        if "min_contour_points" in params:
            self.min_contour_points = params["min_contour_points"]
        if "blur_kernel" in params:
            self.blur_kernel = params["blur_kernel"]
        if "blur_sigma" in params:
            self.blur_sigma = params["blur_sigma"]

    def get_edge_preview(self, frame: np.ndarray) -> np.ndarray:
        """
        获取边缘预览图像（用于调试窗口）

        优先返回检测过程中缓存的边缘图像，避免重复计算。
        仅在无缓存时才重新计算。

        Args:
            frame: BGR格式的输入图像（仅在无缓存时使用）

        Returns:
            Canny边缘图像（灰度）
        """
        # 优先返回检测过程中缓存的边缘图像
        if self._last_canny_preview is not None:
            preview = self._last_canny_preview
            self._last_canny_preview = None  # 清除缓存，避免下一帧误用
            return preview

        # 无缓存时计算（非 edge_contour_ellipse 方法或首次调用）
        if frame is None:
            return None

        # 降采样
        h, w = frame.shape[:2]
        scale = min(320 / h, 320 / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        # 转换为灰度
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # 高斯模糊
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1  # 确保为奇数
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.blur_sigma)

        # Canny边缘检测
        edges = cv2.Canny(
            blurred,
            self.edge_canny_threshold1,
            self.edge_canny_threshold2,
            apertureSize=3,
        )

        return edges

    def _apply_morphology(self, edges: np.ndarray) -> np.ndarray:
        """
        应用形态学操作（性能优化版）

        Args:
            edges: 边缘图像

        Returns:
            处理后的图像
        """
        if self.morph_type == 0:  # none
            return edges

        # 使用缓存的kernel
        if self._morph_kernel_cache is None or self._morph_kernel_cache.shape[0] != self.morph_kernel:
            self._morph_kernel_cache = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.morph_kernel, self.morph_kernel)
            )
        kernel = self._morph_kernel_cache

        if self.morph_type == 1:  # dilate
            return cv2.dilate(edges, kernel, iterations=self.morph_iterations)
        elif self.morph_type == 2:  # erode
            return cv2.erode(edges, kernel, iterations=self.morph_iterations)
        elif self.morph_type == 3:  # open
            return cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel, iterations=self.morph_iterations)
        elif self.morph_type == 4:  # close
            return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=self.morph_iterations)

        return edges

    def detect_circle_targets(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ) -> CircleTargets:
        """
        检测图像中指定颜色的圆形/椭圆目标
        Args:
            frame: BGR格式的输入图像
            target_color: 目标颜色 ('Red', 'Green', 'Blue', 'Black')，None 表示检测所有颜色
        Returns:
            CircleTargets: 包含所有检测到的目标信息的对象
        """
        self.circle_target.clear()

        if self.detect_method == DetectMethod.CONTOUR_ELLIPSE:
            self._detect_by_contour_ellipse(frame, target_color)
        elif self.detect_method == DetectMethod.EDGE_CONTOUR_ELLIPSE:
            self._detect_by_edge_contour_ellipse(frame, target_color)

        return self.circle_target

    def _detect_by_contour_ellipse(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ):
        """
        轮廓检测 + 椭圆拟合
        先缩小图像，再进行颜色二值化和轮廓检测
        """
        h, w = frame.shape[:2]
        scale = min(320 / h, 320 / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        colors_to_detect = (
            [target_color] if target_color else list(self.color_ranges.keys())
        )

        min_area_scaled = self.min_area_threshold * (scale * scale)

        for color_name in colors_to_detect:
            if color_name not in self.color_ranges:
                continue

            # 获取颜色 mask
            ranges = self.color_ranges[color_name]
            mask = None
            for lower, upper in ranges:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area_scaled:
                    continue

                peri = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

                if len(approx) < self.min_contour_points:
                    continue

                try:
                    ellipse = cv2.fitEllipse(approx)
                except cv2.error:
                    continue

                center_orig = (int(ellipse[0][0] / scale), int(ellipse[0][1] / scale))
                axes_orig = (ellipse[1][0] / scale, ellipse[1][1] / scale)
                angle_orig = ellipse[2]
                radius = max(axes_orig) / 2
                area = 3.14159 * (axes_orig[0] / 2) * (axes_orig[1] / 2)
                # 椭圆面积与轮廓面积可能略有差异!
                if scale < 1.0:
                    # 反向缩放轮廓点
                    contour_orig = (contour * (1.0 / scale)).astype(np.int32)
                else:
                    contour_orig = contour
                target_item = CircleTargetItem(
                    index=0,  # 将在 add_target 时更新
                    center_coordinates=center_orig,
                    radius=radius,
                    area=area,
                    shape_type=ShapeType.ELLIPSE,
                    contour_points=contour_orig,  # 还原到原始尺寸
                    bounding_box=(
                        int((center_orig[0] - axes_orig[0] / 2)),
                        int((center_orig[1] - axes_orig[1] / 2)),
                        int(axes_orig[0]),
                        int(axes_orig[1]),
                    ),
                    color=color_name,
                    major_axis=axes_orig[0],
                    minor_axis=axes_orig[1],
                )
                self.circle_target.add_target(target_item)

    def _find_quadrilaterals(self, contours, scale: float) -> list:
        """
        从轮廓中筛选四边形

        Args:
            contours: 轮廓列表
            scale: 图像缩放比例

        Returns:
            面积大于阈值的四边形列表
        """
        quadrilaterals = []
        min_area_scaled = self.min_area_threshold * (scale * scale)

        for contour in contours:
            epsilon = cv2.arcLength(contour, True) * 0.02
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > min_area_scaled:
                    quadrilaterals.append(approx)

        return quadrilaterals

    def _fit_ellipse_in_quad(self, edges: np.ndarray, quad, img_shape: tuple):
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
            masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_ellipse = None
        best_contour = None
        best_score = 0

        for cnt in inner_contours:
            if len(cnt) < self.min_contour_points:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
                axes = ellipse[1]
                major_axis = max(axes)
                minor_axis = min(axes)

                if minor_axis <= 0 or major_axis <= 0:
                    continue

                aspect_ratio = major_axis / minor_axis
                if aspect_ratio > 3.0:
                    continue

                area = np.pi * ellipse[1][0] * ellipse[1][1] / 4
                ellipse_contour = cv2.ellipse2Poly(
                    (int(ellipse[0][0]), int(ellipse[0][1])),
                    (int(major_axis / 2), int(minor_axis / 2)),
                    int(ellipse[2]),
                    0, 360, 5
                )

                similarity = cv2.matchShapes(cnt, ellipse_contour, cv2.CONTOURS_MATCH_I1, 0)
                score = area / (1 + similarity)

                if score > best_score:
                    best_score = score
                    best_ellipse = ellipse
                    best_contour = cnt
            except cv2.error:
                continue

        if best_ellipse is None:
            return None

        return best_ellipse, best_contour, best_score

    def _create_target_item(self, ellipse, contour, scale: float, color: str) -> CircleTargetItem:
        """
        将椭圆数据转换为 CircleTargetItem

        Args:
            ellipse: cv2.fitEllipse 返回的椭圆 (center, axes, angle)
            contour: 轮廓点
            scale: 图像缩放比例
            color: 颜色名称

        Returns:
            CircleTargetItem 实例
        """
        center_orig = (
            int(ellipse[0][0] / scale),
            int(ellipse[0][1] / scale),
        )
        axes_orig = (ellipse[1][0] / scale, ellipse[1][1] / scale)
        radius = max(axes_orig) / 2
        area = np.pi * axes_orig[0] * axes_orig[1] / 4

        if scale < 1.0:
            contour_orig = (contour * (1.0 / scale)).astype(np.int32)
        else:
            contour_orig = contour

        return CircleTargetItem(
            index=0,
            center_coordinates=center_orig,
            radius=radius,
            area=area,
            shape_type=ShapeType.ELLIPSE,
            contour_points=contour_orig,
            bounding_box=(
                int(center_orig[0] - axes_orig[0] / 2),
                int(center_orig[1] - axes_orig[1] / 2),
                int(axes_orig[0]),
                int(axes_orig[1]),
            ),
            color=color,
            major_axis=axes_orig[0],
            minor_axis=axes_orig[1],
        )

    def _detect_by_edge_contour_ellipse(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ):
        """
        边缘检测 + 椭圆拟合（不使用颜色二值化）
        逻辑：先检测四边形，然后检测四边形内部面积最大的椭圆
        """
        h, w = frame.shape[:2]
        scale = min(320 / h, 320 / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.blur_sigma)

        edges = cv2.Canny(
            blurred,
            self.edge_canny_threshold1,
            self.edge_canny_threshold2,
            apertureSize=3,
        )

        morphed = self._apply_morphology(edges)

        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # 保存形态学处理后的边缘图像，供调试窗口使用
        self._last_canny_preview = morphed

        quadrilaterals = self._find_quadrilaterals(contours, scale)

        for quad in quadrilaterals:
            result = self._fit_ellipse_in_quad(morphed, quad, small.shape)
            if result is None:
                continue

            best_ellipse, best_contour, _ = result
            target_item = self._create_target_item(best_ellipse, best_contour, scale, 'Red')
            self.circle_target.add_target(target_item)

    def _detect_contour_color(
        self, hsv: np.ndarray, contour, img_shape: tuple
    ) -> Optional[str]:
        """
        检测轮廓内部的主要颜色

        Args:
            hsv: HSV 格式图像
            contour: 轮廓
            img_shape: 图像形状 (h, w)

        Returns:
            颜色名称或 None
        """
        # 创建轮廓掩码
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # 计算轮廓内部的平均 HSV 值
        mean_val = cv2.mean(hsv, mask=mask)[:3]
        h, s, v = mean_val

        if self.debug_color:
            print(f"[ColorDebug] HSV: H={h:.1f}, S={s:.1f}, V={v:.1f}")

        # 低明度：黑色
        if v < self.color_v_min:
            return "Black"

        # 低饱和度：灰色/白色，无法判断颜色
        if s < self.color_s_min:
            return None

        # 红色判断（跨越 0 度）
        h_low, h_high, h_low2, h_high2 = self.color_h_ranges["Red"]
        if (h_low <= h < h_high) or (h_low2 <= h <= h_high2):
            return "Red"

        # 绿色判断
        h_low, h_high, _, _ = self.color_h_ranges["Green"]
        if h_low <= h < h_high:
            return "Green"

        # 蓝色判断
        h_low, h_high, _, _ = self.color_h_ranges["Blue"]
        if h_low <= h < h_high:
            return "Blue"

        return None

    def _get_color_mask(self, hsv: np.ndarray, color_name: str) -> np.ndarray:
        """获取指定颜色的mask"""
        ranges = self.color_ranges[color_name]
        mask = None
        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)
        return mask
