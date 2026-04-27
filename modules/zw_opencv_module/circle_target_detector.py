from typing import Optional, Tuple, List

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
    EDGE_DRAWING_QUADS = (
        "edge_drawing_quads"  # 边缘检测 + 四边形检测
    )
    TEST_LINE_QUAD = (
        "test_line_quad"  # 测试：线段+四边形+透视变换+霍夫圆
    )


class CircleTargetDetector:
    def __init__(self, name: str = "circle_target"):
        self.name = name
        
        
        self.ed = cv2.ximgproc.createEdgeDrawing()
        ed_params = self.ed.Params()
        ed_params.MinPathLength = 50       # 最小边缘段长度
        ed_params.GradientThresholdValue = 20
        ed_params.NFAValidation = True
        self.ed.setParams(ed_params)

        # EdgeDrawing 参数（可调）
        self.ed_min_path_length = 50
        self.ed_gradient_threshold = 20
        self.ed_nfa_validation = True
        
        self.lsd = cv2.createLineSegmentDetector(0)
        
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.transitionMatrix = np.array([[1,0,1,0], [0,1,0,1], [0,0,1,0], [0,0,0,1]], np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0], [0,1,0,0]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)

        self.tracking_initialized = False
        self.lost_frames = 0  # 连续丢失帧计数
        self.max_lost_frames = 10  # 最大允许丢失帧数，超过则重置Kalman滤波器
        
        self.color_ranges = {
            "Red": [
                # 亮红到暗红（包含黑色调的红色）
                (np.array([0, 30, 0]), np.array([10, 255, 255])),
                (np.array([170, 30, 0]), np.array([180, 255, 255])),
            ],
            "Green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            "Blue": [(np.array([100, 50, 50]), np.array([130, 255, 255]))],
            "Black": [(np.array([0, 0, 0]), np.array([180, 255, 50]))],
            "UV": [
                # 紫色高亮范围 (H: 130-150, S: 低-中, V: 高)
                (np.array([130, 20, 200]), np.array([150, 255, 255])),
                # 蓝紫高亮范围 (H: 100-130, S: 低, V: 高)
                (np.array([100, 10, 220]), np.array([130, 100, 255])),
            ],
        }
        self.circle_target = CircleTargets()
        self.min_area_threshold_quad = 150  # 四边形最小面积阈值
        self.min_area_threshold_ellipse = 100  # 椭圆最小面积阈值
        self.min_contour_points = 15  # 椭圆拟合最少需要20个点

        # 椭圆验证参数
        self.max_aspect_ratio = 2.0  # 长短轴比上限
        self.min_circularity = 0.4  # 圆度下限

        self.detect_method = DetectMethod.EDGE_DRAWING_QUADS

        
        self.quad_participation = False
        
        # EdgeDrawing 参数（替代 Canny）
        self.ed_min_path_length = 50
        self.ed_gradient_threshold = 20
        self.ed_nfa_validation = True

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

        # 目标四边形的长宽比（width / height），1.0 表示正方形
        self.quad_aspect_ratio = 1.35  # 默认正方形，可配置为其他值如 1.5, 2.0 等
        self.is_detected_quad = False  # 当前是否检测到四边形
        self.is_uv_spot_detected = False  # 是否检测到UV点
        self.uv_spot_center = None  # UV点中心坐标

        

        
        # UV 点卡尔曼滤波器
        self.uv_kalman = cv2.KalmanFilter(4, 2)  # 4状态(x,y,vx,vy), 2测量(x,y)
        self.uv_kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.uv_kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.uv_kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.uv_kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
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
        params["min_area_threshold_quad"] = self.min_area_threshold_quad
        params["min_area_threshold_ellipse"] = self.min_area_threshold_ellipse
        params["min_contour_points"] = self.min_contour_points

        if method_name == "edge_contour_ellipse":
            params["ed_min_path_length"] = self.ed_min_path_length
            params["ed_gradient_threshold"] = self.ed_gradient_threshold
            params["ed_nfa_validation"] = self.ed_nfa_validation
            params["morph_type"] = self.morph_type
            params["morph_kernel"] = self.morph_kernel
            params["morph_iterations"] = self.morph_iterations
            params["blur_kernel"] = self.blur_kernel
            params["blur_sigma"] = self.blur_sigma
            params["max_aspect_ratio"] = self.max_aspect_ratio
            params["min_circularity"] = self.min_circularity

        elif method_name == "edge_drawing_quads":
            params["ed_min_path_length"] = self.ed_min_path_length
            params["ed_gradient_threshold"] = self.ed_gradient_threshold
            params["ed_nfa_validation"] = self.ed_nfa_validation
            params["morph_type"] = self.morph_type
            params["morph_kernel"] = self.morph_kernel
            params["morph_iterations"] = self.morph_iterations
            params["blur_kernel"] = self.blur_kernel
            params["blur_sigma"] = self.blur_sigma

        elif method_name == "test_line_quad":
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
        if "min_area_threshold_quad" in params:
            self.min_area_threshold_quad = params["min_area_threshold_quad"]
        if "min_area_threshold_ellipse" in params:
            self.min_area_threshold_ellipse = params["min_area_threshold_ellipse"]
        if "min_contour_points" in params:
            self.min_contour_points = params["min_contour_points"]

        method_name = method.value

        if method_name == "edge_contour_ellipse":
            if "ed_min_path_length" in params:
                self.ed_min_path_length = params["ed_min_path_length"]
            if "ed_gradient_threshold" in params:
                self.ed_gradient_threshold = params["ed_gradient_threshold"]
            if "ed_nfa_validation" in params:
                self.ed_nfa_validation = params["ed_nfa_validation"]
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
            if "max_aspect_ratio" in params:
                self.max_aspect_ratio = params["max_aspect_ratio"]
            if "min_circularity" in params:
                self.min_circularity = params["min_circularity"]
            # 更新 EdgeDrawing 参数
            self._update_ed_params()

        elif method_name == "edge_drawing_quads":
            if "ed_min_path_length" in params:
                self.ed_min_path_length = params["ed_min_path_length"]
            if "ed_gradient_threshold" in params:
                self.ed_gradient_threshold = params["ed_gradient_threshold"]
            if "ed_nfa_validation" in params:
                self.ed_nfa_validation = params["ed_nfa_validation"]
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
            # 更新 EdgeDrawing 参数
            self._update_ed_params()

        elif method_name == "test_line_quad":
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
        if "ed_min_path_length" in params:
            self.ed_min_path_length = params["ed_min_path_length"]
        if "ed_gradient_threshold" in params:
            self.ed_gradient_threshold = params["ed_gradient_threshold"]
        if "ed_nfa_validation" in params:
            self.ed_nfa_validation = params["ed_nfa_validation"]
        if "morph_type" in params:
            self.morph_type = params["morph_type"]
        if "morph_kernel" in params:
            self.morph_kernel = params["morph_kernel"]
        if "morph_iterations" in params:
            self.morph_iterations = params["morph_iterations"]
        if "min_area_threshold_quad" in params:
            self.min_area_threshold_quad = params["min_area_threshold_quad"]
        if "min_area_threshold_ellipse" in params:
            self.min_area_threshold_ellipse = params["min_area_threshold_ellipse"]
        if "min_contour_points" in params:
            self.min_contour_points = params["min_contour_points"]
        if "blur_kernel" in params:
            self.blur_kernel = params["blur_kernel"]
        if "blur_sigma" in params:
            self.blur_sigma = params["blur_sigma"]
        if "max_aspect_ratio" in params:
            self.max_aspect_ratio = params["max_aspect_ratio"]
        if "min_circularity" in params:
            self.min_circularity = params["min_circularity"]
        # 更新 EdgeDrawing 参数
        self._update_ed_params()

    def _update_ed_params(self):
        """更新 EdgeDrawing 参数"""
        ed_params = self.ed.Params()
        ed_params.MinPathLength = self.ed_min_path_length
        ed_params.GradientThresholdValue = self.ed_gradient_threshold
        ed_params.NFAValidation = self.ed_nfa_validation
        self.ed.setParams(ed_params)

    def get_edge_preview(self, frame: np.ndarray) -> np.ndarray:
        """
        获取边缘预览图像（用于调试窗口）

        优先返回检测过程中缓存的边缘图像，避免重复计算。
        仅在无缓存时才重新计算。

        Args:
            frame: BGR格式的输入图像（仅在无缓存时使用）

        Returns:
            EdgeDrawing边缘图像（灰度）
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

        # EdgeDrawing 边缘检测
        try:
            self.ed.detectEdges(blurred)
            edges = self.ed.getEdgeImage()
            if edges is not None:
                return edges
        except cv2.error:
            pass

        return np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)

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
        elif self.detect_method == DetectMethod.EDGE_DRAWING_QUADS:
            self._detect_by_edge_drawing_quads(frame, target_color)
        elif self.detect_method == DetectMethod.TEST_LINE_QUAD:
            self._detect_by_test_line_quad(frame, target_color)

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

        min_area_scaled = self.min_area_threshold_ellipse * (scale * scale)

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

    def _check_quad_aspect_ratio(self, quad: np.ndarray, expected_ratio: float, tolerance: float = 0.15) -> bool:
        """
        检查四边形的长宽比是否符合预期

        使用最小外接矩形估算长宽比，考虑透视变形允许较大容差。

        Args:
            quad: 四边形顶点 (4, 1, 2) 或 (4, 2)
            expected_ratio: 预期的长宽比 (宽/高)
            tolerance: 容差比例，默认0.3表示允许30%偏差
    
        Returns:
            是否符合预期长宽比
        """
        ordered = self._order_quad_points(quad)
        points = ordered.reshape(4, 2).astype(np.float32)

        # 使用最小外接矩形估算长宽比
        rect = cv2.minAreaRect(points.reshape(1, -1, 2))
        w, h = rect[1]

        if h <= 0 or w <= 0:
            return False

        measured_ratio = max(w, h) / min(w, h)

        # 如果期望比例是1.0（正方形），直接比较
        if expected_ratio == 1.0:
            return abs(measured_ratio - 1.0) <= tolerance

        # 考虑透视变形，允许较大容差
        return abs(measured_ratio - expected_ratio) / expected_ratio <= tolerance

    def _find_quadrilaterals_from_contours(self, contours, scale: float) -> list:
        """
        从轮廓中筛选四边形

        Args:
            contours: 轮廓列表
            scale: 图像缩放比例

        Returns:
            符合条件的四边形列表，以轮廓形式返回
        """
        quadrilaterals = []
        min_area_scaled = self.min_area_threshold_quad * (scale * scale)

        for contour in contours:
            epsilon = cv2.arcLength(contour, True) * 0.02
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > min_area_scaled:
                    # 检查长宽比
                    if self._check_quad_aspect_ratio(approx, self.quad_aspect_ratio):
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
            masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        best_ellipse = None
        best_contour = None
        best_score = 0

        for cnt in inner_contours:
            # 1. 轮廓点数检查
            if len(cnt) < self.min_contour_points:
                continue

            # 2. 面积检查
            contour_area = cv2.contourArea(cnt)
            if contour_area < self.min_area_threshold_ellipse:
                continue

            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue

            axes = ellipse[1]
            major_axis = max(axes)
            minor_axis = min(axes)

            # 3. 轴长有效性
            if minor_axis <= 0 or major_axis <= 0:
                continue

            # 4. 长短轴比检查
            aspect_ratio = major_axis / minor_axis
            if aspect_ratio > self.max_aspect_ratio:
                continue

            # 5. 圆度检查
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = 4 * np.pi * contour_area / (perimeter * perimeter)
                if circularity < self.min_circularity:
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
        鲁棒椭圆检测：优先快速回退方法 + 卡尔曼滤波平滑
        """
        import time
        t0 = time.time()

        h, w = frame.shape[:2]
        target_w, target_h = 640, 480
        scale = min(target_h / h, target_w / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        t1 = time.time()
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.blur_sigma)
        t2 = time.time()

        result = None
        morphed_edges = None
        detected_quad = None
        detected_quad_center = None

        fallback_result = self._fallback_ellipse_detection(small, gray, scale)
        t3 = time.time()

        if fallback_result is not None:
            ellipse, contour, quad, morphed_edges, quad_center = fallback_result
            detected_quad = quad
            detected_quad_center = quad_center
            if ellipse is not None:
                center = ellipse[0]
                radius = max(ellipse[1]) / 2
                result = (center, radius, quad, quad_center)

        # 卡尔曼滤波平滑
        final_center = None
        final_radius = None
        final_quad = detected_quad
        final_quad_center = detected_quad_center

        if result is not None:
            center, radius, quad, quad_center = result
            final_radius = radius
            smoothed_center = self._kalman_update(center)
            if smoothed_center is not None:
                final_center = smoothed_center
            else:
                final_center = center
        else:
            # 检测失败，尝试预测
            predicted = self._kalman_update(None)
            if predicted is not None:
                final_center = predicted
                final_radius = None
            # 如果有四边形中心但没有椭圆，使用四边形中心
            elif detected_quad_center is not None:
                final_center = detected_quad_center
                final_radius = None

        if final_center is not None:
            color_name = target_color if target_color else 'Red'
            if final_radius is not None:
                target_item = self._create_target_item_from_center(
                    final_center, final_radius, scale, color_name, final_quad
                )
            else:
                # 没有半径时，创建一个默认半径用于可视化
                target_item = self._create_target_item_from_center(
                    final_center, 20.0, scale, color_name, final_quad
                )
            self.circle_target.add_target(target_item)

        t4 = time.time()

        # 缓存边缘图像用于调试（复用 fallback 的结果，避免重复计算）
        if morphed_edges is not None:
            self._last_canny_preview = morphed_edges
        else:
            # 仅在 fallback 未返回边缘时重新计算
            try:
                self.ed.detectEdges(blurred)
                edges = self.ed.getEdgeImage()
                if edges is not None:
                    self._last_canny_preview = self._apply_morphology(edges)
                else:
                    self._last_canny_preview = np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)
            except cv2.error:
                self._last_canny_preview = np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)
        t5 = time.time()

        #print(f"[EDGE] preprocess: {(t2-t1)*1000:.1f}ms, fallback: {(t3-t2)*1000:.1f}ms, kalman: {(t4-t3)*1000:.1f}ms, preview: {(t5-t4)*1000:.1f}ms, total: {(t5-t0)*1000:.1f}ms")

    def _detect_by_edge_drawing_quads(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ):
        """
        四边形检测：EdgeDrawing边缘检测 + 四边形筛选

        仅检测四边形边框，不进行椭圆拟合。输出四边形轮廓和透视中心。
        """
        # 重置 UV 检测状态
        self.is_uv_spot_detected = False
        self.uv_spot_center = None

        h, w = frame.shape[:2]
        target_w, target_h = 640, 480
        scale = min(target_h / h, target_w / w, 1.0)
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)))
        else:
            small = frame

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.blur_sigma)

        # EdgeDrawing 边缘检测
        try:
            self.ed.detectEdges(blurred)
            edges = self.ed.getEdgeImage()
            if edges is None or np.count_nonzero(edges) < 100:
                self._last_canny_preview = np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)
                return
        except cv2.error:
            self._last_canny_preview = np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)
            return

        # 形态学操作
        morphed = self._apply_morphology(edges)
        self._last_canny_preview = morphed

        # 查找轮廓
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # 查找四边形
        quadrilaterals = self._find_quadrilaterals_from_contours(contours, scale)
        if not quadrilaterals:
            return

        # 按面积排序，取最大的四边形
        largest_quads = sorted(quadrilaterals, key=cv2.contourArea, reverse=True)[:3]

        best_quad = None
        best_quad_center = None
        best_area = 0

        for quad in largest_quads:
            quad_center = self._get_quad_center_perspective(quad)
            if quad_center is not None:
                area = cv2.contourArea(quad)
                if area > best_area:
                    best_area = area
                    best_quad = quad
                    best_quad_center = quad_center

        if best_quad is None or best_quad_center is None:
            self.is_detected_quad = False
            return
        
        self.is_detected_quad = True


        # 检测 UV 点（在四边形区域内）
        uv_center = self._detect_uv_spot_with_search_contour(small, best_quad)
        if uv_center is not None:
            # 将缩放后的坐标转换回原始尺寸
            uv_center_orig = (int(uv_center[0] / scale), int(uv_center[1] / scale))
            # 使用卡尔曼滤波平滑
            self.uv_spot_center = self._uv_kalman_update(uv_center_orig)
            self.is_uv_spot_detected = True
        else:
            # 尝试使用卡尔曼预测
            predicted = self._uv_kalman_update(None)
            if predicted is not None:
                self.uv_spot_center = predicted
                self.is_uv_spot_detected = True
            else:
                self.is_uv_spot_detected = False
                self.uv_spot_center = None

        # 卡尔曼滤波平滑
        smoothed_center = self._kalman_update(best_quad_center)
        if smoothed_center is not None:
            final_center = smoothed_center
        else:
            final_center = best_quad_center

        # 创建四边形目标项
        color_name = target_color if target_color else 'Red'
        target_item = self._create_quad_target_item(
            final_center, best_quad, scale, color_name
        )
        self.circle_target.add_target(target_item)
    
    def _detect_uv_spot_with_search_contour(self, frame: np.ndarray, search_contour: np.ndarray = None) -> Optional[Tuple[int, int]]:
        """
        检测UV点（UV点在特定颜色范围内）
        使用亮度加权质心提高精度

        Args:
            frame: BGR格式的输入图像
            search_contour: 搜索轮廓，只在此轮廓内检测UV点

        Returns:
            UV点坐标 (x, y) 或 None
        """
        uv_ranges = self.color_ranges.get('UV', [])
        if not uv_ranges:
            return None

        if search_contour is None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = None
            for lower, upper in uv_ranges:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

            if mask is None:
                return None

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) < 5:  # 最小面积过滤
                    return None
                # 使用亮度加权质心
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
                M = cv2.moments(masked_gray, binaryImage=False)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'] + 0.5)
                    cy = int(M['m01'] / M['m00'] + 0.5)
                    return (cx, cy)
            return None

        x, y, w, h = cv2.boundingRect(search_contour)
        if w <= 0 or h <= 0:
            return None

        roi_frame = frame[y:y+h, x:x+w]
        roi_hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        shifted_contour = search_contour - [x, y]
        cv2.drawContours(roi_mask, [shifted_contour], -1, 255, -1)

        mask = None
        for lower, upper in uv_ranges:
            color_mask = cv2.inRange(roi_hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

        if mask is None:
            return None

        contour_mask = cv2.bitwise_and(mask, roi_mask)

        contours, _ = cv2.findContours(
            contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) < 5:  # 最小面积过滤
                return None
            # 使用亮度加权质心
            gray_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
            masked_gray = cv2.bitwise_and(gray_roi, gray_roi, mask=contour_mask)
            M = cv2.moments(masked_gray, binaryImage=False)
            if M['m00'] > 0:
                # 计算相对于原图的坐标（ROI 偏移）
                cx = int(M['m10'] / M['m00'] + 0.5) + x
                cy = int(M['m01'] / M['m00'] + 0.5) + y
                return (cx, cy)

        return None

    def _uv_kalman_update(self, uv_center: Optional[Tuple[float, float]]) -> Optional[Tuple[int, int]]:
        """
        UV 点卡尔曼滤波平滑

        Args:
            uv_center: 检测到的 UV 点坐标，None 表示丢失

        Returns:
            平滑后的 UV 点坐标
        """
        if uv_center is not None:
            measurement = np.array([[np.float32(uv_center[0])], [np.float32(uv_center[1])]])
            self.uv_kalman.correct(measurement)

        prediction = self.uv_kalman.predict()
        return (int(prediction[0]), int(prediction[1]))

    def _create_quad_target_item(
        self, center: Tuple[float, float], quad: np.ndarray,
        scale: float, color: str
    ) -> CircleTargetItem:
        """
        从四边形创建 CircleTargetItem

        Args:
            center: 四边形中心坐标（缩放后的图像坐标）
            quad: 四边形顶点（缩放后的坐标）
            scale: 缩放比例
            color: 颜色名称

        Returns:
            CircleTargetItem 实例
        """
        center_orig = (int(center[0] / scale), int(center[1] / scale))

        # 将四边形坐标转换回原始尺寸
        quad_orig = (quad.astype(np.float32) / scale).astype(np.int32)

        # 计算四边形面积
        area = cv2.contourArea(quad_orig)

        # 计算边界框
        x, y, w_box, h_box = cv2.boundingRect(quad_orig)

        return CircleTargetItem(
            index=0,
            center_coordinates=center_orig,
            radius=0.0,  # 四边形没有半径概念
            area=area,
            shape_type=ShapeType.QUAD,
            contour_points=quad_orig,
            bounding_box=(x, y, w_box, h_box),
            color=color,
            major_axis=None,
            minor_axis=None,
            quad_points=quad_orig,
        )

    def _detect_by_test_line_quad(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ):
        """
        测试方法：线段+四边形+透视变换+霍夫圆+卡尔曼滤波

        适用于：遮挡、边缘断裂、相机运动等复杂场景
        性能较慢，但更鲁棒
        """
        import time
        t0 = time.time()

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

        result = None

        try:
            # Step A: 提取线段
            t1 = time.time()
            lines = self._extract_lines_ed_lsd(blurred)
            print(f"[TEST] extract_lines: {time.time()-t1:.3f}s, n_lines={len(lines) if lines is not None else 0}")

            # Step B: 从线段寻找四边形（限制线段数量以避免性能问题）
            if lines is not None and len(lines) >= 4 and len(lines) <= 200:
                t2 = time.time()
                quad = self._find_quad_from_lines(lines, small.shape)
                print(f"[TEST] find_quad: {time.time()-t2:.3f}s, quad={'found' if quad is not None else 'None'}")

                if quad is not None:
                    # Step C: 透视变换 + 霍夫圆检测
                    t3 = time.time()
                    circle_result = self._detect_circle_in_quad(blurred, quad)
                    print(f"[TEST] detect_circle: {time.time()-t3:.3f}s")

                    if circle_result is not None:
                        circle_center, radius = circle_result

                        # Step D: 中心对齐验证
                        quad_center = self._compute_quad_center(quad)
                        max_offset = 20.0 * scale
                        if self._validate_center_alignment(quad_center, circle_center, max_offset):
                            result = (circle_center, radius, quad)

            # 回退到快速方法
            if result is None:
                t4 = time.time()
                fallback_result = self._fallback_ellipse_detection(small, gray, scale)
                print(f"[TEST] fallback: {time.time()-t4:.3f}s")
                if fallback_result is not None:
                    ellipse, contour, quad = fallback_result
                    center = ellipse[0]
                    radius = max(ellipse[1]) / 2
                    result = (center, radius, quad)

        except Exception as e:
            print(f"[TEST_LINE_QUAD] Error: {e}")
            import traceback
            traceback.print_exc()
            # 回退到快速方法
            try:
                fallback_result = self._fallback_ellipse_detection(small, gray, scale)
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
            smoothed_center = self._kalman_update(center)
            if smoothed_center is not None:
                final_center = smoothed_center
            else:
                final_center = center
        else:
            predicted = self._kalman_update(None)
            if predicted is not None:
                final_center = predicted
                final_radius = None

        # 生成结果
        if final_center is not None and final_radius is not None:
            color_name = target_color if target_color else 'Red'
            target_item = self._create_target_item_from_center(
                final_center, final_radius, scale, color_name
            )
            self.circle_target.add_target(target_item)

        # 缓存边缘图像（安全检查）
        try:
            self.ed.detectEdges(gray)
            edge_img = self.ed.getEdgeImage()
            if edge_img is not None:
                self._last_canny_preview = edge_img
            else:
                self._last_canny_preview = np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)
        except Exception:
            self._last_canny_preview = np.zeros((small.shape[0], small.shape[1]), dtype=np.uint8)
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

    def _order_quad_points(self, quad: np.ndarray) -> np.ndarray:
        """
        将四边形的四个顶点按顺时针顺序排列：左上、右上、右下、左下

        Args:
            quad: 四边形顶点，形状为 (4, 1, 2) 或 (4, 2)

        Returns:
            有序的四边形顶点，形状为 (4, 1, 2)
        """
        # 展平为 (4, 2)
        points = quad.reshape(4, 2).astype(np.float32)

        # 计算质心
        center = np.mean(points, axis=0)

        # 计算每个点相对于质心的角度
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])

        # 按角度排序（顺时针）
        sorted_indices = np.argsort(angles)
        sorted_points = points[sorted_indices]

        # 找到左上角点（x+y最小）
        sums = sorted_points[:, 0] + sorted_points[:, 1]
        top_left_idx = np.argmin(sums)

        # 重新排列：从左上角开始顺时针
        ordered = np.roll(sorted_points, -top_left_idx, axis=0)

        return ordered.reshape(4, 1, 2).astype(np.float32)

    def _extract_lines_ed_lsd(self, gray: np.ndarray) -> np.ndarray:
        """
        使用 EdgeDrawing + LSD 提取线段（不做合并，性能优化）

        Args:
            gray: 灰度图像

        Returns:
            np.ndarray: (N, 4) 线段端点数组 [x1, y1, x2, y2]，无线段时返回空数组
        """
        try:
            # EdgeDrawing 检测边缘
            self.ed.detectEdges(gray)
            edges = self.ed.getEdgeImage()

            if edges is None or np.count_nonzero(edges) < 100:
                return np.array([])

            # LSD 检测线段
            lines = self.lsd.detect(edges)[0]

            if lines is None or len(lines) == 0:
                return np.array([])

            # 去除多余的维度，LSD返回 (N, 1, 4)
            lines = lines.reshape(-1, 4)

            # 性能优化：不做线段合并，直接返回
            return lines

        except cv2.error:
            return np.array([])

    def _merge_collinear_lines(self, lines: np.ndarray,
                                angle_threshold: float = 5.0,
                                dist_threshold: float = 10.0) -> np.ndarray:
        """
        合并共线的线段

        Args:
            lines: (N, 4) 线段数组
            angle_threshold: 角度阈值（度）
            dist_threshold: 距离阈值（像素）

        Returns:
            合并后的线段数组
        """
        if len(lines) == 0:
            return lines

        merged = []
        used = np.zeros(len(lines), dtype=bool)

        # 计算每条线的角度和长度
        angles = np.arctan2(lines[:, 3] - lines[:, 1], lines[:, 2] - lines[:, 0])
        angles_deg = np.degrees(angles) % 180  # 归一化到 0-180

        for i in range(len(lines)):
            if used[i]:
                continue

            group = [i]
            used[i] = True

            for j in range(i + 1, len(lines)):
                if used[j]:
                    continue

                # 检查角度是否相近（考虑180度对称）
                angle_diff = abs(angles_deg[i] - angles_deg[j])
                angle_diff = min(angle_diff, 180 - angle_diff)

                if angle_diff < angle_threshold:
                    # 检查距离是否相近
                    dist = self._line_to_line_distance(lines[i], lines[j])
                    if dist < dist_threshold:
                        group.append(j)
                        used[j] = True

            # 合并组内的线段
            if len(group) > 1:
                merged_line = self._merge_line_group(lines[group])
                merged.append(merged_line)
            else:
                merged.append(lines[i])

        return np.array(merged) if merged else np.array([])

    def _line_to_line_distance(self, line1: np.ndarray, line2: np.ndarray) -> float:
        """计算两条线段之间的最小距离"""
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        # 计算线段中点
        mid1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        mid2 = np.array([(x3 + x4) / 2, (y3 + y4) / 2])

        # 返回中点距离
        return np.linalg.norm(mid1 - mid2)

    def _merge_line_group(self, lines: np.ndarray) -> np.ndarray:
        """合并一组共线线段为一条线段"""
        # 收集所有端点
        points = lines.reshape(-1, 2)

        # 计算主方向
        dx = lines[:, 2] - lines[:, 0]
        dy = lines[:, 3] - lines[:, 1]
        main_angle = np.arctan2(np.mean(dy), np.mean(dx))

        # 沿主方向投影
        cos_a, sin_a = np.cos(main_angle), np.sin(main_angle)
        projections = points[:, 0] * cos_a + points[:, 1] * sin_a

        # 找到最远的两个点
        min_idx, max_idx = np.argmin(projections), np.argmax(projections)

        return np.array([points[min_idx, 0], points[min_idx, 1],
                        points[max_idx, 0], points[max_idx, 1]])

    def _find_quad_from_lines(self, lines: np.ndarray, img_shape: tuple) -> Optional[np.ndarray]:
        """
        从线段组合中寻找最佳四边形

        Args:
            lines: (N, 4) 线段数组
            img_shape: 图像形状 (h, w)

        Returns:
            四边形顶点 (4, 1, 2) 或 None
        """
        if lines is None or len(lines) < 4:
            return None

        h, w = img_shape[:2]
        min_line_length = 30  # 最小线段长度

        # 过滤短线段
        lengths = np.sqrt((lines[:, 2] - lines[:, 0])**2 + (lines[:, 3] - lines[:, 1])**2)
        valid_mask = lengths >= min_line_length
        valid_lines = lines[valid_mask]

        if len(valid_lines) < 4:
            return None

        # 计算线段角度
        angles = np.arctan2(valid_lines[:, 3] - valid_lines[:, 1],
                           valid_lines[:, 2] - valid_lines[:, 0])
        angles_deg = np.degrees(angles) % 180  # 归一化到 0-180

        # 直方图聚类找主方向
        hist, bin_edges = np.histogram(angles_deg, bins=36, range=(0, 180))  # 5度一个bin

        # 找到两个峰值（应该相差约90度）
        peak_indices = np.argsort(hist)[-4:]  # 取前4个峰值
        peak_angles = (bin_edges[peak_indices] + bin_edges[peak_indices + 1]) / 2

        # 寻找相差约90度的两个方向
        best_pair = None
        best_score = 0

        for i in range(len(peak_angles)):
            for j in range(i + 1, len(peak_angles)):
                diff = abs(peak_angles[i] - peak_angles[j])
                diff = min(diff, 180 - diff)
                if 70 < diff < 110:  # 允许20度误差
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
        def find_extreme_parallel(lines_group, ref_angle):
            if len(lines_group) < 2:
                return None, None

            # 计算每条线到原点的距离
            distances = []
            for line in lines_group:
                x1, y1, x2, y2 = line
                # 使用点到直线距离公式
                mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                # 垂直于线段方向的距离
                perp_angle = ref_angle + 90
                dist = mid_x * np.cos(np.radians(perp_angle)) + mid_y * np.sin(np.radians(perp_angle))
                distances.append(dist)

            distances = np.array(distances)
            min_idx = np.argmin(distances)
            max_idx = np.argmax(distances)

            return lines_group[min_idx], lines_group[max_idx]

        line1a, line1b = find_extreme_parallel(group1, angle1)
        line2a, line2b = find_extreme_parallel(group2, angle2)

        if line1a is None or line1b is None or line2a is None or line2b is None:
            return None

        # 计算四条线的交点
        def line_intersection(line1, line2):
            x1, y1, x2, y2 = line1
            x3, y3, x4, y4 = line2

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-10:
                return None

            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)

            return (x, y)

        corners = []
        for l1 in [line1a, line1b]:
            for l2 in [line2a, line2b]:
                pt = line_intersection(l1, l2)
                if pt is not None:
                    corners.append(pt)

        if len(corners) != 4:
            return None

        # 验证四边形
        corners = np.array(corners, dtype=np.float32)

        # 检查点是否在图像范围内
        if np.any(corners[:, 0] < -w * 0.1) or np.any(corners[:, 0] > w * 1.1):
            return None
        if np.any(corners[:, 1] < -h * 0.1) or np.any(corners[:, 1] > h * 1.1):
            return None

        # 计算凸包面积
        hull = cv2.convexHull(corners.astype(np.float32))
        area = cv2.contourArea(hull)

        min_area_scaled = self.min_area_threshold_quad * 0.5  # 放宽阈值
        if area < min_area_scaled:
            return None

        # 检查长宽比
        rect = cv2.minAreaRect(hull)
        width, height = rect[1]
        if width > 0 and height > 0:
            aspect = max(width, height) / min(width, height)
            if aspect > 5.0:  # 太扁了
                return None

        # 排序顶点
        ordered = self._order_quad_points(corners)

        return ordered
    def _get_quad_center_perspective(self, quad: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        使用透视变换计算四边形的真实几何中心

        当四边形因透视变形变成梯形时，简单平均中心不是真正的几何中心。
        本方法通过透视变换将四边形校正为矩形，计算中心后再投影回原图。

        Args:
            quad: 四边形顶点，形状为 (4, 1, 2) 或 (4, 2)
            来自 _find_quadrilaterals_from_contours 的 approx

        Returns:
            校正后的中心坐标 (x, y)，或 None（输入无效时）
        """
        # 1. 验证输入
        if quad is None or len(quad) != 4:
            return None

        # 2. 排序四边形顶点（左上、右上、右下、左下）
        ordered_quad = self._order_quad_points(quad)
        src_points = ordered_quad.reshape(4, 2).astype(np.float32)

        # 3. 计算四边形的平均边长作为基准尺寸
        edges = []
        for i in range(4):
            p1 = src_points[i]
            p2 = src_points[(i + 1) % 4]
            edges.append(np.linalg.norm(p2 - p1))

        avg_edge = np.mean(edges)

        # 4. 根据长宽比计算目标矩形的尺寸
        # 使用配置的长宽比，默认为 1.0（正方形）
        aspect_ratio = self.quad_aspect_ratio
        if aspect_ratio >= 1.0:
            target_width = int(avg_edge)
            target_height = int(avg_edge / aspect_ratio)
        else:
            target_width = int(avg_edge * aspect_ratio)
            target_height = int(avg_edge)

        target_width = max(50, target_width)
        target_height = max(50, target_height)

        # 5. 定义目标矩形顶点（左上、右上、右下、左下）
        dst_points = np.array([
            [0, 0],
            [target_width, 0],
            [target_width, target_height],
            [0, target_height]
        ], dtype=np.float32)

        # 6. 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

        # 7. 计算矩形中心（透视校正后的中心）
        corrected_center = np.array([[target_width / 2, target_height / 2]], dtype=np.float32)

        # 8. 将中心投影回原图坐标
        src_center = cv2.perspectiveTransform(
            corrected_center.reshape(1, 1, 2), M_inv
        )[0, 0]

        return (float(src_center[0]), float(src_center[1]))
    def _detect_circle_in_quad(self, gray: np.ndarray, quad: np.ndarray) -> Optional[Tuple[Tuple[float, float], float]]:
        """
        在四边形内使用透视变换和霍夫圆检测

        Args:
            gray: 灰度图像
            quad: 四边形顶点 (4, 1, 2)

        Returns:
            (center, radius) 圆心和半径（原图坐标），或 None
        """
        # 排序四边形顶点
        ordered_quad = self._order_quad_points(quad)
        src_points = ordered_quad.reshape(4, 2).astype(np.float32)

        # 计算四边形面积，自适应目标尺寸
        quad_area = cv2.contourArea(src_points)
        target_size = int(np.sqrt(quad_area) * 0.8)  # 目标尺寸约为四边形边长的80%
        target_size = max(100, min(400, target_size))  # 限制在100-400之间

        # 目标正方形顶点
        dst_points = np.array([
            [0, 0],
            [target_size, 0],
            [target_size, target_size],
            [0, target_size]
        ], dtype=np.float32)

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        M_inv = cv2.getPerspectiveTransform(dst_points, src_points)

        # 透视变换
        warped = cv2.warpPerspective(gray, M, (target_size, target_size))

        # 自适应霍夫圆参数
        min_radius = int(target_size * 0.15)
        max_radius = int(target_size * 0.4)

        # 霍夫圆检测
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

        # 取最佳圆（最接近中心）
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

        # 将圆心变换回原图坐标
        src_center = cv2.perspectiveTransform(
            np.array([[[cx, cy]]], dtype=np.float32), M_inv
        )[0, 0]

        # 半径也需要缩放
        # 计算缩放因子（近似）
        scale_factor = np.sqrt(quad_area) / target_size
        src_radius = r * scale_factor

        return ((float(src_center[0]), float(src_center[1])), float(src_radius))

    def _compute_quad_center(self, quad: np.ndarray) -> Tuple[float, float]:
        """计算四边形中心"""
        points = quad.reshape(4, 2)
        center = np.mean(points, axis=0)
        return (float(center[0]), float(center[1]))

    def _validate_center_alignment(self, quad_center: Tuple[float, float],
                                    circle_center: Tuple[float, float],
                                    max_offset: float = 20.0) -> bool:
        """
        验证四边形中心和圆心是否对齐

        Args:
            quad_center: 四边形中心
            circle_center: 圆心
            max_offset: 最大允许偏移（像素）

        Returns:
            是否对齐
        """
        dist = np.sqrt((quad_center[0] - circle_center[0])**2 +
                      (quad_center[1] - circle_center[1])**2)
        return dist <= max_offset

    def _kalman_update(self, measurement: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """
        使用卡尔曼滤波更新位置

        Args:
            measurement: 测量到的位置 (x, y)，None 表示检测失败

        Returns:
            平滑后的位置 (x, y)，未初始化时返回 None
        """
        if measurement is not None:
            # 有测量值
            x, y = measurement
            measurement_array = np.array([[x], [y]], dtype=np.float32)

            if not self.tracking_initialized:
                # 初始化状态
                self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
                self.kf.errorCovPost = np.eye(4, dtype=np.float32)
                self.tracking_initialized = True
                self.lost_frames = 0
            else:
                # 校正
                self.kf.correct(measurement_array)
                self.lost_frames = 0
        else:
            # 无测量值
            if self.tracking_initialized:
                self.lost_frames += 1

                # 超过最大丢失帧数，重置滤波器
                if self.lost_frames > self.max_lost_frames:
                    self.tracking_initialized = False
                    self.lost_frames = 0
                    return None

        # 预测
        if self.tracking_initialized:
            prediction = self.kf.predict()
            return (float(prediction[0, 0]), float(prediction[1, 0]))

        return None

    def _fallback_ellipse_detection(self, small: np.ndarray, gray: np.ndarray,
                                     scale: float) -> Optional[Tuple]:
        """
        回退方法：使用 EdgeDrawing 进行边缘检测 + 轮廓椭圆拟合

        Args:
            small: 缩放后的BGR图像
            gray: 灰度图像
            scale: 缩放比例

        Returns:
            (ellipse, contour, quad, morphed_edges) 或 None
            morphed_edges 用于调试预览，避免重复计算
        """
        kernel_size = self.blur_kernel
        if kernel_size % 2 == 0:
            kernel_size += 1
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.blur_sigma)

        # EdgeDrawing 边缘检测
        try:
            self.ed.detectEdges(blurred)
            edges = self.ed.getEdgeImage()
            if edges is None or np.count_nonzero(edges) < 100:
                return None
        except cv2.error:
            return None

        # 形态学操作
        morphed = self._apply_morphology(edges)

        # 查找轮廓
        contours, _ = cv2.findContours(
            morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            #这里更改了cv2.RETR_EXTERNAL为cv2.RETR_LIST，以获取所有轮廓，增加检测机会，但是可能导致性能下降，和NONE
            #会降低10+FPS.
        )

        # 查找四边形
        quadrilaterals = self._find_quadrilaterals_from_contours(contours, scale)
        if not quadrilaterals:
            return None

        # 按面积排序
        largest_quads = sorted(quadrilaterals, key=cv2.contourArea, reverse=True)[:5]

        best_quad = None
        best_quad_center = None

        for quad in largest_quads:
            quad_center = self._get_quad_center_perspective(quad)
            if quad_center is not None and best_quad is None:
                best_quad = quad
                best_quad_center = quad_center

            result = self._fit_ellipse_in_quad(morphed, quad, small.shape)
            if result is not None:
                ellipse, contour, score = result
                ellipse_center = ellipse[0]
                if quad_center is not None:
                    x, y = quad_center
                    if self._is_center_aligned(quad_center, ellipse_center, quad, max_offset_ratio=0.15):
                        print(f"[Fallback] Quad center: ({x:.1f}, {y:.1f})")
                        return (ellipse, contour, quad, morphed, quad_center)

        # 没有椭圆对齐，返回最佳四边形信息
        if best_quad is not None and best_quad_center is not None:
            #print(f"[Fallback] No ellipse aligned, best quad center: ({best_quad_center[0]:.1f}, {best_quad_center[1]:.1f})")
            return (None, None, best_quad, morphed, best_quad_center)

        return None
    def _is_center_aligned(self, quad_center, ellipse_center, quad, max_offset_ratio=0.15):
        """判断椭圆中心是否位于四边形中心附近"""
        ordered = self._order_quad_points(quad)
        # 计算四边形平均宽度
        w_top = np.linalg.norm(ordered[1] - ordered[0])
        w_bottom = np.linalg.norm(ordered[2] - ordered[3])
        avg_width = (w_top + w_bottom) / 2.0

        dist = np.hypot(ellipse_center[0] - quad_center[0],
                        ellipse_center[1] - quad_center[1])
        return dist <= avg_width * max_offset_ratio
    
    def _create_target_item_from_center(self, center: Tuple[float, float], radius: float,
                                         scale: float, color: str, quad: np.ndarray = None) -> CircleTargetItem:
        """
        从圆心和半径创建 CircleTargetItem

        Args:
            center: 圆心坐标（缩放后的图像坐标）
            radius: 半径（缩放后的图像坐标）
            scale: 缩放比例
            color: 颜色名称
            quad: 四边形顶点（缩放后的坐标）

        Returns:
            CircleTargetItem 实例
        """
        center_orig = (int(center[0] / scale), int(center[1] / scale))
        radius_orig = radius / scale
        area = np.pi * radius_orig * radius_orig

        # 生成圆形轮廓点用于可视化
        contour_points = cv2.ellipse2Poly(
            center_orig,
            (int(radius_orig), int(radius_orig)),
            0, 0, 360, 5
        )

        # 将四边形坐标转换回原始尺寸
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
