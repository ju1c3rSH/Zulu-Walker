from typing import Optional
from .circle import CircleTargets, CircleTargetItem, ShapeType
import cv2
import numpy as np
from skimage import transform
from enum import Enum


class DetectMethod(Enum):
    CONTOUR_ELLIPSE = "contour_ellipse"  # 轮廓检测 + 椭圆拟合
    HOUGH_ELLIPSE = "hough_ellipse"  # 霍夫椭圆变换
    HOUGH_CIRCLE = "hough_circle"  # 霍夫圆变换


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

        self.detect_method = DetectMethod.CONTOUR_ELLIPSE

        # 霍夫圆参数
        self.hough_circle_dp = 1.2
        self.hough_circle_min_dist = 20
        self.hough_circle_param1 = 50
        self.hough_circle_param2 = 30
        self.hough_circle_min_radius = 5
        self.hough_circle_max_radius = 100

        # 霍夫椭圆参数
        self.hough_ellipse_accuracy = 20
        self.hough_ellipse_threshold = 250
        self.hough_ellipse_min_size = 30
        self.hough_ellipse_max_size = 300

    def set_detect_method(self, method: DetectMethod):
        """设置检测方法"""
        self.detect_method = method

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
        elif self.detect_method == DetectMethod.HOUGH_ELLIPSE:
            self._detect_by_hough_ellipse(frame, target_color)
        elif self.detect_method == DetectMethod.HOUGH_CIRCLE:
            self._detect_by_hough_circle(frame, target_color)

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

    def _detect_by_hough_ellipse(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ):
        """
        霍夫椭圆变换（skimage）
        注意：skimage的hough_ellipse是纯Python实现，非常慢
        已优化：先缩小图像再检测
        """
        try:
            # 步骤 1: 缩小图像
            h, w = frame.shape[:2]
            scale = min(320 / h, 320 / w, 1.0)
            if scale < 1.0:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small = frame

            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)

            colors_to_detect = (
                [target_color] if target_color else list(self.color_ranges.keys())
            )

            # 缩放后的尺寸参数
            min_size_scaled = max(int(self.hough_ellipse_min_size * scale), 10)
            max_size_scaled = int(self.hough_ellipse_max_size * scale)

            for color_name in colors_to_detect:
                if color_name not in self.color_ranges:
                    continue
                mask = self._get_color_mask(hsv, color_name)
                masked_edges = cv2.bitwise_and(edges, edges, mask=mask)

                # 检查边缘图像是否有效
                if masked_edges.max() == 0:
                    continue

                ellipses = transform.hough_ellipse(
                    masked_edges,
                    accuracy=self.hough_ellipse_accuracy,
                    threshold=self.hough_ellipse_threshold,
                    min_size=min_size_scaled,
                    max_size=max_size_scaled,
                )
                if ellipses is not None and len(ellipses) > 0:
                    # 按累加器值排序，只取前几个最佳结果
                    ellipses = sorted(ellipses, key=lambda e: e[-1], reverse=True)[:5]
                    for e in ellipses:
                        cy, cx, a, b, angle, acc = e
                        # 坐标还原到原始图像尺寸
                        cx = int(round(cx / scale))
                        cy = int(round(cy / scale))
                        a = int(round(a / scale))
                        b = int(round(b / scale))
                        radius = int(np.sqrt(a * b))

                        target_item = CircleTargetItem(
                            index=0,
                            center_coordinates=(cx, cy),
                            radius=radius,
                            area=np.pi * a * b,
                            shape_type=ShapeType.ELLIPSE,
                            contour_points=None,
                            bounding_box=(cx - a, cy - b, cx + a, cy + b),
                            color=color_name,
                        )
                        self.circle_target.add_target(target_item)
        except Exception as e:
            print(f"[CircleTargetDetector] Hough ellipse detection error: {e}")
            import traceback

            traceback.print_exc()

    def _detect_by_hough_circle(
        self, frame: np.ndarray, target_color: Optional[str] = None
    ):
        """
        OpenCV 霍夫圆变换
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        colors_to_detect = (
            [target_color] if target_color else list(self.color_ranges.keys())
        )

        for color_name in colors_to_detect:
            if color_name not in self.color_ranges:
                continue
            mask = self._get_color_mask(hsv, color_name)
            masked_gray = cv2.bitwise_and(gray, gray, mask=mask)
            blurred = cv2.GaussianBlur(masked_gray, (9, 9), 2)

            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_circle_dp,
                minDist=self.hough_circle_min_dist,
                param1=self.hough_circle_param1,
                param2=self.hough_circle_param2,
                minRadius=self.hough_circle_min_radius,
                maxRadius=self.hough_circle_max_radius,
            )

            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")

                for cx, cy, r in circles:
                    target_item = CircleTargetItem(
                        shape_type=None,
                        index=0,
                        center_coordinates=(cx, cy),
                        radius=r,
                        area=np.pi * r * r,
                        contour_points=None,  # 霍夫方法没有轮廓点
                        bounding_box=(cx - r, cy - r, cx + r, cy + r),
                        color=color_name,
                    )
                    self.circle_target.add_target(target_item)

    def _get_color_mask(self, hsv: np.ndarray, color_name: str) -> np.ndarray:
        """获取指定颜色的mask"""
        ranges = self.color_ranges[color_name]
        mask = None
        for lower, upper in ranges:
            color_mask = cv2.inRange(hsv, lower, upper)
            mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)
        return mask
