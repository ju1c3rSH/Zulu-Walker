import cv2
import numpy as np
from .cargos import Cargos, CargoItem, ShapeType


class CargoDetector:
    def __init__(self):
        self.Cargos = Cargos()

        self.color_ranges = {
            'Red': [
                (np.array([0, 50, 50]), np.array([10, 255, 255])),
                (np.array([170, 50, 50]), np.array([180, 255, 255]))
            ],
            'Green': [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
            'Blue': [(np.array([100, 50, 50]), np.array([130, 255, 255]))]
        }

        self.min_area_threshold = 100  # 最小面积阈值
        self.next_cargo_index = 0  # 用于给每个检测到的货物分配唯一索引

    def detect_cargo_shape(self, frame: np.ndarray, target_color: str = None) -> Cargos:
        """检测图像中的货物形状

        Args:
            frame: BGR格式的输入图像
            target_color: 目标颜色 ('Red', 'Green', 'Blue')，None 表示检测所有颜色

        Returns:
            Cargos: 包含所有检测到的货物信息的对象
        """
        self.Cargos.clear()
        self.next_cargo_index = 0
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        colors_to_detect = [target_color] if target_color else list(self.color_ranges.keys())

        for color_name in colors_to_detect:
            if color_name not in self.color_ranges:
                continue

            ranges = self.color_ranges[color_name]
            mask = None
            for lower, upper in ranges:
                color_mask = cv2.inRange(hsv, lower, upper)
                mask = color_mask if mask is None else cv2.bitwise_or(mask, color_mask)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_area_threshold:
                    continue

                cargo_item = self._analyze_contour(contour)
                cargo_item.index = self.next_cargo_index
                cargo_item.color = color_name
                self.next_cargo_index += 1
                self.Cargos.add_cargo(cargo_item)

        # 更新整体形状类型
        if self.Cargos.payload:
            shapes = [cargo.shape_type for cargo in self.Cargos.payload]
            self.Cargos.shape = shapes[0] if len(set(shapes)) == 1 else ShapeType.UNKNOWN

        return self.Cargos

    def _analyze_contour(self, contour: np.ndarray) -> CargoItem:
        """分析单个轮廓的形状特征

        Args:
            contour: OpenCV轮廓
            color_name: 颜色名称（可选）

        Returns:
            CargoItem: 包含形状信息的货物对象
        """
        # 计算基本属性
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)

        '''
        质心（Centroid）：轮廓的几何中心，可以通过一阶矩除以零阶矩得到。
        面积（Area）：轮廓的面积，直接由零阶矩给出。
        方向（Orientation）：轮廓的主方向，可以通过二阶矩计算得到。
        '''
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # 使用边界框中心作为备选
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2

        # 获取边界框
        x, y, w, h = cv2.boundingRect(contour)
        bounding_box = (x, y, x + w, y + h)
        aspect_ratio = float(w) / h if h > 0 else 0

        # 近似多边形
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        vertices = len(approx)

        # 判断形状类型
        shape_type = self._determine_shape(contour, vertices, area, aspect_ratio)

        # 创建CargoItem
        cargo_item = CargoItem(
            index=0,  # 稍后在detect_cargo_shape中设置
            center_coordinates=(cx, cy),
            width=w,
            height=h,
            area=area,
            contour_points=contour,
            bounding_box=bounding_box,
            aspect_ratio=aspect_ratio,
            shape_type=shape_type
        )

        # 如果是圆形，计算半径
        if shape_type == ShapeType.CIRCLE:
            (cx_f, cy_f), radius = cv2.minEnclosingCircle(contour)
            cargo_item.radius = radius

        return cargo_item

    def _determine_shape(self, contour: np.ndarray, vertices: int, area: float, aspect_ratio: float) -> ShapeType:
        """根据轮廓特征判断形状类型

        Args:
            contour: OpenCV轮廓
            vertices: 近似多边形的顶点数
            area: 轮廓面积
            aspect_ratio: 长宽比

        Returns:
            ShapeType: 形状类型
        """
        # 根据顶点数判断
        if vertices == 4:
            return ShapeType.RECTANGLE
        if vertices == 3:
            return ShapeType.TRIANGLE

        # 检查是否为圆形
        if self._is_circle(contour, area):
            return ShapeType.CIRCLE

        # 检查是否为矩形（4个顶点且长宽比接近1或接近某个比例）
        if vertices == 4:
            return ShapeType.RECTANGLE

        # 检查是否为三角形
        if vertices == 3:
            return ShapeType.TRIANGLE

        return ShapeType.UNKNOWN

    def _is_circle(self, contour: np.ndarray, area: float) -> bool:
        """判断轮廓是否为圆形

        Args:
            contour: OpenCV轮廓
            area: 轮廓面积

        Returns:
            bool: 是否为圆形
        """
        # 计算最小外接圆
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * radius * radius

        # 比较实际面积与圆面积
        if circle_area > 0:
            area_ratio = area / circle_area
            # 如果实际面积占圆面积的比例接近1，则认为是圆形
            return area_ratio > self.circle_circumference_threshold
        return False

    @property
    def circle_circumference_threshold(self) -> float:
        """圆形面积比例阈值"""
        return 0.85
    