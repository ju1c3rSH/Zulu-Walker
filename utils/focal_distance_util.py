# -*- coding: utf-8 -*-
"""Compute focal length, object distance and camera-frame coordinates.

Core formulas:

* focal length in pixels: ``fx = focal_length_mm * image_width / sensor_width_mm``
* distance: ``distance = (real_size * focal_length_px) / pixel_size``
* camera coordinates: ``X = (x - cx) * Z / fx``, ``Y = (y - cy) * Z / fy``
"""

from dataclasses import dataclass
from typing import Tuple

# Reference physical sizes in mm.
reference_size_dict = {
    "quad": (264.0, 177.0, 220.5),  # mm (width, height, average)
}

@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width: int
    image_height: int

    # Derived focal lengths in pixels.
    fx: float = 0.0  # horizontal focal length (pixels)
    fy: float = 0.0  # vertical focal length (pixels)
    cx: float = 0.0  # principal point x coordinate
    cy: float = 0.0  # principal point y coordinate

    def __post_init__(self):
        self.fx = self.focal_length_mm * self.image_width / self.sensor_width_mm
        self.fy = self.focal_length_mm * self.image_height / self.sensor_height_mm
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0


class FocalDistanceCalculator:
    """Estimate camera distance from a known object size."""

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def calculate_distance(self, real_size_mm: float, pixel_size: float) -> float:
        """Compute distance from an object's physical and pixel sizes.

        Args:
            real_size_mm: Object's physical size in mm.
            pixel_size: Object's apparent size in pixels.

        Returns:
            Distance from the camera to the object in mm.
        """
        if pixel_size <= 0:
            return 0.0
        return (real_size_mm * self.intrinsics.fx) / pixel_size

    def pixel_to_camera_coords(self, pixel_x: float, pixel_y: float,
                                distance_mm: float) -> Tuple[float, float, float]:
        """Convert pixel coordinates to camera-frame coordinates.

        Args:
            pixel_x: Pixel x coordinate.
            pixel_y: Pixel y coordinate.
            distance_mm: Distance to the object in mm.

        Returns:
            ``(X, Y, Z)`` camera-frame coordinates in mm: ``X`` is positive
            to the right, ``Y`` positive downward, ``Z`` positive forward
            (depth).
        """
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.cx, self.intrinsics.cy

        X = (pixel_x - cx) * distance_mm / fx
        Y = (pixel_y - cy) * distance_mm / fy
        Z = distance_mm

        return (X, Y, Z)
