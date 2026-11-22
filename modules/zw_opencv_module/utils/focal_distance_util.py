# -*- coding: utf-8 -*-
"""Compute real-world object distances from camera intrinsics and pixel sizes.

Distance is derived from the target's apparent size in the image using a
pinhole-camera model.
"""
from dataclasses import dataclass
from typing import Optional, Tuple
import math


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters used for distance computations."""
    focal_length_mm: float      # Focal length in millimeters
    sensor_width_mm: float      # Sensor width in millimeters
    sensor_height_mm: float     # Sensor height in millimeters
    image_width: int            # Image width in pixels
    image_height: int           # Image height in pixels

    @property
    def fx(self) -> float:
        """Return the focal length along the x axis in pixels."""
        return self.focal_length_mm * self.image_width / self.sensor_width_mm

    @property
    def fy(self) -> float:
        """Return the focal length along the y axis in pixels."""
        return self.focal_length_mm * self.image_height / self.sensor_height_mm

    @property
    def cx(self) -> float:
        """Return the x coordinate of the principal point in pixels."""
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        """Return the y coordinate of the principal point in pixels."""
        return self.image_height / 2.0


class FocalDistanceCalculator:
    """Compute distances and coordinate conversions with a pinhole camera.

    Distance uses similar triangles:
    distance = focal_pixel * real_size / pixel_size

    focal_pixel is the focal length in pixels, real_size is the object's real
    size in millimeters, and pixel_size is its apparent size in pixels.
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        """Initialize the calculator with camera intrinsics.

        Args:
            intrinsics: Intrinsic parameters used by all conversions.
        """
        self.intrinsics = intrinsics

    def calculate_distance(
        self,
        real_size_mm: float,
        pixel_size: float,
        use_average_focal: bool = True
    ) -> float:
        """Return the distance to a target from its real and pixel sizes.

        Args:
            real_size_mm: Real size of the target in millimeters.
            pixel_size: Apparent size of the target in pixels.
            use_average_focal: Average fx and fy instead of using fx alone.

        Returns:
            Distance to the target in millimeters, or 0 if pixel_size is not positive.
        """
        if pixel_size <= 0:
            return 0.0

        if use_average_focal:
            focal_pixel = (self.intrinsics.fx + self.intrinsics.fy) / 2.0
        else:
            focal_pixel = self.intrinsics.fx

        distance = (focal_pixel * real_size_mm) / pixel_size
        return distance

    def pixel_to_camera_coords(
        self,
        pixel_x: float,
        pixel_y: float,
        distance_mm: float
    ) -> Tuple[float, float, float]:
        """Convert a pixel location to a 3D camera-frame coordinate.

        The camera frame has its origin at the optical center, with X pointing
        right, Y pointing down, and Z pointing forward along the optical axis.

        Args:
            pixel_x: Pixel x coordinate.
            pixel_y: Pixel y coordinate.
            distance_mm: Distance to the target in millimeters, used as Z.

        Returns:
            The (X, Y, Z) camera-frame coordinates in millimeters.
        """
        # Offset from the principal point
        dx = pixel_x - self.intrinsics.cx
        dy = pixel_y - self.intrinsics.cy

        # Project the offset through similar triangles
        Z = distance_mm
        X = dx * Z / self.intrinsics.fx
        Y = dy * Z / self.intrinsics.fy

        return (X, Y, Z)

    def camera_coords_to_pixel(
        self,
        x: float,
        y: float,
        z: float
    ) -> Tuple[float, float]:
        """Convert a 3D camera-frame coordinate to a pixel location.

        Args:
            x: Camera-frame X coordinate in millimeters.
            y: Camera-frame Y coordinate in millimeters.
            z: Camera-frame Z coordinate in millimeters.

        Returns:
            The (pixel_x, pixel_y) pixel coordinate, or the principal point
            when z is not positive.
        """
        if z <= 0:
            return (self.intrinsics.cx, self.intrinsics.cy)

        pixel_x = x * self.intrinsics.fx / z + self.intrinsics.cx
        pixel_y = y * self.intrinsics.fy / z + self.intrinsics.cy

        return (pixel_x, pixel_y)

    def calculate_angular_offset(
        self,
        pixel_x: float,
        pixel_y: float
    ) -> Tuple[float, float]:
        """Return the angular offset of a pixel from the optical axis.

        Args:
            pixel_x: Pixel x coordinate.
            pixel_y: Pixel y coordinate.

        Returns:
            The (angle_x, angle_y) offsets in radians.
        """
        dx = pixel_x - self.intrinsics.cx
        dy = pixel_y - self.intrinsics.cy

        angle_x = math.atan(dx / self.intrinsics.fx)
        angle_y = math.atan(dy / self.intrinsics.fy)

        return (angle_x, angle_y)
