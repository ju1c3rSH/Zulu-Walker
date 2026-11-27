# -*- coding: utf-8 -*-
"""Compute focal length, object distance and camera-frame coordinates.

Core formulas:

* focal length in pixels: ``fx = focal_length_mm * image_width / sensor_width_mm``
* distance: ``distance = (real_size * focal_length_px) / pixel_size``
* camera coordinates: ``X = (x - cx) * Z / fx``, ``Y = (y - cy) * Z / fy``

This is the only implementation.  ``modules/zw_opencv_module/utils`` re-exports
these names instead of keeping a second copy that drifts.
"""

import math
from dataclasses import dataclass
from typing import Tuple

# Reference physical sizes in mm, as (width, height, average).
reference_size_dict = {
    "quad": (264.0, 177.0, 220.5),  # mm (width, height, average)
}


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters; the derived values follow the inputs.

    They are properties rather than cached fields: a dataclass whose derived
    fields can disagree with its inputs is a trap, and recomputing three
    multiplications costs nothing next to the frame it is used with.
    """

    focal_length_mm: float
    sensor_width_mm: float
    sensor_height_mm: float
    image_width: int
    image_height: int

    @property
    def fx(self) -> float:
        """Focal length along the x axis, in pixels."""
        return self.focal_length_mm * self.image_width / self.sensor_width_mm

    @property
    def fy(self) -> float:
        """Focal length along the y axis, in pixels."""
        return self.focal_length_mm * self.image_height / self.sensor_height_mm

    @property
    def cx(self) -> float:
        """X coordinate of the principal point, in pixels."""
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        """Y coordinate of the principal point, in pixels."""
        return self.image_height / 2.0


class FocalDistanceCalculator:
    """Compute distances and coordinate conversions with a pinhole camera.

    Distance uses similar triangles:
    distance = focal_pixel * real_size / pixel_size
    """

    def __init__(self, intrinsics: CameraIntrinsics):
        self.intrinsics = intrinsics

    def calculate_distance(
        self,
        real_size_mm: float,
        pixel_size: float,
        use_average_focal: bool = False,
    ) -> float:
        """Return the distance to a target from its real and pixel sizes.

        Args:
            real_size_mm: Real size of the target in millimeters.
            pixel_size: Apparent size of the target in pixels.
            use_average_focal: Average fx and fy instead of using fx alone.
                The default is fx only, which is what existing callers and
                the distance figures they were tuned against assume.

        Returns:
            Distance in millimeters, or 0 when pixel_size is not positive.
        """
        if pixel_size <= 0:
            return 0.0

        if use_average_focal:
            focal_pixel = (self.intrinsics.fx + self.intrinsics.fy) / 2.0
        else:
            focal_pixel = self.intrinsics.fx

        return (focal_pixel * real_size_mm) / pixel_size

    def pixel_to_camera_coords(
        self,
        pixel_x: float,
        pixel_y: float,
        distance_mm: float,
    ) -> Tuple[float, float, float]:
        """Convert a pixel location to a 3D camera-frame coordinate.

        The camera frame has its origin at the optical center, with X pointing
        right, Y pointing down and Z pointing forward along the optical axis.

        Returns:
            ``(X, Y, Z)`` in millimeters.
        """
        dx = pixel_x - self.intrinsics.cx
        dy = pixel_y - self.intrinsics.cy
        z = distance_mm
        return (dx * z / self.intrinsics.fx, dy * z / self.intrinsics.fy, z)

    def camera_coords_to_pixel(
        self,
        x: float,
        y: float,
        z: float,
    ) -> Tuple[float, float]:
        """Convert a 3D camera-frame coordinate to a pixel location.

        Returns:
            ``(pixel_x, pixel_y)``, or the principal point when z is not
            positive (the projection is undefined there).
        """
        if z <= 0:
            return (self.intrinsics.cx, self.intrinsics.cy)
        return (
            x * self.intrinsics.fx / z + self.intrinsics.cx,
            y * self.intrinsics.fy / z + self.intrinsics.cy,
        )

    def calculate_angular_offset(
        self,
        pixel_x: float,
        pixel_y: float,
    ) -> Tuple[float, float]:
        """Return the angular offset of a pixel from the optical axis.

        Returns:
            ``(angle_x, angle_y)`` in radians.
        """
        dx = pixel_x - self.intrinsics.cx
        dy = pixel_y - self.intrinsics.cy
        return (
            math.atan(dx / self.intrinsics.fx),
            math.atan(dy / self.intrinsics.fy),
        )
