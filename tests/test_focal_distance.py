"""Unit tests for the fused focal-distance utility.

Run: python -m pytest tests/test_focal_distance.py -q

The merge replaced two copies with one.  The only semantic change is the
default of ``use_average_focal`` (the deleted module defaulted to True, the
surviving one to False); these tests pin the arithmetic so that default
cannot drift silently, because distance figures were tuned against fx alone.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.focal_distance_util import (
    CameraIntrinsics,
    FocalDistanceCalculator,
    reference_size_dict,
)

# 640x480 with a sensor whose aspect ratio matches the image: fx == fy.
SQUARE_PIXEL = CameraIntrinsics(
    focal_length_mm=5.0, sensor_width_mm=5.0, sensor_height_mm=3.75,
    image_width=640, image_height=480,
)

# Same image, sensor aspect ratio off: fx != fy, so the two focal choices differ.
STRETCHED = CameraIntrinsics(
    focal_length_mm=5.0, sensor_width_mm=5.0, sensor_height_mm=4.0,
    image_width=640, image_height=480,
)


def test_the_principal_point_is_the_image_centre():
    assert (SQUARE_PIXEL.cx, SQUARE_PIXEL.cy) == (320.0, 240.0)


def test_focal_length_in_pixels_follows_the_inputs():
    assert SQUARE_PIXEL.fx == pytest.approx(640.0)
    assert SQUARE_PIXEL.fy == pytest.approx(640.0)
    assert STRETCHED.fy == pytest.approx(600.0)


def test_the_default_uses_fx_alone():
    """The tuned distance figures assume fx; averaging would shift them."""
    calc = FocalDistanceCalculator(STRETCHED)

    assert calc.calculate_distance(264.0, 100.0) == pytest.approx(
        640.0 * 264.0 / 100.0
    )


def test_the_average_focal_is_opt_in():
    calc = FocalDistanceCalculator(STRETCHED)

    averaged = calc.calculate_distance(264.0, 100.0, use_average_focal=True)

    assert averaged == pytest.approx((640.0 + 600.0) / 2.0 * 264.0 / 100.0)
    assert averaged != pytest.approx(calc.calculate_distance(264.0, 100.0))


def test_a_square_pixel_makes_both_choices_identical():
    calc = FocalDistanceCalculator(SQUARE_PIXEL)

    assert calc.calculate_distance(264.0, 100.0) == pytest.approx(
        calc.calculate_distance(264.0, 100.0, use_average_focal=True)
    )


def test_a_non_positive_pixel_size_yields_zero():
    calc = FocalDistanceCalculator(SQUARE_PIXEL)

    assert calc.calculate_distance(264.0, 0.0) == 0.0
    assert calc.calculate_distance(264.0, -1.0) == 0.0


def test_pixel_and_camera_coordinates_are_inverse():
    calc = FocalDistanceCalculator(STRETCHED)
    pixel = (400.0, 300.0)

    coords = calc.camera_coords_to_pixel(*calc.pixel_to_camera_coords(*pixel, 1000.0))

    assert coords == pytest.approx(pixel)


def test_the_optical_axis_projects_to_the_principal_point():
    calc = FocalDistanceCalculator(STRETCHED)

    assert calc.camera_coords_to_pixel(0.0, 0.0, 500.0) == pytest.approx((320.0, 240.0))


def test_a_non_positive_depth_falls_back_to_the_principal_point():
    calc = FocalDistanceCalculator(STRETCHED)

    assert calc.camera_coords_to_pixel(10.0, 10.0, 0.0) == (320.0, 240.0)


def test_the_principal_point_has_no_angular_offset():
    calc = FocalDistanceCalculator(STRETCHED)

    angle_x, angle_y = calc.calculate_angular_offset(320.0, 240.0)

    assert angle_x == pytest.approx(0.0)
    assert angle_y == pytest.approx(0.0)


def test_the_angular_offset_is_an_arctangent():
    calc = FocalDistanceCalculator(STRETCHED)

    angle_x, _ = calc.calculate_angular_offset(320.0 + 640.0, 240.0)

    assert angle_x == pytest.approx(math.atan(1.0))


def test_the_reference_sizes_survive_the_merge():
    assert "quad" in reference_size_dict
    assert len(reference_size_dict["quad"]) == 3


def test_the_module_package_reexports_the_same_objects():
    from modules.zw_opencv_module import utils as module_utils

    assert module_utils.CameraIntrinsics is CameraIntrinsics
    assert module_utils.FocalDistanceCalculator is FocalDistanceCalculator
