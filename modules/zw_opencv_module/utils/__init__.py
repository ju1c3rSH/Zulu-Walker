# -*- coding: utf-8 -*-
"""Provide shared utilities for the OpenCV module.

The focal maths lives in ``utils.focal_distance_util``; this package re-exports
it so the import path used by the module keeps working.
"""
from utils.focal_distance_util import CameraIntrinsics, FocalDistanceCalculator

__all__ = [
    'CameraIntrinsics',
    'FocalDistanceCalculator',
]
