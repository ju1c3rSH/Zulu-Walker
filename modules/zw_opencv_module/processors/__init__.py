# -*- coding: utf-8 -*-
"""Expose the processor registry and its opt-in activation helper.

Only AIInferenceProcessor is registered by default. The detector-backed
processors (cargo, circle, ring, qr, line-follow) stay inactive until
enable_optional imports them; they are absent from device packages unless the
packaging manifest includes them.

Enabling is explicit (see the "enabling a detector" section of
docs/architecture/framework-guide.md):

    from modules.zw_opencv_module.processors import enable_optional

    enable_optional(["TrackCargoProcessor"])   # before pipelines start

Importing a processor module runs its @register_processor decorator, which
transitively imports the matching detector package.
"""
from __future__ import annotations

import importlib

from .base import Processor, VisionResult, ColorTrackable
from .ai_inference_processor import AIInferenceProcessor

OPTIONAL_PROCESSORS = {
    "TrackCargoProcessor": ".cargo_processor",
    "CircleTargetProcessor": ".circle_target_processor",
    "QRCodeProcessor": ".qr_processor",
    "RingDiscoveryProcessor": ".ring_discovery_processor",
    "LineFollowProcessor": ".line_follow_processor",
}


def enable_optional(names) -> None:
    """Import and register the named opt-in processors; repeated calls are safe.

    Raises:
        KeyError: An unknown name fails at startup instead of silently
            leaving a task unregistered.
    """
    for name in names:
        importlib.import_module(OPTIONAL_PROCESSORS[name], __package__)


__all__ = [
    "Processor", "VisionResult", "ColorTrackable",
    "AIInferenceProcessor",
    "OPTIONAL_PROCESSORS", "enable_optional",
]
