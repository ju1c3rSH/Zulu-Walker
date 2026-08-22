# -*- coding: utf-8 -*-
"""Processor registry facade.

Production default: only the AI inference processor is registered. The
detector-backed solutions (cargo / circle / ring / qr / line-follow) ship in
the repo but stay DORMANT until you explicitly enable them — nothing imports
them, so they cost nothing at runtime and are absent from device packages
unless the packaging manifest includes them.

Enabling is a manual, explicit step (see
docs/architecture/framework-guide.md "启用一个 detector 方案"):

    from modules.zw_opencv_module.processors import enable_optional

    enable_optional(["TrackCargoProcessor"])   # before pipelines start

Importing a processor module triggers its @register_processor decorator,
which transitively imports the matching detector package.
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
    """Register the named opt-in processors (idempotent).

    Raises KeyError on an unknown name so typos fail loud at startup
    instead of silently missing tasks later.
    """
    for name in names:
        importlib.import_module(OPTIONAL_PROCESSORS[name], __package__)


__all__ = [
    "Processor", "VisionResult", "ColorTrackable",
    "AIInferenceProcessor",
    "OPTIONAL_PROCESSORS", "enable_optional",
]
