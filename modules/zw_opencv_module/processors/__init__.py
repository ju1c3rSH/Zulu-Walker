# -*- coding: utf-8 -*-
# Production-only processor exports. The legacy processors (cargo / circle /
# qr / ring_discovery) stay in the repo for reference but are excluded from
# the device package (see app.yaml); importing them here would crash on
# device. Register new production processors by importing their module in the
# task bootstrap path, not by re-exporting here.
from .base import Processor, VisionResult, ColorTrackable
from .ai_inference_processor import AIInferenceProcessor

__all__ = [
    "Processor", "VisionResult", "ColorTrackable",
    "AIInferenceProcessor",
]
