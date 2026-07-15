# -*- coding: utf-8 -*-
from .base import Processor, VisionResult, ColorTrackable
from .circle_target_processor import CircleTargetProcessor
from .qr_processor import QRCodeProcessor
from .cargo_processor import TrackCargoProcessor
from .ring_discovery_processor import RingDiscoveryProcessor
from .ai_inference_processor import AIInferenceProcessor

__all__ = [
    "Processor", "VisionResult", "ColorTrackable",
    "CircleTargetProcessor",
    "QRCodeProcessor",
    "TrackCargoProcessor",
    "RingDiscoveryProcessor",
    "AIInferenceProcessor",
]
