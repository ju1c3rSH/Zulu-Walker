# -*- coding: utf-8 -*-
from .base import Processor, VisionResult, ColorTrackable
from .circle_target_processor import CircleTargetProcessor
from .qr_processor import QRCodeProcessor
from .cargo_processor import TrackCargoProcessor
from .ring_track_processor import RingTrackProcessor

__all__ = [
    "Processor", "VisionResult", "ColorTrackable",
    "CircleTargetProcessor",
    "QRCodeProcessor",
    "TrackCargoProcessor",
    "RingTrackProcessor",
]
