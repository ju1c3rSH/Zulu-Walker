# -*- coding: utf-8 -*-
from .base import Processor, VisionResult
from .qr_processor import QRProcessor
from .cargo_processor import CargoProcessor

__all__ = ["Processor", "VisionResult", "QRProcessor", "CargoProcessor"]
