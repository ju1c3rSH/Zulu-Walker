from .circle_target_detector import CircleTargetDetector
from .circle_target_detector.detection import DetectMethod as CircleDetectMethod
from .cargo_detector import CargoDetector
from .cargo_detector.detection import DetectMethod as CargoDetectMethod

__all__ = [
    "CircleTargetDetector", "CircleDetectMethod",
    "CargoDetector", "CargoDetectMethod",
]
