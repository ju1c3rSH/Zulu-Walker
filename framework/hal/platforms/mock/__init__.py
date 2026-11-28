from .ai import MockAI
from .camera import MockCamera
from .display import MockDisplay
from .uart import MockUart


def create_camera(source, width: int = 640, height: int = 480, **kwargs) -> MockCamera:
    camera_id = kwargs.pop("camera_id", str(source))
    return MockCamera(camera_id=camera_id)


def create_display() -> MockDisplay:
    return MockDisplay()


def create_uart(port: str = "mock", baudrate: int = 921600) -> MockUart:
    return MockUart(port=port, baudrate=baudrate)


def create_ai(**kwargs) -> MockAI:
    return MockAI()


def create_battery() -> "NullBattery":
    """Capability hook: return the -1 default on platforms without a fuel gauge."""
    from framework.hal.interface import NullBattery

    return NullBattery()


__all__ = [
    "MockAI",
    "MockCamera",
    "MockDisplay",
    "MockUart",
    "NullBattery",
    "create_ai",
    "create_battery",
    "create_camera",
    "create_display",
    "create_uart",
]
