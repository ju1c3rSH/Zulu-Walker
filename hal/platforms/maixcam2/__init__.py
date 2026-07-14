from .camera import MaixCam2Camera
from .display import MaixCam2Display
from .uart import MaixCam2Uart


def create_camera(source, width: int = 640, height: int = 480, **kwargs) -> MaixCam2Camera:
    raise NotImplementedError("MaixCAM2 camera not yet implemented")


def create_display() -> MaixCam2Display:
    raise NotImplementedError("MaixCAM2 display not yet implemented")


def create_uart(port: str, baudrate: int = 921600) -> MaixCam2Uart:
    raise NotImplementedError("MaixCAM2 UART not yet implemented")


__all__ = [
    "MaixCam2Camera",
    "MaixCam2Display",
    "MaixCam2Uart",
    "create_camera",
    "create_display",
    "create_uart",
]
