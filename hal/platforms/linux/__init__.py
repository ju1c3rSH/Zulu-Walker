import logging

from utils.log_util import log_print

from .ai import LinuxAI
from .camera import LinuxCamera
from .display import LinuxDisplay
from .uart import LinuxUart

logger = logging.getLogger(__name__)


def create_camera(source, width: int = 640, height: int = 480, **kwargs) -> LinuxCamera | None:
    camera_id = kwargs.pop("camera_id", str(source))
    fps = kwargs.pop("fps", 120)
    queue_size = kwargs.pop("camera_stream_queue_size", 2)
    focal_length_mm = kwargs.pop("focal_length_mm", None)
    sensor_width_mm = kwargs.pop("sensor_width_mm", None)
    sensor_height_mm = kwargs.pop("sensor_height_mm", None)
    cam = LinuxCamera(
        camera_id=camera_id,
        source=source,
        width=width,
        height=height,
        fps=fps,
        queue_size=queue_size,
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
    )
    try:
        cam.start()
    except (RuntimeError, Exception) as e:
        logger.error("Camera '%s' failed to start (skipped): %s", camera_id, e)
        return None

    log_print(
        f"[Camera:{camera_id}] source={source} "
        f"requested={width}x{height}@{fps}fps "
        f"actual={cam.actual_width}x{cam.actual_height}@{cam.fps}fps MJPG"
    )

    return cam


def create_display() -> LinuxDisplay:
    return LinuxDisplay()


def create_uart(port: str, baudrate: int = 921600) -> LinuxUart:
    return LinuxUart(port=port, baudrate=baudrate)


def create_ai() -> LinuxAI:
    return LinuxAI()


__all__ = [
    "LinuxAI",
    "LinuxCamera",
    "LinuxDisplay",
    "LinuxUart",
    "create_ai",
    "create_camera",
    "create_display",
    "create_uart",
]
