import logging

try:
    from utils.log_util import log_print
except ImportError:
    import logging
    log_print = logging.getLogger(__name__).info

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


def create_ai(**kwargs) -> LinuxAI:
    return LinuxAI()


def resolve_camera_source(raw_source):
    """Capability hook (MOD-02): map user-friendly sources to /dev nodes.

    The container probes this by name; platforms without a resolver simply
    omit it and Machine uses the raw source as-is.
    """
    try:
        from utils.camera_misc_util import CameraMiscUtil
    except ImportError:
        return raw_source
    return CameraMiscUtil.resolve_camera_source(raw_source)


def create_sys_info():
    """Capability hook (ARCH-07): /proc/meminfo snapshot, None off-Linux."""
    from .sysinfo import LinuxSysInfo

    return LinuxSysInfo()


__all__ = [
    "LinuxAI",
    "LinuxCamera",
    "LinuxDisplay",
    "LinuxUart",
    "create_ai",
    "create_camera",
    "create_display",
    "create_uart",
    "resolve_camera_source",
    "create_sys_info",
]
