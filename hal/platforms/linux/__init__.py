from .camera import LinuxCamera
from .display import LinuxDisplay
from .uart import LinuxUart


def create_camera(source, width: int = 640, height: int = 480, **kwargs) -> LinuxCamera:
    camera_id = kwargs.pop("camera_id", str(source))
    fps = kwargs.pop("fps", 120)
    queue_size = kwargs.pop("camera_stream_queue_size", 2)
    cam = LinuxCamera(
        camera_id=camera_id,
        source=source,
        width=width,
        height=height,
        fps=fps,
        queue_size=queue_size,
    )
    cam.start()
    return cam


def create_display() -> LinuxDisplay:
    return LinuxDisplay()


def create_uart(port: str, baudrate: int = 921600) -> LinuxUart:
    return LinuxUart(port=port, baudrate=baudrate)


__all__ = [
    "LinuxCamera",
    "LinuxDisplay",
    "LinuxUart",
    "create_camera",
    "create_display",
    "create_uart",
]
