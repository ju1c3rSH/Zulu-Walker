from .camera import MaixCam2Camera
from .display import MaixCam2Display
from .uart import MaixCam2Uart


def create_camera(source, width: int = 640, height: int = 480, **kwargs) -> MaixCam2Camera:
    camera_id = kwargs.pop("camera_id", str(source))
    fps = kwargs.pop("fps", 120)
    buff_num = kwargs.pop("camera_stream_queue_size", 3)
    focal_length_mm = kwargs.pop("focal_length_mm", None)
    sensor_width_mm = kwargs.pop("sensor_width_mm", None)
    sensor_height_mm = kwargs.pop("sensor_height_mm", None)
    return MaixCam2Camera(
        source=source,
        width=width,
        height=height,
        fps=fps,
        camera_id=camera_id,
        buff_num=buff_num,
        focal_length_mm=focal_length_mm,
        sensor_width_mm=sensor_width_mm,
        sensor_height_mm=sensor_height_mm,
    )


def create_display() -> MaixCam2Display:
    return MaixCam2Display()


def create_uart(port: str, baudrate: int = 921600) -> MaixCam2Uart:
    return MaixCam2Uart(port=port, baudrate=baudrate)


__all__ = [
    "MaixCam2Camera",
    "MaixCam2Display",
    "MaixCam2Uart",
    "create_camera",
    "create_display",
    "create_uart",
]
