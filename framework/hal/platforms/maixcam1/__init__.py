from .ai import MaixCam1AI
from .camera import MaixCam1Camera
from .display import MaixCam1Display
from .uart import MaixCam1Uart


def create_camera(source, width: int = 640, height: int = 480, **kwargs) -> MaixCam1Camera:
    camera_id = kwargs.pop("camera_id", str(source))
    fps = kwargs.pop("fps", None)
    # CAM1 取图延时最低 30+ms（即使 buff_num=1 也至少双缓存），
    # 这里默认 2 即可，不必像 CAM2 那样堆缓冲。
    buff_num = kwargs.pop("camera_stream_queue_size", 2)
    focal_length_mm = kwargs.pop("focal_length_mm", None)
    sensor_width_mm = kwargs.pop("sensor_width_mm", None)
    sensor_height_mm = kwargs.pop("sensor_height_mm", None)
    return MaixCam1Camera(
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


def create_display() -> MaixCam1Display:
    return MaixCam1Display()


def create_uart(port: str, baudrate: int = 115200) -> MaixCam1Uart:
    return MaixCam1Uart(port=port, baudrate=baudrate)


def create_ai() -> MaixCam1AI:
    return MaixCam1AI()


__all__ = [
    "MaixCam1AI",
    "MaixCam1Camera",
    "MaixCam1Display",
    "MaixCam1Uart",
    "create_ai",
    "create_camera",
    "create_display",
    "create_uart",
]
