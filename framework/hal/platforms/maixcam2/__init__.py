from .ai import MaixCam2AI
from .camera import MaixCam2Camera
from .display import MaixCam2Display
from .uart import MaixCam2Uart


def create_exit_check():
    """Return the platform exit predicate for the framework main loop.

    The framework calls this without importing maix itself (ARCH-06).
    """
    from maix import app as _app

    return _app.need_exit


def create_watchdog(timeout_s: float = 10.0):
    """Hardware watchdog capability (ARCH-02). Raises when unavailable."""
    from .watchdog import MaixWatchdog

    return MaixWatchdog(timeout_s=timeout_s)


def create_camera(source, width: int = 640, height: int = 480, **kwargs) -> MaixCam2Camera:  # type: ignore[type-arg]
    camera_id = kwargs.pop("camera_id", str(source))
    fps = kwargs.pop("fps", None)
    buff_num = kwargs.pop("camera_stream_queue_size", 3)
    focal_length_mm = kwargs.pop("focal_length_mm", None)
    sensor_width_mm = kwargs.pop("sensor_width_mm", None)
    sensor_height_mm = kwargs.pop("sensor_height_mm", None)
    exposure_us = kwargs.pop("exposure_us", None)
    gain = kwargs.pop("gain", None)
    aec = kwargs.pop("aec", None)
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
        exposure_us=exposure_us,
        gain=gain,
        aec=aec,
    )


def create_display() -> MaixCam2Display:
    return MaixCam2Display()


def create_uart(port: str, baudrate: int = 921600) -> MaixCam2Uart:
    return MaixCam2Uart(port=port, baudrate=baudrate)


def create_ai() -> MaixCam2AI:
    return MaixCam2AI()


_fill_light_led = [None]


def _get_fill_light_gpio():
    if _fill_light_led[0] is None:
        from maix import gpio, pinmap, err

        err.check_raise(pinmap.set_pin_function("B25", "GPIOB25"), "set pin failed")
        _fill_light_led[0] = gpio.GPIO("GPIOB25", gpio.Mode.OUT)
    return _fill_light_led[0]


def set_fill_light(on: bool) -> bool:
    """Turn the onboard fill light (B25, active high) on/off."""
    try:
        led = _get_fill_light_gpio()
        led.value(1 if on else 0)
        return True
    except Exception:
        return False


def enable_fill_light() -> bool:
    """Backward-compatible helper: force the fill light on."""
    return set_fill_light(True)


__all__ = [
    "MaixCam2AI",
    "MaixCam2Camera",
    "MaixCam2Display",
    "MaixCam2Uart",
    "create_ai",
    "create_camera",
    "create_display",
    "create_uart",
    "enable_fill_light",
    "set_fill_light",
]
