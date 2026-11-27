from __future__ import annotations

from typing import Any, Optional

from framework.hal.interface import AIInference

from .ai import MaixCam2AI
from .battery import MaixCam2Battery
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


def create_sys_info():
    """Memory stats capability (ARCH-07): heap + CMM snapshot."""
    from .sysinfo import MaixSysInfo

    return MaixSysInfo()


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


def create_ai(
    engine: str = "",
    *,
    switch_timeout_ms: Optional[int] = None,
    frame: Any = None,
    runtime_options: Optional[dict] = None,
) -> AIInference:
    """Return the AI backend of this board.

    An empty ``engine`` keeps the board's own backend: the firmware's maix.nn
    classes driven straight from the MUD file, which is what the board has
    always done.  A non-empty engine selects the kernel, and the artifact,
    the labels and the frame contract then come from a manifest.

    Both branches satisfy AIInference, so the caller never branches on the
    platform; the choice is a deployment setting, not a code path.

    The kernel imports are local on purpose: a device packaged without
    ``framework/inference`` still starts, because the own backend needs none
    of it.
    """
    if not engine:
        return MaixCam2AI()

    from framework.hal.platforms.frame_spec import frame_spec_from_config
    from framework.inference.facade import DEFAULT_SWITCH_TIMEOUT_MS, AIFacade

    return AIFacade(
        engine,
        switch_timeout_ms=(
            DEFAULT_SWITCH_TIMEOUT_MS
            if switch_timeout_ms is None
            else switch_timeout_ms
        ),
        frame_spec=frame_spec_from_config(frame),
        runtime_options=runtime_options,
    )


def create_canvas(frame):
    """Wrap a maix Image so the vision module can draw the HUD onto it."""
    from framework.hal.platforms.maix_canvas import MaixCanvas

    return MaixCanvas(frame)


def create_battery() -> MaixCam2Battery:
    """Return the battery capability backed by the sysfs fuel gauge (no AXP2101)."""
    return MaixCam2Battery()


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
    "MaixCam2Battery",
    "MaixCam2Camera",
    "MaixCam2Display",
    "MaixCam2Uart",
    "create_ai",
    "create_battery",
    "create_camera",
    "create_display",
    "create_uart",
    "enable_fill_light",
    "set_fill_light",
]
