from __future__ import annotations

from typing import Any, Optional

from framework.hal.interface import AIInference

from .ai import MaixCam1AI
from .battery import MaixCam1Battery
from .camera import MaixCam1Camera
from .display import MaixCam1Display
from .uart import MaixCam1Uart


def create_exit_check():
    """Return the platform exit predicate for the framework main loop.

    The framework calls this without importing maix itself (ARCH-06).
    MaixCAM firmware also provides maix.app.need_exit.
    """
    from maix import app as _app

    return _app.need_exit


def create_camera(source, width: int = 640, height: int = 480, **kwargs) -> MaixCam1Camera:
    camera_id = kwargs.pop("camera_id", str(source))
    fps = kwargs.pop("fps", None)
    # CAM1 capture latency is at least 30 ms even at buff_num=1 because the
    # firmware always double-buffers, so a small value is sufficient here.
    buff_num = kwargs.pop("camera_stream_queue_size", 2)
    focal_length_mm = kwargs.pop("focal_length_mm", None)
    sensor_width_mm = kwargs.pop("sensor_width_mm", None)
    sensor_height_mm = kwargs.pop("sensor_height_mm", None)
    # Machine.create forwards exposure/gain/AEC to the platform factory, but
    # the CAM1 camera constructor rejects them (runtime changes go via set()).
    # Drop them here; passing them through would raise TypeError before the
    # camera can even open.
    kwargs.pop("exposure_us", None)
    kwargs.pop("gain", None)
    kwargs.pop("aec", None)
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


def create_ai(
    engine: str = "",
    *,
    switch_timeout_ms: Optional[int] = None,
    frame: Any = None,
    runtime_options: Optional[dict] = None,
) -> AIInference:
    """Return the AI backend of this board.

    An empty ``engine`` keeps the board's own backend, which loads CVITEK
    .cvimodel files through the firmware.  A non-empty engine selects the
    kernel; both branches satisfy AIInference, so the caller does not branch
    on the platform.

    The kernel imports are local: a package built without
    ``framework/inference`` still starts with the own backend.
    """
    if not engine:
        return MaixCam1AI()

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


def create_battery() -> MaixCam1Battery:
    return MaixCam1Battery()


__all__ = [
    "MaixCam1AI",
    "MaixCam1Battery",
    "MaixCam1Camera",
    "MaixCam1Display",
    "MaixCam1Uart",
    "create_ai",
    "create_battery",
    "create_camera",
    "create_display",
    "create_exit_check",
    "create_uart",
]
