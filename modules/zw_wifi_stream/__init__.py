from __future__ import annotations

import logging
from typing import Optional

from framework.hal import Machine

from .streamer import RtspStreamer

logger = logging.getLogger(__name__)

_streamer: Optional[RtspStreamer] = None

def init(machine: Machine, **kwargs) -> None:
    global _streamer

    try:
        cam = machine.camera_hub.get("main")
        if cam is None:
            logger.warning("[RTSP] main camera not found, skip")
            return
        raw = getattr(cam, "raw_camera", None)
        if raw is None:
            logger.warning("[RTSP] camera has no raw_camera attr, skip")
            return
        try:
            chan = raw.add_channel(cam.width, cam.height)
            cam._read_cam = chan
            logger.info("[RTSP] read channel created (%dx%d)", cam.width, cam.height)
        except Exception as e:
            logger.warning("[RTSP] add_channel failed, fallback raw: %s", e)
    except Exception as e:
        logger.warning("[RTSP] get camera failed: %s", e)
        return

    _streamer = RtspStreamer()
    _streamer.setup_camera(raw)

def get_streamer() -> Optional[RtspStreamer]:
    return _streamer

def start() -> bool:
    return True

def loop():
    pass

def stop():
    global _streamer
    if _streamer is not None:
        _streamer.stop()
        _streamer = None
