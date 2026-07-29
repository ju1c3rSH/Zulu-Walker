from __future__ import annotations

from typing import Optional

from framework.hal import Machine

from .streamer import JpegStreamer

_streamer: Optional[JpegStreamer] = None


def init(machine: Machine, **kwargs) -> None:
    global _streamer
    _streamer = JpegStreamer()


def get_streamer() -> Optional[JpegStreamer]:
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
