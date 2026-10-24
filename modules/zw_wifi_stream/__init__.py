from __future__ import annotations

from typing import Optional

from framework.hal import Machine

from .streamer import JpegStreamer

_streamer: Optional[JpegStreamer] = None
_disabled_reason: str = ""


def supported() -> bool:
    """推流仅在 maix 平台可用: 依赖 maix.http.JpegStreamer + Image.to_jpeg()。"""
    try:
        from maix import http  # noqa: F401
        import maix.image as _mi
    except Exception:
        return False
    return hasattr(_mi.Image, "to_jpeg")


def init(machine: Machine, **kwargs) -> None:
    global _streamer, _disabled_reason
    cfg = (kwargs.get("config") or {}).get("stream") or {}
    if not cfg.get("enabled", False):
        _streamer = None
        _disabled_reason = "stream.enabled=false"
        return
    if not supported():
        _streamer = None
        _disabled_reason = "platform has no maix.http/to_jpeg"
        return
    _streamer = JpegStreamer(
        fps=cfg.get("fps", 5),
        quality=cfg.get("quality", 80),
        max_width=cfg.get("max_width", 320),
    )


def disabled_reason() -> str:
    return _disabled_reason


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
