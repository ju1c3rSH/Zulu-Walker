"""Backwards-compatible re-export of the platform-neutral LCD sink.

``MaixLcdSink`` used to live here, but it only touches ``display.show()`` and
is therefore not MaixCAM2-specific. It now lives in :mod:`framework.hal.sink`
so that platforms whose MaixPy build lacks ``nn.YOLO11`` (MaixCAM-Pro) can
import the sink without pulling in the whole ``maixcam2`` platform package --
importing it would raise AttributeError, not ImportError.
"""

from framework.hal.sink import MaixLcdSink

__all__ = ["MaixLcdSink"]
