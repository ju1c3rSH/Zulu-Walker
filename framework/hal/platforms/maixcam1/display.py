from __future__ import annotations

import logging
from typing import Optional

import maix.display
import maix.err

from utils.boot_timing import boot_stage

logger = logging.getLogger(__name__)

# Physical panel size of the MaixCAM-Pro; width()/height() fall back to it
# when the display object cannot be opened.
_FALLBACK_W, _FALLBACK_H = 552, 368


class MaixCam1Display:
    """Display output for MaixCAM / MaixCAM-Pro.

    Construction simplifies by firmware support: new firmware accepts
    ``open=True`` while older firmware's ``Display()`` rejects the kwarg (the
    official example is ``display.Display()``), so try the kwarg first and
    fall back to no arguments. The MaixCAM-Pro has a 552x368 touchscreen, so
    the tuning page and four-page KEY_OK routing work unchanged.

    Like the camera, fallback can repeat expensive work: after
    ``except TypeError: pass``, the next level is attempted. If
    ``Display(open=True)`` created the display context before raising
    TypeError, retrying reinitializes it. Per-level timing makes the number of
    attempts visible.
    """

    def __init__(self) -> None:
        self._disp: Optional[maix.display.Display] = self._open()

    @staticmethod
    def _open():
        attempts = 0
        attempts += 1
        try:
            with boot_stage("disp.open:L1(open=True)"):
                return maix.display.Display(open=True)
        except TypeError:
            pass
        except Exception as e:
            logger.warning("Display(open=True) failed: %s", e)
        attempts += 1
        try:
            with boot_stage("disp.open:L2(())"):
                return maix.display.Display()
        except Exception as e:
            logger.error("Display() failed: %s", e)
            return None
        finally:
            if attempts > 1:
                from utils.log_util import log_print
                log_print(f"[CAM1] display.attempts={attempts}  <-- 降级重试")

    def show(self, frame) -> bool:
        """Display one frame.

        *frame* may be a maix.image.Image (zero-copy, preferred on the
        display path) or a BGR numpy array (HUD drawn in numpy then passed
        back). Same as MaixCam2Display.
        """
        if self._disp is None or frame is None:
            return False
        try:
            err = self._disp.show(frame)
            return err == maix.err.Err.ERR_NONE
        except Exception:
            return False

    def width(self) -> int:
        """Fall back to the physical panel size, not 0: the debug page's
        resize(0, 0) would fail."""
        try:
            return self._disp.width() or _FALLBACK_W
        except Exception:
            return _FALLBACK_W

    def height(self) -> int:
        try:
            return self._disp.height() or _FALLBACK_H
        except Exception:
            return _FALLBACK_H

    def close(self) -> None:
        if self._disp is not None:
            try:
                self._disp.close()
            except Exception:
                pass
            self._disp = None
