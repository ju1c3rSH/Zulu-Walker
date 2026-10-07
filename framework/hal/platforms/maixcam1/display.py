from __future__ import annotations

import logging
from typing import Optional

import maix.display
import maix.err

logger = logging.getLogger(__name__)


class MaixCam1Display:
    """MaixCAM / MaixCAM-Pro 显示。

    构造签名按 CAM1 固件能力降级：新固件支持 `open=True`，老固件的
    `Display()` 不认该 kwarg（官方示例即`display.Display()`），故先试
    带 kwarg 的、失败退回无参。MaixCAM-Pro 有触摸屏（552x368），调参页
    与四页 KEY_OK 路由因此可原样保留。
    """

    def __init__(self) -> None:
        self._disp: Optional[maix.display.Display] = self._open()

    @staticmethod
    def _open():
        try:
            return maix.display.Display(open=True)
        except TypeError:
            pass
        except Exception as e:
            logger.warning("Display(open=True) failed: %s", e)
        try:
            return maix.display.Display()
        except Exception as e:
            logger.error("Display() failed: %s", e)
            return None

    def show(self, frame) -> bool:
        """显示一帧。

        *frame* 可以是 maix.image.Image（零拷贝，显示路径首选）或 BGR
        numpy（HUD 在 numpy 上绘制后回传）。与 MaixCam2Display 一致。
        """
        if self._disp is None or frame is None:
            return False
        try:
            err = self._disp.show(frame)
            return err == maix.err.Err.ERR_NONE
        except Exception:
            return False

    def width(self) -> int:
        if self._disp is None:
            return 0
        try:
            return self._disp.width()
        except Exception:
            return 0

    def height(self) -> int:
        if self._disp is None:
            return 0
        try:
            return self._disp.height()
        except Exception:
            return 0

    def close(self) -> None:
        if self._disp is not None:
            try:
                self._disp.close()
            except Exception:
                pass
            self._disp = None
