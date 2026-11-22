from __future__ import annotations

import logging
from typing import Optional

import maix.display
import maix.err

from utils.boot_timing import boot_stage

logger = logging.getLogger(__name__)

# MaixCAM-Pro 面板物理尺寸; 显示对象打不开时 width()/height() 退回它。
_FALLBACK_W, _FALLBACK_H = 552, 368


class MaixCam1Display:
    """MaixCAM / MaixCAM-Pro 显示。

    构造签名按 CAM1 固件支持程度逐级简化：新固件支持 `open=True`，老固件的
    `Display()` 不认该 kwarg（官方示例即`display.Display()`），故先试
    带 kwarg 的、失败退回无参。MaixCAM-Pro 有触摸屏（552x368），调参页
    与四页 KEY_OK 路由因此可原样保留。

    与相机同款的白跑风险（2026-10-08 打点）: `except TypeError: pass`
    之后会继续尝试下一级。若 `Display(open=True)` 是"已经建好显示上下文才
    抛 TypeError"，重试就是重复初始化。逐级打点后可见实际试了几次。
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
        """打不开时退回面板物理尺寸而非 0: 0 会让调试页 resize(0,0) 直接失败。"""
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
