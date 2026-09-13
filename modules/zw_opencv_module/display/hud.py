# -*- coding: utf-8 -*-
"""
工科赛近端显示 HUD — 在相机原始 maix Image 上叠加状态与检测结果。

重构自旧 VisionManager._draw_header/_draw_detections:
- 旧显示路径强依赖 AI 模型已加载（self._ai.loaded 门控）, 色块任务无模型时黑屏;
  新路径始终可显示, 且叠加内容来自 gks_task 的 VisionResult（单来源）。
- 顶部状态栏: 模式 / 链路 / FPS; 第二行: 任务码 + 锁存进度 + 坑位结果;
  检测叠加（bbox+标签）来自处理器 hits（RGB 色, maix Image 为 RGB888）。

HUD 不解释像素, 只做呈现; 状态来自 status_fn（coordinator.get_info）。
PC 无 maix 时所有绘制静默跳过（pytest 不受影响）。
"""
from typing import Callable, Dict, Optional

_MODE_TEXT = {
    "EXPLOSIVE": "MODE:EXPLOSIVE",
    "TARGET": "MODE:TARGET",
    "RESCUE": "MODE:RESCUE",
}

# RGB 色（maix Image 为 RGB888; 处理器 hits 约定同色制）
_COLOR_WHITE = (255, 255, 255)
_COLOR_BLACK = (0, 0, 0)
_COLOR_GREEN = (0, 255, 0)
_COLOR_YELLOW = (255, 255, 0)
_COLOR_RED = (255, 64, 64)
_COLOR_GRAY = (160, 160, 160)

_BAR_H = 30
_LINE2_H = 24


def _try_import_maix_image():
    try:
        import maix.image
        return maix.image
    except ImportError:
        return None


class DisplayHUD:
    """状态栏 + 检测叠加。draw() 每显示帧调用一次（显示线程/处理循环）。"""

    def __init__(self, status_fn: Optional[Callable[[], dict]] = None):
        self._status_fn = status_fn
        self._mi = _try_import_maix_image()

    def set_status_provider(self, status_fn: Callable[[], dict]) -> None:
        self._status_fn = status_fn

    # ---- public ----

    def draw(self, img, fps: float, results: Optional[Dict] = None) -> None:
        """img: 原始 maix Image（RGB888）; results: {task_name: VisionResult}"""
        if self._mi is None or img is None:
            return
        status = self._fetch_status()
        self._draw_bar(img)
        self._draw_line1(img, status, fps)
        self._draw_line2(img, status)
        self._draw_hits(img, results or {})

    # ---- internals ----

    def _fetch_status(self) -> dict:
        if self._status_fn is None:
            return {}
        try:
            return self._status_fn() or {}
        except Exception:
            return {}

    def _draw_bar(self, img) -> None:
        try:
            img.draw_rect(0, 0, img.width(), _BAR_H + _LINE2_H,
                          color=self._mi.COLOR_BLACK, thickness=-1)
        except Exception:
            pass

    def _text(self, img, x, y, s, color, scale=1.5) -> None:
        try:
            img.draw_string(x, y, s, color=color, scale=scale, thickness=1)
        except Exception:
            pass

    def _draw_line1(self, img, status: dict, fps: float) -> None:
        mode = str(status.get("state", "IDLE"))
        mode_text = _MODE_TEXT.get(mode, f"MODE:{mode}")
        link = "LINK:OK" if status.get("link_active") else "LINK:--"
        link_color = _COLOR_GREEN if status.get("link_active") else _COLOR_GRAY
        found = "FOUND" if status.get("found") else "-"
        found_color = _COLOR_GREEN if status.get("found") else _COLOR_GRAY

        self._text(img, 6, 4, mode_text, _COLOR_WHITE, scale=1.5)
        try:
            w1 = self._mi.string_size(mode_text, scale=1.5)[0]
        except Exception:
            w1 = 8 * len(mode_text) * 1.5
        x = 6 + int(w1) + 12
        self._text(img, x, 4, link, link_color, scale=1.5)
        self._text(img, img.width() - 130, 4, f"FPS:{fps:4.1f}", _COLOR_WHITE, scale=1.5)
        self._text(img, img.width() - 240, 4, f"T:{found}", found_color, scale=1.5)

    def _draw_line2(self, img, status: dict) -> None:
        mode = str(status.get("state", "IDLE"))
        codes = status.get("codes") or [0, 0, 0]
        y = _BAR_H + 2
        if mode == "IDLE":
            self._text(img, 6, y, "IDLE (wait CMD_REQUEST)", _COLOR_GRAY, scale=1.2)
            return

        parts = [f"CODES:{codes}"]
        order = status.get("latched_order") or []
        pending = status.get("pending_count", 0)
        if order:
            parts.append(f"ORDER:{order}")
        elif pending > 0:
            parts.append(f"LOCK {pending}/2")
        if not status.get("thresholds_ok", True):
            parts.append("THR:MISSING")

        if mode == "EXPLOSIVE":
            parts.append(f"PIT:{status.get('pit', 0)}")
        elif mode == "TARGET":
            parts.append(f"ERR:{int(status.get('err', 0)):+d}")
        elif mode == "RESCUE":
            parts.append(f"POS:{status.get('pos', 0)}")

        color = _COLOR_GREEN if status.get("found") else _COLOR_YELLOW
        self._text(img, 6, y, "  ".join(parts), color, scale=1.2)

    def _draw_hits(self, img, results: Dict) -> None:
        gks = results.get("gks_task")
        if gks is None:
            return
        data = getattr(gks, "result_data", None) or {}
        for hit in data.get("hits", []):
            x, y, w, h = hit["x"], hit["y"], hit["w"], hit["h"]
            color = hit.get("color", _COLOR_GREEN)
            try:
                img.draw_rect(x, y, w, h, color=color, thickness=2)
            except Exception:
                continue
            text = hit.get("text") or ""
            if text:
                self._text(img, x, max(2, y - 16), text, color, scale=1.0)
