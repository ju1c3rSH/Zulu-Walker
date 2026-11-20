# -*- coding: utf-8 -*-
import time
from typing import Callable, Dict, Optional

from modules.zw_uart_module.protocol import NACK_REASON_NAMES

_NACK_BANNER_TTL_S = 5.0
_STALE_FPS_LIMIT = 0.05
_GUIDE_TICK_FRACTIONS = (-0.80, -0.40, -0.20, 0.0, 0.20, 0.40, 0.80)
_TICK_H_RATIO = 0.06

_MODE_TEXT = {
    "EXPLOSIVE": "MODE:EXPLOSIVE",
    "TARGET": "MODE:TARGET",
    "RESCUE": "MODE:RESCUE",
}

# RGB 色（maix Image 为 RGB888; 处理器 hits 约定同色制）。
# 官方文档规定 color 用 image.Color.from_rgb(r,g,b) 构造; 无 maix 时回退 24-bit 打包 int。
def _pack(rgb) -> int:
    r, g, b = rgb
    return (int(r) & 0xFF) << 16 | (int(g) & 0xFF) << 8 | (int(b) & 0xFF)


_BLACK = (0, 0, 0)
_GREEN = (0, 255, 0)
_YELLOW = (255, 255, 0)
_RED = (255, 64, 64)
_GRAY = (160, 160, 160)
_WHITE = (255, 255, 255)
_CYAN = (0, 255, 255)

_BAR_H = 30
_LINE2_H = 24
_BANNER_H = 20

# 电量等级 → 颜色键; 等级在采集侧算好。
_BAT_COLORS = {"charging": "green", "ok": "white", "low": "red", "unknown": "gray"}
_BAT_Y = _BAR_H + 2
_BAT_SCALE = 1.2
_BAT_MARGIN = 8
_TEXT_GAP = 6           # 第二行文字与电量之间的最小间隙


def _try_import_maix_image():
    try:
        import maix.image
        return maix.image
    except ImportError:
        return None


class DisplayHUD:
    """状态栏 + 检测叠加。draw() 每显示帧调用一次（显示线程/处理循环）。"""

    def __init__(self, status_fn: Optional[Callable[[], dict]] = None,
                 battery_fn: Optional[Callable[[], object]] = None,
                 battery_stale_s: float = 30.0):
        self._status_fn = status_fn
        self._battery_fn = battery_fn
        self._battery_stale_s = float(battery_stale_s)
        self._bat_pct = -2          # -2 = 还没画过, 首帧就要量宽
        self._bat_mv = -2
        self._bat_text = ""
        self._bat_w = 0
        self._bat_short = ""        # 与第二行撞车时退化成 "BAT:87%"
        self._bat_short_w = 0
        self._line2_text = ""       # 第二行当前文本/宽度, 供电量行避让
        self._line2_w = 0
        self._mi = _try_import_maix_image()
        self._draw_once_logged = False
        self._diag_logged = {}
        self._c = self._build_colors()

    def _mk_color(self, rgb, const_name=None):
        """优先用 maix.image.COLOR_* 命名常量（旧分支在此设备上验证可用）,
        其次 Color.from_rgb, 最后打包 int"""
        mi = self._mi
        if const_name and hasattr(mi, const_name):
            return getattr(mi, const_name)
        try:
            return mi.Color.from_rgb(*rgb)
        except Exception:
            return _pack(rgb)

    def _build_colors(self) -> dict:
        return {
            "black": self._mk_color(_BLACK, "COLOR_BLACK"),
            "green": self._mk_color(_GREEN, "COLOR_GREEN"),
            "yellow": self._mk_color(_YELLOW, "COLOR_YELLOW"),
            "red": self._mk_color(_RED, "COLOR_RED"),
            "gray": self._mk_color(_GRAY, "COLOR_GRAY"),
            "white": self._mk_color(_WHITE, "COLOR_WHITE"),
            "cyan": self._mk_color(_CYAN, "COLOR_CYAN"),
        }

    @property
    def C(self):
        return self._c

    def set_status_provider(self, status_fn: Callable[[], dict]) -> None:
        self._status_fn = status_fn

    # ---- public ----

    def draw(self, img, fps: float, results: Optional[Dict] = None) -> None:
        """img: 原始 maix Image（RGB888）; results: {task_name: VisionResult}"""
        if self._mi is None or img is None:
            return
        if not self._draw_once_logged:
            self._draw_once_logged = True
            try:
                from utils.log_util import log_print
                log_print(f"[HUD] first draw ok: img={img.width()}x{img.height()} fps={fps:.2f}")
            except Exception:
                pass
        status = self._fetch_status()
        self._draw_bar(img)
        self._draw_line1(img, status, fps)
        self._draw_line2(img, status)
        self._draw_battery(img)
        self._draw_debug(img, status, fps)
        self._draw_hits(img, results or {})
        self._draw_guides(img, status)
        self._draw_alert_banner(img, status)

    # ---- internals ----

    def _fetch_status(self) -> dict:
        if self._status_fn is None:
            return {}
        try:
            return self._status_fn() or {}
        except Exception:
            return {}

    def _diag(self, site: str, exc: Exception) -> None:
        """绘制失败诊断：同一 site 只记一次，避免刷屏。"""
        if site in self._diag_logged:
            return
        self._diag_logged[site] = True
        try:
            from utils.log_util import log_print
            log_print(f"[HUD] draw fail at {site}: {type(exc).__name__}: {exc}")
        except Exception:
            pass

    def _draw_bar(self, img) -> None:
        try:
            img.draw_rect(0, 0, img.width(), _BAR_H + _LINE2_H,
                          color=self._mi.COLOR_BLACK, thickness=-1)
        except Exception as e:
            self._diag("bar", e)

    def _text(self, img, x, y, s, color, scale=1.5) -> None:
        try:
            img.draw_string(x, y, s, color=color, scale=scale, thickness=1)
        except Exception as e:
            self._diag(f"string@({x},{y})", e)

    def _draw_line1(self, img, status: dict, fps: float) -> None:
        mode = str(status.get("state", "IDLE"))
        mode_text = _MODE_TEXT.get(mode, f"MODE:{mode}")
        link = "LINK:OK" if status.get("link_active") else "LINK:--"
        link_color = self._c["green"] if status.get("link_active") else self._c["gray"]
        found = "FOUND" if status.get("found") else "-"
        found_color = self._c["green"] if status.get("found") else self._c["gray"]

        self._text(img, 6, 4, mode_text, self._c["white"], scale=1.5)
        try:
            w1 = self._mi.string_size(mode_text, scale=1.5, thickness=1)[0]
        except Exception:
            w1 = 8 * len(mode_text) * 1.5
        x = 6 + int(w1) + 12
        self._text(img, x, 4, link, link_color, scale=1.5)
        self._text(img, img.width() - 130, 4, f"FPS:{fps:4.1f}", self._c["white"], scale=1.5)
        self._text(img, img.width() - 240, 4, f"T:{found}", found_color, scale=1.5)

    def _draw_line2(self, img, status: dict) -> None:
        mode = str(status.get("state", "IDLE"))
        codes = status.get("codes") or [0, 0, 0]
        y = _BAR_H + 2
        if mode == "IDLE":
            self._line2(img, y, "IDLE (wait CMD_REQUEST)", self._c["gray"])
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

        color = self._c["green"] if status.get("found") else self._c["yellow"]
        self._line2(img, y, "  ".join(parts), color)

    def _line2(self, img, y, text: str, color) -> None:
        """第二行文本 + 缓存宽度（电量行靠它避让，只在文本变化时量一次）。"""
        if text != self._line2_text:
            self._line2_text = text
            self._line2_w = self._measure(text, 1.2)
        self._text(img, 6, y, text, color, scale=1.2)

    # ---- battery (第二行右对齐) ----

    def _measure(self, text: str, scale: float) -> int:
        try:
            return int(self._mi.string_size(text, scale=scale, thickness=1)[0])
        except Exception:
            return int(8 * len(text) * scale)

    def _draw_battery(self, img) -> None:
        """每帧 1 次 draw_string; 文本与宽度只在百分比变化时重算。

        放第二行而不放第一行: 第一行右侧已被 T:FOUND / FPS 占满, 只剩 ~130px。
        这里跑的线程是视觉线程（同线程在跑 find_blobs）: 任何异常都会冒到
        `vision_manager` 的保底 except → traceback 走 stderr + sleep(1.0) 停帧。
        所以失败一次就把输入断开, 与 `_fetch_status` 同约定。
        """
        if self._battery_fn is None:
            return
        try:
            snap = self._battery_fn()
        except Exception as e:
            self._diag("battery", e)
            self._battery_fn = None
            return

        pct, mv = snap.percent, snap.millivolts
        if pct != self._bat_pct or mv != self._bat_mv:
            self._bat_pct, self._bat_mv = pct, mv
            if pct < 0:                 # -1 = 无电池/未采到: 不画
                self._bat_text, self._bat_w = "", 0
                self._bat_short, self._bat_short_w = "", 0
            else:
                if mv >= 0:
                    self._bat_text = f"BAT:{pct}% {mv / 1000:.2f}V"
                else:
                    self._bat_text = f"BAT:{pct}%"
                self._bat_w = self._measure(self._bat_text, _BAT_SCALE)
                self._bat_short = f"BAT:{pct}%"
                self._bat_short_w = (self._measure(self._bat_short, _BAT_SCALE)
                                     if self._bat_text != self._bat_short else self._bat_w)
        if not self._bat_text:
            return

        # 第二行最长出现在 TARGET + THR:MISSING（~57 字符），会压到右对齐的电量。
        # 撞车就退化成只画百分比, 再撞就整行不画 —— 宁可少一个数字也不叠成一团。
        x = img.width() - _BAT_MARGIN - self._bat_w
        text, w = self._bat_text, self._bat_w
        if x < 6 + self._line2_w + _TEXT_GAP:
            text, w = self._bat_short, self._bat_short_w
            x = img.width() - _BAT_MARGIN - w
            if x < 6 + self._line2_w + _TEXT_GAP:
                return

        level = snap.level if time.monotonic() - snap.ts <= self._battery_stale_s else "unknown"
        self._text(img, x, _BAT_Y, text, self._c[_BAT_COLORS.get(level, "gray")],
                   scale=_BAT_SCALE)

    # 顶栏已绘制过的键，调试行不再重复（state/link/fps → 行1; codes/pending → 行2）
    _DEBUG_DUP_KEYS = ("state", "link_active", "fps", "codes", "pending")

    def _draw_debug(self, img, status: dict, fps: float) -> None:
        """调试键值行（黑底白字）：只显示顶栏没有的 DebugConsole 键,
        如 log_drop / det_count 等; 无 maix 或无内容时静默跳过。"""
        items = {}
        try:
            from utils.debug_console import DebugConsole
            for k, v in DebugConsole().get_all().items():
                if k not in self._DEBUG_DUP_KEYS:
                    items[k] = str(v)
        except Exception:
            pass
        if not items:
            return
        line = " ".join(f"{k}={v}" for k, v in items.items())
        if len(line) > 72:
            line = line[:69] + "..."
        y = _BAR_H + _LINE2_H + 4
        try:
            tw, th = self._mi.string_size(line, scale=1, thickness=1)[:2]
        except Exception:
            tw, th = 8 * len(line), 14
        try:
            img.draw_rect(0, y - 3, img.width(), th + 6,
                          color=self._c["black"], thickness=-1)
        except Exception as e:
            self._diag("debug_bar", e)
        self._text(img, 6, y, line, self._c["white"], scale=1)

    def _draw_hits(self, img, results: Dict) -> None:
        gks = results.get("gks_task")
        if gks is None:
            return
        data = getattr(gks, "result_data", None) or {}
        for hit in data.get("hits", []):
            x, y, w, h = hit["x"], hit["y"], hit["w"], hit["h"]
            color = self._mk_color(hit.get("color", (0, 255, 0)))
            try:
                img.draw_rect(x, y, w, h, color=color, thickness=2)
            except Exception as e:
                self._diag("hit_rect", e)
                continue
            text = hit.get("text") or ""
            if text:
                self._text(img, x, max(2, y - 16), text, color, scale=1.0)

    # ---- guides (ROI lines + target error axis) ----

    def _draw_guides(self, img, status) -> None:
        mode = str(status.get("state", "IDLE"))
        if mode == "TARGET":
            self._draw_roi_band(img, status)
            self._draw_track_win(img, status)
            self._draw_error_axis(img, status)
        elif mode == "EXPLOSIVE":
            self._draw_roi_band(img, status, key="explosive_roi", dim=True)
        elif mode == "RESCUE":
            self._draw_roi_band(img, status, dim=True)

    def _draw_roi_band(self, img, status, dim=False, key="target_roi") -> None:
        roi = status.get(key)
        if not roi:
            return
        h = img.height()
        color = self._c["gray"] if dim else self._c["yellow"]
        for fraction in (roi[1], roi[3]):
            y = max(0, min(h - 1, int(fraction * h)))
            try:
                img.draw_rect(0, y, img.width(), 1, color=color, thickness=-1)
            except Exception as e:
                self._diag("roi_line", e)

    def _draw_track_win(self, img, status) -> None:
        """靶标动态跟踪窗青框（无窗不画）。"""
        win = status.get("target_track_win")
        if not win:
            return
        try:
            img.draw_rect(int(win[0]), int(win[1]), int(win[2]), int(win[3]),
                          color=self._c["cyan"], thickness=1)
        except Exception as e:
            self._diag("track_win", e)

    def _draw_error_axis(self, img, status) -> None:
        h = img.height()
        w = img.width()
        mid_y = h // 2
        # 零点: 校准值(laser_cx, TARGET err 的实际零点) 或画面中心。
        zero = int(status.get("target_zero") or 0)
        center_x = zero if 0 < zero < w else w // 2
        color = self._c["green"] if status.get("found") else self._c["gray"]
        tick_len = max(8, int(h * _TICK_H_RATIO))
        try:
            img.draw_rect(0, mid_y, w, 1, color=color, thickness=-1)
        except Exception as e:
            self._diag("axis_mid", e)
        # 校准生效时把零点位置标出来（白色短刻度, 与 ± 刻度区分）。
        if 0 < zero < w:
            try:
                img.draw_rect(center_x, mid_y - tick_len, 2, 2 * tick_len,
                              color=self._c["white"], thickness=-1)
            except Exception as e:
                self._diag("axis_zero", e)
        for fraction in _GUIDE_TICK_FRACTIONS:
            if fraction == 0.0:
                continue
            x = int(center_x + w // 2 * fraction)
            try:
                img.draw_rect(x, mid_y - tick_len // 2, 1, tick_len + 1,
                              color=color, thickness=-1)
                label = f"{fraction:+.0%}"
                self._text(img, min(w - 40, max(2, x - 12)),
                           mid_y + tick_len // 2 + 2, label, color, scale=0.9)
            except Exception as e:
                self._diag("axis_tick", e)

    # ---- alert banner ----

    def _collect_alerts(self, status) -> list:
        alerts = []
        mode = str(status.get("state", "IDLE"))
        if not status.get("thresholds_ok", True):
            alerts.append("THRESHOLDS MISSING (/my-threshold)")
        if mode == "RESCUE" and not status.get("ai_loaded", True):
            alerts.append("AI MODEL NOT LOADED (RESCUE unavailable)")
        if status.get("link_active") and float(status.get("fps", 0.0)) < _STALE_FPS_LIMIT:
            alerts.append("CAMERA STALLED (zero heartbeat streaming)")
        nack = status.get("last_nack")
        if nack:
            reason, at = nack
            if time.monotonic() - at < _NACK_BANNER_TTL_S:
                alerts.append(f"NACK: {NACK_REASON_NAMES.get(reason, reason)}")
        return alerts

    def _draw_alert_banner(self, img, status) -> None:
        alerts = self._collect_alerts(status)
        if not alerts:
            return
        line = " | ".join(alerts)
        y = _BAR_H + _LINE2_H + 26
        try:
            img.draw_rect(0, y, img.width(), _BANNER_H,
                          color=self._c["red"], thickness=-1)
        except Exception as e:
            self._diag("alert_bar", e)
            return
        self._text(img, 6, y + 3, line, self._mi.COLOR_WHITE
                   if hasattr(self._mi, "COLOR_WHITE") else self._c["white"], 1.0)
