# -*- coding: utf-8 -*-
"""
设备端 LAB 阈值调参工具（MaixCAM2 触屏）

两种使用方式:
- 独立模式: `python run.py debug color` — 自建 Camera/Display 阻塞运行;
- 页面模式: 主程序调试页切入, 外部喂帧, `step()` 单帧渲染不复用 Camera/Display。

6 组阈值以 LabThresholdSession 为唯一内存事实来源（本工具只暴露组子集时,
仍对整会话做 SAVE, 保证拆页后不互相覆盖）; 实时预览: 当前组阈值命中像素
绿色高亮 + 色块 bbox + 点选取色(LAB 读数); 保存写到 /my-threshold/my_64threshold.txt,
主程序订阅运行中检测到 mtime 变化自动热重载、不清锁存。

触屏 API 依 MaixPy 官方文档 touchscreen.md。绘图颜色依官方 image_ops.md:
draw_rect/draw_string 的 color 需传入 image.Color.from_rgb(...) /
maix.image.COLOR_* 构造值, 不接受 RGB 元组 — 元组会抛异常且被绘制兜底
吞掉, 表现为界面静默空白。
"""
import time

import numpy as np

from modules.zw_opencv_module.debug.lab_session import (
    GROUP_NAMES,
    THRESHOLD_FILE,
    get_threshold_session,
)
from modules.zw_opencv_module.processors.gks_task_processor import load_gks_config

SLIDERS = ["Lmin", "Lmax", "Amin", "Amax", "Bmin", "Bmax"]
SLIDER_RANGE = {0: (0, 100), 1: (0, 100), 2: (-128, 127), 3: (-128, 127),
                4: (-128, 127), 5: (-128, 127)}

_UI_GROUP_H = 30     # 顶部组选择按钮条高度
_UI_ACTION_H = 30    # Exit/SAVE/RESET 按钮条高度
_UI_SLIDER_H = 50    # 每行滑条高度（底部 2 列 x 3 行）


def _pack(rgb) -> int:
    r, g, b = rgb
    return (int(r) & 0xFF) << 16 | (int(g) & 0xFF) << 8 | (int(b) & 0xFF)


class GksThresholdTuner:
    """触屏 LAB 调参器（组子集视图, 会话级整存整取）。

    独立模式: run() 阻塞运行至点 Exit / app 退出。
    页面模式(主程序内嵌): step(disp, ts, frame_bgr) 每帧调用,
    exit_requested() 为 True 表示用户点了 Exit, 由外部调用方切回主页面。
    group_indices=None 显示全部 6 组; 指定子集时组选择条只放该子集按钮,
    RESET/SAVE 均作用于共享会话整六组。
    """

    def __init__(self, camera_source=None, width: int = 640, height: int = 480,
                 group_indices=None, get_processor=None):
        self._source = camera_source
        self._req_w, self._req_h = width, height
        self._group_indices = tuple(group_indices) if group_indices else None
        self._visible_names = (
            [(i, GROUP_NAMES[i]) for i in self._group_indices]
            if self._group_indices else
            [(i, GROUP_NAMES[i]) for i in range(len(GROUP_NAMES))]
        )
        self._get_processor = get_processor
        self._stop = False
        self._C = self._build_colors()
        self._diag_logged = {}
        self._bind_disp = None
        self._bind_ts = None
        self._session = get_threshold_session()
        self._target_group_index_range = range(3, 6)
        self._roi_cache = None
        self._scale_cache = None

    # ---- 分析降采样档位（配置一次读，之后缓存） ----

    def _scan_scale(self) -> float:
        """调参页分析降采样系数。1.0 = 全帧（回退到旧行为）。

        与主程序 `gks.explosive.scan_scale` 同一套办法（INTER_NEAREST 不插值 +
        面积按 scale² 折算 + 坐标按 1/scale 还原），详见 _analyze docstring。
        独立模式下也生效 —— 那个路径自己建 Camera/Display，不经过调参页 step，
        但同样卡在 findContours 上。
        """
        if self._scale_cache is None:
            self._scale_cache = 1.0
            try:
                raw = load_gks_config().get("tuner", {}).get("scan_scale", 0.5)
                value = float(raw)
                if 0.0 < value <= 1.0:
                    self._scale_cache = value
                else:
                    self._diag("tuner_scan_scale",
                               ValueError(f"out of range: {raw!r}"))
            except Exception as e:
                self._diag("tuner_scan_scale", e)
        return self._scale_cache

    # ---- colors (官方 API: image.Color.from_rgb / maix.image.COLOR_*) ----

    def _mi(self):
        import maix.image
        return maix.image

    def _mk_color(self, rgb, const_name=None):
        try:
            mi = self._mi()
            if const_name and hasattr(mi, const_name):
                return getattr(mi, const_name)
            return mi.Color.from_rgb(*rgb)
        except Exception:
            return _pack(rgb)

    def _build_colors(self) -> dict:
        return {
            "white": self._mk_color((255, 255, 255), "COLOR_WHITE"),
            "black": self._mk_color((0, 0, 0), "COLOR_BLACK"),
            "green": self._mk_color((0, 255, 0), "COLOR_GREEN"),
            "yellow": self._mk_color((255, 255, 0), "COLOR_YELLOW"),
            "gray_off": self._mk_color((200, 200, 200), "COLOR_GRAY"),
            "slider_track": self._mk_color((120, 120, 120), "COLOR_GRAY"),
            "magenta": self._mk_color((255, 0, 255), "COLOR_PURPLE"),
            "cyan": self._mk_color((0, 255, 255), "COLOR_CYAN"),
        }

    # ---- diag ----

    def _diag(self, site: str, exc: Exception) -> None:
        if site in self._diag_logged:
            return
        self._diag_logged[site] = True
        try:
            from utils.log_util import log_print
            log_print(f"[TUNER] draw fail at {site}: {type(exc).__name__}: {exc}")
        except Exception:
            pass

    # ---- lifecycle: 独立模式 ----

    def run(self) -> None:
        from maix import camera, display, app, image, touchscreen

        self._reset_session()
        cam = camera.Camera(width=self._req_w, height=self._req_h,
                            format=image.Format.FMT_RGB888,
                            device=str(self._source) if self._source else None,
                            open=True)
        disp = display.Display()
        ts = touchscreen.TouchScreen()
        while not app.need_exit() and not self._stop:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
            frame_bgr = self._to_bgr(frame)
            img = self._render_step(frame_bgr, disp, ts)
            disp.show(img)
            if self._session_stop:
                break
            time.sleep(0.001)

    def stop(self) -> None:
        self._stop = True

    # ---- lifecycle: 页面模式 ----

    def step(self, disp, ts, frame_bgr) -> None:
        """主程序 KEY_OK 切入后每帧调用: 渲染调参页并处理触屏。

        分项归因见 laser_calib_tuner 同名方法; 这里只分两段（render / show）,
        因为调参页的成本几乎全在 _render_step 里的 findContours。
        """
        _t = time.perf_counter()
        img = self._render_step(frame_bgr, disp, ts)
        self._note_phase("render", time.perf_counter() - _t)
        _t = time.perf_counter()
        disp.show(img)
        self._flush_phase(time.perf_counter() - _t)

    # ---- 分项归因（2026-10-15）----

    def _note_phase(self, key: str, dt_s: float) -> None:
        ph = self.__dict__.get("_phases")
        if ph is None:
            ph = self.__dict__["_phases"] = {"t0": 0.0, "n": 0, "render": 0.0, "show": 0.0}
        ph[key] += dt_s * 1000.0

    def _flush_phase(self, show_s: float) -> None:
        ph = self.__dict__.get("_phases")
        if ph is None:
            return
        ph["show"] += show_s * 1000.0
        ph["n"] += 1
        now = time.monotonic()
        if ph["t0"] == 0.0:
            ph["t0"] = now
            return
        elapsed = now - ph["t0"]
        if elapsed < 1.0:
            return
        n = max(1, ph["n"])
        try:
            from utils.log_util import log_print
            log_print(f"[PERF] tuner page {n / elapsed:.1f}Hz "
                      f"render={ph['render'] / n:.1f}ms show={ph['show'] / n:.1f}ms "
                      f"total={(ph['render'] + ph['show']) / n:.1f}ms")
        except Exception:
            pass
        ph.update(t0=now, n=0, render=0.0, show=0.0)

    def exit_requested(self) -> bool:
        """页面模式下用户点 Exit / 外部 stop()。"""
        return self._session_stop or self._stop

    def begin(self) -> None:
        """页面模式进入调参页：重建会话状态（组/选中/点选/提示）。"""
        self._reset_session()

    def _reset_session(self) -> None:
        self._groups = self._session.groups()
        self._cur = self._group_indices[0] if self._group_indices else 0
        self._pick = None
        self._toast = ""
        self._toast_until = 0.0
        self._pressed_last = False
        self._session_stop = False

    # ---- 单帧渲染 + 触屏交互 ----

    def _render_step(self, frame_bgr, disp, ts):
        groups = self._groups
        cur = self._cur
        pick = self._pick
        pressed_last = self._pressed_last

        disp_w, disp_h = disp.width(), disp.height()

        lab_now = groups[cur]
        mask, blobs = self._analyze(frame_bgr, lab_now, scale=self._scan_scale())
        preview_bgr = self._highlight(frame_bgr, mask)

        img = self._to_maix(preview_bgr)
        if img.width() != disp_w or img.height() != disp_h:
            try:
                img = img.resize(disp_w, disp_h)
            except Exception as e:
                self._diag("resize", e)
        sx = img.width() / frame_bgr.shape[1]
        sy = img.height() / frame_bgr.shape[0]

        x, y, pressed = ts.read()
        clicked = pressed and not pressed_last
        self._pressed_last = pressed

        hit_btn = self._draw_groups(img, cur)
        act = self._draw_actions(img)
        if clicked:
            for gx, gname in hit_btn:
                if self._in(x, y, gx):
                    self._cur = gname
                    self._pick = None
            if self._in(x, y, act["exit"]):
                self._session_stop = True
            if self._in(x, y, act["save"]):
                self._sync_current_to_session()
                self._session.save()
                self._toast = "SAVED -> " + THRESHOLD_FILE
                self._toast_until = time.time() + 1.5
            if self._in(x, y, act["reset"]):
                self._session.reload()
                self._groups = self._session.groups()
                self._toast, self._toast_until = "RELOADED", time.time() + 1.0
            if "mirror" in act and self._in(x, y, act["mirror"]):
                self._toggle_mirror()
            px, py = int(x / sx), int(y / sy)
            if 0 <= px < frame_bgr.shape[1] and 0 <= py < frame_bgr.shape[0] \
                    and y > _UI_GROUP_H + _UI_ACTION_H:
                self._pick = (px, py)

        self._draw_sliders(img, self._groups[self._cur], x, y, pressed)
        self._sync_current_to_session()
        self._draw_info(img, blobs, mask, self._pick, frame_bgr, sx, sy)
        self._draw_roi_lines(img, cur, sx, sy)
        if self._toast and time.time() < self._toast_until:
            self._text(img, 8, img.height() - 24, self._toast,
                       self._C["yellow"], 1.2)
        return img

    def _draw_pit_status(self, img) -> None:
        processor = self._get_processor() if self._get_processor else None
        if processor is None:
            return
        pit = processor.pit_status
        y = _UI_GROUP_H + _UI_ACTION_H + 20
        if any(g < 3 for g in self._visible_names_idx()):
            text = (f"PIT:{pit.get('pit', 0)} "
                    f"LOCK {pit.get('pending', 0)}/{int(processor.config.get('pit_latch_frames', 2))}")
            color = self._C["green"] if pit.get("pit_flag") else self._C["yellow"]
            self._text(img, 8, y, text, color, 1.0)

    def _draw_roi_lines(self, img, group_idx: int, sx, sy) -> None:
        """靶标组画 target.roi 横线; 排爆组画 explosive.roi 横线（与处理器配置同源）。"""
        if group_idx in (0, 1, 2):
            roi = None
            processor = self._get_processor() if self._get_processor else None
            if processor is not None:
                try:
                    roi = processor.explosive_roi()
                except Exception as e:
                    self._diag("explosive_roi", e)
            if roi is None:
                roi = self._roi_from_config_cache("explosive")
            if roi is None:
                return
            h = img.height()
            color = self._C["yellow"]
            for fraction in (float(roi[1]), float(roi[3])):
                line_y = max(0, min(h - 1, int(fraction * h)))
                self._rect(img, 0, line_y, img.width(), 1, color=color, thickness=-1)
            return
        if self._target_group_index_range is None:
            return
        if group_idx not in self._target_group_index_range:
            return
        roi = None
        processor = self._get_processor() if self._get_processor else None
        if processor is not None:
            roi = processor.target_roi()
        if roi is None:
            roi = self._roi_from_config_cache()
        if roi is None:
            return
        h = img.height()
        color = self._C["yellow"]
        for fraction in (float(roi[1]), float(roi[3])):
            line_y = max(0, min(h - 1, int(fraction * h)))
            self._rect(img, 0, line_y, img.width(), 1, color=color, thickness=-1)

    def _roi_from_config_cache(self, section: str = "target"):
        if self._roi_cache is None:
            try:
                roi = load_gks_config().get(section, {}).get("roi")
                if isinstance(roi, (list, tuple)) and len(roi) == 4:
                    self._roi_cache = tuple(float(v) for v in roi)
            except Exception as e:
                self._diag("roi_cfg", e)
        return self._roi_cache

    # ---- data ----

    def _sync_current_to_session(self) -> None:
        self._session.set_group(self._cur, self._groups[self._cur])

    def reset_session_from_file(self) -> None:
        """从配置+阈值文件整载（覆盖未保存的内存态）。"""
        self._session.reload()
        self._groups = self._session.groups()

    # ---- cv helpers ----

    @staticmethod
    def _to_bgr(maix_img):
        import maix.image
        rgb = maix.image.image2cv(maix_img, ensure_bgr=False, copy=True)
        return rgb[:, :, ::-1]

    @staticmethod
    def _to_maix(bgr):
        import maix.image
        return maix.image.cv2image(bgr, bgr=True, copy=True)

    @staticmethod
    def _analyze(frame_bgr, lab_threshold, scale: float = 1.0,
                 min_area_px: int = 200):
        """返回 (mask, blobs); blobs = [{x,y,w,h,area}] 按面积降序前 5。

        **降采样分析（2026-10-07，CAM1 卡顿修复）**：

        实测热点是 `findContours` —— 它的耗时随命中噪点数量爆炸：PC 端低噪声
        0.08ms / 高噪声(11854 轮廓) 10.1ms，**差 126 倍**。也就是说"阈值一放宽
        命中一片噪点 → 轮廓炸到上万 → 逐个 contourArea/boundingRect 的Python
        循环再叠一层 → 页面骤然卡住"。这正是调参时"有时候卡有时候不卡"的来源。
        CAM1(cv181x) 算力本就比 CAM2 弱一个量级，叠加 30+ms 取图延时，手感更差。

        对策与主程序排爆 `explosive.scan_scale` 同一套办法：降采样后分析。

        - INTER_NEAREST **不插值**：插值会把边缘像素与背景混成中间色，
          那正是色块掉出阈值的同一机理（与 gks_task_processor._downscale 同因）。
        - 面积门槛按 `scale²` 折算（`min_area_px * scale²`）—— 保持"物理口径"
          不变，200px@全帧 等价 50px@0.5，与主程序面积折算一致。
        - bbox 坐标与 area 按 `1/scale` / `1/scale²` **还原到全帧坐标**，
          调用方（_draw_info 画框、_pixel_lab 取色）拿到的仍是原坐标系。
        - mask 升回全帧尺寸：_highlight 要在原图上叠加高亮。

        ⚠ 调参页语义是"看颜色命中在哪"，不需要全分辨率；降采样不改变判定结果，
          只影响零碎噪点的可见粒度（1/scale 像素级）。
        """
        import cv2
        fh, fw = frame_bgr.shape[:2]
        scale = float(scale)
        if not (0.0 < scale <= 1.0):
            scale = 1.0

        if scale >= 1.0:
            search = frame_bgr
        else:
            sw = max(1, int(round(fw * scale)))
            sh = max(1, int(round(fh * scale)))
            search = cv2.resize(frame_bgr, (sw, sh),
                                interpolation=cv2.INTER_NEAREST)

        lab_img = cv2.cvtColor(search, cv2.COLOR_BGR2LAB)
        lo = np.array([round(lab_threshold[0] * 2.55), lab_threshold[2] + 128,
                       lab_threshold[4] + 128], dtype=np.uint8).clip(0, 255)
        hi = np.array([round(lab_threshold[1] * 2.55), lab_threshold[3] + 128,
                       lab_threshold[5] + 128], dtype=np.uint8).clip(0, 255)
        mask = cv2.inRange(lab_img, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 面积门槛折算到小图口径（小图像素数少 scale² 倍）
        min_area_small = max(1, int(round(min_area_px * scale * scale)))
        inv = 1.0 / scale
        blobs = []
        for c in contours:
            area_small = int(cv2.contourArea(c))
            if area_small < min_area_small:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            # 还原到全帧坐标，调用方无需知道降采样存在。
            # ⚠ 面积是**放大** inv²（1/scale²）而非缩小 scale²：小图像素数少
            #   scale² 倍，要回到全帧口径必须乘回去。之前误乘 scale² 导致
            #   0.5 档把 10000px 的色块报成 625px（差了 16 倍），台架上读
            #   `max:{area}px` 会被严重误导（AGENTS.md: 排爆面积门槛正是按
            #   调参页读数取 40~50% 反推的）。
            blobs.append({
                "x": int(round(bx * inv)), "y": int(round(by * inv)),
                "w": int(round(bw * inv)), "h": int(round(bh * inv)),
                "area": int(round(area_small * inv * inv)),
            })
        blobs.sort(key=lambda b: b["area"], reverse=True)
        blobs = blobs[:5]

        if scale < 1.0:
            # 升回全帧：_highlight 在原图上按 mask 叠加高亮
            mask = cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
        return mask, blobs

    @staticmethod
    def _highlight(frame_bgr, mask):
        out = frame_bgr.copy()
        out[mask > 0] = (0.4 * out[mask > 0] + 0.6 * np.array([0, 255, 0])).astype(out.dtype)
        return out

    @staticmethod
    def _pixel_lab(frame_bgr, x, y):
        import cv2
        lab = cv2.cvtColor(frame_bgr[y:y + 1, x:x + 1], cv2.COLOR_BGR2LAB)
        l, a, b = int(lab[0, 0, 0]), int(lab[0, 0, 1]) - 128, int(lab[0, 0, 2]) - 128
        return l, a, b

    # ---- ui ----

    @staticmethod
    def _in(x, y, rect):
        return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]

    def _text(self, img, x, y, s, color, scale=1.0):
        try:
            img.draw_string(x, y, s, color=color, scale=scale, thickness=1)
        except Exception as e:
            self._diag(f"string@({x},{y})", e)

    def _rect(self, img, x, y, w, h, color, thickness=2):
        try:
            img.draw_rect(x, y, w, h, color=color, thickness=thickness)
        except Exception as e:
            self._diag(f"rect@({x},{y})", e)

    def _draw_groups(self, img, cur: int):
        rects = []
        buttons = self._visible_names
        bw = img.width() // max(1, len(buttons))
        for i, (group_idx, name) in enumerate(buttons):
            rect = (i * bw + 2, 2, bw - 4, _UI_GROUP_H - 6)
            color = self._C["green"] if group_idx == cur else self._C["gray_off"]
            self._rect(img, rect[0], rect[1], rect[2], rect[3], color=color)
            self._text(img, rect[0] + 8, rect[1] + 4, name, color, 1.2)
            rects.append((rect, group_idx))
        return rects

    def _draw_actions(self, img):
        y = _UI_GROUP_H
        h = _UI_ACTION_H
        rects = {
            "exit": (2, y, 90, h - 4),
            "save": (img.width() - 200, y, 90, h - 4),
            "reset": (img.width() - 104, y, 100, h - 4),
        }
        self._rect(img, *rects["exit"], color=self._C["white"])
        self._rect(img, *rects["save"], color=self._C["green"])
        self._rect(img, *rects["reset"], color=self._C["yellow"])
        self._text(img, rects["exit"][0] + 8, y + 6, "< Exit", self._C["white"], 1.2)
        self._text(img, rects["save"][0] + 8, y + 6, "SAVE", self._C["green"], 1.2)
        self._text(img, rects["reset"][0] + 8, y + 6, "RESET", self._C["yellow"], 1.2)
        if self._mirror_available():
            rects["mirror"] = (rects["exit"][0] + rects["exit"][2] + 6, y, 110, h - 4)
            self._rect(img, *rects["mirror"], color=self._C["cyan"])
            self._text(img, rects["mirror"][0] + 8, y + 6,
                       f"MIR:{'L' if self._mirror_state() else 'R'}", self._C["cyan"], 1.2)
        return rects

    def _mirror_available(self) -> bool:
        processor = self._get_processor() if self._get_processor else None
        return processor is not None and any(g < 3 for g in self._visible_names_idx())

    def _visible_names_idx(self):
        return [group_idx for group_idx, _name in self._visible_names]

    def _mirror_state(self) -> bool:
        processor = self._get_processor()
        return bool(processor.config.get("pit_left_is_1", True)) if processor else True

    def _toggle_mirror(self) -> None:
        processor = self._get_processor()
        if processor is None:
            return
        new_value = not self._mirror_state()
        try:
            processor.set_runtime_param("pit_left_is_1", "pit_left_is_1", new_value)
        except ValueError as e:
            self._diag("mirror_toggle", e)
            return
        self._toast = f"MIRROR={'LEFT=1' if new_value else 'RIGHT=1'}"
        self._toast_until = time.time() + 1.5

    def _draw_sliders(self, img, lab: list, tx, ty, pressed) -> bool:
        """2 列 x 3 行滑条; 按住拖动直接设值。返回是否发生修改。"""
        changed = False
        col_w = img.width() // 2
        top = img.height() - 3 * _UI_SLIDER_H
        for i, name in enumerate(SLIDERS):
            col, row = i % 2, i // 2
            x0 = col * col_w + 10
            y0 = top + row * _UI_SLIDER_H + 6
            w = col_w - 60
            lo, hi = SLIDER_RANGE[i]
            value = lab[i]
            self._rect(img, x0, y0 + 8, w, 8, self._C["slider_track"], thickness=-1)
            kx = x0 + int((value - lo) / (hi - lo) * w)
            self._rect(img, kx - 4, y0 + 2, 8, 20, self._C["green"], thickness=-1)
            self._text(img, x0, y0 - 2, f"{name}:{value}", self._C["white"], 1.0)
            if pressed and x0 <= tx <= x0 + w and y0 - 6 <= ty <= y0 + 30:
                new_v = int(lo + (tx - x0) / max(1, w) * (hi - lo))
                new_v = max(lo, min(hi, new_v))
                if new_v != lab[i]:
                    lab[i] = new_v
                    changed = True
        return changed

    def _draw_info(self, img, blobs, mask, pick, frame_bgr, sx, sy):
        total_px = int((mask > 0).sum())
        # 末尾附降采样档位: 台架据此确认优化生效 (1.0x=全帧, 0.5x=半分辨率)
        scale = self._scan_scale()
        info = f"hit_px:{total_px} blobs:{len(blobs)} s:{scale:g}x"
        if blobs:
            b = blobs[0]
            info += f" max:{b['area']}px ({b['x']},{b['y']},{b['w']},{b['h']})"
        self._text(img, 8, _UI_GROUP_H + _UI_ACTION_H + 4, info,
                   self._C["white"], 1.0)
        self._draw_pit_status(img)
        for b in blobs:
            self._rect(img, int(b["x"] * sx), int(b["y"] * sy),
                       int(b["w"] * sx), int(b["h"] * sy),
                       color=self._C["magenta"])
        if pick is not None:
            l, a, bb = self._pixel_lab(frame_bgr, pick[0], pick[1])
            self._text(img, img.width() - 220, _UI_GROUP_H + _UI_ACTION_H + 4,
                       f"pick({pick[0]},{pick[1]}) L:{l} A:{a} B:{bb}",
                       self._C["cyan"], 1.0)
            try:
                img.draw_circle(int(pick[0] * sx), int(pick[1] * sy), 4,
                                color=self._C["cyan"], thickness=2)
            except Exception as e:
                self._diag("pick_circle", e)
