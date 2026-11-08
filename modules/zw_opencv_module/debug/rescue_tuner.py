# -*- coding: utf-8 -*-
"""人质(RESCUE)调试页 — YOLO 参数触屏调参。

conf_th/iou_th/x_crop_px 三条滑条即时写 GksTaskProcessor 运行时参数
（process() 立即生效）; 顶栏 **SAVE** 把当前值落盘到
/my-threshold/my_rescue_params.txt（key=value 行）—— 主程序按 mtime 热重载
（不清锁存, 语义同 my_64threshold.txt）, eod_config.yaml 保持只读不毁注释。
RESET 回本次进页时的加载值（yaml + 覆盖文件合并结果）。模型状态来自注入的
ai provider（framework Machine.ai）: 显示 active_model/labels 数量, 未加载红字。
画面预览: 本页 tick 由 main.py 喂原始相机帧, 页内自行调用注入的 AI HAL
跑一次 detect()（运行参数 conf/iou 与 x_crop 过滤口径同
GksTaskProcessor._process_rescue）并把检测框叠加到预览上 —— 调试页独占
相机期间主管线已暂停, 页内推理是 YOLO 参数唯一的实时验证途径, 不影响
coordinator 的归零心跳兜底。
检出日志（逐框 label:score 标签）之外, 主流程侧的"变化才打"汇总日志见
GksTaskProcessor._log_rescue_dets。
绘图颜色规范同 threshold_tuner.py 文件头说明。
"""
import os
import time

import numpy as np

from framework.display.gate import warn_slow_format
from modules.zw_opencv_module.processors.gks_task_processor import RESCUE_PARAMS_FILE


_UI_SLIDER_H = 50
_UI_TOPBAR_H = 30


def _pack(rgb) -> int:
    r, g, b = rgb
    return (int(r) & 0xFF) << 16 | (int(g) & 0xFF) << 8 | (int(b) & 0xFF)


class GksRescueTuner:
    """人质调试页渲染器: 喂帧渲染 + 触屏调参。"""

    def __init__(self, get_processor, get_ai):
        self._get_processor = get_processor
        self._get_ai = get_ai
        self._sliders = [
            {"key": "conf_th", "label": "conf", "lo": 0.30, "hi": 0.90},
            {"key": "iou_th", "label": "iou", "lo": 0.30, "hi": 0.70},
            {"key": "x_crop_px", "label": "xcrop", "lo": 0, "hi": 640},
        ]
        self._defaults = {}
        self._stop = False
        self._pressed_last = False
        self._toast = ""
        self._toast_until = 0.0
        self._C = self._build_colors()
        self._diag_logged = {}

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
            "red": self._mk_color((255, 64, 64)),
            "yellow": self._mk_color((255, 255, 0), "COLOR_YELLOW"),
            "gray": self._mk_color((120, 120, 120), "COLOR_GRAY"),
            "cyan": self._mk_color((0, 255, 255), "COLOR_CYAN"),
        }

    # ---- 分项归因 ----

    def _note_phase(self, key: str, dt_s: float) -> None:
        ph = self.__dict__.get("_phases")
        if ph is None:
            ph = self.__dict__["_phases"] = {
                "t0": 0.0, "n": 0, "infer": 0.0, "canvas": 0.0,
                "resize": 0.0, "draw": 0.0, "show": 0.0}
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
            log_print(f"[PERF] rescue page {n / elapsed:.1f}Hz | "
                      f"infer={ph['infer'] / n:.1f} canvas={ph['canvas'] / n:.1f} "
                      f"resize={ph['resize'] / n:.1f} draw={ph['draw'] / n:.1f} "
                      f"show={ph['show'] / n:.1f}")
        except Exception:
            pass
        ph.update(t0=now, n=0, infer=0.0, canvas=0.0, resize=0.0,
                  draw=0.0, show=0.0)

    def _diag(self, site, exc):
        if site not in self._diag_logged:
            self._diag_logged[site] = True
            try:
                from utils.log_util import log_print
                log_print(f"[RESCUE-TUNER] fail at {site}: {type(exc).__name__}: {exc}")
            except Exception:
                pass

    # ---- inference & overlay ----

    def _infer(self, frame_bgr):
        """页内跑一次 YOLO 推理（参数由处理器的实时 config 提供）。

        AI 缺失/未加载/推理失败时返回 [], 预览仍显示原始画面,
        模型状态由顶部栏红字说明 —— 页面自身不因 AI 状态降级崩溃。
        """
        ai = self._get_ai() if callable(self._get_ai) else None
        if ai is None or not getattr(ai, "loaded", False) or frame_bgr is None:
            return []
        try:
            processor = self._get_processor()
            rescue_cfg = (processor.config.get("rescue", {})
                          if processor is not None else {})
            return ai.detect(frame_bgr,
                             conf_th=float(rescue_cfg.get("conf_th", 0.7)),
                             iou_th=float(rescue_cfg.get("iou_th", 0.45)))
        except Exception as e:
            self._diag("infer", e)
            return []

    def _draw_detections(self, img, dets, frame_bgr, disp_w, disp_h) -> None:
        self._draw_latch_status(img, disp_h)
        if not dets or frame_bgr is None:
            return
        sx = disp_w / frame_bgr.shape[1]
        sy = disp_h / frame_bgr.shape[0]
        processor = self._get_processor()
        rescue_cfg = (processor.config.get("rescue", {})
                      if processor is not None else {})
        labels = []
        ai = self._get_ai() if callable(self._get_ai) else None
        if ai is not None:
            labels = list(getattr(ai, "labels", None) or [])
        shape_map = rescue_cfg.get("labels", {})
        x_crop = int(rescue_cfg.get("x_crop_px") or 0)

        for det in dets:
            label = labels[det.class_id] if det.class_id < len(labels) else ""
            shape_num = int(shape_map.get(label, 0))
            # 颜色语义与 _process_rescue hits 对齐:
            # 计入坑位的形状 = 橙色, 未映射标签 = 青色, x_crop 过滤 = 灰色
            if x_crop > 0 and det.x > x_crop:
                color, tag = self._C["gray"], "X"
            elif shape_num:
                color, tag = self._C["yellow"], f"S{shape_num}"
            else:
                color, tag = self._C["cyan"], "?"
            self._rect(img, int(det.x * sx), int(det.y * sy),
                       max(2, int(det.w * sx)), max(2, int(det.h * sy)),
                       color, thickness=2)
            self._text(img, int(det.x * sx), max(2, int(det.y * sy) - 14),
                       f"{tag}:{label or det.class_id}:{det.score:.2f}",
                       color, 0.9)

    def _draw_latch_status(self, img, disp_h) -> None:
        """坑位锁存进度行（坐标口径同 _process_rescue 的锁存数据）。"""
        try:
            processor = self._get_processor()
            ps = processor.pit_status if processor is not None else {}
        except Exception as e:
            self._diag("latch_view", e)
            return
        if ps:
            self._text(img, 8, disp_h - 3 * _UI_SLIDER_H - 16,
                       f"latch:{list(ps.get('latched', []))} "
                       f"shapes:{ps.get('shapes_seen', {})}",
                       self._C["cyan"], 0.9)

    # ---- lifecycle ----

    def begin(self) -> None:
        processor = self._get_processor()
        if processor is not None:
            rescue_cfg = processor.config.get("rescue", {})
            self._defaults = {
                s["key"]: rescue_cfg[s["key"]]
                for s in self._sliders if s["key"] in rescue_cfg
            }
        self._pressed_last = False
        self._toast, self._toast_until = "RESET->code defaults", time.time() + 1.0

    def step(self, disp, ts, frame_bgr) -> None:
        """分项归因: infer / canvas / resize / draw / show 五段。"""
        _t = time.perf_counter()
        dets = self._infer(frame_bgr)
        self._note_phase("infer", time.perf_counter() - _t)
        _t = time.perf_counter()
        img = self._to_maix(frame_bgr)
        self._note_phase("canvas", time.perf_counter() - _t)
        _t = time.perf_counter()
        img = self._fit_display(img, disp)
        self._note_phase("resize", time.perf_counter() - _t)
        _t = time.perf_counter()
        self._rect(img, 0, 0, img.width(), _UI_TOPBAR_H,
                   color=self._C["black"], thickness=-1)
        # 叠加画在 topbar 之后: 检测框永远可见, 后画的 UI 控件优先响应触摸
        self._draw_detections(img, dets, frame_bgr, img.width(), img.height())
        self._draw_model_status(img)

        x, y, pressed = ts.read()
        clicked = pressed and not self._pressed_last
        self._pressed_last = pressed
        if clicked and self._in(x, y, (2, 2, 90, _UI_TOPBAR_H - 6)):
            self._stop = True
            _t = time.perf_counter()
            warn_slow_format(disp, img)   # 告警而非 assert: 抛异常会被上层吞成冻屏
            disp.show(img)
            self._flush_phase(time.perf_counter() - _t)
            return
        if clicked and self._in(x, y, (img.width() - 212, 2, 100, _UI_TOPBAR_H - 6)):
            self._save_params()
        if clicked and self._defaults is not None and self._in(
                x, y, (img.width() - 104, 2, 100, _UI_TOPBAR_H - 6)):
            self._reset_params()
            self._toast = "RESET -> loaded values"
            self._toast_until = time.time() + 1.0
        self._note_phase("draw", time.perf_counter() - _t)
        self._draw_sliders(img, x, y, pressed)
        if self._toast and time.time() < self._toast_until:
            self._text(img, 8, 34, self._toast, self._C["yellow"], 1.2)
        _t = time.perf_counter()
        warn_slow_format(disp, img)   # 告警而非 assert: 抛异常会被上层吞成冻屏
        disp.show(img)
        self._flush_phase(time.perf_counter() - _t)

    def exit_requested(self) -> bool:
        return self._stop

    # ---- internals ----

    @staticmethod
    def _in(x, y, rect):
        return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]

    def _read_param(self, key: str):
        processor = self._get_processor()
        if processor is None:
            return 0.0
        return processor.config.get("rescue", {}).get(key, 0.0)

    def _write_param(self, key: str, value) -> None:
        processor = self._get_processor()
        if processor is not None:
            try:
                processor.set_runtime_param("rescue", key, value)
            except ValueError as e:
                self._diag("set_param", e)

    def _reset_params(self) -> None:
        for s in self._sliders:
            if s["key"] in self._defaults:
                self._write_param(s["key"], self._defaults[s["key"]])

    def _save_params(self) -> None:
        """当前 rescue 参数落盘 RESCUE_PARAMS_FILE（主程序 mtime 热重载）。

        目录不存在自动创建; 写失败只 toast 红字 + [RESCUE-TUNER] 诊断日志,
        页面不崩（与 LAB 阈值页 SAVE 同级容错）。
        """
        processor = self._get_processor()
        if processor is None:
            self._toast, self._toast_until = "SAVE FAILED: no processor", time.time() + 1.5
            return
        try:
            rescue_cfg = processor.config.get("rescue", {})
            lines = ["# GKS rescue tuner SAVE (key=value, mtime hot-reload)"]
            for s in self._sliders:
                key = s["key"]
                if key not in rescue_cfg:
                    continue
                v = rescue_cfg[key]
                lines.append(f"{key}={v:.2f}" if float(s["hi"]) - float(s["lo"]) < 1
                             else f"{key}={int(v)}")
            parent = os.path.dirname(RESCUE_PARAMS_FILE)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(RESCUE_PARAMS_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            try:
                from utils.log_util import log_print
                log_print(f"[RESCUE-TUNER] saved {RESCUE_PARAMS_FILE}: "
                          + " ".join(lines[1:]))
            except Exception:
                pass
            self._toast, self._toast_until = "SAVED -> /my-threshold", time.time() + 1.5
        except Exception as e:
            self._diag("save", e)
            self._toast, self._toast_until = "SAVE FAILED (see log)", time.time() + 1.5

    def _draw_model_status(self, img) -> None:
        ai = self._get_ai() if callable(self._get_ai) else None
        model = getattr(ai, "active_model", None) if ai else None
        labels = getattr(ai, "labels", None) or []
        if model and getattr(ai, "loaded", False):
            text = f"AI:{model} labels:{len(labels)}"
            color = self._C["green"]
        else:
            text = "AI: NOT LOADED (RESCUE returns NACK NOT_READY)"
            color = self._C["red"]
        self._text(img, 100, 4, text, color, 1.2)
        self._text(img, 8, 6, "< Exit", self._C["white"], 1.2)
        # 顶栏右侧: SAVE(绿) | RESET(黄)。点击区见 step(), 宽 100px 各半分。
        self._rect(img, img.width() - 208, 4, 92, _UI_TOPBAR_H - 10,
                   self._C["green"], thickness=-1)
        self._text(img, img.width() - 204, 6, "SAVE", self._C["black"], 1.2)
        self._text(img, img.width() - 100, 6, "RESET", self._C["yellow"], 1.2)

    def _draw_sliders(self, img, tx, ty, pressed) -> None:
        col_w = img.width() // 2
        top = img.height() - 3 * _UI_SLIDER_H
        for i, s in enumerate(self._sliders):
            col, row = i % 2, i // 2
            x0 = col * col_w + 10
            y0 = top + row * _UI_SLIDER_H + 6
            w = col_w - 60
            lo, hi = float(s["lo"]), float(s["hi"])
            value = float(self._read_param(s["key"]))
            track_y = y0 + 8
            self._rect(img, x0, track_y, w, 8, self._C["gray"], thickness=-1)
            kx = x0 + int((value - lo) / max(1e-6, hi - lo) * w)
            self._rect(img, kx - 4, y0 + 2, 8, 20, self._C["green"], thickness=-1)
            self._text(img, x0, y0 - 2, f"{s['label']}:{value:.2f}" if hi - lo < 1
                       else f"{s['label']}:{int(value)}", self._C["white"], 1.0)
            if pressed and x0 <= tx <= x0 + w and y0 - 6 <= ty <= y0 + 30:
                new_v = lo + (tx - x0) / max(1, w) * (hi - lo)
                new_v = max(lo, min(hi, new_v))
                if hi - lo < 1:
                    value_out = round(new_v, 2)
                else:
                    value_out = int(round(new_v))
                self._write_param(s["key"], value_out)

    def _fit_display(self, img, disp):
        if img.width() != disp.width() or img.height() != disp.height():
            from utils.log_util import perf_note, perf_t0
            _t0 = perf_t0()
            try:
                img = img.resize(disp.width(), disp.height())
            except Exception as e:
                self._diag("resize", e)
            else:
                if _t0:
                    perf_note("resize", (time.perf_counter() - _t0) * 1000.0,
                              "{}x{}".format(disp.width(), disp.height()))
        return img

    def _to_maix(self, bgr):
        """BGR numpy -> FMT_RGB888 原生画布, 免 show() 内部整帧转换。"""
        # RGB 字节本就正确; ascontiguousarray 只保证连续(防 pybind11 拒收/镜像)。
        # ⚠ 若去掉 copy=True, 负步长视图会 OOB 或整帧镜像 —— 不是换色通道。
        import maix.image
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        return maix.image.cv2image(rgb, bgr=False, copy=True)

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
