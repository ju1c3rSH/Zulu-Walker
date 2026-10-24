# -*- coding: utf-8 -*-
"""激光校准(LASER)调试页 — 红点检测 + err 零点校准。

职责: Laser 按钮经 0x25 CMD_LASER 帧控制 Brain 板载激光(无应答); 页内全分辨率
分层检测红色激光点(白核+红环+光晕; 旧"绝对红"判据 + "白核+淡晕"签名入口, 见下),
命中画十字+实时坐标;
CALIB 把当前点 x 存为 TARGET err 零点(laser_cx, 立即生效), RESET 回画面中心,
SAVE 落盘 /my-threshold/my_laser_calib.txt(主程序 mtime 热重载)。

检测分层(每层堵上一层的失效模式, 依据 laser 点成像调研):
  1. 候选提取: R>=lum_min 且 R-max(G,B)>=dom_min(红环满足, 白核三通道同饱和被排除);
  2. 颜色确认: dom_min 判据天然排除白墙/灯光高光(白色 R-G≈0);
  3. 质心校正: R 通道加权质心; 权重和为 0 时退回连通域几何质心。
激光点仅 3~30px, **必须全分辨率**运行, 不进主管线(主管线暂停期间页内自算)。

亮环境/白背景签名扩展(2026-10-25; 本页只用于标定激光零点, 与 TARGET 主链路无关):
  台架实测(bbcf8e0): 白背景 + ~50cm 远距时光斑过曝泛白成"亮度极高的白点 + 一点
  红晕", 晕的 dom≈25~45 落在逐像素 dom_min=40 门限两侧, 随 AE 抖动/噪声概率性
  丢失(蒙特卡洛 37/100); 暗背景/近距 dom 富余, 旧判据本来就够 —— 与台架
  "白色背景不行、暗一点就行"的观察一致。
  签名方案: 激光点的指纹 = "三通道全饱和的白核(R,G,B>=prom_min) + 核周淡红晕"。
    1. 第二入口: 白核掩码膨胀 halo_pad 后, 允许核周边亮像素(R>=lum_min)免红判据
       进候选 —— 白化光斑的芯+晕整体进掩码。halo_pad=0 或帧内无白核时整段跳过,
       掩码与旧实现逐像素一致(兼容开关)。
    2. 逐块确认: bbox 外扩 halo_pad 的窗口内 max(dom)>=dom_min(强红, 近距) 或
       (有白核 且 max(dom)>=dom_faint)(白核+淡晕, 远距)。窗口是 bbox 联合外扩,
       晕进没进掩码都覆盖; 白墙/反光只有白核没有晕, 被此层拒掉。
    3. 打分 score=sqrt(area)*bbox内R峰值 与 R 加权质心沿用。
  标定读数: 检出时 Laser 按钮下方显示 a/core/dom —— core=最白像素的三通道最小值
  (prom_min 取 core-10), dom=确认窗口内红度量峰值(dom_faint 取其 1/2~2/3)。
  调参入口同原有四项: LASER 页 SAVE -> /my-threshold/my_laser_calib.txt 热重载。

安全: 激光只在 _laser_on 时亮; Exit/切页(on_leave)/程序退出均自动发关帧。
绘图颜色规范同 threshold_tuner.py 文件头说明。
"""
import math
import os
import time

import cv2
import numpy as np

from framework.display.gate import warn_slow_format
from modules.zw_opencv_module.processors.gks_task_processor import (
    LASER_CALIB_FILE, _LASER_CALIB_KEYS,
)


_UI_TOPBAR_H = 30
_UI_BTN_W = 70
_UI_LASER_BTN = (12, 60, 130, 54)   # Laser 开关按钮 (x, y, w, h)


def _pack(rgb) -> int:
    r, g, b = rgb
    return (int(r) & 0xFF) << 16 | (int(g) & 0xFF) << 8 | (int(b) & 0xFF)


class GksLaserCalibTuner:
    """激光校准调试页渲染器: 喂帧渲染 + 触屏操作。"""

    def __init__(self, get_processor, send_laser):
        self._get_processor = get_processor
        self._send_laser = send_laser
        self._laser_on = False
        self._dot = None            # (lx, ly, area) | None
        self._last_stats = None     # 检出块的标定读数 {area, dom, core} | None
        self._stop = False
        self._pressed_last = False
        self._toast = ""
        self._toast_until = 0.0
        self._C = self._build_colors()
        self._diag_logged = set()

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
            "amber": self._mk_color((240, 160, 40)),
        }

    def _note_phase(self, key: str, dt_s: float) -> None:
        ph = self.__dict__.get("_phases")
        if ph is None:
            ph = self.__dict__["_phases"] = {
                "t0": 0.0, "n": 0, "detect": 0.0, "canvas": 0.0,
                "resize": 0.0, "draw": 0.0, "show": 0.0}
        ph[key] += dt_s * 1000.0

    def _flush_phase(self, show_s: float) -> None:
        """每秒一条 [PERF] laser page: 把一轮 step() 拆成 detect/canvas/resize/draw/show。

        这是回答"激光页为什么慢"的唯一依据: 若 canvas+resize 占大头, 说明瓶颈在
        `cv2image(copy=True)` + `resize`（格式/内存来源问题, 该由 presenter 统一供给）;
        若 detect 占大头, 说明是全分辨率检测的锅。
        """
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
        total = sum(ph[k] for k in ("detect", "canvas", "resize", "draw", "show"))
        try:
            from utils.log_util import log_print
            log_print(
                f"[PERF] laser page {n / elapsed:.1f}Hz total={total / n:.1f}ms | "
                f"detect={ph['detect'] / n:.1f} canvas={ph['canvas'] / n:.1f} "
                f"resize={ph['resize'] / n:.1f} draw={ph['draw'] / n:.1f} "
                f"show={ph['show'] / n:.1f}")
        except Exception:
            pass
        ph.update(t0=now, n=0, detect=0.0, canvas=0.0, resize=0.0,
                  draw=0.0, show=0.0)

    def _diag(self, site, exc):
        if site not in self._diag_logged:
            self._diag_logged.add(site)
            try:
                from utils.log_util import log_print
                log_print(f"[LASER] fail at {site}: {type(exc).__name__}: {exc}")
            except Exception:
                pass

    # ---- laser dot detection (full resolution, layered) ----

    def detect_dot(self, frame_bgr):
        """分层检测红激光点 -> (lx, ly, area) | None。参数来自 cfg["laser"]。

        判据链见文件头: 旧"绝对红"入口 + "白核+淡晕"签名入口(默认开启,
        halo_pad=0 整体关闭回到旧判据)。检出时把 core/dom 实测值存到
        self._last_stats 供 HUD 读数 —— 台架据此定 prom_min/dom_faint。
        """
        if frame_bgr is None:
            return None
        self._last_stats = None
        processor = self._get_processor()
        cfg = (processor.config.get("laser", {}) if processor is not None else {})
        lum_min = int(cfg.get("lum_min", 200))
        dom_min = int(cfg.get("dom_min", 40))
        min_area = int(cfg.get("min_area", 8))
        max_area = int(cfg.get("max_area", 4000))
        # ---- 白核+淡红晕签名参数(见文件头) ----
        dom_faint = int(cfg.get("dom_faint", 12))
        halo_pad = int(cfg.get("halo_pad", 3))
        prom_min = int(cfg.get("prom_min", 240))
        h, w = frame_bgr.shape[:2]
        b, g, r = frame_bgr[:, :, 0], frame_bgr[:, :, 1], frame_bgr[:, :, 2]
        lum_term = cv2.inRange(r, lum_min, 255)
        # cv2.subtract 在 uint8 上会**饱和到 0**, 所以它只保留 r-max(g,b) 的非负部分。
        # 判据 r-max(g,b) >= dom_min 分两种情况才严格等价于原来的 int16 版本:
        #   dom_min >= 1: 饱和丢掉的负数本来就 < 1 <= dom_min, 不可能命中 -> inRange 够用;
        #   dom_min == 0: 原判据要求 r-max(g,b) >= 0 即 **r >= max(g,b)**,
        #                 而饱和后的 0 会让 r < max(g,b) 的像素也通过 0>=0 —— 会放宽判据。
        #                 故 lo==0 必须显式比较, 否则纯白/偏蓝像素会混进候选（已实测复现）。
        lo = max(0, int(dom_min))
        gb = cv2.max(g, b)
        if lo > 0:
            dom_term = cv2.inRange(cv2.subtract(r, gb), lo, 255)
        else:
            dom_term = cv2.compare(r, gb, cv2.CMP_GE)
        # ---- 签名第二入口: 三通道全饱和的白核膨胀 halo_pad 后, 核周边亮像素免红
        # 判据进候选 —— 远距白化光斑(白核+淡晕)由此复活。halo_pad=0 或帧内无白核时
        # 整段跳过, 掩码与旧实现逐像素一致。白核膨胀区这道门本身就是抗噪闸:
        # 白墙/纹理噪声没有白核, 不会因签名入口涌入候选(百分位方案曾因此 32ms)。
        if halo_pad > 0:
            core = ((r >= prom_min) & (g >= prom_min)
                    & (b >= prom_min)).astype(np.uint8) * 255
            if cv2.countNonZero(core):
                pad_kernel = cv2.getStructuringElement(
                    cv2.MORPH_RECT, (2 * halo_pad + 1, 2 * halo_pad + 1))
                mask = lum_term & (dom_term | cv2.dilate(core, pad_kernel))
            else:
                mask = cv2.bitwise_and(lum_term, dom_term)
        else:
            mask = cv2.bitwise_and(lum_term, dom_term)
        # 空掩码占绝大多数帧（激光没开/没打进视场）: 一次 countNonZero 就能整段
        # 短路掉 morphologyEx + CC + 那张 1.2MB 的 label 图。实测这条快路径
        # 把"无点"帧的检测成本压到 ~1/6。
        if cv2.countNonZero(mask) == 0:
            return None
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        # ---- 逐块颜色确认 + 打分选块 ----
        # score=sqrt(area)*peak_R: 面积给规模, 峰值给"是不是激光", 两者相乘
        # 让小而极亮的点赢过大而平庸的块。
        best = None
        best_score = -1.0
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area:
                continue
            ok, dom_peak, core_peak = self._dot_signature(
                frame_bgr, stats, i, b, g, r, lo, dom_faint, halo_pad, prom_min)
            if not ok:
                continue
            peak = int(self._roi_peak(frame_bgr, stats, i))
            score = math.sqrt(area) * peak
            if score > best_score:
                best_score = score
                best = (i, centroids[i], area, dom_peak, core_peak)
        if best is None:
            return None
        idx, centroid, area, dom_peak, core_peak = best
        self._last_stats = {"area": int(area), "dom": int(dom_peak),
                            "core": int(core_peak)}
        cx, cy = self._core_centroid(frame_bgr, stats, idx) or centroid
        return (float(cx), float(cy), int(area))

    # ---- 颜色确认 / 质心(亮环境扩展) ----

    @staticmethod
    def _bbox_pad(stats, idx, pad, h, w):
        """bbox 向外扩 pad 像素后的 (y0, y1, x0, x1), 已按帧边界裁剪。"""
        x, y = int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP])
        bw, bh = int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT])
        return (max(0, y - pad), min(h, y + bh + pad),
                max(0, x - pad), min(w, x + bw + pad))

    @classmethod
    def _roi_peak(cls, frame_bgr, stats, idx):
        """bbox 内 R 通道峰值(打分用)。"""
        h, w = frame_bgr.shape[:2]
        y0, y1, x0, x1 = cls._bbox_pad(stats, idx, 0, h, w)
        if y1 <= y0 or x1 <= x0:
            return 0
        return int(frame_bgr[y0:y1, x0:x1, 2].max())

    @classmethod
    def _dot_signature(cls, frame_bgr, stats, idx, b, g, r,
                       lo, dom_faint, halo_pad, prom_min):
        """逐块颜色确认(见文件头)。返回 (ok, dom峰值, 白核峰值) 供 HUD 读数。

        b1 强红: bbox+halo_pad 窗口内 max(r-max(g,b)) >= dom_min —— 近距饱和红点;
        b2 白核+淡晕: 同窗口内存在白核(R,G,B 全>=prom_min) 且 max(dom) >= dom_faint
            —— 远距白化光斑。窗口是 bbox 联合外扩, 晕在部件内还是部件外都覆盖,
            不依赖"晕进没进掩码"的几何巧合。

        lo==0(dom_min=0) 时 b1 恒过 —— 与旧实现一致: 那时绝对红判据是
        r>=max(g,b) 的逐像素门限, 已由候选掩码承担, 这里不再重复收紧。
        """
        h, w = frame_bgr.shape[:2]
        y0, y1, x0, x1 = cls._bbox_pad(stats, idx, halo_pad, h, w)
        if y1 <= y0 or x1 <= x0:
            return False, 0, 0
        rr = r[y0:y1, x0:x1]
        gg = g[y0:y1, x0:x1]
        bb = b[y0:y1, x0:x1]
        dom = cv2.subtract(rr, cv2.max(gg, bb))
        dom_peak = int(dom.max())
        # "最白像素"的三通道最小值: 白核处 ≈255, 红像素处 ≈G/B。HUD 据此读 prom_min。
        core_peak = int(cv2.min(rr, cv2.min(gg, bb)).max())
        ok = dom_peak >= lo or (core_peak >= prom_min and dom_peak >= dom_faint)
        return ok, dom_peak, core_peak

    @staticmethod
    def _core_centroid(frame_bgr, stats, idx):
        """R 通道加权质心: 白核+光晕一起按亮度拉权, 比原 R>=250&G>=250 硬门限稳
        (远距小点凑不满 3px 白核时会退到很粗的几何质心); 权重和为 0 时返回
        None(调用方退回连通域几何质心)。"""
        x, y = int(stats[idx, cv2.CC_STAT_LEFT]), int(stats[idx, cv2.CC_STAT_TOP])
        bw, bh = int(stats[idx, cv2.CC_STAT_WIDTH]), int(stats[idx, cv2.CC_STAT_HEIGHT])
        roi = frame_bgr[y:y + bh, x:x + bw]
        if roi.size == 0:
            return None
        weight = roi[:, :, 2]
        total = float(weight.sum())
        if total <= 0.0:
            return None
        ys, xs = np.nonzero(weight)
        wv = weight[ys, xs].astype(np.float64)
        return (x + float((xs * wv).sum() / total),
                y + float((ys * wv).sum() / total))

    # ---- calibration & persistence ----

    def _calib(self, lx: int) -> None:
        processor = self._get_processor()
        if processor is None:
            self._toast, self._toast_until = "NO PROCESSOR", time.time() + 1.2
            return
        try:
            processor.set_runtime_param("target", "laser_cx", int(lx))
            self._toast, self._toast_until = f"CALIB cx={int(lx)}", time.time() + 1.5
        except ValueError as e:
            self._diag("calib", e)

    def _reset(self) -> None:
        processor = self._get_processor()
        if processor is None:
            return
        try:
            processor.set_runtime_param("target", "laser_cx", 0)
            self._toast, self._toast_until = "RESET -> center", time.time() + 1.2
        except ValueError as e:
            self._diag("reset", e)

    def _save(self) -> None:
        processor = self._get_processor()
        if processor is None:
            self._toast, self._toast_until = "SAVE FAILED: no processor", time.time() + 1.5
            return
        try:
            cfg = processor.config
            lines = ["# GKS laser calib SAVE (key=value, mtime hot-reload)"]
            for key, (section, _, _) in _LASER_CALIB_KEYS.items():
                if key in cfg.get(section, {}):
                    lines.append(f"{key}={int(cfg[section][key])}")
            parent = os.path.dirname(LASER_CALIB_FILE)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(LASER_CALIB_FILE, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            from utils.log_util import log_print
            log_print(f"[LASER] SAVE {LASER_CALIB_FILE}: " + " ".join(lines[1:]))
            self._toast, self._toast_until = "SAVED -> /my-threshold", time.time() + 1.5
        except Exception as e:
            self._diag("save", e)
            self._toast, self._toast_until = "SAVE FAILED (see log)", time.time() + 1.5

    def _toggle_laser(self) -> None:
        want = not self._laser_on
        ok = False
        try:
            ok = bool(self._send_laser(want))
        except Exception as e:
            self._diag("send_laser", e)
        if ok:
            self._laser_on = want
            from utils.log_util import log_print
            log_print(f"[LASER] tx CMD_LASER on={int(want)}")
        else:
            self._toast, self._toast_until = "LASER TX FAIL", time.time() + 1.5

    def on_leave(self) -> None:
        """切页/退出兜底: 激光开着就自动发关帧（多条路径之一, 幂等）。"""
        if not self._laser_on:
            return
        self._laser_on = False
        try:
            self._send_laser(False)
            from utils.log_util import log_print
            log_print("[LASER] page leave, auto-off sent")
        except Exception as e:
            self._diag("auto_off", e)

    # ---- lifecycle & rendering ----

    def begin(self) -> None:
        self._laser_on = False
        self._dot = None
        self._last_stats = None
        self._pressed_last = False
        processor = self._get_processor()
        zero = processor.target_zero_px() if processor is not None else 0
        self._toast, self._toast_until = (
            f"laser_cx={zero}" if zero else "not calibrated",
            time.time() + 1.2)

    def step(self, disp, ts, frame_bgr) -> None:
        # 分项归因（2026-10-15）: 调试页每轮 ~150ms 而 show 只占 ~18ms, 剩下必须
        # 拆开看。四项全部是墙钟 ms, 每秒汇总一条 [PERF] laser page。
        # flag 把 dump 挪到 show 之后再做（show 由上层代理计时, 这里不重复计）。
        _t = time.perf_counter()
        self._dot = self.detect_dot(frame_bgr)
        self._note_phase("detect", time.perf_counter() - _t)

        _t = time.perf_counter()
        img = self._to_maix(frame_bgr)
        self._note_phase("canvas", time.perf_counter() - _t)

        _t = time.perf_counter()
        img = self._fit_display(img, disp)
        self._note_phase("resize", time.perf_counter() - _t)

        _t = time.perf_counter()
        self._draw_overlay(img)
        self._rect(img, 0, 0, img.width(), _UI_TOPBAR_H,
                   color=self._C["black"], thickness=-1)
        self._draw_topbar(img)
        self._draw_laser_btn(img)

        x, y, pressed = ts.read()
        clicked = pressed and not self._pressed_last
        self._pressed_last = pressed
        if clicked:
            self._handle_click(img, x, y)
        if self._toast and time.time() < self._toast_until:
            self._text(img, 8, 34, self._toast, self._C["yellow"], 1.2)
        self._note_phase("draw", time.perf_counter() - _t)
        _t = time.perf_counter()
        warn_slow_format(disp, img)   # 告警而非 assert: 抛异常会被上层吞成冻屏
        disp.show(img)
        # show 的真实耗时由上层代理统计; 这里只把"非 show"的 step 总时长记下来
        self._flush_phase(time.perf_counter() - _t)

    def _handle_click(self, img, x, y) -> None:
        if self._in(x, y, (2, 2, 90, _UI_TOPBAR_H - 6)):
            self._stop = True
            self.on_leave()
            return
        if self._in(x, y, (img.width() - 3 * _UI_BTN_W - 8, 2,
                           _UI_BTN_W, _UI_TOPBAR_H - 6)):
            if self._dot is not None:
                self._calib(int(self._dot[0]))
            else:
                self._toast, self._toast_until = "NO DOT", time.time() + 1.2
        elif self._in(x, y, (img.width() - 2 * _UI_BTN_W - 6, 2,
                             _UI_BTN_W, _UI_TOPBAR_H - 6)):
            self._save()
        elif self._in(x, y, (img.width() - _UI_BTN_W - 4, 2,
                             _UI_BTN_W, _UI_TOPBAR_H - 6)):
            self._reset()
        elif self._in(x, y, _UI_LASER_BTN):
            self._toggle_laser()

    def _draw_overlay(self, img) -> None:
        if self._dot is None:
            return
        lx, ly, area = self._dot
        try:
            img.draw_cross(int(lx), int(ly), color=self._C["red"], size=10, thickness=2)
        except Exception:
            self._rect(img, int(lx) - 8, int(ly), 17, 1, self._C["red"], thickness=-1)
            self._rect(img, int(lx), int(ly) - 8, 1, 17, self._C["red"], thickness=-1)
        self._text(img, min(img.width() - 120, int(lx) + 12),
                   max(2, int(ly) - 12),
                   f"({int(lx)},{int(ly)}) a={area}", self._C["red"], 0.9)

    def _draw_topbar(self, img) -> None:
        processor = self._get_processor()
        zero = processor.target_zero_px() if processor is not None else 0
        color = self._C["cyan"] if zero else self._C["gray"]
        self._text(img, 8, 6, "< Exit", self._C["white"], 1.2)
        self._text(img, 100, 6, f"zero:{zero or 'center'}", color, 1.0)
        for off, (label, color) in enumerate((("CALIB", self._C["green"]),
                                              ("SAVE", self._C["amber"]),
                                              ("RESET", self._C["yellow"]))):
            x0 = img.width() - (3 - off) * _UI_BTN_W - 6 + off * 2
            self._rect(img, x0, 4, _UI_BTN_W - 4, _UI_TOPBAR_H - 10,
                       color, thickness=-1)
            self._text(img, x0 + 8, 6, label, self._C["black"], 1.0)

    def _draw_laser_btn(self, img) -> None:
        x, y, w, h = _UI_LASER_BTN
        color = self._C["green"] if self._laser_on else self._C["gray"]
        self._rect(img, x, y, w, h, color, thickness=-1)
        label = "Laser:ON" if self._laser_on else "Laser:OFF"
        self._text(img, x + 8, y + h // 2 - 8, label, self._C["black"], 1.1)
        if self._dot is None:
            self._text(img, x, y + h + 6, "NO DOT", self._C["yellow"], 1.0)
        else:
            # 标定读数(见文件头): core -> prom_min 取 core-10; dom -> dom_faint 取其 1/2~2/3
            st = self._last_stats or {}
            self._text(img, x, y + h + 6,
                       f"a={st.get('area', 0)} core={st.get('core', 0)} "
                       f"dom={st.get('dom', 0)}", self._C["cyan"], 1.0)

    def exit_requested(self) -> bool:
        return self._stop

    def laser_is_on(self) -> bool:
        """供退出兜底查询（factory.send_laser_off）; 未开过不发关帧。"""
        return self._laser_on

    # ---- helpers ----

    @staticmethod
    def _in(x, y, rect):
        return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]

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
