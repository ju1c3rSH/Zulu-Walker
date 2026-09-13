# -*- coding: utf-8 -*-
"""
设备端 LAB 阈值调参工具（MaixCAM2 触屏）— `python run.py debug color`

照搬 F:\\2026工科赛\\maixcam\\脱机调阈值\\640480 的使用方式重构:
- 6 组阈值 = 爆红/爆绿/爆蓝 + 靶红/靶绿/靶蓝（与 GksTaskProcessor 组序约定一致）;
- 实时预览: 当前组阈值命中像素绿色高亮 + 色块 bbox + 点选取色(LAB 读数);
- 保存写到 /my-threshold/my_64threshold.txt（组序同 main.py）,
  主程序订阅运行中检测到 mtime 变化自动热重载、不清锁存。

文件行格式: name|Lmin,Lmax,Amin,Amax,Bmin,Bmax（第三段 hsv 本工具不写,
消费端只读 LAB 段）。触屏 API 依 MaixPy 官方文档 touchscreen.md。
"""
import os
import time

import numpy as np

from modules.zw_opencv_module.processors.gks_task_processor import (
    load_gks_config, load_threshold_file,
)

THRESHOLD_FILE = "/my-threshold/my_64threshold.txt"
GROUP_NAMES = ["E-R", "E-G", "E-B", "T-R", "T-G", "T-B"]
FILE_KEYS = ["exp_red", "exp_green", "exp_blue", "tgt_red", "tgt_green", "tgt_blue"]

# 滑条范围: L 0..100, A/B -128..127
SLIDERS = ["Lmin", "Lmax", "Amin", "Amax", "Bmin", "Bmax"]
SLIDER_RANGE = {0: (0, 100), 1: (0, 100), 2: (-128, 127), 3: (-128, 127),
                4: (-128, 127), 5: (-128, 127)}

_UI_GROUP_H = 30     # 顶部组选择按钮条高度
_UI_ACTION_H = 30    # Exit/SAVE/RESET 按钮条高度
_UI_SLIDER_H = 50    # 每行滑条高度（底部 2 列 x 3 行）


class GksThresholdTuner:
    """触屏 LAB 调参器。run() 阻塞运行至点 Exit / app 退出。"""

    def __init__(self, camera_source=None, width: int = 640, height: int = 480):
        self._source = camera_source
        self._req_w, self._req_h = width, height
        self._stop = False

    # ---- lifecycle ----

    def run(self) -> None:
        import cv2
        from maix import camera, display, image, app, touchscreen

        cam = camera.Camera(width=self._req_w, height=self._req_h,
                            format=image.Format.FMT_RGB888,
                            device=str(self._source) if self._source else None,
                            open=True)
        disp = display.Display()
        ts = touchscreen.TouchScreen()
        dw, dh = disp.width(), disp.height()

        groups = self._load_groups()
        cur = 0
        pick = None          # 预览区点选像素 (x, y) -> LAB 读数
        toast = ""           # 操作提示（1.5s 消失）
        toast_until = 0.0
        pressed_last = False

        while not app.need_exit() and not self._stop:
            frame = cam.read()
            if frame is None:
                time.sleep(0.005)
                continue
            frame_bgr = self._to_bgr(frame)

            lab_now = groups[cur]
            mask, blobs = self._analyze(frame_bgr, lab_now)
            preview_bgr = self._highlight(frame_bgr, mask)

            img = self._to_maix(preview_bgr)
            if img.width() != dw or img.height() != dh:
                try:
                    img = img.resize(dw, dh)
                except Exception:
                    pass
            sx = img.width() / frame_bgr.shape[1]
            sy = img.height() / frame_bgr.shape[0]

            x, y, pressed = ts.read()
            clicked = pressed and not pressed_last
            pressed_last = pressed

            hit_btn = self._draw_groups(img, cur)
            act = self._draw_actions(img)
            if clicked:
                for gx, gname in hit_btn:
                    if self._in(x, y, gx):
                        cur = gname
                        pick = None
                if self._in(x, y, act["exit"]):
                    break
                if self._in(x, y, act["save"]):
                    self._save(groups)
                    toast, toast_until = "SAVED -> " + THRESHOLD_FILE, time.time() + 1.5
                if self._in(x, y, act["reset"]):
                    groups = self._load_groups()
                    toast, toast_until = "RELOADED", time.time() + 1.0
                px, py = int(x / sx), int(y / sy)
                if 0 <= px < frame_bgr.shape[1] and 0 <= py < frame_bgr.shape[0] \
                        and y > _UI_GROUP_H + _UI_ACTION_H:
                    pick = (px, py)

            self._draw_sliders(img, groups[cur], x, y, pressed)
            self._draw_info(img, blobs, mask, pick, frame_bgr, sx, sy)
            if toast and time.time() < toast_until:
                self._text(img, 8, img.height() - 24, toast, (255, 255, 0), 1.2)

            disp.show(img)
            time.sleep(0.001)

    def stop(self) -> None:
        self._stop = True

    # ---- data ----

    def _load_groups(self) -> list:
        cfg = load_gks_config()
        exp = cfg["explosive"]["lab"]
        tgt = cfg["target"]["lab"]
        groups = [list(exp["red"]), list(exp["green"]), list(exp["blue"]),
                  list(tgt["red"]), list(tgt["green"]), list(tgt["blue"])]
        if os.path.exists(THRESHOLD_FILE):
            saved = load_threshold_file(THRESHOLD_FILE)
            for i in range(min(6, len(saved))):
                groups[i] = list(saved[i])
        return groups

    def _save(self, groups: list) -> None:
        parent = os.path.dirname(THRESHOLD_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        lines = [f"{FILE_KEYS[i]}|{','.join(str(v) for v in groups[i])}"
                 for i in range(6)]
        with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

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
    def _analyze(frame_bgr, lab_threshold):
        """返回 (mask, blobs); blobs = [{x,y,w,h,area}] 按面积降序前 5。"""
        import cv2
        lab_img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        lo = np.array([round(lab_threshold[0] * 2.55), lab_threshold[2] + 128,
                       lab_threshold[4] + 128], dtype=np.uint8).clip(0, 255)
        hi = np.array([round(lab_threshold[1] * 2.55), lab_threshold[3] + 128,
                       lab_threshold[5] + 128], dtype=np.uint8).clip(0, 255)
        mask = cv2.inRange(lab_img, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < 200:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            blobs.append({"x": bx, "y": by, "w": bw, "h": bh, "area": area})
        blobs.sort(key=lambda b: b["area"], reverse=True)
        return mask, blobs[:5]

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

    @staticmethod
    def _text(img, x, y, s, color, scale=1.0):
        try:
            img.draw_string(x, y, s, color=color, scale=scale, thickness=1)
        except Exception:
            pass

    def _draw_groups(self, img, cur: int):
        rects = []
        bw = img.width() // 6
        for i, name in enumerate(GROUP_NAMES):
            rect = (i * bw + 2, 2, bw - 4, _UI_GROUP_H - 6)
            color = (0, 255, 0) if i == cur else (200, 200, 200)
            try:
                img.draw_rect(rect[0], rect[1], rect[2], rect[3], color=color, thickness=2)
            except Exception:
                pass
            self._text(img, rect[0] + 8, rect[1] + 4, name, color, 1.2)
            rects.append((rect, i))
        return rects

    def _draw_actions(self, img):
        y = _UI_GROUP_H
        h = _UI_ACTION_H
        rects = {
            "exit": (2, y, 90, h - 4),
            "save": (img.width() - 200, y, 90, h - 4),
            "reset": (img.width() - 104, y, 100, h - 4),
        }
        try:
            img.draw_rect(*rects["exit"], color=(255, 255, 255), thickness=2)
            img.draw_rect(*rects["save"], color=(0, 255, 0), thickness=2)
            img.draw_rect(*rects["reset"], color=(255, 255, 0), thickness=2)
        except Exception:
            pass
        self._text(img, rects["exit"][0] + 8, y + 6, "< Exit", (255, 255, 255), 1.2)
        self._text(img, rects["save"][0] + 8, y + 6, "SAVE", (0, 255, 0), 1.2)
        self._text(img, rects["reset"][0] + 8, y + 6, "RESET", (255, 255, 0), 1.2)
        return rects

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
            try:
                img.draw_rect(x0, y0 + 8, w, 8, color=(120, 120, 120), thickness=-1)
                kx = x0 + int((value - lo) / (hi - lo) * w)
                img.draw_rect(kx - 4, y0 + 2, 8, 20, color=(0, 255, 0), thickness=-1)
            except Exception:
                pass
            self._text(img, x0, y0 - 2, f"{name}:{value}", (255, 255, 255), 1.0)
            if pressed and x0 <= tx <= x0 + w and y0 - 6 <= ty <= y0 + 30:
                new_v = int(lo + (tx - x0) / max(1, w) * (hi - lo))
                new_v = max(lo, min(hi, new_v))
                if new_v != lab[i]:
                    lab[i] = new_v
                    changed = True
        return changed

    def _draw_info(self, img, blobs, mask, pick, frame_bgr, sx, sy):
        total_px = int((mask > 0).sum())
        info = f"hit_px:{total_px} blobs:{len(blobs)}"
        if blobs:
            b = blobs[0]
            info += f" max:{b['area']}px ({b['x']},{b['y']},{b['w']},{b['h']})"
        self._text(img, 8, _UI_GROUP_H + _UI_ACTION_H + 4, info, (255, 255, 255), 1.0)
        for b in blobs:
            try:
                img.draw_rect(int(b["x"] * sx), int(b["y"] * sy),
                              int(b["w"] * sx), int(b["h"] * sy),
                              color=(255, 0, 255), thickness=2)
            except Exception:
                pass
        if pick is not None:
            l, a, bb = self._pixel_lab(frame_bgr, pick[0], pick[1])
            self._text(img, img.width() - 220, _UI_GROUP_H + _UI_ACTION_H + 4,
                       f"pick({pick[0]},{pick[1]}) L:{l} A:{a} B:{bb}",
                       (0, 255, 255), 1.0)
            try:
                img.draw_circle(int(pick[0] * sx), int(pick[1] * sy), 4,
                                color=(0, 255, 255), thickness=2)
            except Exception:
                pass
