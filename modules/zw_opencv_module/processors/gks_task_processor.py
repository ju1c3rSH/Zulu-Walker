# -*- coding: utf-8 -*-
"""
工科赛近端任务处理器 — 一处理器三模式（EXPLOSIVE / TARGET / RESCUE）。

识别逻辑逐行照搬 F:\\2026工科赛\\maixcam\\工科赛上端\\main.py 的
task1_explosive / task2_target / task3_rescue（锁存/去抖/映射语义不变），
仅做平台适配：

- 图像来源: 框架单路 640 宽帧（旧版 cam 640x480 + cam1 320x240 双实例消亡）;
  YOLO 输入由 nn.YOLO11 按模型输入尺寸自适应缩放。
- 检测引擎: 设备走 maix_accel C++ find_blobs 快路径（阈值 OpenMV LAB 语义
  1:1 透传），PC/失败自动回退 cv2 inRange（OpenMV→OpenCV LAB 编码转换）。
- 人质: nn.YOLOv5+cvimodel → MaixCam2AI (nn.YOLO11) + axmodel；模型缺失时
  回零值结果（心跳不断流），不崩溃。
- 像素常量全部入 eod_config.yaml `gks:` 段；`/my-threshold/my_64threshold.txt`
  （脱机调阈值工具产出）作运行时覆盖，mtime 变化自动重载、不清锁存。

锁存语义（与 main.py 一致，经审查报告 §5.3 幂等性确认）:
- 坑位锁存 = 同帧三色/三形状齐 + 连续 pit_latch_frames(2) 帧左右序一致；
  一旦锁存不再重锁，直到 set_task 换 data_id/换任务码或 reset()。
- set_task 幂等：同模式同任务码的重订阅不清锁存（Brain 掉线 1s 重订阅安全）。

纯 cv2/numpy + 注入时钟 — 设备与 PC(pytest) 双端可运行。
"""
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from .registry import register_processor
from .base import Processor, VisionResult
from ..detectors.eod_color.maix_accel import find_blobs_maix, maix_image_from_context

MODE_EXPLOSIVE = "EXPLOSIVE"
MODE_TARGET = "TARGET"
MODE_RESCUE = "RESCUE"
_ALL_MODES = (MODE_EXPLOSIVE, MODE_TARGET, MODE_RESCUE)

# 色号 -> (名称, 配置键)
COLOR_KEYS = {1: ("red", "red"), 2: ("green", "green"), 3: ("blue", "blue")}

_DEFAULT_CONFIG = {
    "pit_left_is_1": True,      # True=画面左(小x)=坑1; 摄像头背装镜像时改 False
    "pit_latch_frames": 2,      # 连续 N 帧左右序一致才锁存（main.py 硬编码 2）
    "thresholds_file": "/my-threshold/my_64threshold.txt",
    "explosive": {
        "pixels_threshold": 10000,
        "area_threshold": 10000,
        # 兜底 LAB 阈值（阈值文件缺失时用; 现场用调阈值工具标定后覆盖）
        "lab": {
            "red":   [0, 80, 40, 80, 10, 80],
            "green": [0, 80, -120, -10, 0, 30],
            "blue":  [0, 80, 30, 100, -120, -60],
        },
    },
    "target": {
        "pixels_threshold": 1000,
        "area_threshold": 1000,
        "merge": True,
        # ROI 帧比例 (x0, y0, x1, y1); 默认值 = main.py [0,169,640,197] @640x480
        "roi": [0.0, 169 / 480, 1.0, 366 / 480],
        "lab": {
            "red":   [0, 80, 40, 80, 10, 80],
            "green": [0, 80, -120, -10, 0, 30],
            "blue":  [0, 80, 30, 100, -120, -60],
        },
    },
    "rescue": {
        "conf_th": 0.7,
        "iou_th": 0.45,
        # 旧版 obj.x>240 (320 宽画面) 过滤; cam1 已消亡, 0 = 关闭; 台架复现旧行为可设像素值
        "x_crop_px": 0,
        # 模型标签 -> 形状号（model.mud labels: yao_gu, yuan_zhu, yuan_tai）
        "labels": {"yuan_zhu": 1, "yuan_tai": 2, "yao_gu": 3},
    },
}


def load_gks_config(config_path: Optional[str] = None) -> dict:
    """Load eod_config.yaml `gks:` section over the built-in defaults (shallow per-section)."""
    import yaml

    cfg = {k: (dict(v) if isinstance(v, dict) else list(v)
               if isinstance(v, list) else v)
           for k, v in _DEFAULT_CONFIG.items()}
    for key in ("explosive", "target", "rescue"):
        cfg[key] = dict(_DEFAULT_CONFIG[key])
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "config", "eod_config.yaml")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for key, value in (data.get("gks") or {}).items():
            if isinstance(value, dict) and isinstance(cfg.get(key), dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    except (OSError, yaml.YAMLError):
        pass
    return cfg


def load_threshold_file(path: str) -> list:
    """解析脱机调阈值工具的阈值文件。

    行格式: name|Lmin,Lmax,Amin,Amax,Bmin,Bmax|hsv...（仅取 LAB 段）。
    返回最多 6 组 LAB 阈值列表; 组序与 main.py 约定一致:
    [0..2]=爆炸物 红/绿/蓝, [3..5]=靶标 红/绿/蓝。
    """
    groups: list = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 2:
                    continue
                try:
                    lab = [int(x) for x in parts[1].split(",")]
                except ValueError:
                    continue
                if len(lab) == 6:
                    groups.append(lab)
    except OSError:
        pass
    return groups


def _lab_to_cv_range(lab: list):
    """OpenMV LAB 阈值 (L 0..100, A/B 有符号) -> OpenCV uint8 inRange 上下界。"""
    lo = np.array([
        max(0, min(255, round(lab[0] * 2.55))),
        max(0, min(255, lab[2] + 128)),
        max(0, min(255, lab[4] + 128)),
    ], dtype=np.uint8)
    hi = np.array([
        max(0, min(255, round(lab[1] * 2.55))),
        max(0, min(255, lab[3] + 128)),
        max(0, min(255, lab[5] + 128)),
    ], dtype=np.uint8)
    return lo, hi


@register_processor("GksTaskProcessor")
class GksTaskProcessor(Processor):
    """订阅门控的近端三任务处理器（单模式激活）。"""

    def __init__(self, name: str = "gks_task", config: Optional[dict] = None,
                 now_fn=time.monotonic):
        super().__init__(name)
        self._cfg = config or load_gks_config()
        self._now = now_fn
        self._ai = None

        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        self._mode: Optional[str] = None
        self._codes: Tuple[int, int, int] = (0, 0, 0)

        # 阈值文件运行时覆盖（mtime 变化自动重载）
        self._thresholds_file = str(self._cfg.get("thresholds_file") or "")
        self._thresholds_mtime: Optional[float] = None
        self._explosive_lab: dict = {}
        self._target_lab: dict = {}
        self._reload_thresholds(force=True)

        self._reset_latch()

    # ---- gating ----

    def set_task(self, mode: str, codes: Tuple[int, int, int] = (0, 0, 0)) -> None:
        """订阅切换（coordinator 在主循环中调用）。

        幂等：同模式同任务码的重订阅不清锁存（Brain 掉线 1s 重订阅安全）;
        换 data_id 或换任务码才清（对应 main.py reset_task_latch 时机）。
        """
        if mode not in _ALL_MODES:
            raise ValueError(f"unknown GKS mode: {mode!r}")
        codes = tuple(int(c) for c in codes)
        if mode == self._mode and codes == self._codes:
            return
        self._mode = mode
        self._codes = codes
        self._reset_latch()

    def current_task(self) -> Tuple[Optional[str], Tuple[int, int, int]]:
        """当前订阅模式与任务码快照（coordinator 命令日志判重描述用）。"""
        return self._mode, self._codes

    def reset(self) -> None:
        """CMD_STOP / 主机静默: 清锁存, 保留模式（下次 set_task 决定）。"""
        self._reset_latch()

    _IMMUTABLE_RUNTIME_KEYS = frozenset({"thresholds_file"})

    def set_runtime_param(self, section: str, key: str, value) -> None:
        """调试页运行时改参（立即生效于 process(), 不落盘不清锁存）。

        仅接受 _DEFAULT_CONFIG 已定义的参数: 段参数 (section, key) 或
        顶层标量 (key == section 名), 否则 ValueError。thresholds_file
        属硬性不可改键（重载器在 __init__ 已缓存路径）。
        """
        if key in self._IMMUTABLE_RUNTIME_KEYS:
            raise ValueError(f"{key} is immutable at runtime")
        default_section = _DEFAULT_CONFIG.get(section)
        if isinstance(default_section, dict):
            if key not in default_section:
                raise ValueError(f"unknown runtime param: {section}.{key}")
            if isinstance(default_section[key], dict):
                raise ValueError(f"{section}.{key} is a nested section")
            self._cfg[section][key] = value
            return
        if default_section is None and section not in _DEFAULT_CONFIG:
            raise ValueError(f"unknown runtime param: {section}.{key}")
        if key != section:
            raise ValueError(f"{section} is a top-level param, use key=section")
        self._cfg[section] = value

    def set_ai(self, ai) -> None:
        """Pipeline 自动注入 AI HAL（RESCUE 模式使用; 可为 None）。"""
        self._ai = ai

    @property
    def mode(self) -> Optional[str]:
        return self._mode

    @property
    def config(self) -> dict:
        """当前生效配置（读视图: 修改请走 set_runtime_param）。"""
        return self._cfg

    def target_roi(self):
        """靶标 ROI 归一化界 (x0, y0, x1, y1) 或 None（未配置）。"""
        roi = self._cfg.get("target", {}).get("roi")
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            return tuple(float(v) for v in roi)
        return None

    @property
    def pit_status(self) -> dict:
        """调试页读锁存进度（爆炸物/人质坑位映射当前状态）。"""
        return {
            "pit": self._pos,
            "pit_flag": self._pos_flag,
            "latched": list(self._position),
            "pending": self._pending_count,
            "pit2": self._position2,
            "pending2": self._pending_count2,
            "shapes_seen": dict(self._shape_seen),
        }

    @property
    def codes(self) -> Tuple[int, int, int]:
        return self._codes

    def _reset_latch(self) -> None:
        # task1 爆炸物坑位状态
        self._position = [0, 0, 0]
        self._explosive_cx = 0
        self._pos = 0
        self._pos_flag = 0
        # task3 人质坑位状态
        self._position2 = [0, 0, 0]
        self._shape_seen = {}     # 形状号 -> 最近一次画面中心 x（跨帧刷新）
        # 两帧一致去抖（task1 / task3 各一套）
        self._pending_order: list = []
        self._pending_count = 0
        self._pending_order2: list = []
        self._pending_count2 = 0

    # ---- thresholds ----

    def _reload_thresholds(self, force: bool = False) -> None:
        """阈值文件覆盖（组序: 0..2 爆红/爆绿/爆蓝, 3..5 靶红/靶绿/靶蓝）。"""
        exp_lab = {k: list(v) for k, v in self._cfg["explosive"]["lab"].items()}
        tgt_lab = {k: list(v) for k, v in self._cfg["target"]["lab"].items()}
        if self._thresholds_file:
            try:
                mtime = os.stat(self._thresholds_file).st_mtime
            except OSError:
                mtime = None
            if mtime is not None and (force or mtime != self._thresholds_mtime):
                groups = load_threshold_file(self._thresholds_file)
                if len(groups) >= 6:
                    names = ("red", "green", "blue")
                    for i, n in enumerate(names):
                        exp_lab[n] = groups[i]
                        tgt_lab[n] = groups[3 + i]
                self._thresholds_mtime = mtime
        self._explosive_lab = exp_lab
        self._target_lab = tgt_lab

    def _maybe_reload_thresholds(self) -> None:
        if not self._thresholds_file:
            return
        try:
            mtime = os.stat(self._thresholds_file).st_mtime
        except OSError:
            return
        if mtime != self._thresholds_mtime:
            self._reload_thresholds()

    def _lab_for(self, mode: str, color_code: int) -> Optional[list]:
        key = COLOR_KEYS.get(color_code)
        if key is None:
            return None
        table = self._explosive_lab if mode == MODE_EXPLOSIVE else self._target_lab
        lab = table.get(key[1])
        return list(lab) if lab else None

    # ---- blob scan (accel fast path + cv2 fallback) ----

    @staticmethod
    def _roi_px(roi_fractions, w: int, h: int) -> Optional[list]:
        if not roi_fractions:
            return None
        x0, y0, x1, y1 = roi_fractions
        rx, ry = int(x0 * w), int(y0 * h)
        rw, rh = int((x1 - x0) * w), int((y1 - y0) * h)
        if rw <= 0 or rh <= 0:
            return None
        return [rx, ry, rw, rh]

    def _scan_blobs(self, frame_bgr: np.ndarray, maix_img, lab: list, roi_px,
                    pixels_threshold: int, area_threshold: int, merge: bool) -> list:
        """find_blobs 一次单色扫描, 返回 [{x,y,w,h,cx,cy,pixels}]（像素面积降序不保证）。"""
        if maix_img is not None:
            blobs = find_blobs_maix(
                maix_img, lab, roi=roi_px,
                area_threshold=area_threshold,
                pixels_threshold=pixels_threshold,
                merge=merge)
            if blobs is not None:
                return blobs
            # maix 不可用/失败 -> cv2 回退
        return self._scan_blobs_cv(frame_bgr, lab, roi_px,
                                   pixels_threshold, area_threshold)

    def _scan_blobs_cv(self, frame_bgr: np.ndarray, lab: list, roi_px,
                       pixels_threshold: int, area_threshold: int) -> list:
        if frame_bgr is None:
            return []
        search = frame_bgr
        offset_x = offset_y = 0
        if roi_px is not None:
            rx, ry, rw, rh = roi_px
            fh, fw = frame_bgr.shape[:2]
            rx, ry = max(0, rx), max(0, ry)
            rw, rh = min(rw, fw - rx), min(rh, fh - ry)
            if rw <= 0 or rh <= 0:
                return []
            search = frame_bgr[ry:ry + rh, rx:rx + rw]
            offset_x, offset_y = rx, ry

        lab_img = cv2.cvtColor(search, cv2.COLOR_BGR2LAB)
        lo, hi = _lab_to_cv_range(lab)
        mask = cv2.inRange(lab_img, lo, hi)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)  # 合并近邻色块
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < max(area_threshold, pixels_threshold):
                continue
            x, y, w, h = cv2.boundingRect(c)
            out.append({
                "x": x + offset_x, "y": y + offset_y,
                "w": w, "h": h,
                "cx": x + offset_x + w / 2.0,
                "cy": y + offset_y + h / 2.0,
                "pixels": area,
            })
        return out

    # ---- processing ----

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        self._maybe_reload_thresholds()
        maix_img = maix_image_from_context(context)

        hits: list = []           # HUD 叠加用 [{x,y,w,h,text,color}]
        latched_order: list = []
        pending_count = 0
        thresholds_ok = True
        result = {"cx": 0, "pit": 0, "err_fixed": 0, "pos": 0, "found": False}

        if self._mode == MODE_EXPLOSIVE:
            result, hits, latched_order, pending_count, thresholds_ok = \
                self._process_explosive(frame, maix_img, hits)
        elif self._mode == MODE_TARGET:
            result, hits, thresholds_ok = self._process_target(frame, maix_img, hits)
        elif self._mode == MODE_RESCUE:
            result, hits, latched_order, pending_count = \
                self._process_rescue(frame, hits)

        data = {
            "mode": self._mode,
            "codes": list(self._codes),
            "cx": result.get("cx", 0),
            "pit": result.get("pit", 0),
            "err_fixed": result.get("err_fixed", 0),
            "pos": result.get("pos", 0),
            "found": bool(result.get("found", False)),
            "latched_order": latched_order,
            "pending_count": pending_count,
            "thresholds_ok": thresholds_ok,
            "hits": hits,
        }
        return VisionResult(task_name=self.name, success=True, result_data=data)

    # ---- task1 爆炸物 (main.py task1_explosive 逐行移植) ----

    def _process_explosive(self, frame, maix_img, hits):
        labs = [self._lab_for(MODE_EXPLOSIVE, c) for c in (1, 2, 3)]
        if not all(labs) or frame is None:
            # 阈值缺失不进 find_blobs: 零值心跳不断流, 锁存状态保留（main.py 同语义）
            return ({"cx": 0, "pit": 0, "found": False},
                    hits, list(self._position), self._pending_count, False)
        cfg = self._cfg["explosive"]
        color_cx = {}   # 本帧各色最大色块 {色号1红/2绿/3蓝: cx}
        for i in range(3):
            blobs = self._scan_blobs(frame, maix_img, labs[i], None,
                                     cfg["pixels_threshold"],
                                     cfg["area_threshold"], merge=False)
            if blobs:
                largest = max(blobs, key=lambda b: b["pixels"])
                hits.append({"x": int(largest["x"]), "y": int(largest["y"]),
                             "w": int(largest["w"]), "h": int(largest["h"]),
                             "text": f"C{i + 1}",
                             "color": (0, 255, 0)})
                color_cx[i + 1] = largest["cx"]
        if self._codes[0] in color_cx:
            self._explosive_cx = int(color_cx[self._codes[0]])
        # 顺序映射: 三坑各放一色, 按画面 x 从左到右排;
        # 齐三色且连续两帧顺序一致才锁存, 防单帧误检/部分可见错排
        if self._pos_flag == 0 and len(color_cx) == 3:
            ordered = [c for c, _cx in sorted(color_cx.items(), key=lambda kv: kv[1])]
            if not self._cfg["pit_left_is_1"]:
                ordered.reverse()
            if ordered == self._pending_order:
                self._pending_count += 1
            else:
                self._pending_order = ordered
                self._pending_count = 1
            if self._pending_count >= int(self._cfg["pit_latch_frames"]):
                for j, color_num in enumerate(ordered):
                    self._position[j] = color_num
                if self._codes[0] in ordered:   # 守卫: 非法色号保持 pos=0 等待
                    self._pos = ordered.index(self._codes[0]) + 1
                    self._pos_flag = 1
        result = {
            "cx": self._explosive_cx,
            "pit": self._pos,
            "found": bool(self._pos_flag),
        }
        return result, hits, list(self._position), self._pending_count, True

    # ---- task2 靶标 (main.py task2_target 逐行移植) ----

    def _process_target(self, frame, maix_img, hits):
        lab = self._lab_for(MODE_TARGET, self._codes[1])
        if not lab or frame is None:
            # 无阈值不断流: 零值心跳（main.py 同语义）
            return {"err_fixed": 0, "found": False}, hits, False
        cfg = self._cfg["target"]
        h, w = frame.shape[:2]
        roi_px = self._roi_px(cfg["roi"], w, h)
        blobs = self._scan_blobs(frame, maix_img, lab, roi_px,
                                 cfg["pixels_threshold"], cfg["area_threshold"],
                                 merge=bool(cfg["merge"]))
        if blobs:
            largest = max(blobs, key=lambda b: b["pixels"])
            cx = largest["cx"]
            hits.append({"x": int(largest["x"]), "y": int(largest["y"]),
                         "w": int(largest["w"]), "h": int(largest["h"]),
                         "text": "TGT", "color": (255, 0, 0)})
            # 归一化横向误差 (cx - w/2)/(w/2), clamp ±1, 定点 ×10000
            # (640 宽帧下与旧版 (cx-320)/320 逐位一致)
            half = w / 2.0
            normalized = (cx - half) / half
            normalized = max(-1.0, min(1.0, normalized))
            err_fixed = int(normalized * 10000)   # int(): 与 main.py 同为截断取整
            return {"err_fixed": err_fixed, "found": True}, hits, True
        return {"err_fixed": 0, "found": False}, hits, True

    # ---- task3 人质 (main.py task3_rescue 逐行移植) ----

    def _process_rescue(self, frame, hits):
        ai = self._ai
        if ai is None or not getattr(ai, "loaded", False):
            # 模型缺失优雅降级: 零值心跳不断流, 不崩溃
            return {"pos": 0, "found": False}, hits, list(self._position2), self._pending_count2

        objs = ai.detect(frame, conf_th=self._cfg["rescue"]["conf_th"],
                         iou_th=self._cfg["rescue"]["iou_th"])
        labels = getattr(ai, "labels", None) or []
        label_map = self._cfg["rescue"]["labels"]
        x_crop = int(self._cfg["rescue"].get("x_crop_px") or 0)
        for obj in objs:
            if x_crop > 0 and obj.x > x_crop:
                continue
            label = labels[obj.class_id] if obj.class_id < len(labels) else ""
            shape_num = int(label_map.get(label, 0))
            hits.append({"x": int(obj.x), "y": int(obj.y),
                         "w": int(obj.w), "h": int(obj.h),
                         "text": f"{label}:{obj.score:.2f}",
                         "color": (0, 160, 255)})
            if shape_num == 0:
                continue
            self._shape_seen[shape_num] = obj.x + obj.w / 2.0   # 形状中心 x, 跨帧刷新
        # 顺序映射: 跨帧集齐三形状且连续两帧顺序一致才锁存(按画面 x 从左到右)
        if len(self._shape_seen) == 3 and self._position2[0] == 0:
            ordered = [s for s, _cx in sorted(self._shape_seen.items(), key=lambda kv: kv[1])]
            if not self._cfg["pit_left_is_1"]:
                ordered.reverse()
            if ordered == self._pending_order2:
                self._pending_count2 += 1
            else:
                self._pending_order2 = ordered
                self._pending_count2 = 1
            if self._pending_count2 >= int(self._cfg["pit_latch_frames"]):
                for j, shape_num in enumerate(ordered):
                    self._position2[j] = shape_num
        pos2 = 0
        if self._codes[2] != 0:     # 守卫: 形状码未下发(0)时不得与未锁存 0 自匹配
            for j in range(3):
                if self._codes[2] == self._position2[j]:
                    pos2 = j + 1
                    break
        return ({"pos": pos2, "found": bool(pos2)},
                hits, list(self._position2), self._pending_count2)

    # ---- display (PC cv2 调试; 设备端叠加由 display HUD 完成) ----

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        if frame is None or result.result_data is None:
            return frame
        data = result.result_data
        for hit in data.get("hits", []):
            x, y, w, h = hit["x"], hit["y"], hit["w"], hit["h"]
            bgr = hit["color"][::-1]              # hits 约定 RGB, cv2 需 BGR
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr, 2)
            if hit.get("text"):
                cv2.putText(frame, hit["text"], (x, max(12, y - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 1)
        tag = (f"{data.get('mode') or '-'} "
               f"pit={data.get('pit', 0)} pos={data.get('pos', 0)} "
               f"err={data.get('err_fixed', 0):+d} f={int(bool(data.get('found')))}")
        cv2.putText(frame, tag, (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        return frame

    def release(self) -> None:
        self._ai = None
