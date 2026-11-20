# -*- coding: utf-8 -*-
"""
工科赛近端任务处理器: 单处理器三模式（EXPLOSIVE / TARGET / RESCUE）。

识别逻辑移植自 2026 上端 main.py，锁存、去抖、映射口径不变，只做平台适配:
设备走 maix C++ find_blobs（LAB 阈值 1:1 透传），PC 或失败回退 cv2 inRange；
人质走 YOLO，模型缺失时回零值心跳。像素常量入 eod_config.yaml `gks:` 段，
阈值文件与调参页产物热重载、不清锁存。

锁存规则（爆炸物与人质共用）: 顺序按首次检出先后，同帧多目标按画面左右；
见到 2 个即补出第 3 个，连续 pit_latch_frames 帧同序才锁。换 data_id 或
任务码才清锁存，重复订阅不清。纯 cv2/numpy + 注入时钟，设备与 PC 双端可跑。
"""
import hashlib
import os
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.log_util import log_print

from .registry import register_processor
from .base import Processor, VisionResult
from ..detectors.eod_color.maix_accel import find_blobs_maix, maix_image_from_context

MODE_EXPLOSIVE = "EXPLOSIVE"
MODE_TARGET = "TARGET"
MODE_RESCUE = "RESCUE"
_ALL_MODES = (MODE_EXPLOSIVE, MODE_TARGET, MODE_RESCUE)

# 色号 -> (名称, 配置键)
COLOR_KEYS = {1: ("red", "red"), 2: ("green", "green"), 3: ("blue", "blue")}

# 人质调参页 SAVE 产物，格式与阈值文件相同；文件缺失时用 yaml 值。
RESCUE_PARAMS_FILE = "/my-threshold/my_rescue_params.txt"
# 可被该文件覆盖的键及类型（与调参页滑条一一对应）。
_RESCUE_PARAM_KEYS = {"conf_th": float, "iou_th": float, "x_crop_px": int}

# 激光校准页 SAVE 产物，规则同上。
LASER_CALIB_FILE = "/my-threshold/my_laser_calib.txt"
# 文件键 -> (cfg 段, 键, 类型)。
_LASER_CALIB_KEYS = {
    "laser_cx": ("target", "laser_cx", int),
    "lum_min":  ("laser", "lum_min", int),
    "dom_min":  ("laser", "dom_min", int),
    "min_area": ("laser", "min_area", int),
    "max_area": ("laser", "max_area", int),
    # 白核淡红晕签名扩展；halo_pad=0 关闭，回到旧的「绝对红」单入口。
    "dom_faint": ("laser", "dom_faint", int),   # 白核+淡晕确认: 窗口内 max(dom) 下限
    "halo_pad":  ("laser", "halo_pad", int),    # 白核膨胀半径(0=关闭签名入口)
    "prom_min":  ("laser", "prom_min", int),    # 白核门限(R,G,B 三通道同时>=此值)
}


def file_signature(path: str):
    """返回 (文件大小, 内容 SHA1)；用内容签名而非 mtime，同长改值也能检出。
    先 stat 再读，文件不存在时零 IO。
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError:
        return None
    return (st.st_size, hashlib.sha1(data).hexdigest())


def load_rescue_params_file(path: str) -> dict:
    """解析 key=value 覆盖文件，只认 _RESCUE_PARAM_KEYS 的键，非法行忽略。
    文件缺失返回 {}，调用方据此保持 yaml 值；PC 端无 /my-threshold 也安全。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return {}
    params = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        conv = _RESCUE_PARAM_KEYS.get(key.strip())
        if conv is None:
            continue
        try:
            val = float(value)
        except ValueError:
            continue
        params[key.strip()] = int(val) if conv is int else val
    return params


def load_laser_calib_file(path: str) -> dict:
    """解析激光校准覆盖文件（key=value 行）。键集见 _LASER_CALIB_KEYS, 非法行忽略。"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return {}
    params = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        spec = _LASER_CALIB_KEYS.get(key.strip())
        if spec is None:
            continue
        try:
            params[key.strip()] = int(float(value))
        except ValueError:
            continue
    return params

# 只缓存 maix.image 的可用性，不缓存模块对象；否则测试注入的假模块会被永久屏蔽。
_maix_image_ok = None

_DEFAULT_CONFIG = {
    "pit_left_is_1": True,      # True=画面左(小x)=坑1; 摄像头背装镜像时改 False
    "pit_latch_frames": 2,      # 连续 N 帧左右序一致才锁存（main.py 硬编码 2）
    "thresholds_file": "/my-threshold/my_64threshold.txt",
    "rescue_params_file": RESCUE_PARAMS_FILE,
    "explosive": {
        # 面积门槛下调到 4000: 机位目标小，运动模糊先啃边缘；与 scan_scale 叠加时按 scale² 折算。
        # 以调参页 max:area 的 40~50% 为起点；背景成块误检时回调或先收紧 A/B。
        "pixels_threshold": 4000,
        "area_threshold": 4000,
        # 见到 2 色即锁，未见者补到尾部。补位只推出「是什么」，位置靠 order_by。
        # 若「未见者是最后一坑」不成立会静默锁错坑；日志 inferred=True 时须实物核对 PIT/POS。
        "order_by": "first_seen",
        "infer_missing": "tail",
        # 降采样系数: 1.0 = 640x480 全帧，0.5 = 320x240；面积按 scale² 折算、坐标按 1/scale 还原。
        "scan_scale": 1.0,
        # 阈值文件缺失时的兜底值，与 /my-threshold/my_64threshold.txt 同值。
        # A/B 判色相与饱和度，L 只当噪声地板；用 L 上限排高光会连亮面一起切掉。
        # 值偏保守，现场锁不上再按色放开 L 或 A；改动需同步 target["lab"]。
        # 排爆 ROI 帧比例，None = 全帧。纯提速视窗，不改定序。
        # 三坑必须同时落在 ROI 内，否则漏检后被补到尾部会静默锁错坑。
        "roi": None,
        "lab": {
            "red":   [0, 80, 40, 80, 10, 80],
            "green": [25, 85, -80, -25, 30, 85],
            "blue":  [0, 60, -8, 45, -75, -25],
        },
    },
    "target": {
        "pixels_threshold": 1000,
        "area_threshold": 1000,
        "merge": False,
        "prefer_continuity": False,
        "err_deadband": 0,
        "confirm_frames": 1,
        # find_blobs 的 margin，0 = 不传该 kwarg。margin 会触发膨胀与连通域重标记，CAM1 上尤其贵。
        "merge_margin": 0,
        # ROI 帧比例 (x0,y0,x1,y1)，默认等于 main.py [0,169,640,197] @640x480
        "roi": [0.0, 169 / 480, 1.0, 366 / 480],
        # 降采样两段式（见 _process_target）: 小图定位，全分辨率窄窗复扫并由该次结果算 err_fixed。
        # 只在小图上算误差会放大 cx 的像素级偏差（scale=0.5 时可达 30+ LSB），精修步不可删。
        "scan_scale": 1.0,
        # 精修窗相对粗框的外扩像素（全分辨率口径），按 640 宽标定；换分辨率按比例调。
        "refine_pad_px": 24,
        # fixed 跟踪窗半宽，0 = 关；dynamic 模式下作横向封顶。
        "track_roi_px": 0,
        # 跟踪窗模式: off | fixed | dynamic，非法值按 fixed；未命中均当帧回退全 ROI。
        "track_mode": "fixed",
        # dynamic 窗按上帧 bbox 缩放，横向封顶 2*track_roi_px，纵向封顶 ROI 高。
        "track_win_scale": 4.0,
        "track_win_min_w": 48,
        "track_v_pad_px": 16,
        # 填充率 pixels/(w*h) 过滤: 环约 0.5~0.7，实心约 0.9+；全开区间 = 不过滤。
        "fill_min": 0.0,
        "fill_max": 1.0,
        # 与 explosive 段同值；两种模式物理条件不同，可分头标定，改这里需同步 explosive 段。
        "lab": {
            "red":   [0, 80, 40, 80, 10, 80],
            "green": [25, 85, -80, -25, 30, 85],
            "blue":  [0, 60, -8, 45, -75, -25],
        },
        # 激光零点（像素 x），0 = 用画面中心；只平移零点，不改归一化增益。
        "laser_cx": 0,
    },
    # 红点激光检测参数（LASER 调试页用，判定流程见 laser_calib_tuner.py）。
    "laser": {
        # R 通道亮度门限，白核与红环均满足。
        "lum_min": 200,
        # R - max(G,B) 的下限，用于排白墙与高光。
        "dom_min": 40,
        # 面积过滤（像素，全分辨率口径），光斑直径 3~30px 随距离变化。
        "min_area": 8,
        "max_area": 4000,
        # 白背景远距光斑会泛白，晕的 dom 在门限两侧抖动导致丢点；以下三项为「白核+淡晕」第二入口。
        "dom_faint": 12,
        # 白核膨胀半径/确认窗外扩，0 = 关闭签名扩展。
        "halo_pad": 3,
        # 白核门限；HUD 读数 core=最白像素三通道最小值，现场取 core-10。
        "prom_min": 240,
        "calib_file": LASER_CALIB_FILE,
    },
    "rescue": {
        "conf_th": 0.7,
        "iou_th": 0.45,
        # 丢弃 obj.x（左边缘）超过此值的框，抑制右侧入画区的半出画与运动模糊误判。
        # 0 = 全视野；现场值见 eod_config.yaml（480 = 0.75W）。
        "x_crop_px": 0,
        # 三形状互不相同，见到 2 个即可推出第 3 个；补位只管「是什么」，位置靠 order_by。
        # none = 三形状齐见才锁（旧 main.py 做法，退路）。
        "infer_missing": "tail",
        # 坑位顺序来源: first_seen = 首次检出先后（相机按 pit1→pit3 扫过，跨帧物理序）；
        # x = 按画面 x 排序（三坑同屏时用，镜像需把 pit_left_is_1 置 false）。
        "order_by": "first_seen",
        # 入表所需连续在场帧数，缺席一帧清零。1 = 首帧定案（单测基线，也是回归开关）。
        # 默认留 1 是为了让不读 YAML 的单测保持旧时序；不进 _RESCUE_PARAM_KEYS，
        # 因为热重载只 merge 配置、不清状态，改小门槛会让幽灵立刻入表。
        "shape_hits": 1,
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
    for key in ("explosive", "target", "laser", "rescue"):
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


def _as_numpy(frame):
    """把 LazyFrame 物化成真 numpy；cv2 的 src 参数不接受 __array__ 协议。
    真 numpy / None 原样返回。
    """
    if isinstance(frame, np.ndarray):
        return frame
    getter = getattr(frame, "numpy", None)
    if callable(getter):
        return getter()
    return frame


@register_processor("GksTaskProcessor")
class GksTaskProcessor(Processor):
    """按订阅开关工作的三任务处理器（单模式激活）。"""

    def __init__(self, name: str = "gks_task", config: Optional[dict] = None,
                 now_fn=time.monotonic):
        super().__init__(name)
        self._cfg = config or load_gks_config()
        self._now = now_fn
        self._ai = None

        self._kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._scan_path: str = ""     # 当前 blob 扫描引擎(maix C++ / cv2), 变化时打一次日志

        self._mode: Optional[str] = None
        self._codes: Tuple[int, int, int] = (0, 0, 0)

        # 阈值文件运行时覆盖（内容变化自动重载, 见 file_signature）
        self._thresholds_file = str(self._cfg.get("thresholds_file") or "")
        self._thresholds_sig = None
        self._thresholds_poll_ts: float = 0.0
        self._explosive_lab: dict = {}
        self._target_lab: dict = {}
        self._reload_thresholds(force=True)
        # 人质调参页参数文件，规则同阈值文件，共用 2Hz 轮询。
        self._rescue_params_file = str(self._cfg.get("rescue_params_file") or "")
        self._rescue_params_sig = None
        self._reload_rescue_params(force=True)
        # 激光校准文件，规则同上。
        self._laser_calib_file = str(
            self._cfg.get("laser", {}).get("calib_file") or "")
        self._laser_calib_sig = None
        self._reload_laser_calib(force=True)

        self._reset_latch()

    # ---- gating ----

    def set_task(self, mode: str, codes: Tuple[int, int, int] = (0, 0, 0)) -> None:
        """订阅切换（coordinator 在主循环中调用）。
        同模式同任务码的重订阅不清锁存（Brain 掉线 1s 重订阅安全）；
        换 data_id 或任务码才清。
        """
        if mode not in _ALL_MODES:
            raise ValueError(f"unknown GKS mode: {mode!r}")
        codes = tuple(int(c) for c in codes)
        if mode == self._mode and codes == self._codes:
            return
        self._mode = mode
        self._codes = codes
        self._reset_latch()
        self._log_lab_slots()

    def current_task(self) -> Tuple[Optional[str], Tuple[int, int, int]]:
        """当前订阅模式与任务码快照（coordinator 命令日志判重描述用）。"""
        return self._mode, self._codes

    def reset(self) -> None:
        """CMD_STOP / 主机静默: 清锁存, 保留模式（下次 set_task 决定）。"""
        self._reset_latch()

    _IMMUTABLE_RUNTIME_KEYS = frozenset(
        {"thresholds_file", "rescue_params_file", "calib_file"})

    def set_runtime_param(self, section: str, key: str, value) -> None:
        """调试页运行时改参，立即生效、不落盘、不清锁存。
        只接受 _DEFAULT_CONFIG 已定义的键；三个 *_file 键不可改（路径已在 __init__ 缓存）。
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
            if key == "laser_cx":
                log_print(f"[GKS] target err zero -> "
                          f"{'center' if not value else f'laser_cx={int(value)}'}")
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

    def explosive_roi(self):
        """排爆 ROI 归一化界 (x0, y0, x1, y1) 或 None（全帧/未配置）。"""
        roi = self._cfg.get("explosive", {}).get("roi")
        if isinstance(roi, (list, tuple)) and len(roi) == 4:
            return tuple(float(v) for v in roi)
        return None

    def target_track_win(self):
        """最近一次使用的跟踪窄窗 [x,y,w,h] 全帧像素坐标, 或 None。"""
        return list(self._target_track_win) if self._target_track_win else None

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
            "inferred": self._latch_inferred,
            "shapes_seen": dict(self._shape_seen),
            "seen_order": list(self._seen_order),
        }

    @property
    def codes(self) -> Tuple[int, int, int]:
        return self._codes

    def _reset_latch(self) -> None:
        # task1 爆炸物坑位状态
        self._seen_order = []
        self._color_hits = {1: 0, 2: 0, 3: 0}    # 每色连续命中帧数
        self._position = [0, 0, 0]
        self._explosive_cx = 0
        self._pos = 0
        self._pos_flag = 0
        # task3 人质坑位状态
        self._position2 = [0, 0, 0]
        self._shape_seen = {}     # 形状号 -> 最近一次画面中心 x（跨帧刷新）
        # 三个辅助表与 _shape_seen 同生共死: streak 是连续在场帧数（缺席即清零，
        # 避免累计计数被后来的同名目标救活），first 是首次出现序号（达标时按它重排，
        # 否则坑位顺序会反），peak 是历史最长连续帧数，只用于现场调 shape_hits。
        self._shape_streak: dict = {}
        self._shape_first: dict = {}
        self._shape_peak: dict = {}
        self._latch_inferred = False   # 本次锁存是否用了"未见形状补位"（现场核对用）
        # 检出日志去重基准；换任务/STOP 后清一次，让重订阅首个检出帧重新打一行。
        self._rescue_last_log: dict = {}
        # 两帧一致去抖（task1 / task3 各一套）
        self._pending_order: list = []
        self._pending_count = 0
        self._pending_order2: list = []
        self._pending_count2 = 0
        # 换任务/STOP 后必须清靶标时序状态，否则首帧会继承旧场景的 err。
        # last_err 用 None 表「尚无历史」而非 0: 0 是合法 err_fixed（靶心正中）。
        self._target_last_cx: Optional[float] = None   # 上一帧选中候选的 cx
        self._target_last_err: Optional[int] = None    # 上一帧实际上报的 err_fixed
        self._target_last_bucket: Optional[int] = None  # 死区桶号(连续确认用)
        self._target_confirm_count: int = 0             # 同桶连续帧数
        self._target_drop_count: int = 0                # 连续未确认帧数(日志用)
        self._target_drop_logged: int = 0
        self._target_reject_count: int = 0
        # 跨帧跟踪窄窗（target.track_roi_px）的计数: 用于现场判断 pad 是否偏小。
        # hits 命中=省下全 ROI 扫描; miss=窄窗没命中、当帧回退全 ROI（多花一次窄窗）。
        self._target_track_hits: int = 0
        self._target_track_miss: int = 0
        self._target_last_box: Optional[dict] = None   # 上帧候选 bbox(dynamic 窗用)
        self._target_track_win: Optional[list] = None  # 最近一次跟踪窗(HUD 用)
        self._target_track_scans: int = 0              # 窄窗扫描累计(平均窗面积日志)
        self._target_track_area_sum: int = 0

    # ---- thresholds ----

    def _reload_thresholds(self, force: bool = False) -> None:
        """阈值文件覆盖（组序: 0..2 爆红/爆绿/爆蓝, 3..5 靶红/靶绿/靶蓝）。"""
        exp_lab = {k: list(v) for k, v in self._cfg["explosive"]["lab"].items()}
        tgt_lab = {k: list(v) for k, v in self._cfg["target"]["lab"].items()}
        if self._thresholds_file:
            sig = file_signature(self._thresholds_file)
            if sig is not None and (force or sig != self._thresholds_sig):
                groups = load_threshold_file(self._thresholds_file)
                if len(groups) >= 6:
                    names = ("red", "green", "blue")
                    for i, n in enumerate(names):
                        exp_lab[n] = groups[i]
                        tgt_lab[n] = groups[3 + i]
                self._thresholds_sig = sig
        self._explosive_lab = exp_lab
        self._target_lab = tgt_lab

    # 阈值文件轮询周期（秒）。见 _maybe_reload_thresholds。
    _THRESHOLD_POLL_INTERVAL_S = 0.5

    def _maybe_reload_thresholds(self) -> None:
        """限频 2Hz 检查三个调参页产物的内容签名。
        人手写文件无需更快轮询，每帧省一次 os.stat()；调参页在订阅期使用，
        不能改成只在订阅时检查一次。
        """
        if (not self._thresholds_file and not self._rescue_params_file
                and not self._laser_calib_file):
            return
        now = self._now()
        if now - self._thresholds_poll_ts < self._THRESHOLD_POLL_INTERVAL_S:
            return
        self._thresholds_poll_ts = now
        if self._thresholds_file:
            sig = file_signature(self._thresholds_file)
            if sig is not None and sig != self._thresholds_sig:
                self._reload_thresholds()
        self._reload_rescue_params()
        self._reload_laser_calib()

    def _reload_rescue_params(self, force: bool = False) -> None:
        """加载人质调参页产物。文件缺失或解析为空时保持 yaml 值。
        只 merge 出现的键、不清锁存；签名只在读取成功时缓存，
        保证首次 SAVE 创建文件必触发一次重载。
        """
        if not self._rescue_params_file:
            return
        sig = file_signature(self._rescue_params_file)
        if sig is None:
            return
        if not force and sig == self._rescue_params_sig:
            return
        self._rescue_params_sig = sig
        params = load_rescue_params_file(self._rescue_params_file)
        if params:
            self._cfg["rescue"].update(params)
            log_print(f"[GKS] rescue params "
                      f"{'loaded' if force else 'reloaded'} from "
                      f"{self._rescue_params_file}: {params}")

    def _reload_laser_calib(self, force: bool = False) -> None:
        """激光校准文件（LASER 页 SAVE 产物）加载/热重载。做法跟 _reload_rescue_params 一样。"""
        if not self._laser_calib_file:
            return
        sig = file_signature(self._laser_calib_file)
        if sig is None:
            return
        if not force and sig == self._laser_calib_sig:
            return
        self._laser_calib_sig = sig
        params = load_laser_calib_file(self._laser_calib_file)
        if params:
            for key, value in params.items():
                section, _, _ = _LASER_CALIB_KEYS[key]
                self._cfg[section][key] = value
            log_print(f"[GKS] laser calib "
                      f"{'loaded' if force else 'reloaded'} from "
                      f"{self._laser_calib_file}: {params}")

    def target_zero_px(self) -> int:
        """TARGET err 零点（像素 x）; 0 = 未校准, err 用画面中心。"""
        return int(self._cfg.get("target", {}).get("laser_cx") or 0)

    def _lab_for(self, mode: str, color_code: int) -> Optional[list]:
        key = COLOR_KEYS.get(color_code)
        if key is None:
            return None
        table = self._explosive_lab if mode == MODE_EXPLOSIVE else self._target_lab
        lab = table.get(key[1])
        return list(lab) if lab else None

    def _log_lab_slots(self) -> None:
        """订阅切换时打印实际生效的 LAB 槽位。
        阈值文件按行序绑颜色（忽略 name 前缀），错位是静默的；
        有这行日志可区分文件行序问题与通道序问题。
        """
        if self._mode == MODE_EXPLOSIVE:
            slots = []
            for code in (1, 2, 3):
                slots.append(f"C{code}={COLOR_KEYS[code][1]}{self._lab_for(MODE_EXPLOSIVE, code)}")
            log_print("[GKS] explosive LAB slots: " + "  ".join(slots))
        elif self._mode == MODE_TARGET:
            code = self._codes[1]
            name = COLOR_KEYS.get(code, (None, "?"))[1]
            log_print(f"[GKS] target LAB slot: code={code}({name}) "
                      f"{self._lab_for(MODE_TARGET, code)}")

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
                    pixels_threshold: int, area_threshold: int, merge: bool,
                    margin=None) -> list:
        """单色 find_blobs 扫描，返回 [{x,y,w,h,cx,cy,pixels}]。
        设备优先 maix C++ 快路径，引擎变化时打一次日志便于台架确认是否回退
        （两者帧率差数倍）。margin=None 表示不传该 kwarg。
        """
        if maix_img is not None:
            blobs = find_blobs_maix(
                maix_img, lab, roi=roi_px,
                area_threshold=area_threshold,
                pixels_threshold=pixels_threshold,
                merge=merge,
                margin=margin)
            if blobs is not None:
                self._note_scan_path("maix(C++ find_blobs)")
                return blobs
            # maix 不可用/失败 -> cv2 回退
        self._note_scan_path("cv2 回退(maix 不可用或 find_blobs 失败, 帧率明显偏低)")
        return self._scan_blobs_cv(frame_bgr, lab, roi_px,
                                   pixels_threshold, area_threshold)

    @staticmethod
    def _merge_margin(cfg) -> Optional[int]:
        """merge_margin -> find_blobs 的 margin kwarg；0/非法值 = 不传。
        margin 触发膨胀与连通域重标记，比 merge 本身重。
        """
        try:
            value = int(cfg.get("merge_margin", 0) or 0)
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    def _note_scan_path(self, path: str) -> None:
        if path != self._scan_path:
            self._scan_path = path
            log_print(f"[GKS] blob scan engine = {path}")

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

        # 物化成真 numpy（懒转换配套，见模块级 _as_numpy 的注释）。
        # 按 ROI 切出来的是真 numpy 视图，但 roi_px 为 None 时 search 就是
        # frame_bgr 本身（可能是 LazyFrame 代理）—— cv2 的 src 不吃 __array__。
        search = _as_numpy(search)

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
                self._process_rescue(frame, hits, maix_img)

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
            "inferred": bool(result.get("inferred", False)),
            "hits": hits,
        }
        return VisionResult(task_name=self.name, success=True, result_data=data)

    # ---- 坑位定序（task1 爆炸物 / task3 人质 共用一套）----
    #
    # 2025 方案 task1/task3 的坑位映射是同一个套路: 顺序 = 首次检出先后
    # （相机按 pit1→pit3 扫过, 最先看见的那个就在 pit1）,
    # 且三色/三形状互不相同 -> 见到 2 个即可推出第 3 个"是什么", 补在顺序尾部。
    #
    # 补位只解决"是什么"、不解决"在第几坑"——位置由顺序来源给出。某帧同时检出多个时,
    # 帧内先按画面左右定序（`pit_left_is_1` 决定方向）, 保证结果不受 find_blobs/YOLO
    # 的输出顺序影响（输出序不是空间序）。
    # 锁存仍走"连续 pit_latch_frames 帧同序"去抖, 跟 main.py 一样。

    @staticmethod
    def _flag_on(value, default: str) -> bool:
        """字符串开关判定: none/off/0/false/空 -> False。"""
        if isinstance(value, str):
            return value.strip().lower() not in ("", "none", "off", "0", "false")
        return bool(value if value is not None else default)

    def _infer_missing(self, section: str = "rescue") -> bool:
        """section.infer_missing: 是否允许"只见 2 个即锁存"(未见者补到顺序尾部)。"""
        return self._flag_on(self._cfg[section].get("infer_missing", "tail"), "tail")

    def _order_by_x(self, section: str = "rescue") -> bool:
        """section.order_by == x 时用"全程画面 x 排序"(老 main.py 的做法)。"""
        value = self._cfg[section].get("order_by", "first_seen")
        if isinstance(value, str):
            return value.strip().lower() in ("x", "cx", "image_x", "by_x")
        return bool(value)

    def _ordered_pits(self, section: str, seen):
        """首次出现顺序 seen -> (完整坑位表 | None, 是否用了补位)。

        数量不足（见 infer_missing）返回 None; 缺 1 个时补到顺序尾部。
        """
        if len(seen) < (2 if self._infer_missing(section) else 3):
            return None, False
        ordered = list(seen)
        inferred = False
        if len(ordered) < 3:
            ordered = ordered + [s for s in (1, 2, 3) if s not in seen]
            inferred = True
        return ordered, inferred

    def _latch_debounce(self, ordered, pending_order, pending_count):
        """连续 pit_latch_frames 帧同序才认。-> (pending_order, pending_count, locked)"""
        if ordered == pending_order:
            pending_count += 1
        else:
            pending_order = list(ordered)
            pending_count = 1
        return pending_order, pending_count, \
            pending_count >= int(self._cfg["pit_latch_frames"])

    # ---- task1 爆炸物 ----

    @staticmethod
    def _scan_scale(cfg) -> float:
        """色块扫描降采样系数 `scan_scale`: 0.5 = 320x240 扫, 1.0 = 全帧。

        排爆(explosive)与打靶(target)各有一份, 由调用方传各自的 cfg 段。
        """
        try:
            value = float(cfg.get("scan_scale", 1.0))
        except (TypeError, ValueError):
            return 1.0
        return value if 0.0 < value <= 1.0 else 1.0

    def _downscale(self, frame, maix_img, scale):
        """按 scale 降采样，返回 (BGR 小图, maix 小图或 None)。
        INTER_NEAREST 不插值，避免把边缘混成中间色（等于人为模糊）。
        交给 C++ find_blobs 的小图必须转成 RGB 字节序: MaixCDK 不做通道交换，
        传 BGR 会红蓝互换、坑位定序全错；cv2 回退路径仍用 BGR。
        排爆与打靶共用本函数，面积门槛按 scale²、坐标按 1/scale 还原。
        """
        if scale >= 1.0:
            return frame, maix_img
        h, w = frame.shape[:2]
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(_as_numpy(frame), (sw, sh),
                           interpolation=cv2.INTER_NEAREST)
        # 小图重新包 maix Image: 用 cv2image 而非 img.resize()，保证像素与 NEAREST 结果逐字节一致。
        # 只缓存可用性、每次仍重新 import 模块对象，否则测试注入的假 maix 会被永久屏蔽。
        global _maix_image_ok
        image_mod = None
        if _maix_image_ok is not False:
            try:
                import maix.image as image_mod
                _maix_image_ok = True
            except Exception as e:
                if _maix_image_ok is None:
                    log_print(f"[GKS] maix.image 不可用({type(e).__name__}), "
                              f"降采样小图走 cv2 回退路径")
                _maix_image_ok = False
        small_maix = None
        if image_mod is not None:
            try:
                small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                small_maix = image_mod.cv2image(small_rgb, bgr=False, copy=True)
            except Exception as e:
                log_print(f"[GKS] maix 小图构造失败({w}x{h}->{sw}x{sh}): "
                          f"{type(e).__name__}: {e} (回退 cv2 路径)")
        return small, small_maix

    def _process_explosive(self, frame, maix_img, hits):
        labs = [self._lab_for(MODE_EXPLOSIVE, c) for c in (1, 2, 3)]
        if not all(labs) or frame is None:
            # 阈值缺失不进 find_blobs: 零值心跳不断流, 锁存状态保留（跟 main.py 一样）
            return ({"cx": 0, "pit": 0, "found": False},
                    hits, list(self._position), self._pending_count, False)
        cfg = self._cfg["explosive"]
        scale = self._scan_scale(cfg)
        h, w = frame.shape[:2]
        roi_px = self._roi_px(cfg.get("roi"), w, h)
        frame_s, maix_s = self._downscale(frame, maix_img, scale)
        roi_s = roi_px
        if roi_px is not None and scale != 1.0:
            roi_s = [int(roi_px[0] * scale), int(roi_px[1] * scale),
                     max(1, int(roi_px[2] * scale)), max(1, int(roi_px[3] * scale))]
        # 面积门槛按 scale² 折算（同一物理目标在小图里的像素数是 scale² 倍）
        pixels_th = max(1, int(cfg["pixels_threshold"] * scale * scale))
        area_th = max(1, int(cfg["area_threshold"] * scale * scale))
        color_cx = {}   # 本帧各色最大色块 {色号1红/2绿/3蓝: cx}(已还原到全帧坐标)
        for i in range(3):
            blobs = self._scan_blobs(frame_s, maix_s, labs[i], roi_s,
                                     pixels_th, area_th, merge=False)
            if blobs:
                largest = max(blobs, key=lambda b: b["pixels"])
                hits.append({"x": int(largest["x"] / scale),
                             "y": int(largest["y"] / scale),
                             "w": int(largest["w"] / scale),
                             "h": int(largest["h"] / scale),
                             "text": f"C{i + 1}",
                             "color": (0, 255, 0)})
                color_cx[i + 1] = largest["cx"] / scale
        if self._codes[0] in color_cx:
            self._explosive_cx = int(color_cx[self._codes[0]])
        # 每色连续命中帧数: 现在只作观测(调阈值时看闪烁用), 不再参与锁存判定
        for c in (1, 2, 3):
            self._color_hits[c] = self._color_hits[c] + 1 if c in color_cx else 0

        # 本帧检出色块的空间序 = 扫掠方向上的先后; 首次出现的颜色按该序追加到 _seen_order
        same_frame = sorted(color_cx.keys(), key=lambda c: color_cx[c],
                            reverse=not self._cfg["pit_left_is_1"])
        for color_num in same_frame:
            if color_num not in self._seen_order:
                self._seen_order.append(color_num)
                log_print(f"[GKS] seen += {color_num} -> {self._seen_order}")

        # 锁存: 已见 >=2 色(或 >=3, 由 infer_missing 定)即可补出完整顺序
        if self._pos_flag == 0:
            ordered, inferred = self._ordered_pits("explosive", self._seen_order)
            if ordered is not None:
                self._pending_order, self._pending_count, locked = self._latch_debounce(
                    ordered, self._pending_order, self._pending_count)
                if locked:
                    for j, color_num in enumerate(ordered):
                        self._position[j] = color_num
                    self._latch_inferred = inferred
                    if self._codes[0] in ordered:
                        self._pos = ordered.index(self._codes[0]) + 1
                        self._pos_flag = 1
                    log_print(f"[GKS] latch pit={self._pos} seen={self._seen_order} "
                              f"ordered={ordered} codes={list(self._codes)} "
                              f"inferred={self._latch_inferred}")
        result = {
            "cx": self._explosive_cx,
            "pit": self._pos,
            "found": bool(self._pos_flag),
            "inferred": self._latch_inferred,
        }
        return result, hits, list(self._position), self._pending_count, True

    # ---- task2 靶标 (main.py task2_target 逐行移植) ----

    def _process_target(self, frame, maix_img, hits):
        lab = self._lab_for(MODE_TARGET, self._codes[1])
        if not lab or frame is None:
            # 无阈值不断流: 零值心跳（跟 main.py 一样）
            return {"err_fixed": 0, "found": False}, hits, False
        cfg = self._cfg["target"]
        h, w = frame.shape[:2]
        roi_px = self._roi_px(cfg["roi"], w, h)
        scale = self._scan_scale(cfg)

        # 降采样两段式: 小图只定位，全分辨率窄窗复扫算 err_fixed（见 _DEFAULT_CONFIG）。
        # 跟踪窄窗命中直接用，未命中当帧回退全 ROI，精度与检出率不变。
        # find_blobs 成本正比于搜索面积，是全帧预算里最大的一块。
        blobs = self._target_tracked_scan(frame, maix_img, lab, roi_px, cfg, scale)

        if blobs is None and scale < 1.0:
            pad = int(cfg.get("refine_pad_px", 24))
            coarse = self._target_coarse_scan(frame, maix_img, lab, roi_px, cfg, scale)
            if coarse is None:
                return {"err_fixed": 0, "found": False}, hits, True
            fine_roi = self._target_fine_roi(coarse, roi_px, pad, scale, w, h)
            blobs = self._scan_blobs(frame, maix_img, lab, fine_roi or roi_px,
                                     cfg["pixels_threshold"], cfg["area_threshold"],
                                     merge=bool(cfg["merge"]),
                                     margin=self._merge_margin(cfg))
            if not blobs:
                # 窄窗里没复现（目标移出/阈值边缘抖动）: 用粗结果保底, 保持不断流
                blobs = [{
                    "x": int(coarse["x"] / scale), "y": int(coarse["y"] / scale),
                    "w": int(coarse["w"] / scale), "h": int(coarse["h"] / scale),
                    "cx": coarse["cx"] / scale, "cy": coarse["cy"] / scale,
                    "pixels": coarse["pixels"],
                }]
        elif blobs is None:
            blobs = self._scan_blobs(frame, maix_img, lab, roi_px,
                                     cfg["pixels_threshold"], cfg["area_threshold"],
                                     merge=bool(cfg["merge"]),
                                     margin=self._merge_margin(cfg))
        if blobs:
            largest = self._pick_target_blob(blobs, cfg)
            cx = largest["cx"]
            hits.append({"x": int(largest["x"]), "y": int(largest["y"]),
                         "w": int(largest["w"]), "h": int(largest["h"]),
                         "text": "TGT", "color": (255, 0, 0)})
            # 归一化误差 (cx - zero)/(w/2)，clamp 到 1 后乘 10000。
            # zero 为激光零点或画面中心；除数固定 w/2，Brain 闭环参数不动。
            half = w / 2.0
            zero = self.target_zero_px() or half
            normalized = (cx - zero) / half
            normalized = max(-1.0, min(1.0, normalized))
            err_fixed = int(normalized * 10000)   # int(): 与 main.py 同为截断取整
            # 顺序要紧: 先记last_cx/桶, 再判确认 —— _target_last_err 必须在通过
            # 确认之后才写(见 _target_confirm_ok), 否则被丢弃的帧也会污染死区基准。
            err_fixed = self._apply_target_deadband(err_fixed, cfg)
            self._target_last_cx = cx
            self._target_last_box = {"x": int(largest["x"]), "y": int(largest["y"]),
                                     "w": int(largest["w"]), "h": int(largest["h"]),
                                     "cx": cx, "cy": float(largest["cy"])}
            if not self._target_confirm_ok(err_fixed, cfg):
                return {"err_fixed": 0, "found": False}, hits, True
            return {"err_fixed": err_fixed, "found": True}, hits, True
        return {"err_fixed": 0, "found": False}, hits, True

    # ---- task2 候选选择与时序（防误检）----
    def _target_tracked_scan(self, frame, maix_img, lab, roi_px, cfg, scale):
        """跨帧跟踪窄窗扫描（fixed/dynamic）。返回 blob 列表=命中，None=走全 ROI。

        开关关闭、scale<1、无历史或窗口退化时返回 None；窄窗未命中也返回 None，
        调用方当帧回退全 ROI，因此不会漏检，只是多花一次窄窗的钱。
        只服务 TARGET: 排爆要全帧三色定序、人质走 YOLO 全帧，缩小搜索域会改判定。
        """
        mode = str(cfg.get("track_mode") or "fixed").strip().lower()
        if mode == "off":
            return None
        if mode not in ("fixed", "dynamic"):   # 非法值按 fixed
            mode = "fixed"
        try:
            pad = int(cfg.get("track_roi_px") or 0)
        except (TypeError, ValueError):
            pad = 0
        if pad <= 0 or scale < 1.0 or roi_px is None:
            return None
        if mode == "fixed":
            if self._target_last_cx is None:
                return None
            win = self._target_track_window(self._target_last_cx, roi_px, pad)
        else:
            if self._target_last_box is None:
                return None
            win = self._target_dynamic_window(self._target_last_box, roi_px, pad, cfg)
        if win is None:
            return None
        self._target_track_win = list(win)
        self._target_track_scans += 1
        self._target_track_area_sum += win[2] * win[3]
        blobs = self._scan_blobs(frame, maix_img, lab, win,
                                 cfg["pixels_threshold"], cfg["area_threshold"],
                                 merge=bool(cfg["merge"]),
                                 margin=self._merge_margin(cfg))
        if blobs:
            self._target_track_hits += 1
            if self._target_track_hits % 200 == 1:
                avg = self._target_track_area_sum // max(1, self._target_track_scans)
                log_print(f"[GKS] target 跟踪窄窗命中 x{self._target_track_hits} "
                          f"(win={win[2]}x{win[3]}px, 平均窗 {avg}px², "
                          f"窗扫 {self._target_track_scans} 次)")
            return blobs
        self._note_target_track_miss()
        return None

    @staticmethod
    def _target_track_window(last_cx, roi_px, pad):
        """上一帧 cx 加减 pad 的窄窗，高度撑满 ROI 并夹回 ROI；退化返回 None。
        pad 覆盖目标一帧内的最大位移，偏小会频繁未命中，按「窄窗未命中」日志调。
        """
        rx, ry, rw, rh = roi_px
        x0 = max(rx, int(last_cx) - pad)
        x1 = min(rx + rw, int(last_cx) + pad)
        if x1 - x0 <= 2:
            return None
        return [x0, ry, x1 - x0, rh]

    @staticmethod
    def _target_dynamic_window(box, roi_px, pad, cfg):
        """dynamic 跟踪窗: 上帧 bbox 缩放，双向封顶不大于 fixed 窗；退化返回 None。
        宽 = min(2*pad, max(min_w, scale*bw))，高 = min(rh, bh + 2*v_pad)。
        """
        rx, ry, rw, rh = roi_px
        bw = max(1, int(box["w"]))
        bh = max(1, int(box["h"]))
        try:
            k = float(cfg.get("track_win_scale", 4.0))
            min_w = int(cfg.get("track_win_min_w", 48))
            v_pad = int(cfg.get("track_v_pad_px", 16))
        except (TypeError, ValueError):
            return None
        w = min(2 * pad, max(min_w, int(k * bw)), rw)
        h = min(rh, max(1, bh + 2 * v_pad))
        if w <= 2 or h <= 2:
            return None
        x0 = max(rx, min(int(box["cx"]) - w // 2, rx + rw - w))
        y0 = max(ry, min(int(box["cy"]) - h // 2, ry + rh - h))
        return [x0, y0, w, h]

    def _note_target_track_miss(self) -> None:
        """窄窗未命中的按批汇总日志，不逐帧刷屏。"""
        self._target_track_miss += 1
        if self._target_track_miss % 20 == 1:
            log_print(f"[GKS] target 跟踪窄窗未命中 x{self._target_track_miss} "
                      f"-> 已当帧回退全 ROI（结果与全程全 ROI 一致, 只是多花一次窄窗）;"
                      f" 频繁出现请调大 target.track_roi_px")

    @staticmethod
    def _filter_by_fill(blobs: list, cfg: dict) -> list:
        """候选填充率过滤 pixels/(w*h); 空集回退原列表(只降误检, 不制造漏检)。"""
        try:
            lo = float(cfg.get("fill_min", 0.0))
            hi = float(cfg.get("fill_max", 1.0))
        except (TypeError, ValueError):
            return blobs
        if lo <= 0.0 and hi >= 1.0:
            return blobs
        kept = [b for b in blobs
                if lo <= b["pixels"] / max(1, b["w"] * b["h"]) <= hi]
        if not kept:
            log_print("[GKS] target 填充率过滤后无候选 -> 回退不过滤")
        return kept or blobs

    def _pick_target_blob(self, blobs: list, cfg: dict):
        """从候选里挑一个靶标 blob。
        按 pixels 最大挑，并列时按 cx 取确定性结果，保证同一候选集可复现。
        不用形状先验: 靶标是圆环，实心球阈值会误杀真环。
        prefer_continuity 开启时改为取离上帧 cx 最近者。
        """
        blobs = self._filter_by_fill(blobs, cfg)
        last_cx = self._target_last_cx

        if bool(cfg.get("prefer_continuity")) and last_cx is not None:
            # 连续性优先: 离上一帧最近者胜。同一候选自己与 last_cx 距离 0,
            # 所以"单候选时行为不变"这条性质自然成立, 不需要额外分支。
            return min(blobs, key=lambda b: (abs(float(b["cx"]) - last_cx),
                                            -float(b["pixels"]), float(b["cx"])))

        # 基线路径(默认): 纯 pixels 最大, 但补一个确定性 tiebreak。
        return max(blobs, key=lambda b: (float(b["pixels"]), -float(b["cx"])))

    def _apply_target_deadband(self, err_fixed: int, cfg: dict) -> int:
        """死区: 新值与上次差小于 deadband 时保持上次值。
        不用滑动平均: 打靶时延已超控制周期，再串平均会恶化 PID 相位裕度。
        默认 0 = 关闭。
        """
        deadband = int(cfg.get("err_deadband") or 0)
        if deadband <= 0 or self._target_last_err is None:
            return err_fixed
        if abs(err_fixed - self._target_last_err) < deadband:
            return self._target_last_err
        return err_fixed

    def _target_confirm_ok(self, err_fixed: int, cfg: dict) -> bool:
        """连续确认: err 连续 N 帧落同一死区桶才认。
        n>1 时必须配 err_deadband>0，否则桶退化为 err 本身（要求逐帧完全相同）。
        基准必须在两条 early return 之前写，否则默认 n=1 时永不更新；
        被丢弃的帧不写基准，以免污染死区。
        """
        n = max(1, int(cfg.get("confirm_frames") or 1))
        if n <= 1:
            self._target_last_err = err_fixed
            return True
        deadband = max(1, int(cfg.get("err_deadband") or 0))
        bucket = err_fixed // deadband
        if bucket == self._target_last_bucket:
            self._target_confirm_count += 1
        else:
            self._target_last_bucket = bucket
            self._target_confirm_count = 1
        if self._target_confirm_count < n:
            self._note_target_drop()
            return False
        self._target_drop_count = 0
        self._target_drop_logged = 0
        self._target_last_err = err_fixed
        return True

    def _note_target_drop(self) -> None:
        """连续未确认的按秒汇总(需配 debug 级别, 默认不开)。"""
        self._target_drop_count += 1
        if self._target_drop_count - self._target_drop_logged >= 20:
            self._target_drop_logged = self._target_drop_count
            log_print(f"[GKS] target confirm 未满, 连续丢弃 "
                      f"{self._target_drop_count} 帧(待confirm_frames 帧一致)")

    def _target_coarse_scan(self, frame, maix_img, lab, roi_px, cfg, scale):
        """降采样定位: 返回小图上的最大色块 dict, 无命中返回 None。"""
        frame_s, maix_s = self._downscale(frame, maix_img, scale)
        roi_s = roi_px
        if roi_px is not None:
            roi_s = [int(roi_px[0] * scale), int(roi_px[1] * scale),
                     max(1, int(roi_px[2] * scale)), max(1, int(roi_px[3] * scale))]
        pixels_th = max(1, int(cfg["pixels_threshold"] * scale * scale))
        area_th = max(1, int(cfg["area_threshold"] * scale * scale))
        blobs = self._scan_blobs(frame_s, maix_s, lab, roi_s,
                                 pixels_th, area_th, merge=bool(cfg["merge"]),
                                 margin=self._merge_margin(cfg))
        if not blobs:
            return None
        return max(blobs, key=lambda b: b["pixels"])

    @staticmethod
    def _target_fine_roi(coarse, roi_px, pad, scale, w, h):
        """粗框还原到全帧 + pad 外扩，再夹回画面与 ROI。
        外扩后必须与原 ROI 求交，否则 ROI 排除规则在降采样档失效。
        另留 1px 余量补偿粗框取整误差；窗口退化返回 None。
        """
        bx = coarse["x"] / scale
        by = coarse["y"] / scale
        bw = coarse["w"] / scale
        bh = coarse["h"] / scale
        x0 = int(bx) - pad
        y0 = int(by) - pad
        x1 = int(bx + bw) + pad
        y1 = int(by + bh) + pad
        # 夹回画面
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        # 夹回原 ROI（两者都是 [x, y, w, h]）
        if roi_px is not None:
            rx, ry, rw, rh = roi_px
            x0, y0 = max(x0, rx), max(y0, ry)
            x1, y1 = min(x1, rx + rw), min(y1, ry + rh)
        if x1 - x0 <= 1 or y1 - y0 <= 1:
            return None       # 窗口退化 -> 调用方回退到原 ROI
        return [x0, y0, x1 - x0, y1 - y0]

    # ---- task3 人质 ----

    # ---- task3 人质: 检出日志 ----

    # 同一标签分数波动 ≤ 此值不打新行（"变化才打", 防 30fps 逐帧落盘刷屏）。
    _RESCUE_DET_LOG_DELTA = 0.1

    def _log_rescue_dets(self, dets) -> None:
        """人质检出日志: 形状集合或分数跳变超过 0.1 才打一行，不逐帧刷屏。
        同标签多检出取分最高者；坐标为画面中心。
        格式: [GKS] rescue det: yuan_zhu=0.82(214,165) ...
        """
        cur = {}
        for label, score, cx, cy in dets:
            prev = cur.get(label)
            if prev is None or score > prev[0]:
                cur[label] = (score, cx, cy)
        last = self._rescue_last_log
        changed = set(cur) != set(last)
        if not changed:
            for label, (score, _cx, _cy) in cur.items():
                if abs(score - last[label][0]) > self._RESCUE_DET_LOG_DELTA:
                    changed = True
                    break
        if not changed:
            return
        self._rescue_last_log = cur
        body = " ".join(f"{lb}={sc:.2f}({cx},{cy})"
                        for lb, (sc, cx, cy) in cur.items())
        log_print(f"[GKS] rescue det: {body or '(none)'}")

    def _process_rescue(self, frame, hits, maix_img=None):
        ai = self._ai
        if ai is None or not getattr(ai, "loaded", False):
            # 模型缺失时也不中断: 零值心跳不断流, 不崩溃
            return ({"pos": 0, "found": False, "inferred": self._latch_inferred},
                    hits, list(self._position2), self._pending_count2)

        # 零拷贝推理: maix Image 可用时走 _raw=True，省掉 image2cv 与 cv2image 两次拷贝。
        # 两平台 ai.detect 都 pop 掉 _raw；无 maix 时退回 numpy，行为与改前一致。
        if maix_img is not None:
            objs = ai.detect(maix_img, conf_th=self._cfg["rescue"]["conf_th"],
                             iou_th=self._cfg["rescue"]["iou_th"], _raw=True)
        else:
            objs = ai.detect(frame, conf_th=self._cfg["rescue"]["conf_th"],
                             iou_th=self._cfg["rescue"]["iou_th"])
        # 同一帧内可能同时检出多个形状: 此时"时间先后"无意义, 先按画面左右关系定序
        # (镜像时反向), 保证入表顺序不受模型输出顺序影响(YOLO 的输出序不是空间序)。
        # 跨帧仍保持"首次检出先后"—— 比赛流程固定, 相机按 pit1→pit3 扫过。
        objs = sorted(objs, key=lambda o: o.x, reverse=not self._cfg["pit_left_is_1"])
        labels = getattr(ai, "labels", None) or []
        label_map = self._cfg["rescue"]["labels"]
        x_crop = int(self._cfg["rescue"].get("x_crop_px") or 0)
        shape_hits = max(1, int(self._cfg["rescue"].get("shape_hits", 1) or 1))
        # 本帧出现的形状（shape -> cx）。先聚合成帧、再走连续帧门槛, 而不是在下面的
        # 循环里直接写 _shape_seen —— 同一帧同形状出现两个框时, 逐框累加会让
        # streak 一帧 +2, 门槛形同虚设。
        frame_shapes: dict = {}
        det_log = []    # (label, score, cx, cy) —— "变化才打"检出日志的数据源
        for obj in objs:
            if x_crop > 0 and obj.x > x_crop:
                continue
            label = labels[obj.class_id] if obj.class_id < len(labels) else ""
            shape_num = int(label_map.get(label, 0))
            hits.append({"x": int(obj.x), "y": int(obj.y),
                         "w": int(obj.w), "h": int(obj.h),
                         "text": f"{label}:{obj.score:.2f}",
                         "color": (0, 160, 255)})
            det_log.append((label or f"c{obj.class_id}", float(obj.score),
                            int(obj.x + obj.w / 2.0), int(obj.y + obj.h / 2.0)))
            if shape_num == 0:
                continue
            # 后写覆盖 = 老做法（objs 已按 x 排序, 旧代码也是最后一次赋值生效）。
            # 不要改成"取最高分": 那是白白把结果改掉了, 且会让 shape_hits=1 时的
            # 行为与改前不一致, 失去"默认值即回归开关"的性质。
            frame_shapes[shape_num] = obj.x + obj.w / 2.0
        # 入表门槛: 目标右侧入画首帧半出画 + 运动模糊，YOLO 易判成邻类；
        # 旧代码首帧即写 _shape_seen 永久占位，会静默锁死错误坑位，故要求连续 shape_hits 帧。
        for s in self._shape_streak:
            if s not in frame_shapes:
                self._shape_streak[s] = 0       # 只改值不增删键, 无迭代中改字典异常
        for s, cx in frame_shapes.items():
            self._shape_first.setdefault(s, len(self._shape_first))
            n = self._shape_streak.get(s, 0) + 1
            self._shape_streak[s] = n
            if n > self._shape_peak.get(s, 0):
                self._shape_peak[s] = n
            if n < shape_hits:
                continue                        # 证据不够: 不占坑, 也不刷新 cx
            is_new = s not in self._shape_seen
            self._shape_seen[s] = cx            # 已入表者每帧刷新 cx（HUD / order_by:x 行为不变）
            if is_new and len(self._shape_seen) > 1:
                # 按"首次出现"重排, 而不是按"达标先后": 先入画的坑若中途漏检一两帧,
                # 它的 streak 会清零重计, 可能比后入画的坑更晚达标 —— 按达标顺序
                # 会把它排到后面, 坑位顺序整体反掉。shape_hits=1 时两者等价。
                self._shape_seen = dict(sorted(
                    self._shape_seen.items(),
                    key=lambda kv: self._shape_first.get(kv[0], 0)))
        self._log_rescue_dets(det_log)
        # ---- 顺序映射（定序规则与补位见上方"坑位定序"共用段）----
        if self._position2[0] == 0:
            if self._order_by_x("rescue"):
                seq = [s for s, _cx in sorted(self._shape_seen.items(), key=lambda kv: kv[1])]
                # 镜像翻转必须在补位之前: 未见者补在"扫掠方向的最后一个"
                if not self._cfg["pit_left_is_1"]:
                    seq.reverse()
            else:
                # dict 保持插入顺序 = 首次检出先后 = 物理坑位顺序, 与镜像开关无关
                seq = list(self._shape_seen)
            ordered, inferred = self._ordered_pits("rescue", seq)
            if ordered is not None:
                self._pending_order2, self._pending_count2, locked = self._latch_debounce(
                    ordered, self._pending_order2, self._pending_count2)
                if locked:
                    for j, shape_num in enumerate(ordered):
                        self._position2[j] = shape_num
                    self._latch_inferred = inferred
                    # peak = 见过但未攒够门槛的形状及其最长连续帧数，是幽灵的指纹。
                    # 锁存时 streak 已清零，只有 peak 能证明它曾连续出现几帧。
                    # 现场调 shape_hits 看这里: 未占坑形状的 peak 逼近 hits 时该抬门槛或查模型。
                    peak = {s: n for s, n in self._shape_peak.items()
                            if s not in self._shape_seen and n > 0}
                    log_print(f"[GKS] latch pos2={self._position2} "
                              f"seen={list(self._shape_seen)} "
                              f"hits={shape_hits} peak={peak} "
                              f"order_by={'x' if self._order_by_x('rescue') else 'first_seen'} "
                              f"codes={list(self._codes)} inferred={self._latch_inferred}")
        pos2 = 0
        if self._codes[2] != 0:     # 守卫: 形状码未下发(0)时不得与未锁存 0 自匹配
            for j in range(3):
                if self._codes[2] == self._position2[j]:
                    pos2 = j + 1
                    break
        return ({"pos": pos2, "found": bool(pos2), "inferred": self._latch_inferred},
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
