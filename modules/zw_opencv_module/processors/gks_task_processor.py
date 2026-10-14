# -*- coding: utf-8 -*-
"""
工科赛近端任务处理器 — 一处理器三模式（EXPLOSIVE / TARGET / RESCUE）。

识别逻辑逐行照搬 F:\\2026工科赛\\maixcam\\工科赛上端\\main.py 的
task1_explosive / task2_target / task3_rescue（锁存/去抖/映射语义不变），
仅做平台适配：

- 图像来源: 框架单路 640 宽帧（旧版 cam 640x480 + cam1 320x240 双实例消亡）;
  YOLO 输入由 nn 类按模型输入尺寸自适应缩放（CAM2=nn.YOLO11，**CAM1=nn.YOLOv5**）。
- 检测引擎: 设备走 maix_accel C++ find_blobs 快路径（阈值 OpenMV LAB 语义
  1:1 透传），PC/失败自动回退 cv2 inRange（OpenMV→OpenCV LAB 编码转换）。
- 人质: 模型缺失时回零值结果（心跳不断流），不崩溃。
  - CAM2 分支: nn.YOLO11 + .axmodel（Axera），走 MaixCam2AI
  - CAM1 分支: nn.YOLOv5 + .cvimodel（CVITEK），走 MaixCam1AI
  两者 mud 键名不同（`model_npu` vs `model`），模型文件互不通用。
- 像素常量全部入 eod_config.yaml `gks:` 段；`/my-threshold/my_64threshold.txt`
  （脱机调阈值工具产出）作运行时覆盖，mtime 变化自动重载、不清锁存。
- 人质参数文件 `/my-threshold/my_rescue_params.txt`（调参页 SAVE 产出, conf_th/
  iou_th/x_crop_px）同 mtime 热重载语义, 详见 `_reload_rescue_params`。
- 人质检出日志"变化才打"（`_log_rescue_dets`）: 形状集合变化或分数跳变 >0.1
  才打一行汇总, 防 log_print 逐行落盘被逐帧检出刷爆。
- 排爆扫描可降采样: `explosive.scan_scale`（0.5 = 320x240 扫描）—— 全帧三色串行
  find_blobs 是排爆帧率瓶颈（台架实测 12~19fps, 低于 50ms 推流节奏）。降采样后
  面积门槛按 scale² 折算、cx/框按 1/scale 还原到全帧坐标; 缩放用 INTER_NEAREST
  （不做插值, 避免把边缘像素混成中间色 —— 那正是运动模糊掉色的同一机理）。
  ⚠ 降采样小图必须交 **RGB 字节序**（2026-09-26 台架实测: 交 BGR 会红蓝互换,
  蓝色目标被框成 C1、坑位定序连带全错）—— 机理见 `_downscale` 注释。

锁存语义（task1/task3 同一套，与 2025 视觉方案 task1/task3 一致）:
- 坑位顺序 = **首次检出先后**（相机按 pit1→pit3 扫过, 最先看见的那个就在 pit1）;
  某帧同时检出多个时帧内先按画面左右定序（`pit_left_is_1` 决定方向）;
- 三色/三形状互不相同 → 见到 2 个即可推出第 3 个"是什么", 补在顺序尾部
  （2025 原版写法: `position[2] = 6 - position[0] - position[1]`）;
- 顺序连续 `pit_latch_frames(2)` 帧一致才锁存；一旦锁存不再重锁, 直到
  set_task 换 data_id/换任务码或 reset()。
- 每任务的顺序来源与是否允许补位可分别配置: `<task>.order_by` / `<task>.infer_missing`。
- set_task 幂等：同模式同任务码的重订阅不清锁存（Brain 掉线 1s 重订阅安全）。

纯 cv2/numpy + 注入时钟 — 设备与 PC(pytest) 双端可运行。
"""
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

# 人质调参页 SAVE 落盘文件（逐行 key=value）。mtime 变化热重载, 语义同
# thresholds_file（/my-threshold/my_64threshold.txt）; 文件缺失 = 保持 yaml 值。
RESCUE_PARAMS_FILE = "/my-threshold/my_rescue_params.txt"
# 可被该文件覆盖的键（与调参页滑条一一对应）及类型。
_RESCUE_PARAM_KEYS = {"conf_th": float, "iou_th": float, "x_crop_px": int}


def load_rescue_params_file(path: str) -> dict:
    """解析人质调参覆盖文件（key=value 行, '#' 注释, 空行忽略）。

    只认 _RESCUE_PARAM_KEYS 里的键并做类型收敛（int 键经 float 再取整）;
    非法行/非法值整行忽略。文件缺失/不可读返回 {} —— 调用方据此保持
    yaml 值不动（PC 端无 /my-threshold 也安全 no-op）。
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

# maix.image 是否可用的惰性探测结果（None=未探测 / False=不可用 / True=可用）。
# 只缓存"可用性", 不缓存模块对象 —— 见 GksTaskProcessor._downscale 的注释
# （缓存对象会与测试的 sys.modules monkeypatch 冲突）。PC 端首帧探测一次即 False,
# 之后不再逐帧重抛 ImportError。
_maix_image_ok = None

_DEFAULT_CONFIG = {
    "pit_left_is_1": True,      # True=画面左(小x)=坑1; 摄像头背装镜像时改 False
    "pit_latch_frames": 2,      # 连续 N 帧左右序一致才锁存（main.py 硬编码 2）
    "thresholds_file": "/my-threshold/my_64threshold.txt",
    "rescue_params_file": RESCUE_PARAMS_FILE,
    "explosive": {
        # 面积门槛 2026-09-26 下调: 10000 -> 4000（10000 px ≈ 100x100 ≈ 画面 3.3%,
        # 是 2025 main.py 原值, 对应他们那套机位; 我们机位目标更小, 且运动模糊会先啃掉
        # 边缘一圈 —— 调参页看得见(只要 area>=200)但主流程 found=0 就是这个门槛卡住）。
        # 与 scan_scale 叠加时按 scale² 折算(0.5 -> 小图 1000 像素), 物理口径不变。
        # 精调: 静止读调参页 `max:{area}px` 取 40~50%; 背景成块误检时往回加, 或先收紧 A/B 轴。
        "pixels_threshold": 4000,
        "area_threshold": 4000,
        # 顺序来源 / 未见者补位, 语义同 rescue: 2025 方案 task1 也是"首次检出先后 +
        # 缺色用 6-a-b 推出来"。2026-09-24 台架决策: 排爆**使用推断**(infer_missing: tail,
        # 与 2025 方案一致) —— 见到 2 色即锁, 未见者补到顺序尾部。
        # ⚠ 代价(台架必核对): 补位只推出"是什么"、推不出"在第几坑"; 一旦"未见者 =
        #   最后一个坑"不成立, 会**静默锁错坑 -> 转台打错坑 -> 抓错**, 比锁不上更危险。
        #   核对手段: 日志 `latch … inferred=True` 出现时必须拿实物核对 PIT/POS。
        #   要退回保守行为把它置 none(三色齐见才锁, 锁不上会超时 FAULT 可见)。
        "order_by": "first_seen",
        "infer_missing": "tail",
        # 排爆扫描降采样系数: 1.0 = 原分辨率(640x480) 全帧扫描; 0.5 = 320x240。
        # 全帧三色串行 find_blobs 是排爆帧率瓶颈(台架实测 12~19fps, 慢于 50ms 推流节奏)。
        # 降采样后: 面积门槛按 scale² 折算, cx/框按 1/scale 还原到全帧坐标。
        # 置 1.0 即回到原分辨率。
        "scan_scale": 1.0,
        # 兜底 LAB 阈值（阈值文件缺失时用）; 2026-09-23 台架回写, 与设备
        # /my-threshold/my_64threshold.txt 逐字一致 —— 文件在时以文件为准, 两者同值。
        # 取值原则（纯色目标）: A/B 负责色相与饱和度判别, L 只当噪声地板 ——
        # 不用 L 上限去排除高光（白/高光 a*≈b*≈0, 靠 A/B 天然排除; 用 L 上限切
        # 只会连物体亮面一起切掉, 色块被切成半块 -> 面积/质心双失准）。
        # ⚠ 这组值偏保守, 已知代价（现场"看得见却锁不上 / 色块切半"时按下表放开）:
        #   1) L 上限 red 80 / green 85 / blue 60: 高饱和纯色本身 L 就高(纯绿 87.7,
        #      受光面 74+; 实测绿球亮面 L 到 81) —— 绿球高光面仍可能出界, 可放到 95;
        #      蓝 60 是"只要深蓝"的代价, 蓝目标受光变亮时先放开到 80;
        #   2) 红 A 下界 40: 砍掉红暗面(实测 A=37);
        #   3) 靶标段与本段逐字同值 —— 改这里记得同步 target["lab"]。
        # 绿(2026-09-26 收紧): 旧 [8,67,-83,-10,-47,71] 太松 —— 全帧误命中 18570 px 中
        #   15658 px 是低饱和淡绿/灰绿(A -10~-25、B 0~35)。现靠两根色度轴收口:
        #   Amax -10→-25(同时挡哑光黄 A≈-12~-20)、Bmin -47→30(挡低饱和偏青绿);
        #   代价用 Lmax 67→85 补回(球心 ROI 覆盖 28→48px, 正检 548→525 基本不掉)。
        #   要更狠的选择性: [8,67,-83,-25,30,71](误命中 954, 但球内少约 20%)。
        # 蓝(2026-09-26 收紧): 旧 [0,80,-110,100,-66,-10] 只靠 B 一维且 Bmax=-10 离中性灰
        #   只有 10 —— 全帧误命中 5365 px 中 3644 px 是浅蓝/淡蓝(B -25~-5)。现按"只要深蓝"
        #   收口: Bmax -10→-25、Lmax 80→60、A 由全开 [-110,100] 收到 [-8,45](挡紫/青,
        #   让 A 参与判别)、Bmin -66→-75(补住暗面 -69)。代价: 球内 401→341(-15%),
        #   换误命中 -69% 且淡蓝归零。蓝目标受光变亮导致切块时先放开 Lmax 到 80。
        # 排爆扫描 ROI: 帧比例 (x0, y0, x1, y1); None/缺省 = 全帧(与改动前逐字节一致)。
        # 纯提速视窗: 只缩小 find_blobs 搜索域, 不改定序语义。
        # ⚠ 定序靠"首次检出先后 + 同帧内按画面左右" ⇒ **三坑必须同时落在 ROI 内**,
        #   任一坑被 ROI 切出视野就会静默错序(漏检->补位塞尾部), 比锁不上更危险。
        #   现场先把 ROI 放到覆盖三坑, 再看日志 cx/cy 收窄。
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
        # merge_margin: find_blobs 的 margin（形态学膨胀半径, 加重连通域重标记）。
        #   ⚠ 0/None = **不传该 kwarg**（2026-10-07 台架修正）。本项此前硬编码
        #   margin=1, 但去年基准 main.py:390 是裸 `merge=True` 不传 margin。
        #   CAM1 算力弱一个量级, 带 margin 时订阅后实测只有 5.3fps(189ms/帧),
        #   是"比去年明显滞后"的主要嫌疑。默认回到与去年逐字节一致的调用形态。
        #   需要 margin 时填整数(如 1), 会走有 margin 的分支, 固件不认仍降级到无 margin。
        "merge_margin": 0,
        # ROI 帧比例 (x0, y0, x1, y1); 默认值 = main.py [0,169,640,197] @640x480
        "roi": [0.0, 169 / 480, 1.0, 366 / 480],
        # 靶标扫描降采样（2026-10-07）: 1.0 = 640x480 全帧扫, 0.5 = 320x240 扫。
        #   降采样是**两段式**（见 _process_target 注释）: 小图只定位"有没有/在哪"
        #   （门槛 ×scale²、ROI 面积降到 1/scale², 省时间的大头），再在贴着粗框的
        #   全分辨率窄窗上按原门槛复扫一次, 由该次结果算 cx 与 err_fixed。
        #   ⚠ 不能只降采样了就算误差: cx 是像素级量, 小图 ±1 像素还原到全帧
        #   就是 ±1/scale 像素, scale=0.5 时偏差可达 30+ LSB(实测 7031 vs 7062)。
        #   打靶是云台闭环、误差即控制量, 这点精度不能省 —— 故精修那一步不可删。
        "scan_scale": 1.0,
        # 精修窗口相对粗框的外扩像素（全分辨率口径, 640 宽下 24 ≈ 3.75% 画面宽）。
        #   取小了追不上目标一帧内的位移 -> 窄窗扫不到、回落粗结果（精度打折）;
        #   取大了就吃掉降采样的收益。默认按 640x480 标定, 换分辨率按比例调。
        "refine_pad_px": 24,
        # fixed 跟踪窗半宽; 0 = 关。dynamic 模式下作横向封顶(窗宽 ≤ 2×此值)。
        "track_roi_px": 0,
        # 跟踪窗模式: off | fixed(上帧 cx±track_roi_px) | dynamic(随上帧 bbox 缩放)。
        # 非法值按 fixed。未命中均当帧回退全 ROI, 结果语义不变。
        "track_mode": "fixed",
        # dynamic 窗: 宽 = min(2*track_roi_px, max(min_w, scale*blob_w)),
        #             高 = min(roi高, blob_h + 2*v_pad)。双向封顶不大于 fixed 窗。
        "track_win_scale": 4.0,
        "track_win_min_w": 48,
        "track_v_pad_px": 16,
        # 候选填充率过滤 pixels/(w*h): 环≈0.5~0.7, 实心矩形≈0.9+。
        # 全开区间 = 不过滤; cv2 路径内孔被填实(环≈0.79), max 需留余量, 阈值真机标定。
        "fill_min": 0.0,
        "fill_max": 1.0,
        # 与 explosive 段同值（"什么是红/绿/蓝"不随任务变）; 现场可按 ROI 内实际光比
        # 分头标定 —— 物理条件不同（爆: 全帧三坑 + 10000 面积门槛; 靶: 窄 ROI + 1000）,
        # 同一组值只是起点而非最优; 要分开标定先改这里。
        "lab": {
            "red":   [0, 80, 40, 80, 10, 80],
            "green": [25, 85, -80, -25, 30, 85],
            "blue":  [0, 60, -8, 45, -75, -25],
        },
    },
    "rescue": {
        "conf_th": 0.7,
        "iou_th": 0.45,
        # 旧版 obj.x>240 (320 宽画面) 过滤; cam1 已消亡, 0 = 关闭; 台架复现旧行为可设像素值
        "x_crop_px": 0,
        # 未见形状的补位规则（三坑形状互不相同, 见到 2 个即可推出第 3 个"是什么"）:
        #   "tail" = 未见者就是顺序上的最后一个 = 最后遇到的那个坑, 补在顺序尾部（默认）;
        #   "none"/"" = 不补位, 必须三形状齐见才锁存（旧 main.py 语义, 退路）。
        # 补位只能推出"是什么", 推不出"在第几坑" —— 位置靠 order_by 给。
        "infer_missing": "tail",
        # 坑位顺序的来源:
        #   "first_seen" (默认) = 首次被检测到的先后。比赛流程固定、相机按 pit1→pit3
        #       方向扫过, 所以"最先看见的那个就在 pit1"; 该顺序是物理顺序, 与图像左右
        #       无关, 故 **pit_left_is_1 不参与跨帧定序**（它只用于同帧内的左右定序, 见下）。
        #   "x" = 按画面 x 从左到右排序（旧 main.py 语义; 三坑同屏时用; 背装镜像需
        #       把 pit_left_is_1 置 false 做整体翻转）。
        "order_by": "first_seen",
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


def _as_numpy(frame):
    """把帧物化成真 BGR numpy（是 LazyFrame 就取其缓存的真数组, 否则原样返回）。

    为什么需要（2026-10-09, 懒转换配套）: OpenCV 的 `src` 参数**不吃 `__array__`
    协议** —— `cv2.cvtColor(lazy, ...)` / `cv2.resize(lazy, ...)` 会直接报
    "src is not a numpy array"。而 LazyFrame 只在**真用像素**时才转, 于是凡是
    "要把像素交给 cv2"的地方都必须显式物化。

    代价可控: LazyFrame 内部缓存结果, 同一帧调多少次都只 image2cv 一次;
    而真机走 C++ find_blobs 快路径时这些地方根本执行不到（一次都不转）。

    判定用"有没有 numpy() 方法"的鸭子类型, 因此对真 numpy / None / 其它类型
    一律保持原行为, 不引入任何新语义。
    """
    if isinstance(frame, np.ndarray):
        return frame
    getter = getattr(frame, "numpy", None)
    if callable(getter):
        return getter()
    return frame


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
        self._scan_path: str = ""     # 当前 blob 扫描引擎(maix C++ / cv2), 变化时打一次日志

        self._mode: Optional[str] = None
        self._codes: Tuple[int, int, int] = (0, 0, 0)

        # 阈值文件运行时覆盖（mtime 变化自动重载）
        self._thresholds_file = str(self._cfg.get("thresholds_file") or "")
        self._thresholds_mtime: Optional[float] = None
        self._thresholds_poll_ts: float = 0.0
        self._explosive_lab: dict = {}
        self._target_lab: dict = {}
        self._reload_thresholds(force=True)
        # 人质调参页 SAVE 的参数文件（conf_th/iou_th/x_crop_px）,
        # mtime 热重载语义同阈值文件（轮询共用 _maybe_reload_thresholds 的 2Hz 闸门）。
        self._rescue_params_file = str(self._cfg.get("rescue_params_file") or "")
        self._rescue_params_mtime: Optional[float] = None
        self._reload_rescue_params(force=True)

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
        self._log_lab_slots()

    def current_task(self) -> Tuple[Optional[str], Tuple[int, int, int]]:
        """当前订阅模式与任务码快照（coordinator 命令日志判重描述用）。"""
        return self._mode, self._codes

    def reset(self) -> None:
        """CMD_STOP / 主机静默: 清锁存, 保留模式（下次 set_task 决定）。"""
        self._reset_latch()

    _IMMUTABLE_RUNTIME_KEYS = frozenset({"thresholds_file", "rescue_params_file"})

    def set_runtime_param(self, section: str, key: str, value) -> None:
        """调试页运行时改参（立即生效于 process(), 不落盘不清锁存）。

        仅接受 _DEFAULT_CONFIG 已定义的参数: 段参数 (section, key) 或
        顶层标量 (key == section 名), 否则 ValueError。thresholds_file /
        rescue_params_file 属硬性不可改键（重载器在 __init__ 已缓存路径）。
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
        self._latch_inferred = False   # 本次锁存是否用了"未见形状补位"（现场核对用）
        # 人质检出日志的去重基准（label -> (score, cx, cy)），见 _log_rescue_dets。
        # 换任务/STOP 后清一次: 重订阅后首个检出帧重新打一行, 便于对时间轴。
        self._rescue_last_log: dict = {}
        # 两帧一致去抖（task1 / task3 各一套）
        self._pending_order: list = []
        self._pending_count = 0
        self._pending_order2: list = []
        self._pending_count2 = 0
        # task2 靶标的时序状态。必须在这里清: 换任务/换码/STOP 后重订阅, 首帧不该
        # 继承上一段的 continuity 历史 —— 否则一订阅就报一个来自旧场景的 err。
        # _target_last_err 用 None 表"尚无历史"而非 0: err_fixed=0 是合法值(靶心正
        # 中), 拿0 当哨兵会让死区在首帧之后一直失效。
        self._target_last_cx: Optional[float] = None   # 上一帧选中候选的 cx
        self._target_last_err: Optional[int] = None    # 上一帧**实际上报**的 err_fixed
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

    # 阈值文件 mtime 轮询周期（秒）。见 _maybe_reload_thresholds。
    _THRESHOLD_POLL_INTERVAL_S = 0.5

    def _maybe_reload_thresholds(self) -> None:
        """每帧检查阈值文件 mtime —— 但**限频到 2Hz**。

        台架真机实测（2026-10-08, 未订阅/画面无目标）: `[PERF] vision ... task=1.1ms`。
        那个状态下 `process()` 只跑 `_maybe_reload_thresholds()` + 拼结果 dict，
        识别分支一个都没进（`_mode` 为 None）—— 也就是说 **1.1ms 基本全是这次
        `os.stat()`**，一次每帧都发生的文件系统调用。

        阈值文件是人手在调参页上写出来的，2Hz 轮询远远够用（人手不可能更快），
        换来每帧省 ~1ms：在 40fps 上约 4% 的整帧预算，而在 20ms 的短帧预算里就是 5%+。
        ⚠ 不要为了"省一次 stat"改成只在订阅时检查一次：调参页是**订阅中**用的，
        改语义会让热重载失效。

        2026-10-10: 人质参数文件（rescue_params_file）共用本闸门 —— SAVE 产物
        同样是人手频率的落盘, 2Hz 足够; 但**两个文件各一次 stat**, 任一存在才进入。
        """
        if not self._thresholds_file and not self._rescue_params_file:
            return
        now = self._now()
        if now - self._thresholds_poll_ts < self._THRESHOLD_POLL_INTERVAL_S:
            return
        self._thresholds_poll_ts = now
        if self._thresholds_file:
            try:
                mtime = os.stat(self._thresholds_file).st_mtime
            except OSError:
                mtime = None
            if mtime is not None and mtime != self._thresholds_mtime:
                self._reload_thresholds()
        self._reload_rescue_params()

    def _reload_rescue_params(self, force: bool = False) -> None:
        """人质参数文件（调参页 SAVE 产物）加载/热重载。

        文件缺失/不可读 = 保持当前 cfg 不动（PC 端无 /my-threshold 安全 no-op）;
        文件存在但解析后为空（如全是注释）同样不覆盖。mtime 只在 stat 成功时
        缓存 —— 首次 SAVE 创建文件后必然触发一次重载。只 merge 文件里出现的
        键, 未写的键保持 yaml 值; **不清锁存**（conf 改动只影响后续检出）。
        """
        if not self._rescue_params_file:
            return
        try:
            mtime = os.stat(self._rescue_params_file).st_mtime
        except OSError:
            return
        if not force and mtime == self._rescue_params_mtime:
            return
        self._rescue_params_mtime = mtime
        params = load_rescue_params_file(self._rescue_params_file)
        if params:
            self._cfg["rescue"].update(params)
            log_print(f"[GKS] rescue params "
                      f"{'loaded' if force else 'reloaded'} from "
                      f"{self._rescue_params_file}: {params}")

    def _lab_for(self, mode: str, color_code: int) -> Optional[list]:
        key = COLOR_KEYS.get(color_code)
        if key is None:
            return None
        table = self._explosive_lab if mode == MODE_EXPLOSIVE else self._target_lab
        lab = table.get(key[1])
        return list(lab) if lab else None

    def _log_lab_slots(self) -> None:
        """订阅切换时打一条"色号 -> 实际生效的 LAB 盒"日志（每次 set_task 一条）。

        为什么需要: 阈值文件只按**行序**绑定颜色（`load_threshold_file` 忽略 name
        前缀: row0→红/row1→绿/row2→蓝, 靶标 3~5）。文件若被别的工具按其它顺序写出,
        "C1 槽跑着蓝色的阈值"这种错位是静默的 —— 屏幕上看蓝色被框成 C1, 与
        R/B 通道序错误症状极像。有这行日志, 一眼就能分辨是文件行序问题还是通道序问题。
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
        """find_blobs 一次单色扫描, 返回 [{x,y,w,h,cx,cy,pixels}]（像素面积降序不保证）。

        设备优先 maix C++ find_blobs; 引擎**变化时**打一次日志（台架据此确认到底是
        快路径还是 cv2 回退 —— 两者的帧率差好几倍, 别靠猜）。

        `margin` 默认 None = 不传给 maix find_blobs, 与去年基准 main.py task1/task2
        的裸 `merge=True` 调用一致（见 maix_accel.find_blobs_maix 的注释）。
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
        """`merge_margin` -> find_blobs 的 margin kwarg 值。

        0 / None / 缺省 / 非法 -> **None = 不传该 kwarg**（与去年基准一致）。
        去年 main.py 的 task1/task2 都是裸 `merge=True`, 没有 margin ——
        margin 在 OpenMV 系 find_blobs 里会触发形态学膨胀 + 连通域重标记,
        比 merge 本身重得多, CAM1 上尤其明显（见 _DEFAULT_CONFIG["target"] 注释）。
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

    # ---- 坑位定序（task1 爆炸物 / task3 人质 共用一套, 与 2025 视觉方案一致）----
    #
    # 2025 方案 task1/task3 的坑位映射是同一个套路：
    #   task3: `found_shapes` 累积, 第 1 个 -> position2[0], 第 2 个 -> position2[1],
    #          第 3 个直接算出来 position2[2] = 6 - sum(found_shapes)
    #   task1: `position_count` 累积, 同理 position[2] = 6 - position[0] - position[1]
    # 即：**顺序 = 首次检出先后**（相机按 pit1→pit3 扫过, 最先看见的那个就在 pit1）,
    # 且**三色/三形状互不相同 -> 见到 2 个即可推出第 3 个"是什么"**, 补在顺序尾部。
    #
    # 补位只解决"是什么"、不解决"在第几坑"——位置由顺序来源给出。某帧同时检出多个时,
    # 帧内先按画面左右定序（`pit_left_is_1` 决定方向）, 保证结果不受 find_blobs/YOLO
    # 的输出顺序影响（输出序不是空间序）。
    # 锁存仍走"连续 pit_latch_frames 帧同序"去抖, 与 main.py 同语义。

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
        """section.order_by == x 时用"全程画面 x 排序"(旧 main.py 语义)。"""
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
        """按 scale 降采样, 返回 (frame_bgr_small, maix_small)。

        INTER_NEAREST: 不插值 —— 双线性会把目标边缘像素与背景混成中间色, 那正是
        运动模糊让色块掉出阈值的同一机理。maix 缩放失败时返回 None(自动走 cv2 回退)。

        ⚠ 通道序（2026-09-26 台架实测踩坑, 见 AGENTS.md「已知边界」）: 交给 C++
        find_blobs 的小图必须是 **RGB 字节序 + FMT_RGB888 标记**。MaixCDK 的
        convert_to_imlib_image()（components/vision/src/maix_image_ops.cpp）把
        FMT_BGR888 直接 fall-through 成 PIXFORMAT_RGB888、**不做通道交换**，
        于是 BGR 数据被当 RGB 算 LAB → 红蓝互换: 蓝色目标命中"红色"阈值盒，
        HUD 框写 C1、坑位定序连带全错。故这里 resize 出 BGR 后转 RGB 再以
        bgr=False 包图（cv2 回退路径继续用 BGR 小图, 与 COLOR_BGR2LAB 匹配）。

        排爆(explosive)与打靶(target)共用本降采样器 —— 两者误差/坑位语义
        都与分辨率无关（面积门槛 ×scale²、坐标 ÷scale 还原）, 故共用不引入分叉。
        """
        if scale >= 1.0:
            return frame, maix_img
        h, w = frame.shape[:2]
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = cv2.resize(_as_numpy(frame), (sw, sh),
                           interpolation=cv2.INTER_NEAREST)
        # 小图重新包成 maix Image: 走 C++ find_blobs 快路径且像素与上面 NEAREST 结果
        # 逐字节一致（不用 img.resize(): 固件默认插值会把边缘混成中间色）。
        #
        # import 探测结果模块级缓存（2026-10-07）: 这里的 `import maix.image` 是
        # **函数内**import, PC 端(无 maix) 每次调用都要走一遍 sys.modules 查找 +
        # 抛 ImportError, 实测单帧 0.92ms —— 而 _downscale 的有效工作只有 0.07ms,
        # 即 93% 的时间花在重复失败的 import 上（设备端 import 命中缓存, 开销小,
        # 但缓存对两端都只有好处）。
        #
        # ⚠ 只缓存"是否可用"(_maix_image_ok), **每次仍重新 import 取模块对象**:
        #   sys.modules 命中是纳秒级的字典查找, 而缓存模块对象会与测试的
        #   monkeypatch.setitem(sys.modules, "maix.image", fake) 冲突 ——
        #   先前缓存成 False 的实例会让后续注入的假 maix 永远拿不到。
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
            # 阈值缺失不进 find_blobs: 零值心跳不断流, 锁存状态保留（main.py 同语义）
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
            # 无阈值不断流: 零值心跳（main.py 同语义）
            return {"err_fixed": 0, "found": False}, hits, False
        cfg = self._cfg["target"]
        h, w = frame.shape[:2]
        roi_px = self._roi_px(cfg["roi"], w, h)
        scale = self._scan_scale(cfg)

        # ---- 降采样两段式（2026-10-07）----
        #
        # 只降采样会把 cx 的精度也降下来：cx 是**像素级**量, 小图上 ±1 小图像素
        # 还原到全帧就是 ±1/scale 个像素 —— scale=0.5 时误差能偏 30+ 个 LSB
        # (实测 7031 vs 7062)。打靶是云台闭环, 误差就是控制量, 这点偏差不能省。
        #
        # 故拆成两段: ①**小图只负责"有没有、在哪"**（面积门槛按 scale² 折算）,
        # ROI 面积直接降到 1/scale², 这是省时间的大头; ②拿小图给出的粗位置,
        # 在**全分辨率**上开一个贴着它的窄窗再扫一次, 门槛用原值, 由这一次
        # 的结果算 cx 与 err_fixed —— 与全分辨率逐位一致, 精度不打折。
        # ②的窗口很小(靶标 bbox 外扩 pad), 扫的像素量远小于①, 净收益仍为正。
        # ---- 跨帧跟踪窄窗（2026-10-09, 默认关闭）----
        #
        # 打靶是**云台闭环**: 目标在帧间只移动少量像素。于是可以用上一帧的 cx 开一个
        # 窄窗, 命中就直接用; 未命中**当帧回退全 ROI 复扫**。这个"当帧回退"是关键:
        # 它保证精度与检出率与"全程全 ROI"**完全一致**（同一套阈值／几何／选块代码）,
        # 跟踪只在命中时省时间。而 find_blobs 的成本≈正比于搜索面积, 真机带目标时
        # `[PERF] task` 达 21ms, 是全帧预算里最大的一块。
        # 实测依据（scripts/bench_frame_budget.py @640x480, ROI 高度扫描）:
        #   640x197 = 8.96ms -> 640x120 = 5.10ms -> 640x40 = 1.53ms（近似线性）。
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
                # 窄窗里没复现（目标移出/阈值边缘抖动）: 用粗结果兜底, 保持不断流
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
            # 归一化横向误差 (cx - w/2)/(w/2), clamp ±1, 定点 ×10000
            # (640 宽帧下与旧版 (cx-320)/320 逐位一致)
            half = w / 2.0
            normalized = (cx - half) / half
            normalized = max(-1.0, min(1.0, normalized))
            err_fixed = int(normalized * 10000)   # int(): 与 main.py 同为截断取整
            # 顺序要紧: 先记last_cx/桶, 再判确认 —— _target_last_err 必须在**通过
            # 确认之后**才写(见 _target_confirm_ok), 否则被丢弃的帧也会污染死区基准。
            err_fixed = self._apply_target_deadband(err_fixed, cfg)
            self._target_last_cx = cx
            self._target_last_box = {"x": int(largest["x"]), "y": int(largest["y"]),
                                     "w": int(largest["w"]), "h": int(largest["h"]),
                                     "cx": cx, "cy": float(largest["cy"])}
            if not self._target_confirm_ok(err_fixed, cfg):
                return {"err_fixed": 0, "found": False}, hits, True
            return {"err_fixed": err_fixed, "found": True}, hits, True
        return {"err_fixed": 0, "found": False}, hits, True

    # ---- task2 候选选择与时序（防误检, 2026-10-08）----
    def _target_tracked_scan(self, frame, maix_img, lab, roi_px, cfg, scale):
        """跨帧跟踪窄窗扫描(track_mode: fixed/dynamic)。返回 blob 列表 = 命中;
        **None = 调用方应走全 ROI 路径**。

        返回 None 的两种情况（调用方无需区分, 都会落到全 ROI 那条路）:
          · 不适用: 开关关闭 / scale<1（不与两段式混用）/ 无上一帧历史 / 窗口退化;
          · 窄窗未命中: 此时**同一帧内**回退全 ROI 复扫 —— 所以不会漏检,
            代价只是这一帧多花一次窄窗的钱。miss 计数与日志用于现场判断 pad 是否偏小。

        ⚠ 只服务 TARGET。排爆要全帧三色定序（首次检出先后）, 人质走 YOLO 全帧 ——
        两者都**不能**缩小搜索域, 缩了是改语义而不是省时间。

        ⚠ 窗口以"上一帧 cx"为中心, 因此天然带连续性先验: 画面别处的红色区域不在窗内
        就不会被选中。而 target.roi 是全宽的、prefer_continuity 默认关闭 —— 对
        "背景大片红赢过真球"那个已知风险, 跟踪窗口反而是加固（不是削弱）。
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
        """上一帧 cx ± pad 的窄窗, 高度撑满 ROI, 并夹回 ROI。退化返回 None。

        外扩 pad 覆盖"目标一帧内的最大位移"。取小了会频繁未命中（净收益变负:
        每次都白花一次窄窗 + 一次全 ROI）; 取大了省不下时间。现场按
        「窄窗未命中」日志的出现频率调。
        """
        rx, ry, rw, rh = roi_px
        x0 = max(rx, int(last_cx) - pad)
        x1 = min(rx + rw, int(last_cx) + pad)
        if x1 - x0 <= 2:
            return None
        return [x0, ry, x1 - x0, rh]

    @staticmethod
    def _target_dynamic_window(box, roi_px, pad, cfg):
        """dynamic 跟踪窗: 上帧 bbox 缩放, 横宽纵窄, 夹回 ROI; 退化返回 None。

        宽 = min(2*pad, max(min_w, scale*bw)); 高 = min(rh, bh + 2*v_pad)。
        双向封顶保证任何一帧的窗口不大于 fixed 模式的 2*pad × rh。
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
        """窄窗未命中的按批汇总日志（不逐帧刷屏）。"""
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

        ⚠ **为什么不能是"谁 pixels 最大"**: ROI 是全宽的(roi x0=0.0,x1=1.0),
        场景里任何成片红色区域(地面标记、标语、另一组的球)像素都可能赢过真球,
        而 pixels 只衡量"有多少红像素", 完全不衡量"这是不是靶标"。

        ⚠ **形状判据(roundness/density/solidity)已被删除**: 靶标是**圆环**,
        实心球那套阈值会把真环全部误杀, 且 cv2 回退路径的 RETR_EXTERNAL 会把内孔
        填成实心圆导致两端口径相反。详见 eod_config.yaml target 段的说明。

        排序稳定性: max() 在 pixels 相同时依赖 find_blobs 的输出顺序(非稳定序),
        这里统一用 **(-判据, cx)** 显式比较, 保证同一候选集的挑选结果可复现 ——
        否则现场连"跳没跳"都无法归因。
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
        """死区: |新err - 上次err| < deadband 时保持上次值。

        为什么用死区而不是滑动平均: 打靶总时延已 ~45ms(取图30+ + find_blobs~25 +
        主循环2 + 线1.4), 已经**大于** 28Hz 控制周期 35ms, 再串一个滑动平均会
        继续加相位滞后, 直接恶化 PID 相位裕度。死区在桶内**保持值不动**,
        不引入额外滞后 —— 代价是桶边界仍有一次跳变, 但那比持续抖动轻得多。

        仅在 err_deadband > 0 时生效(默认 0 = 关闭, 与改动前逐字节一致)。
        """
        deadband = int(cfg.get("err_deadband") or 0)
        if deadband <= 0 or self._target_last_err is None:
            return err_fixed
        if abs(err_fixed - self._target_last_err) < deadband:
            return self._target_last_err
        return err_fixed

    def _target_confirm_ok(self, err_fixed: int, cfg: dict) -> bool:
        """连续确认: err 连续 N 帧落在同一死区桶里才认, 否则 False(报 found=0)。

        桶 = err // deadband(死区关闭时桶退化为 err 本身, 即要求逐帧完全相同 ——
        那太苛刻, 所以 confirm_frames > 1 时**必须**同时给 err_deadband > 0)。

        ⚠ **基准写入必须在两条 early-return 之外**(2026-10-08 修正): 此前
        `_target_last_err = err_fixed` 只写在函数末尾, 于是 confirm_frames=1(默认)
        时函数在第一行就return True —— 基准**永远不更新**, 死区恒等形同关闭,
        配了 err_deadband 也毫无效果。现在两条 return 之前都写。
        被丢弃的帧不写基准(否则污染死区), 故确认通过后才更新。

        未确认时按秒汇总打一条, 不逐帧刷屏; 现场判读看: 连续丢弃计数涨 =
        候选在两个远距簇之间摆(选择器问题), 还是单点偶发(= 单帧误检)。
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
        """由小图粗框换算全分辨率精修窗口: 粗框还原到全帧 + pad 外扩, 再夹回 ROI/画面。

        pad 取小了会在目标快速移动时精修窗追不上; 取大了就失去降采样的意义。
        24px @640 宽 ≈ 画面 3.75%, 覆盖 target 段一帧内的最大位移 (云台闭环
        的残差速率下远小于此)。外扩后仍与原 ROI 求交 —— 精修**不得**越出 ROI,
        否则 ROI 的排除语义（test_target_roi_excludes_outside）在降采样档失效。

        另加 1px 余量: 粗框是 cxf()/2 取整还原的, 边界可能差 1, 不留会切掉目标边缘。
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
        """人质检出日志: 形状集合变化或任一分数跳变 >0.1 才打一行汇总。

        log_print 每行一次 open/write/flush 落盘, 不能逐帧打 —— 30fps 下每秒
        30 次开文件且淹没 latch/推流日志。标签集合稳定且分数小幅波动时静默,
        典型频率每秒 0~3 条。同标签多检出只取分最高者参与判定/打印。
        格式: [GKS] rescue det: yuan_zhu=0.82(214,165) yao_gu=0.76(427,220)
        坐标 = 画面中心 (cx, cy), 与 hits/shape_seen 同口径。
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
            # 模型缺失优雅降级: 零值心跳不断流, 不崩溃
            return ({"pos": 0, "found": False, "inferred": self._latch_inferred},
                    hits, list(self._position2), self._pending_count2)

        # ---- 零拷贝推理（2026-10-09）----
        # maix Image 可用时走 `_raw=True`: NPU 直接吃原始图, **省掉两次拷贝**
        # —— ① 相机的 image2cv(BGR numpy) ② detect() 内的 cv2image(numpy→maix)。
        # 这是"干掉白做的 image2cv"在人质任务上的同款收益。
        # ⚠ 两条平台的 `ai.detect` 都认 `_raw`
        #   (maixcam1/ai.py:235、maixcam2/ai.py:179, 均 pop 掉该 kwarg)。
        # maix_img 缺失(PC/无 maix)时退回 numpy 路径, 行为与改前逐位一致。
        if maix_img is not None:
            objs = ai.detect(maix_img, conf_th=self._cfg["rescue"]["conf_th"],
                             iou_th=self._cfg["rescue"]["iou_th"], _raw=True)
        else:
            objs = ai.detect(frame, conf_th=self._cfg["rescue"]["conf_th"],
                             iou_th=self._cfg["rescue"]["iou_th"])
        # 同一帧内可能同时检出多个形状: 此时"时间先后"无意义, 先按画面左右关系定序
        # (镜像时反向), 保证入表顺序**不受模型输出顺序影响**(YOLO 的输出序不是空间序)。
        # 跨帧仍保持"首次检出先后"—— 比赛流程固定, 相机按 pit1→pit3 扫过。
        objs = sorted(objs, key=lambda o: o.x, reverse=not self._cfg["pit_left_is_1"])
        labels = getattr(ai, "labels", None) or []
        label_map = self._cfg["rescue"]["labels"]
        x_crop = int(self._cfg["rescue"].get("x_crop_px") or 0)
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
            self._shape_seen[shape_num] = obj.x + obj.w / 2.0   # 形状中心 x, 跨帧刷新
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
                    # 只反映"真正锁存那一帧"是否用了补位(现场核对: True=锁存时有形状没看到)
                    self._latch_inferred = inferred
                    log_print(f"[GKS] latch pos2={self._position2} "
                              f"seen={list(self._shape_seen)} "
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
