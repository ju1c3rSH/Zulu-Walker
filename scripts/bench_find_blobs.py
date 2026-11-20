# -*- coding: utf-8 -*-
"""
CAM1 find_blobs 台架测量 + 打靶误识别诊断 (2026-10-08)

两件事一起做:
  1. 提速验证: 排爆"三色串行 3 次全帧扫" vs "多阈值 1 次 + stride" 的实测耗时差。
     收益 1 (多阈值一次调用) 与收益 2 (x_stride/y_stride) 的兑现幅度只能实测 ——
     CAM1 算力比 CAM2 弱一个量级, CAM2 的经验不能沿用 (同 scan_scale 的教训)。
  2. 误识别诊断: 在target ROI 内把每个候选 blob 的 pixels/cx/roundness/
     density 全部打出来, 直接看是真环一个还是"背景大片红 + merge 合并框"。
     靶标是圆环 —— roundness 对环 = (1-k)/(1+k) (k=内径比), k=0.5 时只有
        0.333, 比方块的 0.785 还低; 别拿实心球的直觉读这两个数。

用法 (设备上):
    python scripts/bench_find_blobs.py            # 默认 40 帧/变体
    python scripts/bench_find_blobs.py --frames 80
    python scripts/bench_find_blobs.py --dump     # 额外逐帧 dump target 候选明细

只依赖 maix + 标准库, 不 import 项目代码 —— 设备上跑不牵扯模块路径/配置解析,
阈值/LAB 直接写死在这里, 避免"配置文件读的是相机端还没同步的旧值"这种干扰。

阈值来源: 取自仓库 eod_config.yaml 的备用初值, 不是设备
/my-threshold/my_64threshold.txt (那份在 CAM1 上是另一套, 见 bench 输出对照)。
本脚本的目的不是标定, 是相对比较各变体耗时 —— 用哪套阈值对相对值影响很小。
"""
import argparse
import time

from maix import camera, image

# ---- 与 eod_config.yaml / gks_task_processor._DEFAULT_CONFIG 一致的保底 LAB ----
LAB_RED = [0, 80, 40, 80, 10, 80]
LAB_GREEN = [25, 85, -80, -25, 30, 85]
LAB_BLUE = [0, 60, -8, 45, -75, -25]
LABS3 = [LAB_RED, LAB_GREEN, LAB_BLUE]

# 排爆面积门槛(eod_config.yaml explosive.area_threshold), 全帧 scan_scale=1.0
EXPLOSIVE_AREA_TH = 4000
EXPLOSIVE_PIXELS_TH = 4000
# 打靶门槛(eod_config.yaml target.*)
TARGET_AREA_TH = 1000
TARGET_PIXELS_TH = 1000
# target.roi = [0.0, 0.352, 1.0, 0.763] -> 640x480 像素
TARGET_ROI = [0, 169, 640, 197]

W, H = 640, 480


# ---------------------------------------------------------------- helpers ----
def _safe(fn, default=-1):
    """blob 的可选属性在老固件上可能不存在, 缺了不能崩 (只影响诊断打印)。"""
    try:
        return fn()
    except Exception:
        return default


def _blob_code(b):
    """blob 命中的是第几个阈值 —— 多阈值调用的关键。

    老固件没有 code() 时返回 -1, 此时多阈值变体只能看总耗时, 无法核对分色结果。
    """
    return _safe(lambda: b.code(), -1)


def _dump(b):
    return (f"code={_blob_code(b):2d} cx={_safe(lambda: b.cx(), -1):4d} "
            f"cy={_safe(lambda: b.cy(), -1):4d} w={_safe(lambda: b.w(), -1):4d} "
            f"h={_safe(lambda: b.h(), -1):4d} pix={_safe(lambda: b.pixels(), -1):6d} "
            f"round={_safe(lambda: round(b.roundness(), 3), -1):6} "
            f"dens={_safe(lambda: round(b.density(), 3), -1):6}")


def _time_variant(name, fn, frames, warmup=8):
    """跑 fn(img) frames 次, 返回 (ms/帧, 每次的命中数列表)。"""
    for _ in range(warmup):
        fn(img)
    hits = []
    t0 = time.time()
    for _ in range(frames):
        n = fn(img)
        hits.append(n)
    dt = (time.time() - t0) * 1000.0 / frames
    hit_str = f"hit={sum(hits)}/{frames}"
    print(f"  {name:<46s} {dt:7.1f} ms/帧   {hit_str}")
    return dt, hits


# ------------------------------------------------------------- variants ----
def v_baseline_3x(img):
    """【当前实现】三色串行, 每次全帧, merge=False。"""
    total = 0
    for lab in LABS3:
        blobs = img.find_blobs([lab], invert=False,
                               area_threshold=EXPLOSIVE_AREA_TH,
                               pixels_threshold=EXPLOSIVE_PIXELS_TH)
        if blobs:
            total += 1
    return total


def v_multi_1x(img, x_stride=2, y_stride=1, merge=False):
    """【收益1+2】一次调用传三阈值, 用 code() 分色; stride 由参数给。"""
    kw = dict(invert=False, x_stride=x_stride, y_stride=y_stride,
              area_threshold=EXPLOSIVE_AREA_TH,
              pixels_threshold=EXPLOSIVE_PIXELS_TH)
    if merge:
        kw["merge"] = True
    blobs = img.find_blobs(list(LABS3), **kw)
    return len(blobs or [])


def v_target_current(img):
    """【当前实现】target: merge=True, 无 stride (默认 x_stride=2)。"""
    blobs = img.find_blobs([LAB_RED], roi=list(TARGET_ROI),
                           area_threshold=TARGET_AREA_TH,
                           pixels_threshold=TARGET_PIXELS_TH,
                           merge=True)
    return len(blobs or [])


def v_target_nomerge(img):
    """【收益3】target: merge=False —— 提速 + 让候选不被合并成一个横跨框。"""
    blobs = img.find_blobs([LAB_RED], roi=list(TARGET_ROI),
                           area_threshold=TARGET_AREA_TH,
                           pixels_threshold=TARGET_PIXELS_TH)
    return len(blobs or [])


def v_target_stride(img, x_stride, y_stride, merge=False):
    kw = dict(invert=False, x_stride=x_stride, y_stride=y_stride,
              roi=list(TARGET_ROI),
              area_threshold=TARGET_AREA_TH, pixels_threshold=TARGET_PIXELS_TH)
    if merge:
        kw["merge"] = True
    blobs = img.find_blobs([LAB_RED], **kw)
    return len(blobs or [])


# ------------------------------------------------------------------ main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--dump", action="store_true",
                    help="逐帧 dump target ROI 内每个候选 blob 的明细")
    args = ap.parse_args()

    print("=" * 74)
    print("CAM1 find_blobs 台架测量 (CV181x)")
    print("=" * 74)

    cam = camera.Camera(W, H, image.Format.FMT_RGB888)
    # buff_num=1 降低取图延时(CAM1 至少双缓存, 只能到 ~30ms); 让 bench 更贴近
    # pipeline 实测的 ~28fps 而不是被缓冲掩盖。
    try:
        cam = camera.Camera(W, H, image.Format.FMT_RGB888, buff_num=1)
    except Exception:
        pass
    img = cam.read()
    print(f"[bench] 取图 {img.width()}x{img.height()}  fmt={img.format()}")

    # 固件能力探测 —— stride/多阈值不支持时要有话说, 而不是静默跑出假数据
    print("\n[bench] 固件能力探测")
    try:
        b = img.find_blobs(list(LABS3), invert=False, x_stride=4, y_stride=2,
                           area_threshold=10, pixels_threshold=10)
        supports_stride = True
        supports_multi = True
    except TypeError as e:
        supports_stride = False
        supports_multi = "thresholds" in str(e) is False
        print(f"[bench] 多阈值/x_stride 不被接受: {e}")
        print("[bench]   -> 固件较老, 收益 1/2 需先升级 MaixPy 才能吃到")
    else:
        print(f"[bench] 多阈值 OK, code() = {_blob_code(b[0]) if b else 'N/A'} "
              f"(首个 blob 命中的阈值序号)")
    if not hasattr(img, "find_blobs"):
        raise SystemExit("[bench] 致命: maix.image.Image 没有 find_blobs")

    # ---------------- 排爆: 提速 ----------------
    print("\n" + "-" * 74)
    print("[1] 排爆耗时 (全帧, merge=False, area/pixels_th=4000)")
    print("-" * 74)
    base, _ = _time_variant("基线: 3 次串行单色 (当前实现)", v_baseline_3x, args.frames)

    results = {}
    if supports_stride:
        for xs, ys in ((2, 1), (4, 1), (4, 2), (6, 3)):
            dt, _ = _time_variant(
                f"多阈值 1 次调用  x_stride={xs} y_stride={ys}",
                lambda im, a=xs, b=ys: v_multi_1x(im, a, b), args.frames)
            results[(xs, ys)] = dt

    print("\n  提速比 (相对基线):")
    if supports_stride:
        for (xs, ys), dt in results.items():
            print(f"    x_stride={xs} y_stride={ys}: {base / dt:.2f}x  "
                  f"({base:.1f} -> {dt:.1f} ms)")

    # ---------------- 打靶: 误识别诊断 + 提速 ----------------
    print("\n" + "-" * 74)
    print("[2] 打靶耗时 (ROI=640x197, area/pixels_th=1000)")
    print("-" * 74)
    t_cur, _ = _time_variant("基线: merge=True (当前实现)", v_target_current, args.frames)
    t_nm, _ = _time_variant("候选: merge=False", v_target_nomerge, args.frames)
    print(f"    merge=False 提速 {t_cur / t_nm:.2f}x")
    if supports_stride:
        for xs, ys in ((4, 1), (4, 2)):
            t, _ = _time_variant(
                f"候选: merge=False + x_stride={xs} y_stride={ys}",
                lambda im, a=xs, b=ys: v_target_stride(im, a, b), args.frames)
            print(f"    -> 提速 {t_cur / t:.2f}x")

    # ---------------- 打靶误识别诊断 ----------------
    print("\n" + "-" * 74)
    print("[3] 打靶误识别诊断: target ROI 内候选 blob 明细")
    print("-" * 74)
    print("怎么读这一段:")
    print("  · 每次 find_blobs 只出 1 个候选, 但它 cx 在两个位置间跳")
    print("    -> merge 把邻近色块并成一个横跨框 (当前 merge=True) => err 大跳的元凶")
    print("  · 出现多个候选, 且背景那片 pixels 远大于真环")
    print("    -> max(pixels) 选错了 => 开prefer_continuity(治候选间摇摆),")
    print("       或收窄 LAB/加像素门槛; 形状阈值对圆环不适用(见文件头)")
    print("  · 候选 pixels 本身就在阈值边缘(几百~一千几)")
    print("    -> LAB 阈值过窄, 运动模糊下断块 => 先放宽 L 区间")
    print()

    for label, kw in (("merge=True  (当前)", dict(merge=True)),
                      ("merge=False", dict())):
        print(f"  --- {label} ---")
        seen = {}
        for i in range(max(10, args.frames // 2)):
            blobs = img.find_blobs([LAB_RED], roi=list(TARGET_ROI),
                                   area_threshold=TARGET_AREA_TH,
                                   pixels_threshold=TARGET_PIXELS_TH, **kw)
            if not blobs:
                continue
            # 复刻 gks_task_processor 的选取标准: 纯按 pixels 最大
            best = max(blobs, key=lambda b: _safe(lambda: b.pixels(), 0))
            key = _safe(lambda: best.cx(), -1)
            seen.setdefault(key // 20 * 20, 0)     # 按 20px 分桶看 cx 分布
            seen[key // 20 * 20] += 1
            if i < 6 or args.dump:
                others = [_dump(b) for b in blobs if b is not best]
                print(f"    [{i:3d}] 选中 {_dump(best)}")
                for o in others:
                    print(f"          落选 {o}")
        if seen:
            buckets = ", ".join(f"cx~{k}×{n}" for k, n in
                                sorted(seen.items()))
            print(f"    >>> 选中块 cx 分布: {buckets}")
            if len(seen) > 1:
                print("    >>> cx 在多个远距簇间跳 -> 确认误识别, "
                      "先改 merge=False 再看是否消失")
        else:
            print("    >>> 无候选 (目标不在 ROI/阈值不匹配)")

    # ---------------- 结论 ----------------
    print("\n" + "=" * 74)
    print("结论怎么用")
    print("=" * 74)
    print(f"  排爆基线 {base:.1f} ms/帧"
          f" -> 改多阈值调用后见上表; 能吃到就是最大一块零精度收益。")
    print(f"  打靶 merge=True {t_cur:.1f} -> merge=False {t_nm:.1f} ms "
          f"({t_cur / t_nm:.2f}x), 且 merge=False 同时修 err 大跳。")
    print("  注意: 这些是**扫描**耗时, 不含取图。CAM1 取图 ≥30ms 且不可压,")
    print("        所以 pipeline 帧率上限 ~30Hz; 排爆段才是唯一有优化空间的地方。")
    print("  stride 是否安全: 排爆只看色块有没有 + 左右定序, cx 精度不敏感;")
    print("        **打靶不能加 stride** (err 是云台控制量, cx 精度即控制精度)。")


if __name__ == "__main__":
    main()