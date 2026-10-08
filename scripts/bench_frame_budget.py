# -*- coding: utf-8 -*-
"""
CAM1 打靶 err 反馈速度 — 帧周期分项测量 (2026-10-08)

回答一个问题: `[FPS] pipeline 20.3` + `[UART TX] DATA_STREAM 20.1Hz` 说明推流链路
已经不是瓶颈 (结果驱动生效了), 但 50ms 的单帧周期**到底花在哪**, 现有日志只有一个
总数, 分不出下面两种情形:

  假设 A — read 实际是阻塞的:
      周期 = 33ms(等新帧) + 17ms(处理) ≈ 50ms, 相机与处理**串行**相加。
      解法: 采集与处理解耦 (独立采集线程 + 只取最新帧)。
  假设 B — read 非阻塞, 相机在 80fps 产帧:
      周期 = 50ms ≈ 全是 find_blobs 的时间。
      解法: 降 find_blobs 成本 (收窄 ROI / 两段式降采样 / 收紧 LAB 降噪点)。

两个假设的解法完全不同, 而且**改错方向会白花一天台架时间**, 所以先测再改。
本脚本同时给出"串行 vs 解耦"的实测对比, 直接回答"加采集线程到底有没有用"。

用法 (设备上, SSH):
    python scripts/bench_frame_budget.py                 # 每项默认 3 秒
    python scripts/bench_frame_budget.py --seconds 5

只依赖 maix + 标准库, **不 import 项目代码** —— 避免读到设备上还没同步的旧配置。
阈值/LAB 取自仓库 eod_config.yaml 的兜底初值; 本脚本只做**相对**比较, 不用于标定。

读结果 (2026-10-08 首轮实测基线, MaixCAM-Pro / GC4653 720P):
  [1] 产帧 59.7Hz, 间隔 16.7ms        -> fps=60 生效, 相机不是瓶颈
  [2] image2cv 4.73ms/帧              -> TARGET 走 C++ find_blobs, 这份 numpy 白做
  [3] find_blobs ROI 8.90ms / 全帧 20.25ms
  [4] ROI 197/120/80/40 行 = 8.96/5.10/3.34/1.53ms   -> 成本≈线性于像素数
  [5] 串行 read+find_blobs **59.7Hz** -> 识别链路本身能跑满 60fps!
  [7] display.show() **23.4ms/次**    -> 每 2 帧一次 ≈ 每秒 117ms 的 GIL 独占

  ⇒ 结论: 识别**不是**瓶颈。真机 [FPS] pipeline 只有 20.3Hz, 与识别链路的
    59.7Hz 差 ~35ms/帧 —— 差额不在 read/find_blobs 里, 而在**别的线程抢 GIL**
    (显示线程的 show() 是首位嫌疑, 其次是 debug_console 的 rich 10Hz 重绘)。
    实测方法: 真机跑主程序看 `[PERF] vision ... | 差额 X ms` 那一行。

  ⚠ 本脚本的 [6] 曾给出"解耦 31.7Hz"的反常结果 —— 那是**测量假象**: 采集线程用
    `time.sleep(0.0005)` 忙轮询, 持续抢 GIL 把消费线程饿死。现已改为低自旋写法,
    并在输出里同时给"非阻塞+1ms"与"阻塞读"两种采集线程模型, 便于对照判断
    阻塞读到底会不会释放 GIL。
"""
import argparse
import statistics
import threading
import time

from maix import camera, display, image

W, H = 640, 480
TARGET_ROI = [0, 169, 640, 197]          # eod_config.yaml target.roi @640x480
LAB_RED = [0, 80, 40, 80, 10, 80]        # 打靶默认只扫 codes[1] 一色
TARGET_AREA_TH = 1000
TARGET_PIXELS_TH = 1000

BAR = "=" * 74


# ------------------------------------------------------------------ camera ----
def open_camera():
    """按能力从多到少构造, 并**明确报告命中了哪一级**。

    `[BOOT] cam.open:CRASH ... sensor timeout` + `camera.attempts=3` 说明台架上
    第一次构造就炸了、最后落在位置参数签名 —— 那一级的 fps/buff_num **根本没传
    进相机**, MaixPy 对 <=1280x720 会默认给 80fps。这里把命中级别打出来,
    后面所有帧率数字才解释得通。
    """
    levels = [
        ("L1 full-kwargs (fps=60 buff_num=1)",
         lambda: camera.Camera(width=W, height=H, format=image.Format.FMT_RGB888,
                               fps=60, buff_num=1, open=True)),
        ("L2 no open=    (fps=60 buff_num=1)",
         lambda: camera.Camera(width=W, height=H, format=image.Format.FMT_RGB888,
                               fps=60, buff_num=1)),
        ("L3 positional  (**fps/buff_num 丢失**)",
         lambda: camera.Camera(W, H, image.Format.FMT_RGB888)),
        ("L4 minimal     (**fps/buff_num 丢失**)",
         lambda: camera.Camera(W, H)),
    ]
    for i, (name, fn) in enumerate(levels, 1):
        t0 = time.time()
        try:
            cam = fn()
        except Exception as e:
            print(f"  {name:<40s} 失败 {type(e).__name__}: {e} "
                  f"({(time.time() - t0) * 1000:.0f}ms)")
            continue
        ms = (time.time() - t0) * 1000
        print(f"  {name:<40s} 命中  ({ms:.0f}ms)")
        if i >= 3:
            print("  ⚠ 命中降级级: project_config 的 fps/buff_num 没生效,"
                  " 实际帧率以 [1] 为准")
        return cam, i
    return None, 0


def probe_block_kwarg(cam):
    """read(block=False) 是否被固件接受。

    官方 examples/vision/display/display_multi_channel.py 用了这个 kwarg, 说明
    新版固件是支持的; 但项目代码里有一条"TypeError -> 静默降级为阻塞读"的分支,
    真命中时停帧保护失效、且每帧要等满一个帧间隔。必须先确认在哪一支里。
    """
    try:
        cam.read(block=False)
        return True
    except TypeError:
        return False
    except Exception:
        return True


# --------------------------------------------------------------- primitives ----
def time_capture_cadence(cam, seconds, block_ok):
    """非阻塞轮询 N 秒: 统计产帧间隔 + 轮询落空率 (告诉我们谁更快)。"""
    gaps, misses = [], 0
    last = None
    t_end = time.time() + seconds
    while time.time() < t_end:
        if block_ok:
            img = cam.read(block=False)
        else:
            img = cam.read()
        if img is None:
            misses += 1
            time.sleep(0.0005)
            continue
        now = time.time()
        if last is not None:
            gaps.append((now - last) * 1000.0)
        last = now
    if not gaps:
        return None
    gaps.sort()
    return {
        "n": len(gaps),
        "p50": gaps[len(gaps) // 2],
        "p95": gaps[int(len(gaps) * 0.95)],
        "misses": misses,
        "rate": len(gaps) / seconds,
    }


def bench(image_callable, n):
    """跑 n 次取中位数毫秒。image_callable() 内做完整个操作。"""
    for _ in range(5):
        image_callable()
    samples = []
    for _ in range(n):
        t0 = time.perf_counter()
        image_callable()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return samples[len(samples) // 2]


def scan(img):
    return img.find_blobs([LAB_RED], invert=False, roi=list(TARGET_ROI),
                          area_threshold=TARGET_AREA_TH,
                          pixels_threshold=TARGET_PIXELS_TH)


# ------------------------------------------------------------- simulations ----
def sim_serial(cam, seconds, block_ok):
    """【当前实现】同一线程: read -> find_blobs, 串行相加。"""
    n, t_end = 0, time.time() + seconds
    while time.time() < t_end:
        img = cam.read(block=False) if block_ok else cam.read()
        if img is None:
            time.sleep(0.001)
            continue
        scan(img)
        n += 1
    dt = seconds
    return n / dt


def sim_decoupled(cam, seconds, block_ok, blocking_read=False):
    """【候选】采集线程只保留最新帧, 处理线程只做 find_blobs。

    两种采集线程模型都给出来, 因为它们的 GIL 行为完全不同:
      blocking_read=False -> `read(block=False)` + 1ms 轮询。低自旋, 但每秒仍有
                            1000 次唤醒抢 GIL。
      blocking_read=True  -> 直接用阻塞 `read()`。零自旋, 但**前提是固件在阻塞
                            等待期间释放 GIL**; 若 pybind 绑定全程持 GIL, 采集
                            线程会把消费线程彻底饿死, 实测 Hz 会明显低于串行。

    ⚠ 不要写成 `while True: read(); time.sleep(0.0005)` 的忙轮询 —— 那测的是
      GIL 争抢, 不是 pipeline 设计（首轮就是这个假象, 见文件头）。
    """
    slot = {"img": None, "seq": 0}
    produced = {"n": 0}
    stop = threading.Event()

    def capture():
        while not stop.is_set():
            try:
                if blocking_read:
                    img = cam.read()
                else:
                    img = cam.read(block=False) if block_ok else cam.read()
            except Exception:
                time.sleep(0.001)
                continue
            if img is None:
                time.sleep(0.001)
                continue
            slot["img"] = img        # 覆盖式: 旧帧直接丢
            slot["seq"] += 1
            produced["n"] += 1

    th = threading.Thread(target=capture, daemon=True)
    th.start()
    n, used, t_end = 0, -1, time.time() + seconds
    while time.time() < t_end:
        if slot["seq"] == used:
            time.sleep(0.0005)
            continue
        used = slot["seq"]
        img = slot["img"]
        if img is None:
            continue
        scan(img)
        n += 1
    stop.set()
    th.join(timeout=1.0)
    return n / seconds, produced["n"] / seconds


# -------------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3.0,
                    help="每项测量的时长(秒)")
    args = ap.parse_args()
    secs = args.seconds

    print(BAR)
    print("CAM1 帧周期分项测量 (MaixCAM-Pro / cv181x)")
    print(BAR)

    print("\n[camera] 构造")
    cam, level = open_camera()
    if cam is None:
        raise SystemExit("[bench] 相机全部构造失败, 无法继续")
    block_ok = probe_block_kwarg(cam)
    print(f"  read(block=False) 支持 = {block_ok}"
          + ("" if block_ok else "   ⚠ 已降级为阻塞读, 停帧保护失效且每帧等满一个间隔"))
    img = cam.read() if not block_ok else cam.read(block=False)
    if img is None:
        img = cam.read()
    print(f"  取图 {img.width()}x{img.height()}  fmt={img.format()}")

    print("\n" + "-" * 74)
    print("[1] 相机产帧能力 (非阻塞轮询)")
    print("-" * 74)
    cad = time_capture_cadence(cam, secs, block_ok)
    if cad:
        print(f"  产帧 {cad['rate']:.1f} Hz   间隔 p50={cad['p50']:.1f}ms "
              f"p95={cad['p95']:.1f}ms   轮询落空 {cad['misses']} 次")
        guess = {16.7: "60fps", 12.5: "80fps(fps 参数没生效)", 33.3: "30fps(CAM1 上限)"}
        near = min(guess, key=lambda g: abs(g - cad["p50"]))
        if abs(near - cad["p50"]) < 5:
            print(f"  -> 像是 {guess[near]}")
        if cad["misses"] > 0:
            print("  -> 有落空, 说明处理比产帧慢; 落空多则等帧时间是白等的")

    print("\n" + "-" * 74)
    print("[2] 每帧固定开销 —— 912KB 拷贝值不值")
    print("-" * 74)
    print("  应用当前用的是 image2cv(ensure_bgr=False, copy=True) + [:, :, ::-1]")
    print("  ⚠ TARGET 走 C++ find_blobs（直接吃 camera.last_raw），这份 numpy 只服务")
    print("    cv2 回退路径与 RESCUE/调参页 —— 正常路径下它是**纯白做**的")
    variants = [
        ("copy=True  ensure_bgr=False  (当前)", dict(ensure_bgr=False, copy=True)),
        ("copy=False ensure_bgr=False  (零拷贝?)", dict(ensure_bgr=False, copy=False)),
        ("copy=True  ensure_bgr=True   (C++换序)", dict(ensure_bgr=True, copy=True)),
    ]
    image2cv_ms = 0.0
    seen_copy = False
    for name, kw in variants:
        try:
            t = bench(lambda k=kw: image.image2cv(img, **k), 20)
        except Exception as e:
            print(f"  {name:<38s} 不支持/失败: {type(e).__name__}: {e}")
            continue
        print(f"  {name:<38s} {t:6.2f} ms/帧")
        if not seen_copy and kw.get("copy"):
            seen_copy = True
            image2cv_ms = t          # 应用当前用的那一档, 结论区要用
    try:
        t_view = bench(lambda: image.image2cv(img, ensure_bgr=False, copy=True)[:, :, ::-1], 20)
        print(f"  上面 + [:, :, ::-1]（负步幅视图）        {t_view:6.2f} ms/帧"
              "   (视图本身≈0; 但它是**非连续**数组, 后续 cv2 可能要多拷一次)")
    except Exception:
        pass
    print("  ⚠ copy=False 是零拷贝视图, 但底层缓冲会被下一次 read() 回收 ->")
    print("    只能在「当帧内立即用完」时使用, 不能跨帧持有（否则读到下一帧的数据）")

    print("\n" + "-" * 74)
    print(f"[3] find_blobs (target ROI={TARGET_ROI}, 面积门槛 {TARGET_AREA_TH})")
    print("-" * 74)
    t_scan = bench(lambda: scan(img), 40)
    print(f"  单色 ROI 扫描                        {t_scan:6.2f} ms/帧")
    try:
        t_full = bench(lambda: img.find_blobs([LAB_RED], invert=False,
                                              area_threshold=TARGET_AREA_TH,
                                              pixels_threshold=TARGET_PIXELS_TH), 20)
        print(f"  同色全帧扫描(对照, 用于估 ROI 收益)  {t_full:6.2f} ms/帧"
              f"   ROI 只占 {(1 - t_scan / t_full) * 100:.0f}% 的省时")
    except Exception as e:
        print(f"  全帧对照失败: {e}")

    print("\n" + "-" * 74)
    print("[4] find_blobs 成本 vs ROI 高度 (收窄 ROI 值不值得)")
    print("-" * 74)
    for h in (197, 120, 80, 40):
        y0 = 169 + (197 - h) // 2
        roi = [0, y0, 640, h]
        t = bench(lambda r=roi: img.find_blobs([LAB_RED], invert=False, roi=r,
                                               area_threshold=TARGET_AREA_TH,
                                               pixels_threshold=TARGET_PIXELS_TH), 20)
        print(f"  ROI 高 {h:3d} 行 ({h * 640:6d} px)            {t:6.2f} ms/帧")
    print("  -> 台架: 把靶锁在画面中心, 用 --dump 类工具读靶的 y 范围, 取"
          " ±(球半径+行程余量)")
    print("     收太窄会在云台大偏差时把目标切出 ROI -> found=0, 留足行程")

    print("\n" + "-" * 74)
    print(f"[5] 模拟当前实现 (read -> find_blobs 串行, {secs:.0f}s)")
    print("-" * 74)
    hz_serial = sim_serial(cam, secs, block_ok)
    print(f"  串行                                 {hz_serial:5.1f} Hz"
          f"   (单帧 {1000 / hz_serial:.0f}ms)")

    print("\n" + "-" * 74)
    print(f"[6] 模拟解耦 (采集线程 + 只取最新帧, {secs:.0f}s)")
    print("-" * 74)
    hz_nb, hz_prod_nb = sim_decoupled(cam, secs, block_ok, blocking_read=False)
    print(f"  采集线程=非阻塞+1ms轮询               {hz_nb:5.1f} Hz"
          f"   (单帧 {1000 / hz_nb:.0f}ms, 采集产帧 {hz_prod_nb:.1f} Hz)")
    hz_blk, hz_prod_blk = sim_decoupled(cam, secs, block_ok, blocking_read=True)
    print(f"  采集线程=阻塞 read()                  {hz_blk:5.1f} Hz"
          f"   (单帧 {1000 / hz_blk:.0f}ms, 采集产帧 {hz_prod_blk:.1f} Hz)")
    if hz_blk < hz_nb * 0.8:
        print("  -> 阻塞读版本明显更慢 = 固件在阻塞等待期间**不释放 GIL**,")
        print("     采集线程会把消费线程饿死 -> 采集解耦这条路在 CAM1 上不可行")

    print("\n" + "-" * 74)
    print("[7] display.show() 单次开销 (GIL 独占的元凶?)")
    print("-" * 74)
    try:
        disp = display.Display()
        t = bench(lambda: disp.show(img), 20)
        print(f"  show()                               {t:6.2f} ms/次"
              f"   (每 2 帧一次 => 每秒 {t * 5:.0f}ms 的独占开销)")
    except Exception as e:
        print(f"  show() 测量失败: {e}")

    print("\n" + BAR)
    print("结论怎么用")
    print(BAR)
    print(f"  相机 {cad['rate']:.1f}Hz (帧间隔 {1000 / cad['rate']:.1f}ms), "
          f"find_blobs {t_scan:.1f}ms, image2cv {image2cv_ms:.1f}ms")
    print(f"  串行 {hz_serial:.1f}Hz | 解耦(非阻塞) {hz_nb:.1f}Hz | "
          f"解耦(阻塞读) {hz_blk:.1f}Hz")
    if hz_serial >= cad["rate"] * 0.9:
        print("  -> 识别链路**已经跑满相机帧率**: read + find_blobs 不是瓶颈。")
        print("     此时若真机 [FPS] pipeline 明显低于这个串行 Hz, 差额就是被**其它")
        print("     线程**吃掉的 GIL —— 先看 [7] show() 的量级（持 GIL 不释放）,")
        print("     再关掉 debug_console 对照。别再往 find_blobs 上使劲。")
        print("     真机上直接看主程序打的那行: [PERF] vision ... | 差额 X ms")
    else:
        print("  -> 识别链路没跑满相机, 瓶颈在 find_blobs 本身:")
        print("     · 按 [4] 收窄 ROI 高度（宽度不能动, 误差是全宽归一化）")
        print("     · 重开两段式 target.scan_scale: 0.5（精度零损失, 有 PC 端测试钉住）")
        print("     · 收紧 LAB 降噪点 —— 噪声连通域是 find_blobs 耗时的大头")
        print("       （调参页 findContours 从 0.08ms 爆到 10.1ms 就是这个机理）")
    if hz_blk >= hz_nb * 0.9:
        print("  · 采集线程: 阻塞读与非阻塞轮询相当, 说明阻塞等待时释放了 GIL,")
        print("    解耦在原理上可行（但若串行已跑满就无法再涨, 别白改）")
    else:
        print("  · 采集线程: 阻塞读版本明显更慢 -> 固件在阻塞等待期间持 GIL,")
        print("    解耦会把消费线程饿死, 这条路在 CAM1 上不要走")
    print("  注意: 上面是**扫描+取图**耗时, 不含主循环与串口(合计 ~3ms);")
    print("        真机周期 = 这些 + 主循环 tick + 被其它线程抢走的 GIL 时间")

    try:
        cam.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
