# -*- coding: utf-8 -*-
"""看门狗（WDT）退出重启 —— 设备端实验探针。

## 为什么要这个脚本

现象: MaixCAM-Pro 上程序**运行期一切正常，一退出设备就重启**。
已确认的机制（MaixCDK 源码 `components/peripheral/port/linux_common/maix_wdt.cpp`）:

    WDT::WDT(id, feed_ms) → open("/dev/watchdog") + WDIOC_SETTIMEOUT(feed_ms/1000 秒) + close
                            ⇒ 一旦构造，硬件倒计时就开始跑（构造即自动启动）
    WDT::feed()           → open + WDIOC_KEEPALIVE + close   （每喂一次就重启倒计时）
    WDT::stop()           → 只打一行 "this operation is not supported"，**return 0**（空实现！）

于是: 程序退出后没人喂狗 ⇒ 窗口(本项目 10s)到点 ⇒ SoC 硬复位。
又因为 stop() 是空实现，**在 Python 里调 `wdt.stop()` 关不掉它** —— 日志打印
"stopped" 是假的（它返回 0=成功，代码无从分辨）。

本脚本用**最少的重启次数**把剩下的问题问清楚:
  ① 这台机器的看门狗是什么驱动、能不能被关（nowayout?）;
  ② 绕过 MaixPy、直接对 /dev/watchdog 下 `WDIOS_DISABLECARD` 能不能真正关掉;
  ③ 确认 `stop()` 确实是空实现。

## 用法（设备上、仓库根目录）

    python scripts/wdt_probe.py info      # 只读，不武装任何东西（安全，随时可跑）
    python scripts/wdt_probe.py stopapi   # 证明 maix 的 stop() 是空实现（**结束后设备会重启**）
    python scripts/wdt_probe.py disarm    # 试真正 disarm（**若失败, 结束后设备会重启**）
    python scripts/wdt_probe.py arm       # 对照实验: 武装后直接退出 ⇒ 预期 ~10s 后重启

预期结论怎么读:
  · `info` 里 `parameters/nowayout` = 1 ⇒ 内核层**禁止**停止看门狗（DISABLECARD 会 EBUSY）,
    唯一出路是"别武装它"（`project_config.yaml: wdt_enabled: false`）。
  · `disarm` 跑完设备**没有**重启 ⇒ DISABLECARD 有效，可以把它做进退出路径。
  · `arm` 跑完设备**重启了** ⇒ 机制确认（这一步是阴性对照, 可以不做, 省一次重启）。

⚠ 本脚本只读 sysfs / 只用标准 Linux watchdog ioctl，不依赖 maix 其它模块。
⚠ `arm` / `stopapi` / `disarm` 在**武装之后**无论怎么退出都会走到"关不掉就 10s 后重启"，
   这是被测对象本身的行为，不是脚本的 bug。
"""
from __future__ import annotations

import ctypes
import fcntl
import os
import struct
import subprocess
import sys
import time

# ---- Linux watchdog uapi（<linux/watchdog.h>）----
WATCHDOG_DEV = "/dev/watchdog"
WDT_SYSFS = "/sys/class/watchdog/watchdog0"

# ⚠ ioctl 码必须按**有符号 32 位**交给 fcntl / ioctl。
#   典型码 0x80045707 最高位是 1 ⇒ 作为 C int 其实是**负数**(-2147199225)。
#   传正数时内核回 **errno=14 EFAULT Bad address** —— 不是权限、也不是 nowayout,
#   纯粹是请求码不认（把它当另一个 ioctl 解了）。
WDIOC_GETTIMEOUT = 0x80045707      # _IOR('W', 7, int)
WDIOC_KEEPALIVE = 0x80045705       # _IOR('W', 5, int)
WDIOC_SETOPTIONS = 0x80045704      # _IOR('W', 4, int)
WDIOC_SETTIMEOUT = 0xC0045706      # _IOWR('W', 6, int)——仅打印用, 脚本不主动改窗口
WDIOS_DISABLECARD = 0x0001
WDIOS_ENABLECARD = 0x0002

WINDOW_MS = 10000                  # 与 app/main.py 的 _WDT_WINDOW_MS 保持一致


def _i32(value: int) -> int:
    """把 ioctl 码转成有符号 32 位（C int）。"""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def _ioctl(fd: int, request: int, arg) -> int:
    """对 fd 下 ioctl, 兼容 CPython 的 method-vs-function 两种签名。

    `fcntl.ioctl(fd, req, arg)` 在部分平台上要求 arg 是**可变缓冲区**
    （用于 _IOR/_IOWR 的回填），整数会被拒（TypeError/OverflowError）。
    这里统一用 `ioctl(fd, req, buf, True)` 的 mutate 形式, 并返回 buf 里的 int。
    """
    buf = bytearray(struct.pack("i", int(arg)))
    fcntl.ioctl(fd, _i32(request), buf, True)
    return struct.unpack("i", bytes(buf))[0]


def _say(msg: str = "") -> None:
    """裸 print + flush —— 不经过 logging/rich, 保证退出前一定落到串口。"""
    print(msg, flush=True)


def _read(path: str):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception as e:
        return f"<读不到: {e.__class__.__name__}>"


def _timeleft():
    """看门狗剩余秒数。递减 ⇒ 倒计时在跑(= 武装中)。

    ⚠ sysfs 属性名在不同内核上不一样, 别只试 `timeleft`:
    早先只读 `timeleft` 导致全读到 FileNotFoundError, 结论直接成了"判不了"。
    """
    for name in ("timeleft", "watchdog0_timeleft", "wd_timeleft"):
        try:
            with open(f"{WDT_SYSFS}/{name}") as f:
                return int(f.read().strip())
        except Exception:
            continue
    return None


def _armed(sample_gap_s: float = 0.6):
    """采样两次判断倒计时是否在跑。None = 读不到 sysfs, 判不了。"""
    a = _timeleft()
    if a is None:
        return None
    time.sleep(sample_gap_s)
    b = _timeleft()
    if b is None:
        return None
    return b < a


def _cmd(cmd: str) -> str:
    try:
        out = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                             timeout=5)
        return ((out.stdout or "") + (out.stderr or "")).strip() or "<空>"
    except Exception as e:
        return f"<执行失败: {e}>"


def _uptime_s():
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return -1.0


def _banner(title: str) -> None:
    _say("")
    _say("=" * 62)
    _say(f" {title}")
    _say(f" 系统已运行 {_uptime_s():.1f}s （重启后这个数会归零 ⇒ 可用来判定是否真的重启）")
    _say("=" * 62)


# ---------------------------------------------------------------- info


def cmd_info() -> None:
    """只读探测: 驱动是谁、窗口多大、能不能被关。不武装任何东西。"""
    _banner("WDT 只读探测（不武装，安全）")

    _say("\n[1] 设备节点")
    _say(_cmd(f"ls -l /dev/watchdog* /sys/class/watchdog/ 2>&1"))

    _say("\n[2] sysfs 属性")
    _say("  实际存在的属性文件:")
    _say("    " + _cmd(f"ls {WDT_SYSFS}/ 2>&1"))
    _say("  常见属性（注意内核多用 `<name>_timeout` 而不是 `timeout`）:")
    for attr in ("identity", "state", "status", "timeout", "timeleft",
                 "watchdog0_timeout", "watchdog0_timeleft", "pretimeout",
                 "bootstatus", "max_timeout", "nowayout"):
        _say(f"    {attr:20s} = {_read(f'{WDT_SYSFS}/{attr}')}")

    _say("\n[2b] 绕过 sysfs, 用标准 ioctl 直接问内核窗口")
    _say("  " + _read_kernel_wdt_info())

    _say("\n[3] nowayout（=1 表示内核/驱动禁止停止看门狗, DISABLECARD 会返回 EBUSY）")
    _say(_cmd("grep -Hs . /sys/module/*/parameters/nowayout 2>/dev/null || echo '<无 nowayout 参数>'>"))
    _say("  注: 设备实测为 `soph_wdt ... nowayout:N`（N = No）, 即内核**允许**停止 —— "
         "但 MaixPy 的 WDT::stop() 是空实现, 从不真的去停。")

    _say("\n[4] 驱动与内核看门狗日志（重点看是否有 'watchdog did not stop'）")
    _say(_cmd("dmesg 2>/dev/null | grep -i -E 'watchdog|wdt' | tail -15"))

    _say("\n[5] 'watchdog did not stop' 出现次数"
         "（MaixPy 的 feed() 每次 open/close 不写魔法字符 'V', 每次都会触发这条内核打印）")
    _say("  " + _cmd("dmesg 2>/dev/null | grep -c 'watchdog did not stop'"))

    _say("\n[6] maix.wdt 是否可用")
    try:
        from maix import wdt as _m
        _say(f"  from maix import wdt -> OK ({_m})")
    except Exception as e:
        _say(f"  from maix import wdt -> FAIL: {e.__class__.__name__}: {e}")

    _say("\n[7] 试一记 DISABLECARD（当前若没在跑则无害；在跑则可能直接被关掉, 是好事）")
    _say("  " + _disarm_raw(verbose=True))

    _say("\n[8] 结论怎么读")
    _say("  · nowayout:N + 上面 DISABLECARD OK  ⇒ 退出前能真正关掉（保留死锁兜底 + 退出不重启）")
    _say("  · nowayout:Y 或 DISABLECARD 返回 EBUSY ⇒ 内核禁止停止, **唯一出路是不武装**"
         "（project_config.yaml: wdt_enabled: false，已默认如此）")
    _say("  · `watchdog did not stop` 的次数就是'喂狗次数', 与 uptime 同量级 ⇒ "
         "证实看门狗从构造起一直在跑, 进程退出也不停（这正是重启的机制）")


def cmd_leave() -> None:
    """**决定性实验**: 用 maix 的 stop() 关狗, 然后退出, 看设备是否真的不再重启。

    这是唯一能一次回答"到底能不能关掉"的实验。判定:
      · 退出后设备**没有**重启 ⇒ stop() 能干活（或 raw fallback 生效）, 可用 `stop_after` 方案;
      · 退出后设备**重启了** ⇒ 只能用 project_config.yaml 的 `wdt_enabled: false` 规避。
    两种结果都是有效信息, 不会白跑。
    """
    _banner("实验 D（决定性）: stop() 关狗后退出，观察是否重启")
    from maix import wdt as _wdt
    w = _wdt.WDT(0, WINDOW_MS)
    _say(f"  已武装 WDT(0, {WINDOW_MS}): {w}")
    time.sleep(1.0)
    _say(f"  武装后: timeleft={_timeleft()}  armed={_armed()}")

    # 关狗前**先建立带外信号**: 写一条内核日志。
    # 重启是否发生就能用 `dmesg | grep WDTLEAVE` 一眼判定 ——
    # 比"重启后 uptime 归零"更可靠: uptime 归零也可能是你手动断的电。
    _say("\n  写一条带外时间戳标记（重启后可用 `dmesg | grep WDT_LEAVE` 判定）")
    _cmd("cat > /dev/kmsg <<< 'WDT_LEAVE probe_exit @@ $(date +%s)'; echo done")

    _say("\n  调用 w.stop()（MaixCDK 里是空实现, 但我们还没在真机上验过）")
    try:
        rc = w.stop()
        _say(f"  w.stop() rc={rc}（0 = ERR_NONE, 但**不可信**）")
    except Exception as e:
        _say(f"  w.stop() 抛异常: {e.__class__.__name__}: {e}")

    time.sleep(1.0)
    a, b = _timeleft(), None
    time.sleep(1.2)
    b = _timeleft()
    _say(f"  stop() 后 timeleft: {a} -> {b}  "
         f"{'仍在递减 ⇒ stop() 无效' if (a and b and b < a) else '未递减/读不到'}")

    if _armed() is not False:
        _say("\n  看起来还在跑, 补一记 raw DISABLECARD（修正后的 ioctl 传参 + O_NONBLOCK）")
        _say("  " + _disarm_raw(verbose=True))
        _say(f"  之后: timeleft={_timeleft()}  armed={_armed()}")

    _say("\n  最终状态")
    _say(f"  armed = {_armed()}")
    _exit_now("实验 D 结束 —— 请观察设备是否重启; 若重启, 之后 "
              "`dmesg | grep WDT_LEAVE` 应能查到上面那条标记")


# ------------------------------------------------------- raw ioctl 工具


def _disarm_raw(verbose: bool = False) -> str:
    """绕过 MaixPy, 直接对 /dev/watchdog 下 WDIOS_DISABLECARD。

    这是 `WDT::stop()` 本该做却**没做**的事（MaixCDK 里它是空实现）。
    内核层面 nowayout=1 时会返回 EBUSY(=16) —— 那就说明没有任何用户态办法能关。

    ⚠ 打开时**必须带 O_NONBLOCK**（2026-10-09 实测教训）: 该驱动在
    「看门狗已启动」的状态下, 不带 nonblock 的 open 会**阻塞在
    watchdog_open() 里等标志位**, 表现为脚本挂死（而不是报错）。
    """
    fd = None
    try:
        flags = os.O_RDWR
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK
        fd = os.open(WATCHDOG_DEV, flags)
    except OSError as e:
        return f"open({WATCHDOG_DEV}) 失败: errno={e.errno} {e.strerror}"

    msgs = []
    try:
        if verbose:
            try:
                to = _ioctl(fd, WDIOC_GETTIMEOUT, 0)
                msgs.append(f"GETTIMEOUT = {to}s")
            except OSError as e:
                msgs.append(f"GETTIMEOUT 失败 errno={e.errno} {e.strerror}"
                            + ("（EFAULT=请求码不认, 不是权限問題）" if e.errno == 14 else ""))
        try:
            _ioctl(fd, WDIOC_SETOPTIONS, WDIOS_DISABLECARD)
            msgs.append("DISABLECARD -> OK")
        except OSError as e:
            msgs.append(f"DISABLECARD 失败 errno={e.errno} {e.strerror}"
                        + ("（nowayout=1: 内核禁止停止看门狗, 用户态无解）"
                           if e.errno == 16 else ""))
    finally:
        # 正常关闭走内核的 stop 逻辑; 关闭失败（EBUSY 等）也要留痕, 别静默
        try:
            os.close(fd)
        except OSError as e:
            msgs.append(f"close 失败 errno={e.errno} {e.strerror}")
    return "; ".join(msgs) or "（无输出）"


def _read_kernel_wdt_info():
    """用标准 watchdog ioctl 取内核的窗口/状态（脚本自带 sysfs 缺属性时的兜底）。

    ⚠ sysfs 属性在多数内核上是 `<name>_timeout` / `<name>_timeleft` / `status`
    / `identity`，而不是 `timeout` / `timeleft` —— 见 info 模式里的多路径尝试。
    这里再补一条完全绕过 sysfs 的路。
    """
    try:
        flags = os.O_RDWR | getattr(os, "O_NONBLOCK", 0)
        fd = os.open(WATCHDOG_DEV, flags)
    except OSError as e:
        return f"open 失败: errno={e.errno} {e.strerror}"
    try:
        try:
            to = _ioctl(fd, WDIOC_GETTIMEOUT, 0)
            return f"GETTIMEOUT = {to} s（当前窗口）"
        except OSError as e:
            return f"GETTIMEOUT 失败 errno={e.errno} {e.strerror}"
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _arm():
    """用 maix.wdt 武装（与 app/main.py 完全相同的调用）。"""
    from maix import wdt
    w = wdt.WDT(0, WINDOW_MS)
    _say(f"  已武装: wdt.WDT(0, {WINDOW_MS})  ->  {w}")
    return w


def _exit_now(why: str) -> None:
    _say(f"\n>>> {why}")
    _say(f">>> 现在 os._exit(0) 退出。请盯住串口/屏幕: "
         f"若约 {WINDOW_MS // 1000}s 后设备重启, 说明看门狗仍在跑。")
    _say(f">>> 退出时刻 uptime={_uptime_s():.1f}s")
    sys.stdout.flush()
    os._exit(0)


# ------------------------------------------------------- 实验


def cmd_stopapi() -> None:
    """证明 `wdt.stop()` 是空实现: 调用它之后再采样 timeleft。

    预期: 调用后倒计时**仍在递减** ⇒ stop() 没生效（源码也印证: 空实现 return 0）。
    为了让脚本自己能跑完，全程持续喂狗；跑完把狗停掉再退出（大概率停不掉 ⇒ 重启）。
    """
    _banner("实验 A: maix 的 stop() 到底有没有用")
    w = _arm()
    time.sleep(1.0)
    _say(f"  武装后 timeleft={_timeleft()}s  armed={_armed()}")

    _say("\n调用 w.stop() ……（若要在串口看到 'this operation is not supported', 注意它由 C 层打印）")
    try:
        rc = w.stop()
        _say(f"  w.stop() 返回 rc={rc}   （0 = 源码里的 ERR_NONE, 但其实什么都没做）")
    except Exception as e:
        _say(f"  w.stop() 抛异常: {e.__class__.__name__}: {e}")

    _say("\nstop() 之后再采样倒计时（关键判据: 还在递减 ⇒ stop() 无效）")
    a, b = _timeleft(), None
    time.sleep(1.2)
    b = _timeleft()
    _say(f"  timeleft: {a} -> {b}   {'仍在递减 ⇒ stop() 无效' if (a and b and b < a) else '未递减/读不到'}")
    _say(f"  armed = {_armed()}")

    _say("\n结论: 停止看门狗需要内核标准接口 WDIOC_SETOPTIONS/WDIOS_DISABLECARD, "
         "而 MaixCDK 的 stop() 并未实现它。")
    _say("\n试着补一记 raw DISABLECARD, 免得白重启一次:")
    _say("  " + _disarm_raw(verbose=True))
    _say(f"  之后 armed = {_armed()}")
    _exit_now("实验 A 结束（若上面 DISABLECARD 也失败, 设备将在约 10s 后重启）")


def cmd_disarm() -> None:
    """武装 → 立刻用 raw ioctl 真正 disarm → 立即退出，看设备会不会重启。"""
    _banner("实验 B: 绕过 MaixPy 直接 DISABLECARD 能否真正关掉")
    _arm()
    time.sleep(0.8)
    _say(f"  武装后 timeleft={_timeleft()}s  armed={_armed()}")
    _say("  " + _disarm_raw(verbose=True))
    time.sleep(0.8)
    _say(f"  disarm 后 timeleft={_timeleft()}s  armed={_armed()}")
    _say("\n判读: 上面 armed=False ⇒ 关成功, 本次退出**不该**重启, "
         "可以把这段逻辑做进 app/main.py 的退出路径;")
    _say("      armed=True/None ⇒ 关不掉(nowayout), 只能靠 project_config.yaml "
         "的 wdt_enabled: false 规避。")
    _exit_now("实验 B 结束")


def cmd_arm() -> None:
    """阴性对照: 武装后直接退出, 不喂狗。预期 ~10s 后重启（用来确认这套机制）。"""
    _banner("实验 C（对照）: 武装后不喂狗直接退出")
    _arm()
    _say(f"  timeleft={_timeleft()}s")
    _exit_now("实验 C 结束（预期: 约 10s 后设备重启 —— 这就是原始现象）")


MODES = {
    "info": cmd_info,
    "stopapi": cmd_stopapi,
    "disarm": cmd_disarm,
    "arm": cmd_arm,
    "leave": cmd_leave,
}


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in MODES:
        _say(__doc__)
        _say(f"可用模式: {', '.join(MODES)}")
        return 2
    MODES[mode]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
