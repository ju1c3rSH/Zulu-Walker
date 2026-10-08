# -*- coding: utf-8 -*-
"""`debug_console_enabled: false` 必须真的"不导入 rich"，而不只是"不渲染"。

背景（2026-10-09）:
  · `utils/debug_console.py` 在**模块顶层** `import rich`。rich 是纯 Python 大包，
    而 CAM1 只有 128MB 用户内存（台架实测 `[MEM] user=103/128MB` = 已占 80%）。
  · 原 `run.py` 在读到配置**之前**就无条件 `DebugConsole().start()` +
    `ConsoleCapture.install()`, 于是 `debug_console_enabled: false` **完全无效**:
    rich 照跑 10Hz 全屏重绘, 且 `sys.stdout` 被换成 Python 对象（此后每一行
    `print` 都变成一次 log_print + 一次日志文件 open/close）。

所以要钉住的不变量是 **"未导入"** 而不是"未渲染" —— 这两者差别就是 rich 的
内存与启动时间。用子进程测（`sys.modules` 是全局状态, 同进程内测不准）。

本文件不依赖 project_config.yaml 当前取什么值: 两条分支都用注入的返回值驱动,
这样台架把开关切回 true 也不会让测试红。
"""
import os
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT, capture_output=True, text=True, timeout=90,
    )


def test_disabled_does_not_import_rich():
    """禁用时: rich / utils.debug_console 都不许进 sys.modules, stdout 不许被包裹。"""
    r = _run("""
        import sys
        sys.path.insert(0, ".")
        import run

        run._console_enabled_from_config = lambda: False
        run._init_debug_console()

        assert "rich" not in sys.modules, "禁用时导入 rich = 白付内存与启动时间"
        assert "utils.debug_console" not in sys.modules
        from utils.log_util import debug_console_active
        assert debug_console_active() is False
        assert type(sys.stdout).__name__ == "TextIOWrapper", \\
            "禁用时不应安装 ConsoleCapture（否则每行 print 都变成一次文件 open/close）"
        print("DISABLED_OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "DISABLED_OK" in r.stdout, r.stdout


def test_enabled_still_works():
    """启用时: 必须真的导入并启动（否则"切回 true"这条路被改坏了没人知道）。"""
    r = _run("""
        import sys
        sys.path.insert(0, ".")
        import run

        run._console_enabled_from_config = lambda: True
        run._init_debug_console()

        assert "utils.debug_console" in sys.modules, "启用时应当导入调试台"
        assert "rich" in sys.modules, "启用时 rich 必然被导入（顶层 import）"
        from utils.log_util import debug_console_active
        assert debug_console_active() is True
        assert type(sys.stdout).__name__ == "ConsoleCapture"
        run._cleanup()
        print("ENABLED_OK")
    """)
    assert r.returncode == 0, r.stderr
    assert "ENABLED_OK" in r.stdout, r.stdout


def test_disabled_path_still_logs_to_stdout():
    """禁用 ≠ 静音: `log_print` 必须仍把内容写到真 stdout（台架靠它看 [PERF]）。"""
    r = _run("""
        import sys, time
        sys.path.insert(0, ".")
        import run
        from utils.log_util import start_log_writer, stop_log_writer, log_print

        run._console_enabled_from_config = lambda: False
        run._init_debug_console()
        start_log_writer()
        log_print("[TEST] 禁用调试台后这行仍应出现")
        time.sleep(0.3)
        stop_log_writer()
        sys.__stdout__.write("STDOUT_OK\\n")
    """)
    assert r.returncode == 0, r.stderr
    assert "[TEST] 禁用调试台后这行仍应出现" in r.stdout, \
        "禁用调试台不得把日志一起吞掉"
