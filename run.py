# -*- coding: utf-8 -*-
"""
Zulu-Walker 启动器

使用方法:
    python run.py main                          # 运行主程序
    python run.py debug color                   # 设备端 LAB 阈值调参器（触屏）
"""
import argparse
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


_RUNNER_MAP = {
    "color":  ("modules.zw_opencv_module.debug.threshold_tuner", "GksThresholdTuner"),
}


def run_main():
    from app.main import main
    main()


def run_debug(args):
    module_path, class_name = _RUNNER_MAP[args.detector]
    module = importlib.import_module(module_path)
    runner_cls = getattr(module, class_name)
    camera_source = args.camera
    try:
        camera_source = int(args.camera)
    except ValueError:
        pass
    runner = runner_cls(
        camera_source=camera_source,
        width=args.width,
        height=args.height,
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.stop()


def _console_enabled_from_config() -> bool:
    """project_config.yaml 的 `debug_console_enabled`（缺省/读不到 -> True）。

    必须在启动任何东西之前判定。若先无条件 `DebugConsole().start()` +
    `ConsoleCapture.install()`, 则 `debug_console_enabled: false` 完全无效:
    rich 的 10Hz 全屏重绘照跑, 且 sys.stdout 被换成 Python 对象, 此后每一行
    print 都变成一次 log_print + 一次文件 open/close。

    另外这里连 `utils.debug_console` 都不导入 —— 那个模块顶层 `import rich`,
    而 rich 是纯 Python 大包, CAM1 只有 128MB 用户内存, 不值得为一个已被禁用的
    功能付这份内存与启动时间。读不到配置时返回 True = 保持默认行为。
    """
    try:
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "project_config.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return bool(cfg.get("debug_console_enabled", True))
    except Exception:
        return True


def _init_debug_console() -> None:
    from utils.log_util import log_print
    if not _console_enabled_from_config():
        log_print("[DCONSOLE] 已禁用: 不启动 rich 渲染线程、不导入 rich、"
                  "不安装 stdout 捕获（日志仍照常写 logs/debug.log 与控制台）")
        return
    from utils.debug_console import DebugConsole
    from utils.console_capture import ConsoleCapture
    DebugConsole().start()
    ConsoleCapture.install()


def _cleanup():
    # 只在调试台确实被导入过时才去碰它 —— 否则为了收尾把 rich 拉进内存。
    mod = sys.modules.get("utils.debug_console")
    if mod is not None:
        mod.DebugConsole().stop()


def main():
    parser = argparse.ArgumentParser(
        description="Zulu-Walker 启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    subparsers.add_parser("main", help="运行主程序")

    debug_parser = subparsers.add_parser("debug", help="运行检测调试器")
    debug_parser.add_argument(
        "detector", nargs="?", default="color",
        choices=list(_RUNNER_MAP.keys()),
        help="检测器类型 (默认: color 阈值调参)"
    )
    debug_parser.add_argument("-c", "--camera", type=str, default="0", help="摄像头索引或设备路径 (默认: 0)")
    debug_parser.add_argument("-W", "--width", type=int, default=640, help="画面宽度 (默认: 640)")
    debug_parser.add_argument("-H", "--height", type=int, default=480, help="画面高度 (默认: 480)")

    _init_debug_console()

    args = parser.parse_args()

    try:
        if args.mode == "main":
            run_main()
        elif args.mode == "debug":
            run_debug(args)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
