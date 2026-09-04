# -*- coding: utf-8 -*-
"""
考核骨架启动器

使用方法:
    python run.py debug circle                  # 圆形检测调试器 (教学样板)
    python run.py debug circle -c 1             # 指定摄像头
    python run.py debug circle -W 1280 -H 720   # 指定分辨率
"""
import argparse
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


from utils.console_capture import ConsoleCapture
from utils.debug_console import DebugConsole


_RUNNER_MAP = {
    "circle": ("modules.zw_opencv_module.detectors.circle_target_detector.debug.runner", "CircleTargetDebugRunner"),
}


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


def _init_debug_console():
    DebugConsole().start()
    ConsoleCapture.install()


def _cleanup():
    DebugConsole().stop()


def main():
    parser = argparse.ArgumentParser(
        description="考核骨架启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    debug_parser = subparsers.add_parser("debug", help="运行检测调试器")
    debug_parser.add_argument(
        "detector", nargs="?", default="circle",
        choices=list(_RUNNER_MAP.keys()),
        help="检测器类型 (默认: circle)"
    )
    debug_parser.add_argument("-c", "--camera", type=str, default="0", help="摄像头索引或设备路径 (默认: 0)")
    debug_parser.add_argument("-W", "--width", type=int, default=640, help="画面宽度 (默认: 640)")
    debug_parser.add_argument("-H", "--height", type=int, default=480, help="画面高度 (默认: 480)")

    _init_debug_console()

    args = parser.parse_args()

    try:
        if args.mode == "debug":
            run_debug(args)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
