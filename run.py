# -*- coding: utf-8 -*-
"""Launch the Zulu-Walker application and detector debug tools.

Usage::

    python run.py main                          # run the main program
    python run.py debug [detector]              # run a detector debugger
    python run.py debug cargo                   # cargo-block detector debugger
    python run.py debug circle                  # circle-target detector debugger
    python run.py debug ring                    # ring detector debugger
    python run.py debug line                    # black-line tracker debugger
    python run.py debug pendulum_calib          # pendulum calibration debugger
    python run.py debug cargo -c 1              # select camera 1
    python run.py debug cargo -W 1280 -H 720    # set frame resolution
"""
import argparse
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


from utils.console_capture import ConsoleCapture
from utils.debug_console import DebugConsole


_RUNNER_MAP = {
    "cargo":  ("modules.zw_opencv_module.detectors.cargo_detector.debug.runner", "CargoDebugRunner"),
    "circle": ("modules.zw_opencv_module.detectors.circle_target_detector.debug.runner", "CircleTargetDebugRunner"),
    "ring":   ("modules.zw_opencv_module.detectors.ring_detector.debug.runner", "RingDebugRunner"),
    "line":   ("modules.zw_opencv_module.detectors.line_detector.debug.runner", "LineDebugRunner"),
    "pendulum_calib": ("modules.zw_opencv_module.detectors.pendulum_calibrator.debug.runner", "PendulumCalibDebugRunner"),
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


def _init_debug_console():
    DebugConsole().start()
    ConsoleCapture.install()


def _cleanup():
    DebugConsole().stop()


def main():
    parser = argparse.ArgumentParser(
        description="Zulu-Walker 启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    subparsers.add_parser("main", help="运行主程序")

    debug_parser = subparsers.add_parser("debug", help="运行检测调试器")
    debug_parser.add_argument(
        "detector", nargs="?", default="cargo",
        choices=list(_RUNNER_MAP.keys()),
        help="检测器类型 (默认: cargo)"
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
