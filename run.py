# -*- coding: utf-8 -*-
"""
Zulu-Walker 启动器

使用方法:
    python run.py main              # 运行主程序
    python run.py debug             # 运行物料块检测调试器
    python run.py debug -c 1        # 调试器使用摄像头1
    python run.py debug -W 1280 -H 720  # 指定分辨率
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))


def run_main():
    from main import main
    main()


def run_debug(args):
    from modules.zw_opencv_module.detectors.cargo_detector.debug.runner import CargoDebugRunner
    runner = CargoDebugRunner(
        camera_source=args.camera,
        width=args.width,
        height=args.height,
    )
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Zulu-Walker 启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    subparsers.add_parser("main", help="运行主程序")

    debug_parser = subparsers.add_parser("debug", help="运行物料块检测调试器")
    debug_parser.add_argument("-c", "--camera", type=int, default=0, help="摄像头索引 (默认: 0)")
    debug_parser.add_argument("-W", "--width", type=int, default=640, help="画面宽度 (默认: 640)")
    debug_parser.add_argument("-H", "--height", type=int, default=480, help="画面高度 (默认: 480)")

    args = parser.parse_args()

    if args.mode == "main":
        run_main()
    elif args.mode == "debug":
        run_debug(args)


if __name__ == "__main__":
    main()
