# -*- coding: utf-8 -*-
"""
Zulu-Walker 启动器

使用方法:
    python run.py main              # 运行主程序
    python run.py debug             # 运行调试器
    python run.py debug -c 1        # 调试器使用摄像头1
    python run.py debug -W 1280 -H 720  # 指定分辨率
    python run.py debug --debug-uv  # 启用 UV 调试面板
"""
import argparse
import sys
import os

# 确保项目根目录在 path
sys.path.insert(0, os.path.dirname(__file__))


def run_main():
    """运行主程序"""
    from main import main
    main()


def run_debug(args):
    """运行调试器"""
    from modules.zw_opencv_module.debug.detector import DebugDetector
    detector = DebugDetector(
        camera_source=args.camera,
        width=args.width,
        height=args.height,
        config_path=args.config,
        debug_uv=args.debug_uv,
        debug_cam=args.debug_cam
    )
    try:
        detector.start()
    except KeyboardInterrupt:
        detector.stop()


def main():
    parser = argparse.ArgumentParser(
        description="Zulu-Walker 启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    # main 子命令
    subparsers.add_parser("main", help="运行主程序")

    # debug 子命令
    debug_parser = subparsers.add_parser("debug", help="运行调试器")
    debug_parser.add_argument("-c", "--camera", type=int, default=0, help="摄像头索引 (默认: 0)")
    debug_parser.add_argument("-W", "--width", type=int, default=640, help="画面宽度 (默认: 640)")
    debug_parser.add_argument("-H", "--height", type=int, default=480, help="画面高度 (默认: 480)")
    debug_parser.add_argument("-f", "--config", type=str, default=None, help="配置文件路径")
    debug_parser.add_argument("--debug-uv", action="store_true", help="启用 UV 调试面板")
    debug_parser.add_argument("--debug-cam", action="store_true", help="启用摄像头参数调试面板")

    args = parser.parse_args()

    if args.mode == "main":
        run_main()
    elif args.mode == "debug":
        run_debug(args)


if __name__ == "__main__":
    main()
