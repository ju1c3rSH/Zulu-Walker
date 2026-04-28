# -*- coding: utf-8 -*-
"""
Zulu-Walker 启动器

使用方法:
    python run.py main              # 运行主程序
    python run.py debug             # 运行调试器
    python run.py debug -c 1        # 调试器使用摄像头1
    python run.py debug -W 1280 -H 720  # 指定分辨率
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


def main():
    parser = argparse.ArgumentParser(
        description="Zulu-Walker 启动器",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode", required=True, help="运行模式")

    # main 子命令
    subparsers.add_parser("main", help="运行主程序")

    args = parser.parse_args()

    if args.mode == "main":
        run_main()


if __name__ == "__main__":
    main()
