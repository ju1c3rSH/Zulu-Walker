# -*- coding: utf-8 -*-
"""Processor 包导出。

只导出本分支打包运行所需的最小集（app.yaml files 白名单不含遗留处理器,
此处导入它们会在设备端 ModuleNotFoundError）。遗留处理器文件保留在仓库
供 PC 调试, 需要时直接按子模块路径导入。
"""
from .base import Processor, VisionResult, ColorTrackable
from .registry import register_processor, get_processor
# 导入即注册 @register_processor("GksTaskProcessor")（设备端流水线经
# registry 构建任务, 缺此导入会静默注册失败 → 全零输出）
from .gks_task_processor import GksTaskProcessor

__all__ = [
    "Processor", "VisionResult", "ColorTrackable",
    "register_processor", "get_processor",
    "GksTaskProcessor",
]
