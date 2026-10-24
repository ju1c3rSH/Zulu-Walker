# -*- coding: utf-8 -*-
"""模块 init 契约: 必须容忍 ModuleManager 注入的公共关键字参数。

背景: module_manager 用同一组 `**module_init_kwargs` 调用所有模块的 init()。
把 `config=...` 加进这组参数后, 任何一个签名不收 **kwargs 的模块都会
`TypeError: init() got an unexpected keyword argument 'config'` ——
而 ModuleManager.load 会 try/except 住它, 表现为**模块静默未注册**。
"""
import inspect
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODULES = ["zw_opencv_module", "zw_uart_module", "zw_wifi_stream"]
INJECTED_KWARGS = ("machine", "event_bus", "wdt_feed", "config")


@pytest.mark.parametrize("name", MODULES)
def test_module_init_accepts_injected_kwargs(name):
    import importlib
    mod = importlib.import_module("modules." + name)
    assert hasattr(mod, "init"), f"{name}.init 缺失"
    sig = inspect.signature(mod.init)
    params = sig.parameters
    has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
    missing = [k for k in INJECTED_KWARGS
               if k not in params and not has_var_kw]
    assert not missing, (
        f"{name}.init 无法接受 {missing}; module_manager 会抛 TypeError "
        f"并静默跳过该模块。签名: {sig}")


def test_module_manager_passes_config():
    """确认 module_manager 确实会把 config 注入 —— 与上面测试配对, 防止只改一边。"""
    import framework.module_manager as mm
    src = inspect.getsource(mm.ModuleManager.load)
    assert "**self._module_init_kwargs" in src
