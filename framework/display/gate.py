# -*- coding: utf-8 -*-
"""格式检查: 帧格式不在 disp.caps().accepted 里就打印一行, 每个平台只打一次。

不用 assert: 两个调用点的 ``except Exception`` 会把它吞成"静默冻屏"。
"""
from __future__ import annotations

from framework.display.types import PixelFormat
from utils.log_util import log_print

# accepted 只跟 display 类型有关; 按类型缓存一次, 免得每帧付 caps() 的构造开销。
_ACCEPTED_BY_TYPE: dict = {}
_WARNED = set()


def warn_slow_format(disp, frame) -> bool:
    """格式不被本平台接受时返回 True 并告警一次; 判不出一律当正常, 绝不抛异常。"""
    try:
        # 只认公开的 caps(): 代理靠 __getattr__ 透传, 嗅探私有属性会在 C3 加一层后永久失聪
        if not hasattr(disp, "caps"):
            return False
        fmt = PixelFormat.of(frame)
        if fmt is None:
            return False
        key = type(disp)
        accepted = _ACCEPTED_BY_TYPE.get(key)
        if accepted is None:
            accepted = _ACCEPTED_BY_TYPE[key] = frozenset(disp.caps().accepted)
        if fmt in accepted:
            return False
        if (key, fmt) not in _WARNED:
            _WARNED.add((key, fmt))
            log_print("[DISP] fmt not accepted: fmt=%s accepted={%s}"
                      " (slow or unsupported path)" % (
                          fmt.name, ",".join(sorted(f.name for f in accepted))))
        return True
    except Exception:
        return False
