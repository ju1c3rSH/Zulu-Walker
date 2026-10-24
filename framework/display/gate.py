# -*- coding: utf-8 -*-
"""格式闸门: 帧格式不在 disp.caps().accepted 里就打印一行, 每个平台只打一次。

⚠ 不用 assert: 两个调用点的 ``except Exception`` 会把它吞成"静默冻屏"。
"""
from __future__ import annotations

from framework.display.types import PixelFormat
from utils.log_util import log_print

# accepted 只跟平台类有关, 按类缓存一次即可, 免得每帧付 caps() 的构造开销。
_ACCEPTED_BY_TYPE: dict = {}
_WARNED = set()


def _caps_owner(disp):
    """取真正的 display 类: 调试页拿到的是每帧新建的代理, 必须看它包着的那个。"""
    return type(getattr(disp, "_real", None) or disp)


def warn_slow_format(disp, frame) -> bool:
    """命中慢路径返回 True 并告警一次; 判不出格式一律当正常, 绝不抛异常。"""
    try:
        fmt = PixelFormat.of(frame)
        if fmt is None:
            return False
        key = _caps_owner(disp)
        accepted = _ACCEPTED_BY_TYPE.get(key)
        if accepted is None:
            accepted = _ACCEPTED_BY_TYPE[key] = disp.caps().accepted
        if fmt in accepted:
            return False
        if (key, fmt) not in _WARNED:
            _WARNED.add((key, fmt))
            log_print("[DISP] slow path: fmt=%s not in caps.accepted={%s}"
                      % (fmt.name, ",".join(sorted(f.name for f in accepted))))
        return True
    except Exception:
        return False
