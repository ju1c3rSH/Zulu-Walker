from dataclasses import dataclass, field


@dataclass
class CmdRequestEvent:
    """CMD_REQUEST(0x20) 订阅事件。

    5B 完整订阅帧携带任务码 (爆色/靶色/形状)；短帧 = 重订阅,
    codes 为 None 表示沿用上次任务码（重复发也不要清视觉锁存）。
    """
    data_type: int
    min_interval_ms: int
    explosive_color: int = 0
    target_color: int = 0
    rescue_shape: int = 0
    has_codes: bool = False

    @property
    def codes(self) -> tuple:
        return (self.explosive_color, self.target_color, self.rescue_shape)


@dataclass
class CmdStopEvent:
    """CMD_STOP(0x23) 停止推送事件。"""
    pass
