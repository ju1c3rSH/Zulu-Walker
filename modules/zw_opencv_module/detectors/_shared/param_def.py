from dataclasses import dataclass


@dataclass
class ParamDef:
    name: str
    display: str
    default: int
    min_val: int
    max_val: int
    step: int = 1
    scale: float = 1.0
    odd: bool = False
