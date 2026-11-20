from __future__ import annotations

import logging
import os
import platform
import threading
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_DEFAULT_VISION_PROCESSING = {1}
_DEFAULT_CAMERA_CAPTURE = {0}
_DEFAULT_MAIN_LOOP = {0}
_DEFAULT_HEARTBEAT = {0}
_DEFAULT_UART_RECEIVER = {0}
_DEFAULT_DEBUG_CONSOLE = {0}

_global_config: Optional["CpuAffinityConfig"] = None


class CpuAffinityConfig:
    def __init__(self, cfg: Optional[dict] = None) -> None:
        cfg = cfg or {}
        self.enabled = cfg.get("enabled", True) and platform.system() == "Linux"
        self.vision_processing = _parse_cores(cfg.get("vision_processing"), _DEFAULT_VISION_PROCESSING)
        self.camera_capture = _parse_cores(cfg.get("camera_capture"), _DEFAULT_CAMERA_CAPTURE)
        self.main_loop = _parse_cores(cfg.get("main_loop"), _DEFAULT_MAIN_LOOP)
        self.heartbeat = _parse_cores(cfg.get("heartbeat"), _DEFAULT_HEARTBEAT)
        self.uart_receiver = _parse_cores(cfg.get("uart_receiver"), _DEFAULT_UART_RECEIVER)
        self.debug_console = _parse_cores(cfg.get("debug_console"), _DEFAULT_DEBUG_CONSOLE)
        self.opencv_threads = cfg.get("opencv_threads", 0)

    def get_cores(self, role: str) -> Optional[Set[int]]:
        return getattr(self, role, None)


def _parse_cores(value, default: Set[int]) -> Set[int]:
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        return set(int(c) for c in value)
    return default


def _validate_cores(cfg: "CpuAffinityConfig") -> Optional[str]:
    """Return a description of out-of-range core numbers, or None if valid.

    Context:
        Defaults target MaixCAM2 (AX630C). On boards with fewer cores
        (e.g. MaixCAM-Pro cv181x), ``sched_setaffinity`` would silently fail
        and leave binding entirely ineffective; catch it up front.
    """
    try:
        ncpu = os.cpu_count() or 0
    except Exception:
        return None
    if ncpu <= 0:
        return None
    roles = (
        "vision_processing", "camera_capture", "main_loop",
        "heartbeat", "uart_receiver", "debug_console",
    )
    invalid = []
    for role in roles:
        cores = cfg.get_cores(role)
        if not cores:
            continue
        if any(c < 0 or c >= ncpu for c in cores):
            invalid.append(f"{role}={sorted(cores)}")
    if not invalid:
        return None
    return f"configured/default cores exceed CPU count ({ncpu}): {', '.join(invalid)}"


def configure(config_dict: Optional[dict]) -> None:
    global _global_config
    _global_config = CpuAffinityConfig(config_dict)
    if _global_config.enabled:
        problem = _validate_cores(_global_config)
        if problem:
            _global_config.enabled = False
            logger.warning("CPU affinity DISABLED: %s", problem)
    if _global_config.enabled:
        logger.info(
            "CPU affinity enabled. Vision=%s Camera=%s MainLoop=%s "
            "Heartbeat=%s UART=%s Debug=%s OpenCV_threads=%s",
            _global_config.vision_processing,
            _global_config.camera_capture,
            _global_config.main_loop,
            _global_config.heartbeat,
            _global_config.uart_receiver,
            _global_config.debug_console,
            _global_config.opencv_threads,
        )
    else:
        logger.info("CPU affinity disabled (platform=%s)", platform.system())


def get_config() -> Optional[CpuAffinityConfig]:
    return _global_config


def bind_current_thread(role: str) -> None:
    config = _global_config
    if config is None or not config.enabled:
        return
    cores = config.get_cores(role)
    if cores is None:
        logger.warning(
            "Unknown CPU affinity role '%s' for thread '%s'",
            role,
            threading.current_thread().name,
        )
        return
    try:
        os.sched_setaffinity(0, cores)
        logger.debug(
            "Bound thread '%s' to cores %s",
            threading.current_thread().name,
            cores,
        )
    except (AttributeError, OSError, PermissionError) as e:
        logger.warning(
            "Failed to bind thread '%s' to cores %s: %s",
            threading.current_thread().name,
            cores,
            e,
        )
