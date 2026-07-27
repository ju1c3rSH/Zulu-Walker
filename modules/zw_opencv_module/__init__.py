from __future__ import annotations

import os
from typing import Optional

import cv2


from framework.hal import Machine

from .vision_manager import VisionManager, _LegacyCameraManagerShim

_module_dir = os.path.dirname(__file__)
_vision_manager: Optional[VisionManager] = None
_legacy_shim: Optional[_LegacyCameraManagerShim] = None
_running: bool = False

try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass


def init(machine: Machine, event_bus=None) -> None:
    from utils.cpu_affinity import get_config as get_cpu_affinity_config
    _cpu_cfg = get_cpu_affinity_config()
    if _cpu_cfg and _cpu_cfg.opencv_threads > 0:
        try:
            cv2.setNumThreads(_cpu_cfg.opencv_threads)
        except Exception:
            pass
    else:
        try:
            cv2.setNumThreads(1)
        except Exception:
            pass
    global _vision_manager, _legacy_shim

    config_path = os.path.join(_module_dir, "config", "vision_config.yaml")
    _vision_manager = VisionManager(
        camera_hub=machine.camera_hub,
        config_path=config_path,
        ai=machine.ai,
    )
    if event_bus is not None:
        _vision_manager.set_event_bus(event_bus)
    _legacy_shim = _LegacyCameraManagerShim(_vision_manager)


def start() -> None:
    global _running
    if _vision_manager is None:
        return
    _vision_manager.start()
    _running = True


def loop() -> None:
    pass


def stop() -> None:
    global _running, _vision_manager, _legacy_shim
    _running = False
    _legacy_shim = None
    if _vision_manager is not None:
        _vision_manager.release()
        _vision_manager = None


def get_vision_manager() -> Optional[VisionManager]:
    return _vision_manager


def get_camera_manager() -> _LegacyCameraManagerShim:
    return _legacy_shim


def set_event_bus(bus) -> None:
    if _vision_manager is not None:
        _vision_manager.set_event_bus(bus)


def get_all_results():
    if _vision_manager:
        return _vision_manager.get_all_results()
    return {}
