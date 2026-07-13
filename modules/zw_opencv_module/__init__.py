# -*- coding: utf-8 -*-
"""
zw_opencv_module - OpenCV Camera Vision Module

Provides camera stream management, QR code detection, and RTMP streaming capabilities.
"""

import os
import cv2

from .camera_manager import CameraManager

# Module directory for config path
_module_dir = os.path.dirname(__file__)

# Module-level instance
_camera_manager: CameraManager = None
_config_path: str = None
_running: bool = False


def init(event_bus=None):
    """Module initialization (called by ModuleManager)"""
    global _camera_manager, _config_path

    if _camera_manager is not None:
        if event_bus is not None:
            _camera_manager.set_event_bus(event_bus)
        return

    print("[zw_opencv_module] Initializing...")
    print(cv2.getBuildInformation())
    _config_path = os.path.join(_module_dir, "config", "camera_config.yaml")

    _camera_manager = CameraManager()
    if event_bus is not None:
        _camera_manager.set_event_bus(event_bus)
    else:
        from context import EventBus
        _camera_manager.set_event_bus(EventBus())

    print("[zw_opencv_module] Initialized successfully")

# hi
def start():
    """Module start (called by ModuleManager)"""
    global _camera_manager, _running

    if _camera_manager is None:
        print("[zw_opencv_module] Error: Module not initialized")
        return False

    print("[zw_opencv_module] Starting...")

    try:
        # Load configuration
        if os.path.exists(_config_path):
            _camera_manager.load_config(_config_path)
            print(f"[zw_opencv_module] Loaded config from {_config_path}")
        else:
            print(f"[zw_opencv_module] Warning: Config file not found at {_config_path}")
            print("[zw_opencv_module] Starting without camera configuration")

        # Start processing loop
        _camera_manager.start()
        _running = True

        print("[zw_opencv_module] Started successfully")
        return True

    except Exception as e:
        print(f"[zw_opencv_module] Failed to start: {e}")
        import traceback
        traceback.print_exc()
        return False


def loop():
    """Module main loop (called by ModuleManager on main thread)"""
    if _camera_manager:
        _camera_manager.display_frame()


def stop():
    """Module stop (called by ModuleManager)"""
    global _camera_manager, _running

    print("[zw_opencv_module] Stopping...")

    _running = False

    if _camera_manager is not None:
        _camera_manager.release()
        _camera_manager = None

    print("[zw_opencv_module] Stopped")


# === Optional: Direct access functions ===

def get_camera_manager() -> CameraManager:
    """Get the CameraManager instance"""
    return _camera_manager


def get_camera(camera_id: str):
    """Get a specific camera by ID"""
    if _camera_manager:
        return _camera_manager.get_camera(camera_id)
    return None


def get_all_results():
    """Get all camera processing results"""
    if _camera_manager:
        return _camera_manager.process_all()
    return None, {}
