from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Optional

import yaml

from hal.camera_hub import CameraHub
from hal.interface import Display, Uart

logger = logging.getLogger(__name__)


class Machine:
    def __init__(
        self,
        camera_hub: CameraHub,
        display: Display,
        uart: Uart,
        ai=None,
    ) -> None:
        self.camera_hub = camera_hub
        self.display = display
        self.uart = uart
        self.ai = ai

    @classmethod
    def create(cls, config_path: str = "project_config.yaml") -> "Machine":
        path = Path(config_path)
        if not path.exists():
            logger.warning("%s not found, using platform=linux defaults", config_path)
            platform = "linux"
            cameras_config = []
            uart_config = {"port": "/dev/ttyS4", "baudrate": 921600}
        else:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            platform = cfg.get("platform", "linux")
            cameras_config = cfg.get("cameras", [])
            uart_defaults = cfg.get("uart_defaults", {})
            uart_config = {
                "port": uart_defaults.get("port", "/dev/ttyS4"),
                "baudrate": uart_defaults.get("baudrate", 921600),
            }

        platform_mod = importlib.import_module(f"hal.platforms.{platform}")

        hub = CameraHub.init_instance(platform)

        for cam_cfg in cameras_config:
            cid = cam_cfg.get("camera_id", str(cam_cfg.get("source", "")))
            hub.open(
                camera_id=cid,
                source=cam_cfg["source"],
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                fps=cam_cfg.get("fps", 120),
                camera_stream_queue_size=cam_cfg.get("camera_stream_queue_size", 2),
                focal_length_mm=cam_cfg.get("focal_length_mm"),
                sensor_width_mm=cam_cfg.get("sensor_width_mm"),
                sensor_height_mm=cam_cfg.get("sensor_height_mm"),
            )

        display = platform_mod.create_display()
        uart = platform_mod.create_uart(**uart_config)

        return cls(camera_hub=hub, display=display, uart=uart)

    def close(self) -> None:
        if self.camera_hub:
            try:
                self.camera_hub.release_all()
            except Exception as e:
                logger.error("CameraHub release_all error: %s", e)
        if self.display:
            try:
                self.display.close()
            except Exception as e:
                logger.error("Display close error: %s", e)
        if self.uart:
            try:
                self.uart.disconnect()
            except Exception as e:
                logger.error("UART disconnect error: %s", e)
