from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Optional

import yaml

from framework.hal.camera_hub import CameraHub
from framework.hal.interface import AIInference, Display, Uart
try:
    from utils.cpu_affinity import configure as configure_cpu_affinity
except ImportError:
    def configure_cpu_affinity(cfg): pass

logger = logging.getLogger(__name__)


class Machine:
    def __init__(
        self,
        camera_hub: CameraHub,
        display: Display,
        uart: Uart,
        ai: Optional[AIInference] = None,
    ) -> None:
        self.camera_hub = camera_hub
        self.display = display
        self.uart = uart
        self.ai = ai

    @classmethod
    def create(cls, config_path: str = "project_config.yaml") -> "Machine":
        path = Path(config_path)
        if not path.exists():
            logger.warning("%s not found, using platform=maixcam2 defaults", config_path)
            platform = "maixcam2"
            cameras_config = []
            uart_config = {"port": "/dev/ttyS1", "baudrate": 921600}
            ai_config = None
        else:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            platform = cfg.get("platform", "maixcam2")
            cameras_config = cfg.get("cameras", [])
            uart_defaults = cfg.get("uart_defaults", {})
            uart_config = {
                "port": uart_defaults.get("port", "/dev/ttyS1"),
                "baudrate": uart_defaults.get("baudrate", 921600),
            }
            ai_config = cfg.get("ai")
            cpu_affinity_cfg = cfg.get("cpu_affinity")
            configure_cpu_affinity(cpu_affinity_cfg)

        platform_mod = importlib.import_module(f"framework.hal.platforms.{platform}")

        hub = CameraHub.init_instance(platform)

        for cam_cfg in cameras_config:
            cid = cam_cfg.get("camera_id", str(cam_cfg.get("source", "")))
            raw_source = cam_cfg["source"]
            source = raw_source
            try:
                from utils.camera_misc_util import CameraMiscUtil
            except ImportError:
                CameraMiscUtil = None
            if platform == "linux" and CameraMiscUtil is not None:
                resolved = CameraMiscUtil.resolve_camera_source(raw_source)
                if resolved != raw_source:
                    logger.info("Camera '%s': source %s -> %s", cid, raw_source, resolved)
                source = resolved
            hub.open(
                camera_id=cid,
                source=source,
                width=cam_cfg.get("width", 640),
                height=cam_cfg.get("height", 480),
                fps=cam_cfg.get("fps"),
                camera_stream_queue_size=cam_cfg.get("camera_stream_queue_size", 2),
                focal_length_mm=cam_cfg.get("focal_length_mm"),
                sensor_width_mm=cam_cfg.get("sensor_width_mm"),
                sensor_height_mm=cam_cfg.get("sensor_height_mm"),
                exposure_us=cam_cfg.get("exposure_us"),
                gain=cam_cfg.get("gain"),
                aec=cam_cfg.get("aec"),
            )

        display = platform_mod.create_display()
        try:
            uart = platform_mod.create_uart(**uart_config)
        except Exception as e:
            logger.warning("UART creation failed (non-fatal): %s", e)
            uart = None

        # --- AI initialization ---
        ai: Optional[AIInference] = None
        if ai_config is not None:
            try:
                ai = platform_mod.create_ai()
                models_cfg = ai_config.get("models", [])
                for m in models_cfg:
                    ai.add(
                        nick_name=m["nick_name"],
                        model_path=m["model"],
                        model_type=m.get("model_type", "auto"),
                    )
                active = ai_config.get("active")
                if active:
                    if active in ai.models:
                        ai.switch(active)
                    else:
                        logger.warning(
                            "AI active model '%s' not in registered models %s",
                            active,
                            ai.models,
                        )
            except Exception as e:
                logger.error("Failed to initialize AI: %s", e)
                ai = None

        return cls(camera_hub=hub, display=display, uart=uart, ai=ai)

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
        if self.ai is not None:
            try:
                self.ai.unload()
            except Exception as e:
                logger.error("AI unload error: %s", e)
