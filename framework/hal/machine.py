from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Optional

import yaml

from framework.hal.camera_hub import CameraHub
from framework.hal.interface import AIInference, Battery, Display, Uart
from utils.boot_timing import BootTimer
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
        battery: Optional[Battery] = None,
    ) -> None:
        self.camera_hub = camera_hub
        self.display = display
        self.uart = uart
        self.ai = ai
        self.battery = battery

    @classmethod
    def create(cls, config_path: str = "project_config.yaml") -> "Machine":
        # 启动打点: 2026-10-08 实测 MaixCAM-Pro 上这段串行构造耗时约 8 秒
        # (04:05:54 -> 04:06:02), 而改造前**全程无打点**, 导致无法判断是
        # 相机/显示/UART0 抢线/模型加载哪一段吃掉了时间。五步逐段打点后,
        # 台架一次启动就能定位到段。零风险: 打点只读时钟, 失败即静默退化。
        bt = BootTimer("machine")
        path = Path(config_path)
        # 缺省平台取project_config.yaml 的首选值（当前分支 maixcam1）；
        # 端口/波特率缺省与 project_config.yaml `uart_defaults` 保持一致
        #（当前 CAM1 分支 = /dev/ttyS0 + 115200）—— 官方警告其他波特率误码率高，
        # 且端口与 Brain 侧接线强绑定，配置缺失时静默用错端口等于链路不通。
        default_platform = "maixcam1"
        default_uart = {"port": "/dev/ttyS0", "baudrate": 115200}
        if not path.exists():
            logger.warning("%s not found, using platform=%s defaults %s",
                           config_path, default_platform, default_uart)
            platform = default_platform
            cameras_config = []
            uart_config = dict(default_uart)
            ai_config = None
        else:
            with open(path) as f:
                cfg = yaml.safe_load(f)
            platform = cfg.get("platform", default_platform)
            cameras_config = cfg.get("cameras", [])
            uart_defaults = cfg.get("uart_defaults", {})
            uart_config = {
                "port": uart_defaults.get("port", default_uart["port"]),
                "baudrate": uart_defaults.get("baudrate", default_uart["baudrate"]),
            }
            ai_config = cfg.get("ai")
            cpu_affinity_cfg = cfg.get("cpu_affinity")
            configure_cpu_affinity(cpu_affinity_cfg)

        platform_mod = importlib.import_module(f"framework.hal.platforms.{platform}")
        bt.mark("import_platform")

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
            bt.mark(f"camera.open({cid})")
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
            )

        display = platform_mod.create_display()
        bt.mark("display")
        try:
            uart = platform_mod.create_uart(**uart_config)
        except Exception as e:
            logger.warning("UART creation failed (non-fatal): %s", e)
            uart = None
        bt.mark("uart")

        # 电量: 构造不碰硬件（Pro 的 PMU 句柄在第一次 read 时懒建）, 所以不打点。
        # 失败不致命: 没有电量计就退化成"不显示电量"。
        try:
            battery = platform_mod.create_battery()
        except Exception as e:
            logger.warning("Battery creation failed (non-fatal): %s", e)
            battery = None

        # --- AI initialization ---
        ai: Optional[AIInference] = None
        if ai_config is not None:
            try:
                ai = platform_mod.create_ai()
                bt.mark("ai.create")
                models_cfg = ai_config.get("models", [])
                for m in models_cfg:
                    ai.add(
                        nick_name=m["nick_name"],
                        model_path=m["model"],
                        model_type=m.get("model_type", "auto"),
                    )
                bt.mark("ai.register")
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
                # 单独一段: NPU 从 flash 加载 .cvimodel 是本函数里最可能
                # 出现"几百毫秒级"的一段, 且**完全可延后** —— 人质任务在
                # RESCUE 订阅才需要, 排爆/打靶阶段用不到。拆出来是为了让
                # "启动慢"能区分出是模型加载慢还是别的慢。
                bt.mark("ai.switch(load)")
            except Exception as e:
                logger.error("Failed to initialize AI: %s", e)
                ai = None

        bt.summary("Machine.create")
        return cls(camera_hub=hub, display=display, uart=uart, ai=ai, battery=battery)

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
