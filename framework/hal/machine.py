from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Optional

import yaml

from framework.hal.camera_hub import CameraHub
from framework.hal.interface import AIInference, Battery, Display, Uart
try:
    from utils.cpu_affinity import configure as configure_cpu_affinity
except ImportError:
    def configure_cpu_affinity(cfg): pass
try:
    from utils.boot_timing import BootTimer
except ImportError:
    BootTimer = None

logger = logging.getLogger(__name__)


def _runtime_options(raw: Any) -> dict:
    """The accelerator block of the config, or an empty dict when absent.

    The block is read from ``ai.runtime`` and from ``ai.models[].runtime``.
    It is passed to the engine verbatim: the engine layer owns the
    vocabulary (providers, device, fallback), so the configuration loader
    stays a loader and does not grow a second copy of the schema.
    """
    if isinstance(raw, dict):
        return dict(raw)
    if raw is not None:
        logger.warning("ai.runtime must be a mapping, ignoring %r", raw)
    return {}


class Machine:
    def __init__(
        self,
        camera_hub: CameraHub,
        display: Display,
        uart: Uart,
        ai: Optional[AIInference] = None,
        exit_check: Optional[object] = None,
        sys_info: Optional[object] = None,
        platform: Optional[str] = None,
        battery: Optional[Battery] = None,
        canvas_factory: Optional[object] = None,
    ) -> None:
        self.camera_hub = camera_hub
        self.display = display
        self.uart = uart
        self.ai = ai
        self.platform = platform
        # Platform capability hooks (probed via getattr, never hard imports):
        self.exit_check = exit_check
        self.sys_info = sys_info
        # Turns a frame into a UI Canvas; None on a platform without one, in
        # which case the vision module simply renders no HUD.
        self.canvas_factory = canvas_factory
        # Optional fuel gauge: an instance on platforms with a PMU/sysfs gauge,
        # None elsewhere. Constructed in create(), so the main loop is untouched.
        self.battery = battery

    @classmethod
    def create(cls, config_path: str = "project_config.yaml") -> "Machine":
        """Build a Machine from config, marking each slow stage on stdout.

        Args:
            config_path: YAML config; missing file falls back to maixcam2 defaults.

        Returns:
            Fully assembled Machine.

        Context:
            Timing marks are observation-only; BootTimer absence is a no-op.
        """
        bt = BootTimer("machine") if BootTimer else None
        path = Path(config_path)
        display_enabled = True
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
            display_enabled = bool(cfg.get("display_enabled", True))
            cameras_config = cfg.get("cameras", [])
            uart_defaults = cfg.get("uart_defaults", {})
            uart_config = {
                "port": uart_defaults.get("port", "/dev/ttyS1"),
                "baudrate": uart_defaults.get("baudrate", 921600),
            }
            ai_config = cfg.get("ai")
            cpu_affinity_cfg = cfg.get("cpu_affinity")
            configure_cpu_affinity(cpu_affinity_cfg)

        if bt:
            bt.mark("config")
        platform_mod = importlib.import_module(f"framework.hal.platforms.{platform}")

        hub = CameraHub.init_instance(platform)

        # Optional platform capabilities: probe by name instead of branching
        # on the platform string (ARCH-06/MOD-02/ARCH-07).
        exit_factory = getattr(platform_mod, "create_exit_check", None)
        exit_check = exit_factory() if callable(exit_factory) else None
        source_resolver = getattr(platform_mod, "resolve_camera_source", None)
        canvas_factory = getattr(platform_mod, "create_canvas", None)
        if not callable(canvas_factory):
            canvas_factory = None
        sysinfo_factory = getattr(platform_mod, "create_sys_info", None)
        try:
            sys_info = sysinfo_factory() if callable(sysinfo_factory) else None
        except Exception as e:
            logger.warning("SysInfo unavailable: %s", e)
            sys_info = None

        for cam_cfg in cameras_config:
            cid = cam_cfg.get("camera_id", str(cam_cfg.get("source", "")))
            raw_source = cam_cfg["source"]
            if callable(source_resolver):
                source = source_resolver(raw_source)
                if source != raw_source:
                    logger.info("Camera '%s': source %s -> %s", cid, raw_source, source)
            else:
                source = raw_source
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
        if bt:
            bt.mark("cameras")

        # `display_enabled: false` is a hard "no LCD" switch: skipping creation
        # avoids the panel init cost and guarantees nothing downstream can
        # present (the sink group only exists when a display object does).
        display = platform_mod.create_display() if display_enabled else None
        if display is None:
            logger.info("Display disabled by config (display_enabled=false)")
        try:
            uart = platform_mod.create_uart(**uart_config)
        except Exception as e:
            logger.warning("UART creation failed (non-fatal): %s", e)
            uart = None

        # Battery, like display/uart, is initialized in create() (on Pro this
        # opens i2c-4, ~2.4 ms). Deferring it to the first read would put a cold
        # import into a main-loop tick. Platforms without create_battery keep
        # None: capability probing, no hard import.
        battery: Optional[Battery] = None
        battery_factory = getattr(platform_mod, "create_battery", None)
        if callable(battery_factory):
            try:
                battery = battery_factory()
            except Exception as e:
                logger.warning("Battery creation failed (non-fatal): %s", e)
                battery = None

        # --- AI initialization ---
        # Two paths.  An explicit engine in the config selects the kernel
        # driven backend, which owns the artifact and the manifest; without
        # an engine the platform keeps its own backend (the MaixCAM
        # platforms bind maix.nn directly and read the MUD file).
        ai: Optional[AIInference] = None
        if ai_config is not None:
            engine_id = str(ai_config.get("engine") or "").strip()
            # The accelerator is a deployment property, not an artifact
            # property: the same .onnx runs on a CPU laptop, a DirectML
            # desktop and a CUDA host, and only this block differs.
            runtime_options = _runtime_options(ai_config.get("runtime"))
            try:
                if engine_id:
                    ai = platform_mod.create_ai(
                        engine=engine_id,
                        switch_timeout_ms=int(
                            ai_config.get("switch_timeout_ms", 2000)
                        ),
                        frame=ai_config.get("frame"),
                        runtime_options=runtime_options,
                    )
                else:
                    ai = platform_mod.create_ai()
                models_cfg = ai_config.get("models", [])
                for m in models_cfg:
                    manifest_path = m.get("manifest")
                    model_runtime = _runtime_options(m.get("runtime"))
                    if manifest_path:
                        ai.add(
                            nick_name=m["nick_name"],
                            model_path=m["model"],
                            model_type=m.get("model_type", "auto"),
                            manifest_path=manifest_path,
                            runtime_options=model_runtime,
                        )
                    else:
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
                if engine_id:
                    # An explicitly configured engine that cannot be brought
                    # up is a startup failure.  A silent downgrade would run
                    # the system an order of magnitude slower with no visible
                    # cause, so the failure stays visible.
                    raise
                ai = None
        if bt:
            bt.summary("TOTAL")

        return cls(
            camera_hub=hub,
            display=display,
            uart=uart,
            ai=ai,
            exit_check=exit_check,
            sys_info=sys_info,
            platform=platform,
            battery=battery,
            canvas_factory=canvas_factory,
        )

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
