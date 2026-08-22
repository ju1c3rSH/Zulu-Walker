from __future__ import annotations

import importlib
import logging
import time
from typing import Callable, Dict, Optional

from framework.hal import Machine
from utils.log_util import log_print

logger = logging.getLogger(__name__)


def supports_platform(module_or_cls, platform: Optional[str]) -> bool:
    """Platform gating for optional modules (ARCH-03/05).

    A module may declare ``PLATFORMS = ("maixcam2", ...)``; absent or empty
    means "runs anywhere".
    """
    declared = getattr(module_or_cls, "PLATFORMS", None)
    return not declared or platform in declared


class ModuleManager:
    MAIN_LOOP_DELAY = 0.002

    def __init__(
        self,
        machine: Machine,
        event_bus=None,
        wdt_feed=None,
        exit_check: Optional[Callable[[], bool]] = None,
        **module_init_kwargs,
    ) -> None:
        self._machine = machine
        self._event_bus = event_bus
        self.modules: Dict[str, object] = {}
        self._loop_methods: Dict[str, callable] = {}
        self._running = True
        self._wdt_feed = wdt_feed or (lambda: None)
        self._wdt_count = 0
        self._exit_check = exit_check
        if wdt_feed is not None:
            module_init_kwargs.setdefault('wdt_feed', wdt_feed)
        self._module_init_kwargs = module_init_kwargs

    def register(self, name: str) -> bool:
        """Register and load a single module by name."""
        return self.load(name)

    def register_many(self, names: list[str]) -> None:
        """Register and load multiple modules by name."""
        for name in names:
            self.load(name)

    def load(self, name: str) -> bool:
        try:
            full_name = f"modules.{name}"
            mod = importlib.import_module(full_name)
            if not supports_platform(mod, getattr(self._machine, "platform", None)):
                logger.info(
                    "Module '%s' skipped on platform '%s' (declares PLATFORMS=%s)",
                    name,
                    getattr(self._machine, "platform", None),
                    getattr(mod, "PLATFORMS", None),
                )
                return True
            self.modules[name] = mod
            if hasattr(mod, "init"):
                mod.init(machine=self._machine, event_bus=self._event_bus, **self._module_init_kwargs)
            if hasattr(mod, "start"):
                mod.start()
            if hasattr(mod, "loop"):
                self._loop_methods[name] = mod.loop
            return True
        except Exception as e:
            import traceback
            traceback.print_exc()
            return False

    def get_module(self, name: str):
        return self.modules.get(name)

    def run_main_loop(self, coordinator=None, tick_callback=None, display_callback: Optional[Callable] = None) -> None:
        try:
            from utils.cpu_affinity import bind_current_thread
            bind_current_thread("main_loop")
        except ImportError:
            pass

        while self._running:
            try:
                self._wdt_feed()
                self._wdt_count += 1
                if self._wdt_count % 200 == 0:
                    log_print(f"[WDT] main feed #{self._wdt_count}")

                for name, loop_method in self._loop_methods.items():
                    try:
                        loop_method()
                    except Exception:
                        logger.exception("Module '%s' loop() failed", name)

                if coordinator:
                    try:
                        coordinator.loop()
                    except Exception:
                        now = time.monotonic()
                        if now - getattr(self, "_coord_err_last", 0.0) >= 1.0:
                            self._coord_err_last = now
                            logger.exception("coordinator.loop() failed")
                    if tick_callback:
                        try:
                            tick_callback(coordinator)
                        except Exception:
                            logger.exception("tick_callback() failed")

                if display_callback:
                    try:
                        display_callback()
                    except Exception:
                        logger.exception("display_callback() failed")

                # Platform-provided predicate (ARCH-06): the framework never
                # imports a concrete platform to learn how "exit" looks.
                if self._exit_check and self._exit_check():
                    break

                time.sleep(self.MAIN_LOOP_DELAY)
            except KeyboardInterrupt:
                break
            except Exception:
                import traceback
                traceback.print_exc()
                time.sleep(1.0)

        self.stop_all()

    def stop_all(self) -> None:
        self._running = False
        for name, mod in self.modules.items():
            if hasattr(mod, "stop"):
                try:
                    mod.stop()
                except Exception:
                    ...
        self._loop_methods.clear()
        if self._machine:
            self._machine.close()
