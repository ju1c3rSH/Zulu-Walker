from __future__ import annotations

import importlib
import time
from typing import Callable, Dict, Optional

from framework.hal import Machine


class ModuleManager:
    MAIN_LOOP_DELAY = 0.00333

    def __init__(self, machine: Machine, event_bus=None) -> None:
        self._machine = machine
        self._event_bus = event_bus
        self.modules: Dict[str, object] = {}
        self._loop_methods: Dict[str, callable] = {}
        self._running = True

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
            self.modules[name] = mod
            if hasattr(mod, "init"):
                mod.init(machine=self._machine, event_bus=self._event_bus)
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
                for name, loop_method in self._loop_methods.items():
                    try:
                        loop_method()
                    except Exception:
                        ...

                if coordinator:
                    coordinator.loop()
                    if tick_callback:
                        try:
                            tick_callback(coordinator)
                        except Exception:
                            ...

                if display_callback:
                    try:
                        display_callback()
                    except Exception:
                        ...

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
