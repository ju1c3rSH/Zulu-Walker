from __future__ import annotations

import importlib
import time
from typing import Dict, Optional

from hal import Machine


class ModuleManager:
    MAIN_LOOP_DELAY = 0.00333

    def __init__(self, machine: Machine, event_bus=None) -> None:
        self._machine = machine
        self._event_bus = event_bus
        self.modules: Dict[str, object] = {}
        self._loop_methods: Dict[str, callable] = {}
        self._running = True
        self._auto_start = [
            "zw_opencv_module",
            "zw_uart_module",
        ]

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

    def start_all(self) -> None:
        for name in self._auto_start:
            self.load(name)

    def run_main_loop(self, coordinator=None, tick_callback=None) -> None:
        from utils.cpu_affinity import bind_current_thread
        bind_current_thread("main_loop")

        while self._running:
            try:
                for name, loop_method in self._loop_methods.items():
                    try:
                        loop_method()
                    except Exception as e:
                        ...

                if coordinator:
                    coordinator.loop()
                    if tick_callback:
                        try:
                            tick_callback(coordinator)
                        except Exception:
                            ...

                if self._machine and self._machine.display:
                    vision_mod = self.modules.get("zw_opencv_module")
                    if vision_mod:
                        vm = getattr(vision_mod, "get_vision_manager", lambda: None)()
                        if vm:
                            frame = vm.compose_frame()
                            if frame is not None:
                                if not self._machine.display.show(frame):
                                    self._running = False

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
