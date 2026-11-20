from __future__ import annotations

import importlib
import logging
import time
from typing import Callable, Dict, Optional

from framework.hal import Machine
from utils.boot_timing import thread_cpu as _thread_cpu
from utils.log_util import log_print, perf_interval_s

logger = logging.getLogger(__name__)


class ModuleManager:
    MAIN_LOOP_DELAY = 0.002

    def __init__(self, machine: Machine, event_bus=None, wdt_feed=None, **module_init_kwargs) -> None:
        self._machine = machine
        self._event_bus = event_bus
        self.modules: Dict[str, object] = {}
        self._loop_methods: Dict[str, callable] = {}
        self._running = True
        self._wdt_feed = wdt_feed or (lambda: None)
        self._wdt_count = 0
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
            # traceback 走 stderr, 设备控制台只显示 stdout —— 失败必须留一行看得见的
            log_print(f"[MODULE] {name} 加载失败: {type(e).__name__}: {e}")
            return False
    def get_module(self, name: str):
        return self.modules.get(name)

    def run_main_loop(self, coordinator=None, tick_callback=None, display_callback: Optional[Callable] = None) -> None:
        try:
            from utils.cpu_affinity import bind_current_thread
            bind_current_thread("main_loop")
        except ImportError:
            pass

        _CHECK_EXIT = False
        try:
            from maix import app as _maix_app
            _CHECK_EXIT = True
        except ImportError:
            pass

        # 主循环分项打点（每 `perf_interval_s()` 秒一条 [PERF], 默认 10s）。
        # 为什么需要: 主循环以 MAIN_LOOP_DELAY=2ms 空转 = 每秒 500 次, 每次都要走
        # coordinator.loop() + tick_callback(DebugConsole.set ×6) +
        # display_callback(touch.read() 硬件读)。这些是 Python 侧工作, 累加后可能
        # 与视觉线程抢 GIL —— 差额必须能逐线程、逐子步骤归因。
        # 同时报 body(墙上) 与 cpu(thread_time): 两者之差就是等 GIL 的时间。
        #   只看 body 会把"纯等待"误读成"这一路在吃 CPU"。
        _perf = {
            "n": 0, "t0": time.perf_counter(), "cpu0": _thread_cpu(),
            "body": 0.0, "cpu": 0.0,
            "wdt": 0.0, "mods": 0.0, "coord": 0.0, "tick": 0.0, "disp": 0.0,
        }

        while self._running:
            try:
                _body0, _cpu0 = time.perf_counter(), _thread_cpu()

                _t = time.perf_counter()
                self._wdt_feed()
                self._wdt_count += 1
                if self._wdt_count % 600 == 0:
                    log_print(f"[WDT] main feed #{self._wdt_count}")
                _perf["wdt"] += time.perf_counter() - _t

                _t = time.perf_counter()
                for name, loop_method in self._loop_methods.items():
                    try:
                        loop_method()
                    except Exception:
                        logger.exception("Module '%s' loop() failed", name)
                _perf["mods"] += time.perf_counter() - _t

                _t = time.perf_counter()
                if coordinator:
                    coordinator.loop()
                    _c = time.perf_counter()
                    _perf["coord"] += _c - _t
                    if tick_callback:
                        try:
                            tick_callback(coordinator)
                        except Exception:
                            logger.exception("tick_callback() failed")
                    _perf["tick"] += time.perf_counter() - _c

                _t = time.perf_counter()
                if display_callback:
                    try:
                        display_callback()
                    except Exception:
                        logger.exception("display_callback() failed")
                _perf["disp"] += time.perf_counter() - _t

                if _CHECK_EXIT and _maix_app.need_exit():
                    break

                _perf["body"] += time.perf_counter() - _body0
                _perf["cpu"] += _thread_cpu() - _cpu0
                _perf["n"] += 1
                _el = time.perf_counter() - _perf["t0"]
                if _el >= perf_interval_s():
                    n = max(1, _perf["n"])
                    body_ms = _perf["body"] / n * 1000.0
                    cpu_ms = _perf["cpu"] / n * 1000.0
                    log_print(
                        f"[PERF] mainloop {n / _el:.0f}Hz body={body_ms:.2f}ms "
                        f"cpu={cpu_ms:.2f}ms gil_wait={body_ms - cpu_ms:.2f}ms | "
                        f"wdt={_perf['wdt'] / n * 1000:.2f} "
                        f"coord={_perf['coord'] / n * 1000:.2f} "
                        f"tick={_perf['tick'] / n * 1000:.2f} "
                        f"touch={_perf['disp'] / n * 1000:.2f} | "
                        f"每秒 CPU {_perf['cpu'] / _el * 1000:.0f}ms")
                    _perf.update(n=0, t0=time.perf_counter(),
                                 cpu0=_thread_cpu(), body=0.0, cpu=0.0,
                                 wdt=0.0, mods=0.0, coord=0.0, tick=0.0, disp=0.0)

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
