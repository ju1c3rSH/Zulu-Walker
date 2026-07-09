#main.py
import gc
import os
import sys
import time
import importlib
from typing import Dict, Any
# 添加项目根目录到 sys.path，确保模块只被加载一次
sys.path.insert(0, os.path.dirname(__file__))

class ModuleManager:
    def __init__(self):
        self.modules = {}
        self.running = True
        self._loop_methods = {}

    def load_module(self,module_name):
        try:
            # 使用 modules.{name} 路径加载，与绝对导入一致
            full_name = f'modules.{module_name}'
            module = __import__(full_name, fromlist=[module_name])
            self.modules[module_name] = module
            if hasattr(module, 'init'):
                module.init()
            if SystemConfig.DEBUG:
                print(f"Module '{module_name}' loaded successfully")
            return True
        except Exception as e:
            print(f"Failed to load module '{module_name}': {e}")
            return False

    def start_all(self):
        """启动所有模块
        检查模块是否有Loop和Start方法，分别调用，并将Loop方法注册到主循环调用列表
        """
        for module_name in SystemConfig.AUTO_START_MODULES:
            if self.load_module(module_name):
                module = self.modules.get(module_name)
                
                if module and hasattr(module, 'start'):
                    try:
                        module.start()
                    except Exception as e:
                        print(f"Failed to start {module_name}: {e}")
                
                if module and hasattr(module, 'loop'):
                    self._loop_methods[module_name] = module.loop
                    
    def stop_all(self):
        """停止所有模块，并且清除所有的loop方法"""
        self.running = False
        for module_name, module in self.modules.items():
            if hasattr(module, 'stop'):
                try:
                    module.stop()
                except Exception as e:
                    print(f"Failed to stop {module_name}: {e}")
        self._loop_methods.clear()

    def run_main_loop(self, coordinator=None):
        """主循环，定期调用模块的loop方法"""

        print("Entering main loop...")
        try:
            while self.running:
                try:
                    #gc.collect()
                    #这里先移除了gc，排查是不是gc引起的性能问题，后续再优化
                    for module_name, loop_method in self._loop_methods.items():
                        try:
                            loop_method()
                        except Exception as e:
                            print(f"Error in {module_name} loop: {e}")

                    if coordinator:
                        coordinator.loop()

                    time.sleep(SystemConfig.MAIN_LOOP_DELAY)
                except KeyboardInterrupt:
                    print("Program interrupted")
                    break
                except Exception as e:
                    print(f"Error in main loop: {e}")
        finally:
            self.stop_all()    
        
        
class SystemConfig:
    DEBUG = True
    
    WATCHDOG_TIMEOUT = 60
    
    MAIN_LOOP_DELAY = 0.01
    AUTO_START_MODULES = [
        #'uart_test',
        'zw_opencv_module',
        # 'zw_uart_module' -- loaded manually in main() with event_bus injection
    ]


            
def main():
    """主入口"""
    print("0xfb709394")

    # 创建中枢层：Coordinator ← EventBus ← 各模块
    from context import EventBus, MissionCoordinator
    bus = EventBus()
    coordinator = MissionCoordinator(bus)

    # 注入 EventBus 到各模块（必须在 start_all() 之前）
    import modules.zw_opencv_module as opencv_mod
    import modules.zw_uart_module as uart_mod
    opencv_mod.init(event_bus=bus)
    uart_mod.init(event_bus=bus)

    # 启动模块（UART 不在 AUTO_START_MODULES 里，手动启动）
    manager = ModuleManager()
    manager.start_all()
    uart_mod.start()

    # 桥接：Coordinator ↔ 各模块（必须在 start 之后，拿到实例引用）
    from modules.zw_opencv_module import get_camera_manager
    from modules.zw_uart_module import get_interface

    cm = get_camera_manager()
    if cm:
        coordinator.connect_camera(cm)

    uart = get_interface()
    if uart:
        coordinator.set_uart_sender(uart.send_raw)

    coordinator.start()
    manager.run_main_loop(coordinator)
    
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)