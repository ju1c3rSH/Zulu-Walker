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

    def run_main_loop(self):
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
        'zw_uart_module',
    ]


            
def main():
    """主入口"""
    print("0xfb709394")
    manager = ModuleManager()
    manager.start_all()
    manager.run_main_loop()
    
    
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)