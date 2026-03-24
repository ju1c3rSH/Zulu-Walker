#main.py
import gc
import sys
import time
import importlib
from typing import Dict, Any

class ModuleManager:
    def __init__(self):
        self.modules = {}
        self.running = True
    
    def load_module(self,module_name):
        try:
            module = __import__(module_name)
            self.modules[module_name] = module
            if hasattr(module, 'init'):
                module.init()
            if SystemConfig.DEBUG:
                print(f"Module '{module_name}' loaded successfully")
        except Exception as e:
            print(f"Failed to load module '{module_name}': {e}")
    def start_all(self):
        """启动所有模块"""
        for module_name in SystemConfig.AUTO_START_MODULES:
            if self.load_module(module_name):
                # 如果模块有start方法，调用它
                module = self.modules.get(module_name)
                if module and hasattr(module, 'start'):
                    try:
                        module.start()
                    except Exception as e:
                        print(f"Failed to start {module_name}: {e}")           
    def stop_all(self):
        """停止所有模块"""
        self.runnning = False
        for module_name, module in self.modules.items():
            if hasattr(module, 'stop'):
                try:
                    module.stop()
                except Exception as e:
                    print(f"Failed to stop {module_name}: {e}")
    def run_main_loop(self):
        """主循环，定期调用模块的loop方法"""

        print("Entering main loop...")
        while self.running:
            try:
                for module_name, module in self.modules.items():
                        if hasattr(module, 'loop'):
                            try:
                                module.loop()
                            except Exception as e:
                                print(f"Error in {module_name} loop: {e}")
                gc.collect()
                time.sleep(SystemConfig.MAIN_LOOP_DELAY)
            except KeyboardInterrupt:
                print("Program interrupted")
                self.stop()
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
            finally:
                self.stop_all()    
        
class SystemConfig:
    DEBUG = True
    
    WATCHDOG_TIMEOUT = 60
    
    MAIN_LOOP_DELAY = 0.1
    AUTO_START_MODULES = [
        'uart_test',
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