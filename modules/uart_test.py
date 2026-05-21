import queue
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import SerialController
from utils.log_util import UARTModuleLogger

u = UARTModuleLogger("SerialController")

class UartTest:
    def __init__(self, port: str = "/dev/ttyS4", baudrate: int = 921600):
        self.port = port
        self.baudrate = baudrate
        self.serial = SerialController(port, baudrate)
        self.test_results = []
        self.running = False
        self.connected = False
        self.last_test_time = 1
        self.test_interval = 5  # 定期测试间隔（秒）
        
        self.loopback_testing = False
        self.loopback_thread = None
        self.loopback_data_queue = queue.Queue()
        self.loopback_stats = {
            'sent': 0,
            'received': 0,
            'matched': 0,
            'errors': 0,
            'last_send_time': 0,
            'last_receive_time': 0,
            'avg_latency': 0
        }
        self.loopback_latencies = []    
    
    def init(self):
        u.log_info(f"UART Test module initializing...")
        u.log_info(f"Port: {self.port}, Baudrate: {self.baudrate}")
    
    def loop(self):
        if not self.running:
            return
        
        # 检查连接状态
        if not self.serial.is_connected:
            if self.connected:
                self.connected = False
                u.log_warning("Serial connection lost")
            self._try_reconnect()
            return
        
        # 确保连接状态同步
        if not self.connected:
            self.connected = True
            u.log_info("Connection restored")
    
        import time
        current_time = time.time()
        
        # 定期执行测试
        if current_time - self.last_test_time >= self.test_interval:
            self.last_test_time = current_time
            self.run_periodic_test()
                
    def _try_reconnect(self):
        import time
        if not self.running:
            return
        
        # 避免重复重连
        if hasattr(self, '_reconnecting') and self._reconnecting:
            return
        
        self._reconnecting = True
        try:
            u.log_info("Attempting to reconnect to UART...")
            
            # 只有在确实需要断开时才断开
            if self.serial.is_connected:
                self.serial.disconnect()
                self.connected = False
            
            time.sleep(2)
            
            # 尝试重新连接
            if self.serial.connect():
                self.connected = True
                u.log_info("Reconnected to UART successfully")
            else:
                u.log_error("Failed to reconnect to UART")
        finally:
            self._reconnecting = False

    def run_periodic_test(self):
        """执行周期性测试"""
        u.log_info("=" * 50)
        u.log_info("Running periodic UART test...")
        
        # 运行自回传测试
        if self.loopback_testing:
            stats = self.get_loopback_stats()
            u.log_info(f"Loopback Test Stats - Sent: {stats['sent']}, "
                      f"Received: {stats['received']}, "
                      f"Matched: {stats['matched']}, "
                      f"Errors: {stats['errors']}, "
                      f"Avg Latency: {stats['avg_latency']:.3f}ms")
        else:
            # 简单发送测试
            test_msg = f"Periodic Test at {time.strftime('%H:%M:%S')}\r\n"
            bytes_sent = self.serial.send(test_msg)
            if bytes_sent > 0:
                u.log_info(f"Sent periodic test data: {bytes_sent} bytes")
        
        u.log_info("=" * 50)

    
    def start(self) -> bool:
        """
        启动串口连接
        
        Returns:
            bool: 连接是否成功
        """
        self.connected = self.serial.connect()
        self.running = self.connected
        if self.connected:
            u.log_info(f"{self.serial.port} started successfully at {self.baudrate} bps")
        else:
            u.log_error(f"Failed to start {self.serial.port}")
        
        return self.connected  # 直接返回连接状态
    def send_test_data(self, data: str = "Hello, UART!"):
        """
        发送测试数据
        
        Args:
            data: 要发送的测试数据，默认值："Hello, UART!"
        """
        if self.serial.is_connected:
            bytes_sent = self.serial.send(data)
            if bytes_sent > 0:
                u.log_info(f"Sent {bytes_sent} bytes: {data}")
            else:
                u.log_error("Failed to send data")
        else:
            u.log_error("Serial port not connected. Cannot send data.")
            
    def receive_line(self) -> str:
        """
        接收一行测试数据
        
        Returns:
            str: 接收到的行数据
        """
        if not self.serial.is_connected:
            u.log_error("Serial port not connected. Cannot receive data.")
            return None
        
        line = self.serial.receive_line()
        if line:
            u.log_info(f"Received line: {line}")
            return line
        else:
            u.log_info("No line received")
            return None
    
    def echo_test(self, test_string: str = "Echo Test") -> bool:
        """
        回环测试：发送数据并等待接收相同的响应
        
        Args:
            test_string: 测试字符串
            
        Returns:
            bool: 测试是否通过
        """
        if not self.serial.is_connected:
            u.log_error("Serial port not connected. Cannot run echo test.")
            return False
        
        u.log_info(f"\n=== Echo Test ===")
        u.log_info(f"Sending: {test_string}")
        
        # 发送数据
        bytes_sent = self.serial.send(test_string + "\r\n")
        
        if bytes_sent == 0:
            u.log_error("Failed to send data")
            return False
        
        # 等待响应
        import time
        time.sleep(0.5)
        
        # 接收响应
        response = self.serial.receive_line()
        
        if response:
            print(f"Received: {response}")
            if response == test_string:
                print("✓ Echo test passed!")
                return True
            else:
                print(f"✗ Echo test failed: Expected '{test_string}', got '{response}'")
                return False
        else:
            print("✗ Echo test failed: No response received")
            return False
    def stop(self):
            """停止并断开连接"""
            if self.connected:
                self.serial.disconnect()
                self.connected = False
                self.running = False
                print("UART test stopped")
                
                
uart_test_instance = None

def init():
    """模块初始化函数（供ModuleManager调用）"""
    global uart_test_instance
    uart_test_instance = UartTest(port="/dev/ttyS4", baudrate=921600)
    uart_test_instance.init()

def start():
    """模块启动函数（供ModuleManager调用）"""
    global uart_test_instance
    if uart_test_instance:
        return uart_test_instance.start()
    return False

def loop():
    """模块主循环函数（供ModuleManager调用）"""
    global uart_test_instance
    if uart_test_instance:
        uart_test_instance.loop()

def stop():
    """模块停止函数（供ModuleManager调用）"""
    global uart_test_instance
    if uart_test_instance:
        uart_test_instance.stop()

                