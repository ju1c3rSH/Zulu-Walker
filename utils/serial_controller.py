import serial
import serial.tools.list_ports
import threading
import queue
import time
from typing import Optional, Callable

from .log_util import UARTModuleLogger



class SerialController:
    def __init__(self,  port: str = "/dev/ttyS0", baudrate: int = 115200):
        """
        @Brief: 初始化串口控制器
        
        @Args:
            port: 串口设备，如 /dev/ttyS0, /dev/ttyUSB0
            baudrate: 波特率
            
        """
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_connected = False
        self._receive_thread = None
        self._running = False
        self._callback = None
        
    def connect(self) -> bool:
        try:
            self.serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=8,
                parity='N',
                stopbits=1,
                timeout=1.0,
                write_timeout=1.0
            )
            self.is_connected = True
            print(f"Connected to serial port successfully: {self.port} @ {self.baudrate}bps")
            return True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return False

    def disconnect(self):
        """@Brief: 断开当前串口连接"""
        self._running = False
        if self._receive_thread:
            self._receive_thread.join(timeout=2)
        if self.serial and self.serial.is_open:
            self.serial.close()
        self.is_connected = False
        print("串口已断开")

    def send(self, data: str) -> int:
        """
        @Brief: 发送数据
        
        Args:
            data: 要发送的字符串
            
        Returns:
            发送的字节数
        """
        if not self.is_connected or not self.serial:
            return 0
        
        try:
            bytes_written = self.serial.write(data.encode('utf-8'))
            self.serial.flush()
            return bytes_written
        except Exception as e:
            print(f"发送失败: {e}")
            return 0

    def send_bytes(self, data: bytes) -> int:
        """@Brief: 发送字节数据"""
        if not self.is_connected or not self.serial:
            return 0
        
        try:
            bytes_written = self.serial.write(data)
            self.serial.flush()
            return bytes_written
        except Exception as e:
            print(f"发送失败: {e}")
            return 0

    def receive(self, size: int = 1) -> Optional[bytes]:
        """
        @Brief: 接收数据（阻塞）
        
        Args:
            size: 要接收的字节数
            
        Returns:
            接收到的数据，超时返回 None
        """
        if not self.is_connected or not self.serial:
            return None
        
        try:
            data = self.serial.read(size)
            return data if data else None
        except Exception as e:
            print(f"接收失败: {e}")
            return None

    def receive_line(self) -> Optional[str]:
        """@Brief: 接收一行数据（直到换行符）"""
        if not self.is_connected or not self.serial:
            return None
        
        try:
            line = self.serial.readline()
            return line.decode('utf-8', errors='ignore').strip() if line else None
        except Exception as e:
            print(f"接收行失败: {e}")
            return None

    def receive_all(self) -> Optional[bytes]:
        """@Brief: 接收所有可用数据（非阻塞）"""
        if not self.is_connected or not self.serial:
            return None
        
        try:
            waiting = self.serial.in_waiting
            if waiting > 0:
                return self.serial.read(waiting)
            return None
        except Exception as e:
            print(f"接收失败: {e}")
            return None
        
    def start_receiver(self, callback: Callable[[bytes], None]):
        """
        启动异步接收（后台线程）
        
        Args:
            callback: 接收到数据时的回调函数
        """
        self._callback = callback
        self._running = True
        self._receive_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._receive_thread.start()
        print("异步接收已启动")
        
    def _receive_loop(self):
        """接收循环（内部方法）"""
        while self._running and self.is_connected:
            try:
                data = self.receive_all()
                if data and self._callback:
                    self._callback(data)
                time.sleep(0.01)
            except Exception as e:
                print(f"接收循环错误: {e}")
                time.sleep(0.1)
    
    def stop_receiver(self):
        """停止异步接收"""
        self._running = False
        if self._receive_thread:
            self._receive_thread.join(timeout=2)
        print("异步接收已停止")
    

    def clear_buffer(self):
        """清空缓冲区"""
        if self.serial:
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
    
    @staticmethod
    def list_ports():
        """列出所有可用串口"""
        ports = []
        for port in serial.tools.list_ports.comports():
            ports.append({
                'device': port.device,
                'description': port.description,
                'manufacturer': port.manufacturer
            })
        return ports

        """数据接收callback:
    def on_data_received(data: bytes):
        try:
            text = data.decode('utf-8', errors='ignore')
            print(f"收到: {text}")
        except:
            print(f"收到原始数据: {data.hex()}")
            
            
            
        """