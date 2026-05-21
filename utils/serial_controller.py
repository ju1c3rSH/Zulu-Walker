import os
import serial
import serial.tools.list_ports
import threading
import queue
import time
from typing import Optional, Callable

from .log_util import UARTModuleLogger



class SerialController:
    def __init__(self,  port: str = "/dev/ttyS4", baudrate: int = 921600):
        """
        @Brief: 初始化串口控制器

        @Args:
            port: 串口设备，如 /dev/ttyS4, /dev/ttyUSB0
            baudrate: 波特率

        """
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.is_connected = False
        self._receive_thread = None
        self._running = False
        self._callback = None

        # 异步发送队列
        self._send_queue: queue.Queue = None
        self._send_thread = None

    def _set_thread_affinity(self, cores):
        """设置当前线程的 CPU 亲和性"""
        try:
            os.sched_setaffinity(0, cores)
        except (AttributeError, OSError, PermissionError):
            pass  # Windows 或权限不足时忽略
        
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
            # 启动异步发送线程
            self._send_queue = queue.Queue()
            self._running = True
            self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
            self._send_thread.start()
            print(f"Connected to serial port successfully: {self.port} @ {self.baudrate}bps")
            return True
        except Exception as e:
            print(f"Error connecting to serial port: {e}")
            return False

    def disconnect(self):
        """@Brief: 断开当前串口连接"""
        self._running = False
        # 停止发送线程
        if self._send_queue:
            self._send_queue.put(None)  # 发送停止信号
        if self._send_thread:
            self._send_thread.join(timeout=2)
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
            #self.disconnect()
            return 0

    def send_bytes(self, data: bytes) -> int:
        """
        @Brief: 发送字节数据（非阻塞）

        将数据放入发送队列后立即返回，实际发送由后台线程完成。
        """
        if not self.is_connected or not self.serial:
            return 0

        self._send_queue.put(data)
        return len(data)

    def _write_sync(self, data: bytes) -> int:
        """实际写入串口（由后台线程调用）"""
        try:
            bytes_written = self.serial.write(data)
            # 不调用 flush()，让 OS 缓冲区处理，避免阻塞
            return bytes_written
        except Exception as e:
            print(f"发送失败: {e}")
            self.disconnect()
            return 0

    def _send_loop(self):
        """后台发送线程循环"""
        # 绑定到小核心 (RK3588: 0-3 是小核心 A55)
        self._set_thread_affinity([0, 1, 2, 3])

        while self._running:
            try:
                data = self._send_queue.get(timeout=0.1)
                if data is None:  # 停止信号
                    break
                self._write_sync(data)
            except queue.Empty:
                continue

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
            self.disconnect()
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
            self.disconnect()
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
            self.disconnect()
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
        # 绑定到小核心 (RK3588: 0-3 是小核心 A55)
        self._set_thread_affinity([0, 1, 2, 3])

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