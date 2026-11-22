import queue
import sys
import os
import time
from utils.log_util import log_print


sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils import SerialController
from utils.log_util import UARTModuleLogger

u = UARTModuleLogger("SerialController")

class UartTest:
    def __init__(self, port: str = "/dev/ttyS1", baudrate: int = 921600):
        self.port = port
        self.baudrate = baudrate
        self.serial = SerialController(port, baudrate)
        self.test_results = []
        self.running = False
        self.connected = False
        self.last_test_time = 1
        self.test_interval = 5  # Periodic test interval in seconds

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

        # Check the connection state.
        if not self.serial.is_connected:
            if self.connected:
                self.connected = False
                u.log_warning("Serial connection lost")
            self._try_reconnect()
            return

        # Keep the tracked state in sync with the port.
        if not self.connected:
            self.connected = True
            u.log_info("Connection restored")

        import time
        current_time = time.time()

        # Run the periodic test when it is due.
        if current_time - self.last_test_time >= self.test_interval:
            self.last_test_time = current_time
            self.run_periodic_test()

    def _try_reconnect(self):
        import time
        if not self.running:
            return

        # Avoid overlapping reconnection attempts.
        if hasattr(self, '_reconnecting') and self._reconnecting:
            return

        self._reconnecting = True
        try:
            u.log_info("Attempting to reconnect to UART...")

            # Disconnect only when a live connection exists.
            if self.serial.is_connected:
                self.serial.disconnect()
                self.connected = False

            time.sleep(2)

            # Attempt to reconnect.
            if self.serial.connect():
                self.connected = True
                u.log_info("Reconnected to UART successfully")
            else:
                u.log_error("Failed to reconnect to UART")
        finally:
            self._reconnecting = False

    def run_periodic_test(self):
        """Run the periodic UART test."""
        u.log_info("=" * 50)
        u.log_info("Running periodic UART test...")

        # Exercise the loopback path when it is active.
        if self.loopback_testing:
            stats = self.get_loopback_stats()
            u.log_info(f"Loopback Test Stats - Sent: {stats['sent']}, "
                      f"Received: {stats['received']}, "
                      f"Matched: {stats['matched']}, "
                      f"Errors: {stats['errors']}, "
                      f"Avg Latency: {stats['avg_latency']:.3f}ms")
        else:
            # Otherwise just send a timestamped probe.
            test_msg = f"Periodic Test at {time.strftime('%H:%M:%S')}\r\n"
            bytes_sent = self.serial.send(test_msg)
            if bytes_sent > 0:
                u.log_info(f"Sent periodic test data: {bytes_sent} bytes")

        u.log_info("=" * 50)


    def start(self) -> bool:
        """
        Open the serial connection.

        Returns:
            True when the connection was established.
        """
        self.connected = self.serial.connect()
        self.running = self.connected
        if self.connected:
            u.log_info(f"{self.serial.port} started successfully at {self.baudrate} bps")
        else:
            u.log_error(f"Failed to start {self.serial.port}")

        return self.connected  # Return the connection state directly.
    def send_test_data(self, data: str = "Hello, UART!"):
        """
        Send test data over the serial port.

        Args:
            data: Payload to transmit.
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
        Receive one line of test data.

        Returns:
            The received line, or None when none is available.
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
        Send a string and wait for the same string to be echoed back.

        Args:
            test_string: Payload to send and expect in return.

        Returns:
            True when the echoed line matches, False otherwise.
        """
        if not self.serial.is_connected:
            u.log_error("Serial port not connected. Cannot run echo test.")
            return False

        u.log_info(f"\n=== Echo Test ===")
        u.log_info(f"Sending: {test_string}")

        # Send the probe.
        bytes_sent = self.serial.send(test_string + "\r\n")

        if bytes_sent == 0:
            u.log_error("Failed to send data")
            return False

        # Allow time for the echo to arrive.
        import time
        time.sleep(0.5)

        # Read the echoed line.
        response = self.serial.receive_line()

        if response:
            log_print(f"Received: {response}")
            if response == test_string:
                log_print("✓ Echo test passed!")
                return True
            else:
                log_print(f"✗ Echo test failed: Expected '{test_string}', got '{response}'")
                return False
        else:
            log_print("✗ Echo test failed: No response received")
            return False
    def stop(self):
            """Stop testing and disconnect."""
            if self.connected:
                self.serial.disconnect()
                self.connected = False
                self.running = False
                log_print("UART test stopped")


uart_test_instance = None

def init():
    """Initialize the module instance (called by ModuleManager)."""
    global uart_test_instance
    uart_test_instance = UartTest(port="/dev/ttyS1", baudrate=921600)
    uart_test_instance.init()

def start():
    """Start the module instance (called by ModuleManager)."""
    global uart_test_instance
    if uart_test_instance:
        return uart_test_instance.start()
    return False

def loop():
    """Run the module main loop (called by ModuleManager)."""
    global uart_test_instance
    if uart_test_instance:
        uart_test_instance.loop()

def stop():
    """Stop the module instance (called by ModuleManager)."""
    global uart_test_instance
    if uart_test_instance:
        uart_test_instance.stop()
