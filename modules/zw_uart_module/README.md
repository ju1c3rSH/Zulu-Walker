# zw_uart_module

UART communication module for STM32 microcontroller communication on Orange Pi.

## Purpose

Provides binary frame-based UART communication for:
- Receiving zone events from STM32 (ARRIVED, PICK, SET)
- Sending visual error feedback to STM32

## Design Principle

**Passive State Maintenance**: The module maintains internal state through state machine parsing of STM32 events. Other modules query state through interfaces - no active callbacks to other modules. This achieves loose coupling.

## Dependencies

- `pyserial` - Serial port communication
- `logging` - Standard library
- `threading` - Standard library

## Usage

### As a Module (via ModuleManager)

Add to `AUTO_START_MODULES` in `main.py`:

```python
AUTO_START_MODULES = [
    'zw_uart_module',
    # ... other modules
]
```

Query state from other modules:

```python
from modules.zw_uart_module import get_current_zone, get_last_arrived_zone, send_error

# Get current zone
zone = get_current_zone()

# Send visual error
send_error(0, -3)  # X direction error, value -3
```

### Direct Usage

```python
from zw_uart_module import STM32UartInterface

with STM32UartInterface("/dev/ttyS4", baudrate=115200) as uart:
    uart.send_error(0, -3)  # Send error frame

    # Query state in main loop
    while True:
        current = uart.get_current_zone()
        arrived = uart.get_last_arrived_zone()
        pick = uart.get_last_pick_zone()
        print(f"Zone: cur={current}, arrived={arrived}, pick={pick}")
        time.sleep(0.1)
```

## Protocol

### Frame Structure

| Field | Size | Description |
|-------|------|-------------|
| SOF | 1 | Start of Frame, fixed 0xAA |
| Length | 1 | Bytes from Type to Checksum |
| Type | 1 | Frame type identifier |
| Payload | 0~252 | Variable length data |
| Checksum | 1 | XOR of Type to Payload bytes |

### Frame Types

| Type | Name | Direction | Payload |
|------|------|-----------|---------|
| 0x01 | EVENT_ERROR | Orange Pi → STM32 | error_type(1B) + error_value(2B, int16 LE) |
| 0x02 | EVENT_ARRIVED_AT_ZONE | STM32 → Orange Pi | zone_id(1B) |
| 0x03 | EVENT_PICK_AT_ZONE | STM32 → Orange Pi | zone_id(1B) |
| 0x04 | EVENT_SET_ZONE | STM32 → Orange Pi | zone_id(1B) |

### Error Types

| Value | Description |
|-------|-------------|
| 0 | X direction error |
| 1 | Y direction error |
| 2 | Z direction error |
| 3 | Other error |

### Example Frames

```
# SET_ZONE for zone 5
AA 03 04 05 01
# SOF=AA, Len=03, Type=04, Payload=05, Checksum=01
# Length = Type(1) + Payload(1) + Checksum(1) = 3

# ERROR frame: type=0 (X), value=-3
AA 05 01 00 FD FF 03
# SOF=AA, Len=05, Type=01, error_type=00, error_value=FDFF(-3), Checksum=03
# Length = Type(1) + Payload(3) + Checksum(1) = 5
```

## API Reference

### STM32UartInterface

| Method | Description |
|--------|-------------|
| `start() -> bool` | Connect and start receiver |
| `stop()` | Disconnect and cleanup |
| `get_current_zone() -> int` | Get current zone ID |
| `get_last_arrived_zone() -> int` | Get last arrived zone ID |
| `get_last_pick_zone() -> int` | Get last pick zone ID |
| `send_error(type, value) -> bool` | Send error frame |
| `set_log_level(level)` | Set logging level |
| `set_debug_hex(enabled)` | Enable hex dump logging |

### Module Functions

| Function | Description |
|----------|-------------|
| `init()` | Module initialization |
| `start() -> bool` | Module start |
| `loop()` | Module main loop (no-op) |
| `stop()` | Module stop |
| `get_interface() -> STM32UartInterface` | Get interface instance |
| `get_current_zone() -> int` | Convenience function |
| `get_last_arrived_zone() -> int` | Convenience function |
| `get_last_pick_zone() -> int` | Convenience function |
| `send_error(type, value) -> bool` | Convenience function |

## Files

```
zw_uart_module/
├── __init__.py      # Module interface
├── uart_driver.py   # STM32UartInterface, FrameParser
├── protocol.py      # Protocol constants, frame building/parsing
├── exceptions.py    # Custom exceptions
└── README.md        # This file
```
