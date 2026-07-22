# -*- coding: utf-8 -*-
"""
zw_uart_module - Custom exceptions

Defines module-specific exceptions for UART communication errors.
"""


class UartError(Exception):
    """Base exception for UART module errors."""
    pass


class InvalidFrameError(UartError):
    """Raised when a frame has invalid format or structure."""
    pass


class ChecksumError(InvalidFrameError):
    """Raised when frame checksum validation fails."""
    pass


class ParameterError(UartError):
    """Raised when invalid parameters are provided."""
    pass
