# -*- coding: utf-8 -*-
"""Coordinator QR one-shot event semantics + min_interval throttling logic."""
import time

from app.coordinator import EodCoordinator
from modules.zw_uart_module.protocol import (
    build_data_stream_frame, parse_frame,
    DEV_FLAG_FOUND,
)


def make_coordinator():
    class _Bus:
        def subscribe(self, *a, **k):
            pass

    return EodCoordinator(_Bus())


def _drain(coordinator, send_log):
    coordinator._uart_sender = send_log.append
    coordinator.loop()


def test_qr_pending_arms_on_request_and_sends_once():
    coordinator = make_coordinator()
    coordinator._latest_qr = {"decoded": True, "code": "312",
                              "code_domain_valid": True, "latched": True,
                              "changed": True}

    class _Evt:
        data_type = 0x01
        min_interval_ms = 0
        reserved = 0

    coordinator._on_cmd_request(_Evt())
    assert coordinator._qr_pending

    sent = []
    _drain(coordinator, sent)
    assert len(sent) == 1
    frame = parse_frame(sent[0])
    assert frame is not None and frame.frame_type == 0x24
    assert frame.payload[1] == 0x01
    body = frame.payload[2:]
    assert body[0] == DEV_FLAG_FOUND and body[1] == 3
    assert body[2:5] == b"312"
    assert not coordinator._qr_pending

    # silence afterwards: same code must not push twice (content dedup)
    sent.clear()
    _drain(coordinator, sent)
    assert sent == []


def test_qr_silent_without_latched_decode():
    coordinator = make_coordinator()
    coordinator._latest_qr = {"decoded": True, "code": "312",
                              "code_domain_valid": True, "latched": False,
                              "changed": True}

    class _Evt:
        data_type = 0x01
        min_interval_ms = 0
        reserved = 0

    coordinator._on_cmd_request(_Evt())
    sent = []
    _drain(coordinator, sent)
    assert sent == []


def test_qr_invalid_domain_never_sent():
    coordinator = make_coordinator()
    coordinator._latest_qr = {"decoded": True, "code": "999",
                              "code_domain_valid": False, "latched": True,
                              "changed": True}

    class _Evt:
        data_type = 0x01
        min_interval_ms = 0
        reserved = 0

    coordinator._on_cmd_request(_Evt())
    sent = []
    _drain(coordinator, sent)
    assert sent == []


def test_device_stream_uses_align_cache_and_zero_when_unfound():
    coordinator = make_coordinator()
    coordinator._latest_eod = {"mode": "BOMB", "error_x": 1234, "error_y": -567,
                               "id": 2, "flags": DEV_FLAG_FOUND}

    class _Evt:
        data_type = 0x02
        min_interval_ms = 0
        reserved = 2

    coordinator._on_cmd_request(_Evt())
    sent = []
    _drain(coordinator, sent)
    frame = parse_frame(sent[0])
    sub = frame.payload[2:]
    assert int.from_bytes(sub[0:2], "little", signed=True) == 1234
    assert int.from_bytes(sub[2:4], "little", signed=True) == -567
    assert sub[4] == 2

    # flags=0 (no target) -> error fields MUST be zero on the wire
    coordinator._latest_eod = {"mode": "BOMB", "error_x": 1234, "error_y": -567,
                               "id": 2, "flags": 0}
    sent.clear()
    _drain(coordinator, sent)
    frame = parse_frame(sent[0])
    assert frame.payload[2:] == b"\x00" * 6


def test_min_interval_throttles_streaming():
    coordinator = make_coordinator()
    coordinator._latest_eod = {"mode": "BOMB", "error_x": 0, "error_y": 0,
                               "id": 1, "flags": DEV_FLAG_FOUND}

    class _Evt:
        data_type = 0x02
        min_interval_ms = 20
        reserved = 1

    coordinator._on_cmd_request(_Evt())

    sent = []
    def fake_send(frame):
        sent.append(frame)
        return True
    coordinator._uart_sender = fake_send
    coordinator.loop()          # first frame allowed (last_send reset on request)
    assert len(sent) == 1
    coordinator.loop()          # 0ms later -> throttled
    coordinator.loop()          # still throttled
    assert len(sent) == 1
    time.sleep(0.025)           # > 20ms
    coordinator.loop()
    assert len(sent) == 2


def test_stop_resets_qr_and_gates_idle():
    coordinator = make_coordinator()
    coordinator._qr_pending = True

    class _Evt:
        pass

    coordinator._on_cmd_stop(_Evt())
    assert coordinator._qr_pending is False
    assert coordinator._streaming_type == 0
