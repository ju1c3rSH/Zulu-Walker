"""Boot-time WiFi bring-up (ARCH-03: extracted from app/main.py).

Project-specific feature with a declared platform scope: ``PLATFORMS``
states where this works, and callers gate through
``framework.module_manager.supports_platform`` instead of relying on
ImportError side effects.
"""

from __future__ import annotations

from utils.log_util import log_print

PLATFORMS = ("maixcam2",)


def init(cfg: dict) -> None:
    """Start AP or report STA state per ``cfg['streaming']``. Boot-step only."""
    streaming = cfg.get("streaming", {})
    mode = streaming.get("wifi_mode", "off")
    if mode not in ("ap", "sta"):
        return None
    try:
        from maix.network import wifi as _mw

        w = _mw.Wifi()
        if mode == "ap":
            if w.is_connected():
                log_print("[WiFi] Disconnecting existing STA before AP...")
                e = w.disconnect()
                if e != 0:
                    log_print(f"[WiFi] Disconnect failed, err={e}")
            # Credentials must come from project_config.yaml - no hardcoded
            # fallback (a missing key now skips AP start instead of bringing
            # up a well-known-password network).
            ssid = streaming.get("ap_ssid")
            password = streaming.get("ap_password")
            if not ssid or not password:
                log_print("[WiFi] ap_ssid/ap_password missing in config, skip AP start")
                return None
            e = w.start_ap(ssid, password, ip="192.168.1.1")
            if e != 0:
                log_print(f"[WiFi] AP start failed, err={e}")
                return None
            import time

            time.sleep(0.5)
            if not w.is_ap_mode():
                log_print("[WiFi] AP mode not confirmed after start_ap")
                return None
            ip = w.get_ip()
            log_print(f"[WiFi] AP mode started, SSID={ssid}, IP={ip}")
            return ip
        elif mode == "sta":
            ip = w.get_ip()
            if ip:
                log_print(f"[WiFi] STA mode, system-managed, IP={ip}")
            else:
                log_print("[WiFi] STA mode, no IP yet (waiting for system connection)")
            return ip
    except Exception as e:
        log_print(f"[WiFi] init FAIL: {e}")
        return None
