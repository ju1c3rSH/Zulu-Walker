from typing import Optional

from .event_bus import EventBus
from .mission_state_machine import (
    MissionStateMachine, MissionContext,
    MissionState, Zone,
)
from modules.zw_uart_module.protocol import (
    build_status_from_vision_frame, build_visual_servo_data_frame,
    build_qr_result_frame, build_color_result_frame,
    build_heartbeat_frame, build_emergency_stop_frame,
    CMD_START_QR, CMD_START_COLOR_DETECT,
    CMD_TRACK_TARGET, CMD_TRACK_RING, CMD_TRACK_TOP,
    CMD_STOP_VISUAL,
    TYPE_CMD_FROM_MCU, TYPE_ACTION_DONE,
)
from modules.zw_opencv_module.models.cargo import CargoSet


class MissionCoordinator:

    def __init__(self):
        self.event_bus = EventBus()
        self.mission_sm = MissionStateMachine()
        self.cargo_set = CargoSet.create_standard()
        self._uart_sender: Optional[callable] = None

    def set_uart_sender(self, sender: callable) -> None:
        self._uart_sender = sender

    def _send(self, frame: bytes) -> bool:
        if self._uart_sender:
            return self._uart_sender(frame)
        return False

    def on_mcu_cmd(self, cmd_id: int, args: bytes) -> None:
        if cmd_id == CMD_START_QR:
            self.mission_sm.start()
            self._send(build_heartbeat_frame(
                0, self.mission_sm.current_state_id, 0))

        elif cmd_id == CMD_START_COLOR_DETECT:
            pass

        elif cmd_id == CMD_TRACK_TARGET:
            if len(args) >= 1:
                pass

        elif cmd_id == CMD_TRACK_RING:
            if len(args) >= 1:
                pass

        elif cmd_id == CMD_TRACK_TOP:
            if len(args) >= 1:
                pass

        elif cmd_id == CMD_STOP_VISUAL:
            pass

    def on_arrived(self, zone_id: int) -> None:
        self.mission_sm.on_arrived(zone_id)

    def on_action_done(self, action_id: int, result: int) -> None:
        self.mission_sm.on_action_done(action_id, result)

    def on_qr_result(self, qr_str: str) -> None:
        success = self.mission_sm.on_qr_result(qr_str)
        if success:
            self._send(build_qr_result_frame(qr_str))

    def on_servo_data(self, error_x: int, error_y: int,
                      distance: int, state: int) -> None:
        self._send(build_visual_servo_data_frame(
            error_x, error_y, distance, state))

    def on_visual_status_update(self, visual_state: int, flags: int) -> None:
        self.mission_sm.on_visual_status(visual_state, flags)
        self._send(build_status_from_vision_frame(
            self.mission_sm.current_state_id,
            visual_state, flags,
            self.mission_sm.context.cargo_count,
        ))

    def get_info(self) -> dict:
        return {
            "mission": self.mission_sm.get_info(),
            "cargo_batch1": [
                {"index": i.index, "color": i.color.name, "available": i.available}
                for i in self.cargo_set.get_batch(1)
            ],
            "cargo_batch2": [
                {"index": i.index, "color": i.color.name, "available": i.available}
                for i in self.cargo_set.get_batch(2)
            ],
        }
