# -*- coding: utf-8 -*-
"""LAB 阈值会话 — 6 组阈值的内存态唯一事实来源。

爆炸物页(组 0-2)与靶标页(组 3-5)共享同一会话, SAVE 时整文件落盘,
避免按组拆页后互相覆盖; 会话跨页面切换持有, RESET 从配置+阈值文件整载。
文件行格式: name|Lmin,Lmax,Amin,Amax,Bmin,Bmax（组序与 GksTaskProcessor 一致）。
"""
import os
import threading

from modules.zw_opencv_module.processors.gks_task_processor import (
    load_gks_config,
    load_threshold_file,
)

THRESHOLD_FILE = "/my-threshold/my_64threshold.txt"
GROUP_NAMES = ["E-R", "E-G", "E-B", "T-R", "T-G", "T-B"]
FILE_KEYS = ["exp_red", "exp_green", "exp_blue", "tgt_red", "tgt_green", "tgt_blue"]
GROUP_COUNT = 6


class LabThresholdSession:
    """六组 LAB 阈值的进程内单例会话。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reload()

    def reload(self) -> None:
        groups = self._load()
        with self._lock:
            self._groups = groups

    def groups(self) -> list:
        with self._lock:
            return [list(g) for g in self._groups]

    def group(self, index: int) -> list:
        with self._lock:
            return list(self._groups[index])

    def set_group(self, index: int, values: list) -> None:
        with self._lock:
            self._groups[index] = list(values)

    def save(self) -> None:
        with self._lock:
            snapshot = [list(g) for g in self._groups]
        self._save_file(snapshot)

    def _load(self) -> list:
        cfg = load_gks_config()
        explosive_lab = cfg["explosive"]["lab"]
        target_lab = cfg["target"]["lab"]
        groups = [
            list(explosive_lab["red"]), list(explosive_lab["green"]), list(explosive_lab["blue"]),
            list(target_lab["red"]), list(target_lab["green"]), list(target_lab["blue"]),
        ]
        if os.path.exists(THRESHOLD_FILE):
            saved = load_threshold_file(THRESHOLD_FILE)
            for i in range(min(GROUP_COUNT, len(saved))):
                groups[i] = list(saved[i])
        return groups

    def _save_file(self, groups: list) -> None:
        parent = os.path.dirname(THRESHOLD_FILE)
        if parent:
            os.makedirs(parent, exist_ok=True)
        lines = [f"{FILE_KEYS[i]}|{','.join(str(v) for v in groups[i])}"
                 for i in range(GROUP_COUNT)]
        with open(THRESHOLD_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


_session: LabThresholdSession = None
_session_lock = threading.Lock()


def get_threshold_session() -> LabThresholdSession:
    """返回进程级共享会话（页面切换间保持内存态）。"""
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                _session = LabThresholdSession()
    return _session
