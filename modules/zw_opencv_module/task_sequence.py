# -*- coding: utf-8 -*-
from typing import List, Optional

COLOR_MAP = {
    '1': 'Red',
    '2': 'Green',
    '3': 'Blue'
}

class TaskSequence:
    def __init__(self, batch1: List[str] = None, batch2: List[str] = None):
        self.batch1 = batch1 or []
        self.batch2 = batch2 or []
        self.current_batch = 1
        self.current_item_index = 0

    @classmethod
    def from_qr_data(cls, qr_data: str) -> 'TaskSequence':
        parts = qr_data.split('+')
        batch1 = [COLOR_MAP.get(c, c) for c in parts[0]] if len(parts) > 0 else []
        batch2 = [COLOR_MAP.get(c, c) for c in parts[1]] if len(parts) > 1 else []
        return cls(batch1, batch2)

    def is_complete(self) -> bool:
        if self.current_batch == 1:
            return False
        elif self.current_batch == 2:
            return self.current_item_index >= len(self.batch2)
        return True

    def get_current_item(self) -> Optional[str]:
        if self.current_batch == 1 and self.current_item_index < len(self.batch1):
            return self.batch1[self.current_item_index]
        elif self.current_batch == 2 and self.current_item_index < len(self.batch2):
            return self.batch2[self.current_item_index]
        else:
            return None

    def get_next_target(self) -> Optional[str]:
        if self.current_batch == 1:
            if self.current_item_index < len(self.batch1):
                return self.batch1[self.current_item_index]
            else:
                self.current_batch = 2
                self.current_item_index = 0
                return self.get_next_target()  # 递归调用获取下一批次的第一个目标
        elif self.current_batch == 2:
            if self.current_item_index < len(self.batch2):
                return self.batch2[self.current_item_index]
            else:
                return None  # 两批次都完成了

        return None

    def advance(self):
        self.current_item_index += 1
