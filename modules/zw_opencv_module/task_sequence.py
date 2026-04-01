# -*- coding: utf-8 -*-
from typing import List, Optional

class TaskSequence:
    def __init__(self, batch1: List[str], batch2: List[str]):
        self.batch1 = batch1
        self.batch2 = batch2
        self.current_batch = 1
        self.current_item_index = 0

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
