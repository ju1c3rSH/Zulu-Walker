"""Typed command queue: ordered, mergeable commands for the main loop.

Producers (UART RX thread, UI callbacks, input sources) enqueue Cmd items;
the main loop drains the queue once per tick and executes commands in order.
Every command is a named, inspectable value - never an anonymous lambda - so
the queue doubles as a pending-action audit log and is trivial to unit test.

Selection rule (docs/architecture/thread_tick_topology.md): use a CmdQueue
when an action must not be lost but repeats can be coalesced (calibration
requests, record toggles, exit).
"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass(frozen=True)
class Cmd:
    """A named action request. ``kind`` values are defined by the consumer."""

    kind: str
    payload: Dict = field(default_factory=dict)


class CmdQueue:
    """Bounded FIFO of typed commands with same-kind coalescing.

    Coalescing policy: by default an incoming command whose kind already has
    a pending twin is dropped (button spam). With ``replace=True`` the pending
    command's payload is swapped instead. The queue never grows beyond
    ``maxlen``; overflow silently drops the oldest command.
    """

    def __init__(self, maxlen: int = 64) -> None:
        self._items: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def put(self, cmd: Cmd, replace: bool = False) -> bool:
        """Enqueue ``cmd``.

        Returns True when the command entered the queue (fresh or replacing a
        pending same-kind twin), False when it was coalesced away.
        """
        with self._lock:
            for i, existing in enumerate(self._items):
                if existing.kind == cmd.kind:
                    if replace:
                        self._items[i] = cmd
                        return True
                    return False
            self._items.append(cmd)
            return True

    def drain(self) -> List[Cmd]:
        """Atomically remove and return all pending commands, oldest first."""
        with self._lock:
            out = list(self._items)
            self._items.clear()
            return out

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)
