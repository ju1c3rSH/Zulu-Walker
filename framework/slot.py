"""Latest-value slot: publish, overwrite, read newest.

One writer (usually a worker thread) publishes the most recent value;
readers poll for anything newer than the generation they last saw. This
replaces the hand-rolled "store newest + detect change" patterns (object
identity checks, serial comparisons, drain deques) scattered across the
codebase.

Selection rule (docs/architecture/thread_tick_topology.md): use a Slot for
continuous state where losing intermediate values is acceptable - camera
frames, detection results, fps/link status.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Stamped(Generic[T]):
    """A value together with the slot generation that produced it."""

    value: T
    gen: int


class Slot(Generic[T]):
    """Single-slot mailbox holding only the newest published value."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._gen: int = 0

    def publish(self, value: T) -> int:
        """Overwrite the slot and bump the generation. Returns the new gen."""
        with self._lock:
            self._value = value
            self._gen += 1
            return self._gen

    def load(self, seen_gen: int = 0) -> Optional[Stamped[T]]:
        """Return the newest value if its gen differs from ``seen_gen``.

        None means "nothing new" - the reader should keep whatever it already
        has. Pass back ``stamped.gen`` on the next call.
        """
        with self._lock:
            if self._value is None or self._gen == seen_gen:
                return None
            return Stamped(self._value, self._gen)

    def latest(self) -> Optional[Stamped[T]]:
        """Return the current value regardless of any reader's generation."""
        with self._lock:
            if self._value is None:
                return None
            return Stamped(self._value, self._gen)

    @property
    def gen(self) -> int:
        """Current generation number (0 until the first publish)."""
        with self._lock:
            return self._gen
