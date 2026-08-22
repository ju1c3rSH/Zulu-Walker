"""Display-pump contracts: FrameSink, SinkGroup and InputSource.

Architecture (DISP-03/D3/D4 decisions):

* Composition (drawing overlays) happens in the vision thread through the
  composer hook - it never opens threads.
* Presentation goes through FrameSink objects that do NOT own threads:
  the producer pushes the latest composed frame, and the main loop pumps
  ``flush()`` once per tick. Each sink throttles itself against its own
  fps limit and skips frames whose content did not change.
* User input never rides on ``show()`` return values. InputSource objects
  translate raw device events into typed Cmds on a CmdQueue; the main loop
  owns exit semantics (Cmd kind "EXIT").
"""

from __future__ import annotations

from typing import List, Protocol

from framework.cmd_queue import CmdQueue


class FrameSink(Protocol):
    """A consumer of composed frames (LCD, cv2 window, JPEG streamer, ...).

    Implementations must accept the platform-native pixel container for
    their target (maix.Image on MaixCAM2, ndarray elsewhere) - there is no
    canonical pixel format by design (D1).
    """

    def push(self, frame) -> None:
        """Store the latest composed frame. Non-blocking; overwrites unsent."""
        ...

    def flush(self) -> None:
        """Present the stored frame if this sink is due (fps-limited)."""
        ...

    def close(self) -> None:
        """Release device resources. Idempotent."""
        ...


class SinkGroup:
    """Fan-out container pumping several sinks from one call site."""

    def __init__(self) -> None:
        self._sinks: List[FrameSink] = []

    def add(self, sink: FrameSink) -> None:
        self._sinks.append(sink)

    def remove(self, sink: FrameSink) -> None:
        if sink in self._sinks:
            self._sinks.remove(sink)

    @property
    def sinks(self) -> List[FrameSink]:
        return list(self._sinks)

    def push(self, frame) -> None:
        for sink in self._sinks:
            sink.push(frame)

    def flush(self) -> None:
        for sink in self._sinks:
            sink.flush()

    def close(self) -> None:
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:
                pass
        self._sinks.clear()


class InputSource(Protocol):
    """A producer of user commands (touchscreen, keyboard, ...).

    Implementations translate raw device events into typed Cmds. They must
    not block longer than one main-loop tick.
    """

    def poll(self, cmds: CmdQueue) -> None:
        """Read pending device events and enqueue corresponding Cmds."""
        ...
