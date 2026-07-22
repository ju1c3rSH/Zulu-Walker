import threading
from collections import deque
from typing import Any, Callable, Dict, List


class EventBus:
    def __init__(self, history_size: int = 1000):
        self._subscribers: Dict[type, List[Callable]] = {}
        self._lock = threading.RLock()
        self._history: deque = deque(maxlen=history_size)

    def subscribe(self, event_type: type, callback: Callable) -> None:
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: type, callback: Callable) -> bool:
        with self._lock:
            subs = self._subscribers.get(event_type, [])
            if callback in subs:
                subs.remove(callback)
                return True
            return False

    def publish(self, event: Any) -> None:
        with self._lock:
            self._history.append(event)
            event_type = type(event)
            for subscriber in self._subscribers.get(event_type, []):
                try:
                    subscriber(event)
                except Exception as e:
                    import traceback
                    traceback.print_exc()

    @property
    def history(self) -> List[Any]:
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()
