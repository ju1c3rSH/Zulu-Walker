# -*- coding: utf-8 -*-
"""Provide reusable state-machine base classes with transitions and callbacks."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, List, Set
from enum import Enum
from dataclasses import dataclass
from time import time
import threading
from framework.log import fw_log as log_print


class State(ABC):
    """Base class for states; implement on_enter, on_execute, and on_exit.

    Returning a target state name from ``on_execute`` triggers a transition.
    """

    @abstractmethod
    def on_enter(self, context: Any, from_state: str) -> None:
        """Initialize the state on entry.

        Args:
            context: State-machine context.
            from_state: Name of the state being left.
        """
        pass

    @abstractmethod
    def on_execute(self, context: Any) -> Optional[str]:
        """Run one frame of state logic.

        Args:
            context: State-machine context.

        Returns:
            A target state name to trigger a transition, or None to stay.
        """
        pass

    @abstractmethod
    def on_exit(self, context: Any, to_state: str) -> None:
        """Clean up when leaving the state.

        Args:
            context: State-machine context.
            to_state: Name of the state being entered.
        """
        pass


@dataclass
class Transition:
    """A transition between two states, with an optional condition or event."""

    from_state: str
    to_state: str
    condition: Optional[Callable[[Any], bool]] = None
    event: Optional[str] = None


class BaseStateMachine(ABC):
    """Manage state registration, transitions, and event-driven changes.

    Subclasses supply concrete states and transitions.

    Example:
        >>> class MyStateMachine(BaseStateMachine):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.register_state("IDLE", IdleState())
        ...         self.register_state("RUNNING", RunningState())
        ...         self.set_initial_state("IDLE")
        ...
        >>> sm = MyStateMachine()
        >>> sm.trigger("START")  # fire an event
        >>> sm.update()           # run the current state
    """

    def __init__(self):
        self._states: Dict[str, State] = {}
        self._transitions: Dict[str, Dict[str, Transition]] = {}
        self._event_transitions: Dict[str, Dict[str, str]] = {}
        self._current_state: Optional[str] = None
        self._previous_state: Optional[str] = None
        self._context: Any = None

        self._lock = threading.RLock()
        self._state_callbacks: List[Callable[[str, str, Optional[str]], None]] = []
        self._enter_callbacks: Dict[str, List[Callable[[Any, str], None]]] = {}
        self._exit_callbacks: Dict[str, List[Callable[[Any, str], None]]] = {}

        self._state_entry_time: float = 0.0
        self._state_frame_count: int = 0

    # State registration

    def register_state(self, name: str, state: State) -> None:
        """Register a state under ``name``."""
        with self._lock:
            self._states[name] = state
            if name not in self._transitions:
                self._transitions[name] = {}

    def register_transition(
        self,
        from_state: str,
        to_state: str,
        condition: Optional[Callable[[Any], bool]] = None,
        event: Optional[str] = None
    ) -> None:
        """Register a transition, optionally gated by a condition or event.

        Args:
            from_state: Source state name.
            to_state: Target state name.
            condition: Optional predicate evaluated on each update.
            event: Optional event name that triggers the transition.
        """
        with self._lock:
            if from_state not in self._transitions:
                self._transitions[from_state] = {}

            self._transitions[from_state][to_state] = Transition(
                from_state=from_state,
                to_state=to_state,
                condition=condition,
                event=event
            )

            if event:
                if from_state not in self._event_transitions:
                    self._event_transitions[from_state] = {}
                self._event_transitions[from_state][event] = to_state

    def set_initial_state(self, state_name: str) -> None:
        """Set the initial state and reset its timing counters."""
        with self._lock:
            if state_name not in self._states:
                raise ValueError(f"State '{state_name}' not registered")
            self._current_state = state_name
            self._state_entry_time = time()
            self._state_frame_count = 0

    # Core operations

    def trigger(self, event: str, data: Any = None) -> bool:
        """Fire an event and transition if the current state handles it.

        Args:
            event: Event name.
            data: Payload stored on ``context.event_data``.

        Returns:
            True if a transition occurred.
        """
        with self._lock:
            if self._current_state is None:
                return False

            # Ignore events the current state does not handle.
            event_map = self._event_transitions.get(self._current_state, {})
            if event not in event_map:
                return False

            target_state = event_map[event]

            # Expose the event to state logic through the context.
            if self._context is not None:
                setattr(self._context, 'event_data', data)
                setattr(self._context, 'triggered_event', event)

            return self._do_transition(target_state, event)

    def update(self) -> None:
        """Run one frame: execute the current state, then check automatic transitions."""
        with self._lock:
            if self._current_state is None:
                return

            current_state_obj = self._states.get(self._current_state)
            if current_state_obj is None:
                return

            self._state_frame_count += 1

            target_state = current_state_obj.on_execute(self._context)

            # Honor a transition requested by the state itself.
            if target_state is not None and target_state != self._current_state:
                self._do_transition(target_state, None)
                return

            # Fall back to condition-based transitions.
            transitions = self._transitions.get(self._current_state, {})
            for transition in transitions.values():
                if transition.condition and transition.condition(self._context):
                    self._do_transition(transition.to_state, None)
                    return

    def run_to_completion(self, max_steps: int = 10) -> int:
        """Call ``update`` repeatedly until the state stops changing.

        Useful when an entry triggers chained automatic transitions (for
        example RING_DISCOVERY -> ALIGN_ROUGH -> PLACE_ROUGH) so callers need
        not invoke ``update`` several times. ``max_steps`` bounds the loop
        against endless cascades; stable states whose ``on_execute`` returns
        None are handled.

        Returns:
            The number of transitions performed.
        """
        steps = 0
        for _ in range(max_steps):
            prev = self._current_state
            self.update()
            if self._current_state == prev:
                break
            steps += 1
        return steps

    def _do_transition(self, target_state: str, event: Optional[str]) -> bool:
        """Move to another state, running exit/entry hooks and callbacks."""
        if target_state not in self._states:
            return False

        if self._current_state == target_state:
            return False

        old_state = self._current_state
        new_state = target_state

        if old_state and old_state in self._states:
            self._states[old_state].on_exit(self._context, new_state)
            self._trigger_exit_callbacks(old_state, new_state)

        self._previous_state = old_state
        self._current_state = new_state
        self._state_entry_time = time()
        self._state_frame_count = 0

        self._states[new_state].on_enter(self._context, old_state or "")
        self._trigger_enter_callbacks(new_state, old_state or "")

        self._notify_callbacks(old_state or "", new_state, event)

        return True

    # Callback management

    def add_state_change_callback(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        """Register a callback invoked on every state change.

        Args:
            callback: Receives ``(old_state, new_state, event)``.
        """
        self._state_callbacks.append(callback)

    def remove_state_change_callback(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        """Unregister a state-change callback."""
        if callback in self._state_callbacks:
            self._state_callbacks.remove(callback)

    def on_state_enter(
        self,
        state: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        """Register a callback for entry into ``state``: ``callback(context, from_state)``."""
        self.add_enter_callback(state, callback)

    def on_state_exit(
        self,
        state: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        """Register a callback for exit from ``state``: ``callback(context, to_state)``."""
        self.add_exit_callback(state, callback)

    def on_state_change(
        self,
        callback: Callable[[str, str, Optional[str]], None]
    ) -> None:
        """Register a callback for any change: ``callback(old_state, new_state, event)``."""
        self.add_state_change_callback(callback)

    def add_enter_callback(
        self,
        state_name: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        """Add an entry callback for ``state_name``."""
        if state_name not in self._enter_callbacks:
            self._enter_callbacks[state_name] = []
        self._enter_callbacks[state_name].append(callback)

    def add_exit_callback(
        self,
        state_name: str,
        callback: Callable[[Any, str], None]
    ) -> None:
        """Add an exit callback for ``state_name``."""
        if state_name not in self._exit_callbacks:
            self._exit_callbacks[state_name] = []
        self._exit_callbacks[state_name].append(callback)

    def _notify_callbacks(self, old_state: str, new_state: str, event: Optional[str]) -> None:
        """Invoke global state-change callbacks, logging any errors."""
        for callback in self._state_callbacks:
            try:
                callback(old_state, new_state, event)
            except Exception as e:
                log_print(f"[StateMachine] Callback error: {e}")

    def _trigger_enter_callbacks(self, state_name: str, from_state: str) -> None:
        """Invoke entry callbacks for a state, logging any errors."""
        for callback in self._enter_callbacks.get(state_name, []):
            try:
                callback(self._context, from_state)
            except Exception as e:
                log_print(f"[StateMachine] Enter callback error: {e}")

    def _trigger_exit_callbacks(self, state_name: str, to_state: str) -> None:
        """Invoke exit callbacks for a state, logging any errors."""
        for callback in self._exit_callbacks.get(state_name, []):
            try:
                callback(self._context, to_state)
            except Exception as e:
                log_print(f"[StateMachine] Exit callback error: {e}")

    # Properties

    @property
    def current_state(self) -> Optional[str]:
        """Name of the current state."""
        with self._lock:
            return self._current_state

    @property
    def previous_state(self) -> Optional[str]:
        """Name of the state before the current one."""
        with self._lock:
            return self._previous_state

    @property
    def context(self) -> Any:
        """Shared context object passed to state hooks."""
        return self._context

    @context.setter
    def context(self, value: Any) -> None:
        """Replace the shared context object."""
        self._context = value

    @property
    def state_duration(self) -> float:
        """Seconds spent in the current state."""
        return time() - self._state_entry_time

    @property
    def state_frame_count(self) -> int:
        """Number of updates run in the current state."""
        return self._state_frame_count

    def is_in_state(self, state_name: str) -> bool:
        """Return whether the current state is ``state_name``."""
        with self._lock:
            return self._current_state == state_name

    def can_handle_event(self, event: str) -> bool:
        """Return whether the current state handles ``event``."""
        with self._lock:
            if self._current_state is None:
                return False
            event_map = self._event_transitions.get(self._current_state, {})
            return event in event_map

    def reset(self) -> None:
        """Clear the current state and timing counters."""
        with self._lock:
            self._current_state = None
            self._previous_state = None
            self._state_entry_time = 0.0
            self._state_frame_count = 0
