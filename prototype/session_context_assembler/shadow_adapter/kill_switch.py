"""Fail-closed kill switch with atomic side-effect guards."""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock
from typing import Callable, Iterator

from .errors import ShadowAdapterError


class KillSwitch:
    def __init__(self, active: bool = False, reason: str | None = None) -> None:
        self._active = active
        self._reason = reason
        self._generation = 0
        self._lock = RLock()

    @property
    def active(self) -> bool:
        with self._lock:
            return self._active

    @property
    def reason(self) -> str | None:
        with self._lock:
            return self._reason

    def activate(self, reason: str, rollback: Callable[[], None] | None = None) -> None:
        with self._lock:
            self._active = True
            self._reason = reason
            self._generation += 1
            if rollback is not None:
                rollback()

    def deactivate_for_test(self) -> None:
        """Test-only reset; no runtime re-enable path is exposed by the adapter."""
        with self._lock:
            self._active = False
            self._reason = None
            self._generation += 1

    def begin(self) -> int:
        with self._lock:
            self._require_enabled_unlocked()
            return self._generation

    def require_enabled(self, expected_generation: int | None = None) -> None:
        with self._lock:
            self._require_enabled_unlocked(expected_generation)

    @contextmanager
    def guard(self, expected_generation: int) -> Iterator[None]:
        """Atomically check and perform one bounded side effect."""
        with self._lock:
            self._require_enabled_unlocked(expected_generation)
            yield

    def _require_enabled_unlocked(self, expected_generation: int | None = None) -> None:
        if self._active or (
            expected_generation is not None and expected_generation != self._generation
        ):
            raise ShadowAdapterError(
                "KILL_SWITCH_ACTIVE", "Shadow adapter is disabled.", retryable=False
            )
