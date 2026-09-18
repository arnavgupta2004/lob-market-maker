"""Deterministic future-event queue for the discrete-event simulator."""
from __future__ import annotations

import heapq
from typing import Any


class EventQueue:
    """Min-heap ordered by ``(time_ns, insertion_sequence)``.

    The insertion counter makes ties deterministic (FIFO), so a run depends only on the seed
    and configuration, never on heap-internal ordering.
    """

    __slots__ = ("_heap", "_n")

    def __init__(self) -> None:
        self._heap: list[tuple[int, int, int, Any, Any]] = []
        self._n = 0

    def push(self, time: int, kind: int, a: Any = None, b: Any = None) -> None:
        self._n += 1
        heapq.heappush(self._heap, (time, self._n, kind, a, b))

    def pop(self) -> tuple[int, int, int, Any, Any]:
        return heapq.heappop(self._heap)

    def peek_time(self) -> int | None:
        return self._heap[0][0] if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)
