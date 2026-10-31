# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""``k_heap`` as a reader sees it."""

from zview.backend.base import HeapInfo
from zview.kinds.kind import Kind, State, level, waiters_text


class Heaps(Kind):
    key = "heaps"
    label = "HEP"
    title = "All Heaps"
    # A heap with nothing free blocks the next allocation.
    pressure = "exhausted"

    def state(self, info: HeapInfo) -> State:
        return State(
            info.usage_percent,
            f"{info.allocated_bytes}/{info.total_bytes}B",
            level(is_blocked=info.is_exhausted, is_busy=bool(info.waiters)),
        )

    def dump(self, info: HeapInfo) -> str:
        return (
            f"heap {info.name:20s} free={info.free_bytes} "
            f"alloc={info.allocated_bytes} max={info.max_allocated_bytes} "
            f"waiters={waiters_text(info.waiters)}"
        )
