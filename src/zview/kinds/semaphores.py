# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""``k_sem`` as a reader sees it."""

from zview.backend.base import SemaphoreInfo
from zview.kinds.kind import Kind, State, level, waiters_text


def _count(info: SemaphoreInfo, unlimited: str, uninitialized: str) -> str:
    """The count over the limit. First match wins."""
    return next(
        text
        for applies, text in (
            (not info.is_initialized, uninitialized),
            (info.is_unbounded, f"{info.count}{unlimited}"),
            (True, f"{info.count}/{info.limit}"),
        )
        if applies
    )


class Semaphores(Kind):
    key = "semaphores"
    label = "SEM"
    title = "All Semaphores"

    def state(self, info: SemaphoreInfo) -> State:
        return State(
            info.fill_percent,
            _count(info, " / unlimited", "NOT INITIALIZED"),
            level(busy=bool(info.waiters)),
        )

    def dump(self, info: SemaphoreInfo) -> str:
        count = _count(info, "/-", "not initialized")
        return f"sem  {info.name:20s} {count}  waiters={waiters_text(info.waiters)}"
