# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""``k_mutex`` as a reader sees it."""

from zview.backend.base import MutexInfo, MutexState
from zview.kinds.kind import Kind, Level, State, level, waiters_text


class Mutexes(Kind):
    key = "mutexes"
    label = "MTX"
    title = "All Mutexes"

    def state(self, info: MutexInfo) -> State:
        if not info.is_locked:
            return State(0.0, "FREE", Level.QUIET)

        owner = info.owner_name or f"0x{info.owner_address:X}"
        text = f"LOCKED ● {owner}"
        if info.lock_count > 1:
            text += f" ×{info.lock_count}"
        return State(100.0, text, level(blocked=info.state is MutexState.CONTENDED, busy=True))

    def in_use(self, info: MutexInfo) -> bool:
        return info.is_locked

    def dump(self, info: MutexInfo) -> str:
        owner = f"owner={info.owner_name or hex(info.owner_address)} depth={info.lock_count}"
        return (
            f"mtx  {info.name:20s} {'LOCKED' if info.is_locked else 'FREE':6s} {owner} "
            f"waiters={waiters_text(info.waiters)}"
        )
