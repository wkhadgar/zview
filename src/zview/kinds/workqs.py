# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""``k_work_q`` as a reader sees it."""

from zview.backend.base import WorkqInfo
from zview.kinds.kind import Kind, State, level, waiters_text


class Workqs(Kind):
    key = "workqs"
    label = "WKQ"
    title = "All Work Queues"
    # Plugged or draining, it refuses submissions from outside.
    pressure = "refusing"

    def state(self, info: WorkqInfo) -> State:
        # First match wins.
        text = next(
            word
            for applies, word in (
                (info.is_busy, "BUSY"),
                (info.is_started, "IDLE"),
                (True, "NOT STARTED"),
            )
            if applies
        )
        if info.depth:
            more = "+" if info.is_pending_truncated else ""
            text += f" ● {info.depth}{more} pending"
        return State(
            100.0 if info.is_busy else 0.0,
            text,
            level(
                is_blocked=info.is_plugged or info.is_draining,
                is_busy=bool(info.depth or info.waiters),
            ),
        )

    def in_use(self, info: WorkqInfo) -> bool:
        return info.is_busy or bool(info.depth)

    def dump(self, info: WorkqInfo) -> str:
        items = ",".join(info.pending) + (",..." if info.is_pending_truncated else "")
        return (
            f"wkq  {info.name:20s} {info.depth} pending flags={','.join(info.states) or '-'} "
            f"thread={info.thread_name or '-'} items={items or '-'} "
            f"waiters={waiters_text(info.waiters)}"
        )
