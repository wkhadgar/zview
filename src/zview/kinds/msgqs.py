# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""``k_msgq`` as a reader sees it."""

from zview.backend.base import MsgqInfo
from zview.kinds.kind import Kind, State, level, waiters_text


class Msgqs(Kind):
    key = "msgqs"
    label = "MSG"
    title = "All Queues"

    def state(self, info: MsgqInfo) -> State:
        # A full queue blocks its senders.
        return State(
            info.fill_percent,
            f"{info.used_msgs}/{info.max_msgs} × {info.msg_size}B",
            level(is_blocked=info.is_full, is_busy=bool(info.waiters)),
        )

    def dump(self, info: MsgqInfo) -> str:
        return (
            f"msgq {info.name:20s} {info.used_msgs}/{info.max_msgs} x {info.msg_size}B "
            f"waiters={waiters_text(info.waiters)}"
        )
