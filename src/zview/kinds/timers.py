# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""``k_timer`` as a reader sees it."""

from zview.backend.base import TimerInfo
from zview.kinds.kind import Kind, State, level, waiters_text


def _caption(info: TimerInfo) -> str:
    """
    Stopped, or how often it expires. A timer without a callback is read by
    polling its status, so its unread expiries follow. First match wins.
    """
    period = next(
        text
        for applies, text in (
            (not info.is_running, "stopped"),
            (info.is_oneshot, "oneshot"),
            (True, f"every {info.period_ticks}t"),
        )
        if applies
    )
    if info.is_running and info.expiry_fn is None and info.status:
        return f"{period}, {info.status} unread"
    return period


class Timers(Kind):
    key = "timers"
    label = "TMR"
    title = "All Timers"

    def state(self, info: TimerInfo) -> State:
        return State(
            100.0 if info.is_running else 0.0,
            _caption(info),
            level(is_busy=bool(info.waiters)),
        )

    def dump(self, info: TimerInfo) -> str:
        running = "running" if info.is_running else "stopped"
        period = "oneshot" if info.is_oneshot else f"every {info.period_ticks}t"
        expiry = info.expiry_fn_symbol or "-"
        return (
            f"tmr  {info.name:20s} {running} {period} status={info.status} "
            f"expiry={expiry} waiters={waiters_text(info.waiters)}"
        )
