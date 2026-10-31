# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Threads, which the thread views show rather than a table of objects."""

from zview.backend.base import ThreadInfo
from zview.kinds.kind import Kind, State, level


class Threads(Kind):
    key = "threads"

    def state(self, info: ThreadInfo) -> State | None:
        """Its CPU share, busy when it ran in the last frame, on a build that counts usage."""
        runtime = info.runtime
        if runtime is None or runtime.cpu_normalized is None:
            return None
        text = "running" if runtime.is_active else "waiting"
        return State(runtime.cpu_normalized, text, level(is_busy=runtime.is_active))

    def listed(self, info: ThreadInfo) -> bool:
        """Every thread but the idle thread of each CPU."""
        return not info.is_idle

    def dump(self, info: ThreadInfo) -> str:
        watermark = getattr(info.runtime, "stack_watermark", None)
        watermark = "-" if watermark is None else watermark
        size = "-" if info.stack_size is None else info.stack_size
        return f"thread {info.name:30s} stack={size:>6}  watermark={watermark}"
