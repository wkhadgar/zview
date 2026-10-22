# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Threads, which the thread views show rather than a table of objects."""

from zview.backend.base import ThreadInfo
from zview.kinds.kind import Kind


class Threads(Kind):
    key = "threads"

    def dump(self, info: ThreadInfo) -> str:
        watermark = getattr(info.runtime, "stack_watermark", None)
        watermark = "-" if watermark is None else watermark
        size = "-" if info.stack_size is None else info.stack_size
        return f"thread {info.name:30s} stack={size:>6}  watermark={watermark}"
