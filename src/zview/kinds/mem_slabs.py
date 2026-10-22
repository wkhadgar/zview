# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""``k_mem_slab`` as a reader sees it."""

from zview.backend.base import MemSlabInfo
from zview.kinds.kind import Kind, State, level, waiters_text


class MemSlabs(Kind):
    key = "mem_slabs"
    label = "SLB"
    title = "All Slabs"
    # An exhausted slab blocks the next allocation.
    pressure = "exhausted"

    def state(self, info: MemSlabInfo) -> State:
        return State(
            info.fill_percent,
            f"{info.num_used}/{info.num_blocks} × {info.block_size}B",
            level(blocked=info.is_exhausted, busy=bool(info.waiters)),
        )

    def dump(self, info: MemSlabInfo) -> str:
        peak = "-" if info.max_used is None else info.max_used
        return (
            f"slab {info.name:20s} {info.num_used}/{info.num_blocks} x {info.block_size}B "
            f"peak={peak} waiters={waiters_text(info.waiters)}"
        )
