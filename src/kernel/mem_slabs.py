# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_mem_slab`` walker."""

from backend.base import AbstractScraper, MemSlabInfo
from backend.elf_inspector import ElfInspector
from kernel.layout import KernelLayout
from kernel.object_names import label_instances
from kernel.structs import struct_words
from kernel.wait_queues import resolve_waiter_names, walk_wait_queue


def walk_mem_slabs(
    scraper: AbstractScraper,
    elf: ElfInspector,
    addresses: dict[str, list[int]],
    layout: KernelLayout,
    thread_names: dict[int, str] | None = None,
    *,
    walk_waiters: bool = True,
) -> dict[str, MemSlabInfo]:
    """
    Read every statically declared ``k_mem_slab`` and return ``{label: MemSlabInfo}``.

    ``addresses`` maps a symbol name to its instance addresses. Each slab costs
    one bulk struct read plus one read per waiting thread.
    ``walk_waiters=False`` skips the wait queue and leaves ``waiters`` as
    ``None``; required on a non-walkable (scalable) layout.

    ``max_used`` is read only where the layout resolved it.
    """
    words_to_read = struct_words(elf, "k_mem_slab")
    blocks_idx = layout.mem_slab_num_blocks // 4
    size_idx = layout.mem_slab_block_size // 4
    used_idx = layout.mem_slab_num_used // 4
    wait_q_idx = layout.mem_slab_wait_q // 4

    slabs: dict[str, MemSlabInfo] = {}
    for label, address in label_instances(addresses):
        words = scraper.read32(address, words_to_read)

        waiters: tuple[str, ...] | None = None
        if walk_waiters and layout.thread_qnode is not None:
            waiters = resolve_waiter_names(
                walk_wait_queue(
                    scraper,
                    address + layout.mem_slab_wait_q,
                    layout.thread_qnode,
                    head=words[wait_q_idx],
                ),
                thread_names,
            )

        max_used = None
        if layout.mem_slab_max_used is not None:
            max_used = words[layout.mem_slab_max_used // 4]

        slabs[label] = MemSlabInfo(
            name=label,
            address=address,
            num_blocks=words[blocks_idx],
            block_size=words[size_idx],
            num_used=words[used_idx],
            max_used=max_used,
            waiters=waiters,
        )

    return slabs
