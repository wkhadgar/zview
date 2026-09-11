# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_mutex`` walker."""

from backend.base import AbstractScraper, MutexInfo
from backend.elf_inspector import ElfInspector
from kernel.layout import KernelLayout
from kernel.object_names import label_instances
from kernel.wait_queues import resolve_waiter_names, walk_wait_queue


def walk_mutexes(
    scraper: AbstractScraper,
    elf: ElfInspector,
    addresses: dict[str, list[int]],
    layout: KernelLayout,
    thread_names: dict[int, str] | None = None,
    *,
    walk_waiters: bool = True,
) -> dict[str, MutexInfo]:
    """
    Read every statically declared ``k_mutex`` and return ``{label: MutexInfo}``.

    The owner pointer is resolved against ``thread_names``. Each mutex costs
    one bulk struct read plus one read per waiting thread. A lock taken and
    released between two calls is not observed.
    """
    words_to_read = (elf.get_struct_size("k_mutex") + 3) // 4
    owner_idx = layout.mutex_owner // 4
    lock_count_idx = layout.mutex_lock_count // 4
    wait_q_idx = layout.mutex_wait_q // 4

    mutexes: dict[str, MutexInfo] = {}
    for label, address in label_instances(addresses):
        words = scraper.read32(address, words_to_read)
        owner_address = words[owner_idx]

        waiters: tuple[str, ...] | None = None
        if walk_waiters and layout.thread_qnode is not None:
            waiters = resolve_waiter_names(
                walk_wait_queue(
                    scraper,
                    address + layout.mutex_wait_q,
                    layout.thread_qnode,
                    head=words[wait_q_idx],
                ),
                thread_names,
            )

        mutexes[label] = MutexInfo(
            name=label,
            address=address,
            lock_count=words[lock_count_idx],
            owner_address=owner_address,
            owner_name=(thread_names or {}).get(owner_address) if owner_address else None,
            waiters=waiters,
        )

    return mutexes
