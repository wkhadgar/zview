# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_sem`` walker."""

from backend.base import AbstractScraper, SemaphoreInfo
from backend.elf_inspector import ElfInspector
from kernel.layout import KernelLayout
from kernel.object_names import label_instances
from kernel.wait_queues import resolve_waiter_names, walk_wait_queue


def walk_semaphores(
    scraper: AbstractScraper,
    elf: ElfInspector,
    addresses: dict[str, list[int]],
    layout: KernelLayout,
    thread_names: dict[int, str] | None = None,
    *,
    walk_waiters: bool = True,
) -> dict[str, SemaphoreInfo]:
    """
    Read every statically declared ``k_sem`` and return ``{label: SemaphoreInfo}``.

    ``addresses`` maps a symbol name to its instance addresses. Each semaphore
    costs one bulk struct read plus one read per waiting thread.
    ``walk_waiters=False`` skips the wait queue and leaves ``waiters`` as
    ``None``; required on a non-walkable (scalable) layout.
    """
    words_to_read = _struct_words(elf, "k_sem")
    count_idx = layout.sem_count // 4
    limit_idx = layout.sem_limit // 4
    wait_q_idx = layout.sem_wait_q // 4

    semaphores: dict[str, SemaphoreInfo] = {}
    for label, address in label_instances(addresses):
        words = scraper.read32(address, words_to_read)

        waiters: tuple[str, ...] | None = None
        if walk_waiters and layout.thread_qnode is not None:
            waiters = resolve_waiter_names(
                walk_wait_queue(
                    scraper,
                    address + layout.sem_wait_q,
                    layout.thread_qnode,
                    head=words[wait_q_idx],
                ),
                thread_names,
            )

        semaphores[label] = SemaphoreInfo(
            name=label,
            address=address,
            count=words[count_idx],
            limit=words[limit_idx],
            waiters=waiters,
        )

    return semaphores


def _struct_words(elf: ElfInspector, struct_name: str) -> int:
    """Word count covering a struct, rounded up so a trailing partial word is read."""
    return (elf.get_struct_size(struct_name) + 3) // 4
