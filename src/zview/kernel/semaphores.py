# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_sem`` walker."""

from zview.backend.base import SemaphoreInfo
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.kernel.structs import struct_words
from zview.kernel.wait_queues import resolve_waiter_names, walk_wait_queue


def walk_semaphores(session: Session, addresses: dict[str, list[int]]) -> dict[str, SemaphoreInfo]:
    """
    Read every statically declared ``k_sem`` and return ``{label: SemaphoreInfo}``.

    ``addresses`` maps a symbol name to its instance addresses. Each semaphore
    costs one bulk struct read plus one read per waiting thread. On a wait
    queue layout that cannot be walked (scalable), ``waiters`` is ``None``.
    """
    scraper, elf, layout = session.scraper, session.elf, session.layout
    walk_waiters, thread_names = session.walks_waiters, session.thread_names
    words_to_read = struct_words(elf, "k_sem")
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
