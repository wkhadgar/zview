# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_mutex`` group."""

from zview.backend.base import MutexInfo
from zview.kernel.compat import FieldTable
from zview.kernel.group import Group
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.kernel.structs import struct_words
from zview.kernel.wait_queues import resolve_waiter_names, walk_wait_queue

MUTEX_FIELDS: FieldTable = {
    "mutex_wait_q": ((("k_mutex", "wait_q"),),),
    "mutex_owner": ((("k_mutex", "owner"),),),
    "mutex_lock_count": ((("k_mutex", "lock_count"),),),
}


class Mutexes(Group):
    key = "mutexes"
    struct = "k_mutex"
    required = (MUTEX_FIELDS,)

    def walk(self, session: Session, addresses: dict[str, list[int]]) -> dict[str, MutexInfo]:
        """
        Read every statically declared ``k_mutex`` and return ``{label: MutexInfo}``.

        The owner pointer is resolved against the session's thread names. Each
        mutex costs one bulk struct read plus one read per waiting thread. A lock
        taken and released between two calls is not observed.
        """
        scraper, elf, layout = session.scraper, session.elf, session.layout
        walk_waiters, thread_names = session.is_waitq_walkable, session.thread_names
        words_to_read = struct_words(elf, "k_mutex")
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
