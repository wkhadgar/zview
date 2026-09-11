# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_msgq`` walker."""

from backend.base import AbstractScraper, MsgqInfo
from backend.elf_inspector import ElfInspector
from kernel.layout import KernelLayout
from kernel.object_names import label_instances
from kernel.structs import struct_words
from kernel.wait_queues import resolve_waiter_names, walk_wait_queue


def walk_msgqs(
    scraper: AbstractScraper,
    elf: ElfInspector,
    addresses: dict[str, list[int]],
    layout: KernelLayout,
    thread_names: dict[int, str] | None = None,
    *,
    walk_waiters: bool = True,
) -> dict[str, MsgqInfo]:
    """
    Read every statically declared ``k_msgq`` and return ``{label: MsgqInfo}``.

    ``addresses`` maps a symbol name to its instance addresses. Each queue
    costs one bulk struct read plus one read per waiting thread.
    ``walk_waiters=False`` skips the wait queue and leaves ``waiters`` as
    ``None``; required on a non-walkable (scalable) layout.
    """
    words_to_read = struct_words(elf, "k_msgq")
    used_idx = layout.msgq_used_msgs // 4
    max_idx = layout.msgq_max_msgs // 4
    size_idx = layout.msgq_msg_size // 4
    wait_q_idx = layout.msgq_wait_q // 4

    msgqs: dict[str, MsgqInfo] = {}
    for label, address in label_instances(addresses):
        words = scraper.read32(address, words_to_read)

        waiters: tuple[str, ...] | None = None
        if walk_waiters and layout.thread_qnode is not None:
            waiters = resolve_waiter_names(
                walk_wait_queue(
                    scraper,
                    address + layout.msgq_wait_q,
                    layout.thread_qnode,
                    head=words[wait_q_idx],
                ),
                thread_names,
            )

        msgqs[label] = MsgqInfo(
            name=label,
            address=address,
            used_msgs=words[used_idx],
            max_msgs=words[max_idx],
            msg_size=words[size_idx],
            waiters=waiters,
        )

    return msgqs
