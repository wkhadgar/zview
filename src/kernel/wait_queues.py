# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr wait queue (``_wait_q_t``) walker."""

from backend.base import AbstractScraper

# A sys_dlist_t and a sys_dnode_t are the same two-pointer _dnode: the list's
# first word is its head, a node's first word is its next.
_NEXT_WORD = 0


def walk_wait_queue(
    scraper: AbstractScraper,
    list_address: int,
    qnode_offset: int,
    *,
    head: int | None = None,
    max_waiters: int = 32,
) -> list[int]:
    """
    Return the addresses of the threads queued on a dlist wait queue.

    Threads come back in queue order, highest priority first.
    ``qnode_offset`` is the offset of ``_thread_base.qnode_dlist`` within
    ``k_thread``; a queued thread's address is its node address minus that
    offset. ``head`` supplies an already-read head pointer.

    Valid only for the ``simple`` wait queue layout (see
    ``compat.waitq_flavor``). An empty dlist is self-referential
    (``head == list_address``); traversal also stops on a null pointer, on a
    repeated node, and at ``max_waiters``.
    """
    if list_address == 0:
        return []

    node = scraper.read32(list_address)[0] if head is None else head

    waiters: list[int] = []
    seen: set[int] = set()

    while node not in (0, list_address) and len(waiters) < max_waiters:
        if node in seen:
            break
        seen.add(node)

        waiters.append(node - qnode_offset)
        node = scraper.read32(node + _NEXT_WORD)[0]

    return waiters


def resolve_waiter_names(
    waiter_addresses: list[int],
    thread_names: dict[int, str] | None,
) -> tuple[str, ...]:
    """
    Label waiting threads, falling back to the address when the name is unknown.

    An address absent from ``thread_names`` renders as ``thread @ 0x...``.
    """
    names = thread_names or {}

    return tuple(names.get(address, f"thread @ 0x{address:X}") for address in waiter_addresses)
