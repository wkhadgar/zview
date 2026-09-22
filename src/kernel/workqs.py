# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_work_q`` walker."""

from backend.base import AbstractScraper, WorkqInfo
from backend.elf_inspector import ElfInspector
from kernel.layout import KernelLayout
from kernel.object_names import label_instances
from kernel.wait_queues import resolve_waiter_names, walk_wait_queue

_MAX_PENDING = 16


def walk_pending(
    scraper: AbstractScraper,
    elf: ElfInspector,
    head: int,
    layout: KernelLayout,
    *,
    max_items: int = _MAX_PENDING,
) -> tuple[tuple[str, ...], bool]:
    """
    Name the work items queued on a ``sys_slist_t``, in queue order.

    Returns ``(names, truncated)``. A null pointer is the only terminator,
    and the reads are unsynchronized, so a relinked list can appear to loop.
    """
    names: list[str] = []
    seen: set[int] = set()
    node = head

    while node and len(names) < max_items:
        if node in seen:
            break

        seen.add(node)
        item = node - layout.work_node
        handler = scraper.read32(item + layout.work_handler)[0]
        names.append(_item_name(elf, item, handler))
        node = scraper.read32(node)[0]

    return tuple(names), bool(node)


def _item_name(elf: ElfInspector, item: int, handler: int) -> str:
    """``tick_work (work_handler)``, or the handler alone for an item with no symbol."""
    handler_name = elf.get_symbol_name_at(handler) or f"0x{handler:X}"
    item_name = elf.get_symbol_name_at(item)
    return f"{item_name} ({handler_name})" if item_name else handler_name


def walk_workqs(
    scraper: AbstractScraper,
    elf: ElfInspector,
    addresses: dict[str, list[int]],
    layout: KernelLayout,
    thread_names: dict[int, str] | None = None,
    *,
    walk_waiters: bool = True,
    max_pending: int = _MAX_PENDING,
) -> dict[str, WorkqInfo]:
    """
    Read every statically declared ``k_work_q`` and return ``{label: WorkqInfo}``.

    ``addresses`` maps a symbol name to its instance addresses. Each queue
    costs one read of the span its fields sit in, plus one per pending item and
    per draining thread; the struct itself is mostly a deprecated embedded
    ``k_thread``. ``walk_waiters=False`` leaves ``waiters`` as ``None``;
    required on a non-walkable (scalable) layout.
    """
    offsets = [layout.workq_thread_id, layout.workq_pending, layout.workq_flags]
    if layout.workq_drainq is not None:
        offsets.append(layout.workq_drainq)

    base = min(offsets)
    span = (max(offsets) + 4 - base) // 4

    def word(words: tuple[int, ...], offset: int) -> int:
        return words[(offset - base) // 4]

    workqs: dict[str, WorkqInfo] = {}
    for label, address in label_instances(addresses):
        words = tuple(scraper.read32(address + base, span))

        pending, truncated = walk_pending(
            scraper,
            elf,
            word(words, layout.workq_pending),
            layout,
            max_items=max_pending,
        )

        waiters: tuple[str, ...] | None = None
        if walk_waiters and layout.workq_drainq is not None and layout.thread_qnode is not None:
            waiters = resolve_waiter_names(
                walk_wait_queue(
                    scraper,
                    address + layout.workq_drainq,
                    layout.thread_qnode,
                    head=word(words, layout.workq_drainq),
                ),
                thread_names,
            )

        workqs[label] = WorkqInfo(
            name=label,
            address=address,
            flags=word(words, layout.workq_flags),
            pending=pending,
            pending_truncated=truncated,
            thread_name=(thread_names or {}).get(word(words, layout.workq_thread_id)),
            waiters=waiters,
        )

    return workqs
