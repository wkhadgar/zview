# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_work_q`` group."""

from zview.backend.base import AbstractScraper, WorkqInfo
from zview.backend.elf_inspector import ElfInspector
from zview.kernel.compat import FieldTable
from zview.kernel.group import Group
from zview.kernel.layout import KernelLayout
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.kernel.wait_queues import resolve_waiter_names, walk_wait_queue

_MAX_PENDING = 16

# k_work has no wait queue; pending work sits on its queue's list.
WORK_FIELDS: FieldTable = {
    "work_node": ((("k_work", "node"),),),
    "work_handler": ((("k_work", "handler"),),),
    "work_queue": ((("k_work", "queue"),),),
    "work_flags": ((("k_work", "flags"),),),
}

WORKQ_FIELDS: FieldTable = {
    "workq_pending": ((("k_work_q", "pending"),),),
    "workq_flags": ((("k_work_q", "flags"),),),
}

# The queue's own thread: thread_id from v4.2.0, which deprecates the k_thread
# embedded in the queue that older releases name it by.
WORKQ_THREAD_ID_FIELDS: FieldTable = {
    "workq_thread_id": ((("k_work_q", "thread_id"),),),
}
WORKQ_THREAD_FIELDS: FieldTable = {
    "workq_thread": ((("k_work_q", "thread"),),),
}

# Threads blocked in k_work_queue_drain. notifyq holds the queue's own thread.
WORKQ_OPTIONAL_FIELDS: FieldTable = {
    "workq_drainq": ((("k_work_q", "drainq"),),),
}


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


class Workqs(Group):
    key = "workqs"
    struct = "k_work_q"
    # A queue's pending items are k_work nodes, so both must resolve.
    required = (WORKQ_FIELDS, WORK_FIELDS)
    optional = (WORKQ_OPTIONAL_FIELDS, WORKQ_THREAD_ID_FIELDS, WORKQ_THREAD_FIELDS)

    def walk(
        self,
        session: Session,
        addresses: dict[str, list[int]],
        *,
        max_pending: int = _MAX_PENDING,
    ) -> dict[str, WorkqInfo]:
        """
        Read every statically declared ``k_work_q`` and return ``{label: WorkqInfo}``.

        ``addresses`` maps a symbol name to its instance addresses. Each queue
        costs one read of the span its fields sit in, plus one per pending item and
        per draining thread; the struct itself is mostly a deprecated embedded
        ``k_thread``. On a wait queue layout that cannot be walked (scalable),
        ``waiters`` is ``None``.
        """
        scraper, elf, layout = session.scraper, session.elf, session.layout
        walk_waiters, thread_names = session.is_waitq_walkable, session.thread_names
        offsets = [layout.workq_pending, layout.workq_flags]
        offsets += [o for o in (layout.workq_thread_id, layout.workq_drainq) if o is not None]

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

            thread = None
            if layout.workq_thread_id is not None:
                thread = word(words, layout.workq_thread_id)
            elif layout.workq_thread is not None:
                thread = address + layout.workq_thread

            workqs[label] = WorkqInfo(
                name=label,
                address=address,
                flags=word(words, layout.workq_flags),
                pending=pending,
                is_pending_truncated=truncated,
                thread_name=(thread_names or {}).get(thread),
                waiters=waiters,
                decl_site=session.elf.decl_site(address),
            )

        return workqs
