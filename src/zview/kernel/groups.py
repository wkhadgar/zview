# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""The groups a frame reads, each declared alike, in the order a frame reads them."""

from collections.abc import Callable
from dataclasses import dataclass

from zview.kernel import compat
from zview.kernel.heaps import walk_heaps
from zview.kernel.mem_slabs import walk_mem_slabs
from zview.kernel.msgqs import walk_msgqs
from zview.kernel.mutexes import walk_mutexes
from zview.kernel.semaphores import walk_semaphores
from zview.kernel.session import Session
from zview.kernel.workqs import walk_workqs


@dataclass(frozen=True)
class Group:
    """
    A kind of kernel object a frame reads.

    ``key`` names it in a frame, a recording's features and a selection. The
    group is read only when every ``required`` table resolves, and each
    ``optional`` table resolves field by field. Each of its ``parts`` turns on
    some of its reads: on live, on in a replay only when the recording names
    it, as it does every part that was on. ``discover`` finds its rows once per
    ELF, and ``walk`` reads them every frame as ``{label: row}``.
    """

    key: str
    struct: str
    required: tuple[compat.FieldTable, ...]
    optional: tuple[compat.FieldTable, ...]
    parts: tuple[str, ...]
    discover: Callable[[Session, "Group"], dict[str, list[int]]]
    walk: Callable[[Session, dict[str, list[int]]], dict]


def discover_globals(session: Session, group: Group) -> dict[str, list[int]]:
    """``{symbol: addresses}`` of the struct's statically declared instances."""
    return session.elf.find_struct_instances(group.struct)


GROUPS: tuple[Group, ...] = (
    Group(
        key="heaps",
        struct="k_heap",
        required=(compat.HEAP_FIELDS,),
        optional=(compat.HEAP_OPTIONAL_FIELDS,),
        # Each heap's wait queue, walked since after heaps were first read.
        parts=("heap_waiters",),
        discover=discover_globals,
        walk=walk_heaps,
    ),
    Group(
        key="semaphores",
        struct="k_sem",
        required=(compat.SEMAPHORE_FIELDS,),
        optional=(),
        parts=(),
        discover=discover_globals,
        walk=walk_semaphores,
    ),
    Group(
        key="mutexes",
        struct="k_mutex",
        required=(compat.MUTEX_FIELDS,),
        optional=(),
        parts=(),
        discover=discover_globals,
        walk=walk_mutexes,
    ),
    Group(
        key="msgqs",
        struct="k_msgq",
        required=(compat.MSGQ_FIELDS,),
        optional=(),
        parts=(),
        discover=discover_globals,
        walk=walk_msgqs,
    ),
    Group(
        key="mem_slabs",
        struct="k_mem_slab",
        required=(compat.MEM_SLAB_FIELDS,),
        optional=(compat.MEM_SLAB_OPTIONAL_FIELDS,),
        parts=(),
        discover=discover_globals,
        walk=walk_mem_slabs,
    ),
    Group(
        key="workqs",
        struct="k_work_q",
        # A queue's pending items are k_work nodes, so both must resolve.
        required=(compat.WORKQ_FIELDS, compat.WORK_FIELDS),
        optional=(compat.WORKQ_OPTIONAL_FIELDS,),
        parts=(),
        discover=discover_globals,
        walk=walk_workqs,
    ),
)
