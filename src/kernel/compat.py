# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
Zephyr kernel offset and Kconfig compatibility tables.

Field offsets are read from DWARF at runtime. This module holds the candidate
member paths for each logical field, the Kconfig probes and the Zephyr version
lookup.

``resolve_offset`` tries candidate paths in order, summing the steps of a path,
so a field reached through a sub-struct such as ``k_mem_slab.info.num_used`` is
one entry. ``has_kconfig`` reads the ``CONFIG_*`` absolute symbols from the ELF
symbol table. ``detect_zephyr_version`` reads ``KERNEL_VERSION_STRING`` from the
build tree.

The tables match the struct definitions in Zephyr v3.0.0, v3.3.0, v3.7.0,
v4.0.0 and v4.4.0. Where a field has more than one candidate, the comment names
the release that differs.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from backend.elf_inspector import ElfInspector

# A member path whose per-step offsets are summed: (("k_mem_slab", "info"),
# ("k_mem_slab_info", "num_used")) resolves a field inside a sub-struct.
MemberPath = Sequence[tuple[str, str]]
FieldTable = dict[str, tuple[MemberPath, ...]]

CONFIG_WAITQ_SIMPLE = "CONFIG_WAITQ_SIMPLE"
CONFIG_WAITQ_SCALABLE = "CONFIG_WAITQ_SCALABLE"

WaitQFlavor = Literal["simple", "scalable", "unknown"]

_VERSION_HEADER = Path("include") / "generated" / "zephyr" / "version.h"
_VERSION_PATTERN = re.compile(r'#define\s+KERNEL_VERSION_STRING\s+"([^"]+)"')
_VERSION_SEARCH_DEPTH = 4

# --- Threads -----------------------------------------------------------------

# Required for thread walking.
THREAD_FIELDS: FieldTable = {
    "threads_head": ((("z_kernel", "threads"),),),
    "thread_next": ((("k_thread", "next_thread"),),),
    "stack_start": ((("k_thread", "stack_info"), ("_thread_stack_info", "start")),),
    "stack_size": ((("k_thread", "stack_info"), ("_thread_stack_info", "size")),),
}

# CONFIG_THREAD_NAME.
THREAD_NAME_FIELDS: FieldTable = {
    "thread_name": ((("k_thread", "name"),),),
}

# CONFIG_THREAD_RUNTIME_STATS. z_kernel.usage exists from v3.7.0; on v3.0 and
# v3.3 this group does not resolve.
USAGE_FIELDS: FieldTable = {
    "cpu_usage": ((("z_kernel", "usage"),),),
    "thread_usage": (
        (("k_thread", "base"), ("_thread_base", "usage"), ("k_cycle_stats", "total")),
    ),
}

# Metadata mirroring `kernel threads list`. Resolved individually.
THREAD_META_FIELDS: FieldTable = {
    "thread_priority": ((("k_thread", "base"), ("_thread_base", "prio")),),
    "thread_state": ((("k_thread", "base"), ("_thread_base", "thread_state")),),
    "thread_user_options": ((("k_thread", "base"), ("_thread_base", "user_options")),),
    "thread_entry": (
        (("k_thread", "entry"), ("__thread_entry", "pEntry")),
        # Single-underscore struct name carried by some vendor trees.
        (("k_thread", "entry"), ("_thread_entry", "pEntry")),
    ),
}

# Wait queues hold ``_thread_base.qnode_dlist`` nodes; a thread address is a
# node address minus this offset.
THREAD_QNODE_FIELDS: FieldTable = {
    "thread_qnode": ((("k_thread", "base"), ("_thread_base", "qnode_dlist")),),
}

# --- Heaps -------------------------------------------------------------------

# CONFIG_SYS_HEAP_RUNTIME_STATS. max_allocated_bytes exists from v3.3.0; on
# v3.0 this group does not resolve.
HEAP_FIELDS: FieldTable = {
    "heap_free_bytes": ((("z_heap", "free_bytes"),),),
    "heap_allocated_bytes": ((("z_heap", "allocated_bytes"),),),
    "heap_max_allocated_bytes": ((("z_heap", "max_allocated_bytes"),),),
    "heap_end_chunk": ((("z_heap", "end_chunk"),),),
}

# --- Synchronization primitives ----------------------------------------------

SEMAPHORE_FIELDS: FieldTable = {
    "sem_wait_q": ((("k_sem", "wait_q"),),),
    "sem_count": ((("k_sem", "count"),),),
    "sem_limit": ((("k_sem", "limit"),),),
}

MUTEX_FIELDS: FieldTable = {
    "mutex_wait_q": ((("k_mutex", "wait_q"),),),
    "mutex_owner": ((("k_mutex", "owner"),),),
    "mutex_lock_count": ((("k_mutex", "lock_count"),),),
}

MSGQ_FIELDS: FieldTable = {
    "msgq_wait_q": ((("k_msgq", "wait_q"),),),
    "msgq_msg_size": ((("k_msgq", "msg_size"),),),
    "msgq_max_msgs": ((("k_msgq", "max_msgs"),),),
    "msgq_used_msgs": ((("k_msgq", "used_msgs"),),),
}

EVENT_FIELDS: FieldTable = {
    "event_wait_q": ((("k_event", "wait_q"),),),
    "event_events": ((("k_event", "events"),),),
}

# The counters moved into a ``k_mem_slab_info`` sub-struct in v3.7.0; v3.0 and
# v3.3 carry them directly on k_mem_slab.
MEM_SLAB_FIELDS: FieldTable = {
    "mem_slab_wait_q": ((("k_mem_slab", "wait_q"),),),
    "mem_slab_num_blocks": (
        (("k_mem_slab", "info"), ("k_mem_slab_info", "num_blocks")),
        (("k_mem_slab", "num_blocks"),),
    ),
    "mem_slab_block_size": (
        (("k_mem_slab", "info"), ("k_mem_slab_info", "block_size")),
        (("k_mem_slab", "block_size"),),
    ),
    "mem_slab_num_used": (
        (("k_mem_slab", "info"), ("k_mem_slab_info", "num_used")),
        (("k_mem_slab", "num_used"),),
    ),
}

# CONFIG_MEM_SLAB_TRACE_MAX_UTILIZATION, and only inside the sub-struct layout.
MEM_SLAB_OPTIONAL_FIELDS: FieldTable = {
    "mem_slab_max_used": ((("k_mem_slab", "info"), ("k_mem_slab_info", "max_used")),),
}

# k_work has no wait queue; pending work sits on its queue's list.
WORK_FIELDS: FieldTable = {
    "work_handler": ((("k_work", "handler"),),),
    "work_queue": ((("k_work", "queue"),),),
    "work_flags": ((("k_work", "flags"),),),
}


@dataclass(frozen=True)
class ObjectSpec:
    """A kernel object type, its C struct, and the fields needed to read it."""

    label: str
    struct: str
    fields: FieldTable
    optional_fields: FieldTable | None = None


# Kernel object types and the fields needed to read them. Instances are found
# by searching DWARF for globals of ``struct``, so only statically declared
# objects appear.
KERNEL_OBJECTS: dict[str, ObjectSpec] = {
    "semaphores": ObjectSpec("SEM", "k_sem", SEMAPHORE_FIELDS),
    "mutexes": ObjectSpec("MTX", "k_mutex", MUTEX_FIELDS),
    "msgqs": ObjectSpec("MSG", "k_msgq", MSGQ_FIELDS),
    "events": ObjectSpec("EVT", "k_event", EVENT_FIELDS),
    "mem_slabs": ObjectSpec("SLB", "k_mem_slab", MEM_SLAB_FIELDS, MEM_SLAB_OPTIONAL_FIELDS),
    "work": ObjectSpec("WRK", "k_work", WORK_FIELDS),
}


def resolve_offset(elf: ElfInspector, candidates: Sequence[MemberPath]) -> int | None:
    """
    Return the offset of the first candidate path this ELF resolves.

    Every step of a path must resolve for that path to match. Returns ``None``
    when no candidate resolves.
    """
    for path in candidates:
        offset = 0
        for struct_name, member in path:
            try:
                offset += elf.get_struct_member_offset(struct_name, member)
            except LookupError:
                break
        else:
            return offset

    return None


def resolve_fields(elf: ElfInspector, fields: FieldTable) -> dict[str, int] | None:
    """
    Resolve a whole field group.

    Returns ``None`` if any member is missing; a group never resolves
    partially.
    """
    resolved: dict[str, int] = {}
    for name, candidates in fields.items():
        offset = resolve_offset(elf, candidates)
        if offset is None:
            return None
        resolved[name] = offset

    return resolved


def resolve_optional_fields(elf: ElfInspector, fields: FieldTable) -> dict[str, int]:
    """
    Resolve the members of a group this ELF has, skipping the rest.

    For groups whose fields are independent, such as the thread metadata.
    """
    resolved: dict[str, int] = {}
    for name, candidates in fields.items():
        offset = resolve_offset(elf, candidates)
        if offset is not None:
            resolved[name] = offset

    return resolved


def has_kconfig(elf: ElfInspector, option: str) -> bool:
    """
    Report whether a Kconfig option was enabled in the build.

    Zephyr emits each enabled ``CONFIG_*`` as an absolute symbol; a disabled
    option is absent from the symbol table.
    """
    try:
        elf.get_symbol_info(option, "address")
    except LookupError:
        return False

    return True


def waitq_flavor(elf: ElfInspector) -> WaitQFlavor:
    """
    Report the ``_wait_q_t`` layout this build uses.

    ``simple`` is a ``sys_dlist_t``, ``scalable`` a red-black tree
    (``struct _priq_rb``). Only ``simple`` is walkable. ``unknown`` means the
    build advertises neither and is treated as not walkable.
    """
    if has_kconfig(elf, CONFIG_WAITQ_SIMPLE):
        return "simple"
    if has_kconfig(elf, CONFIG_WAITQ_SCALABLE):
        return "scalable"

    return "unknown"


def detect_zephyr_version(elf_path: Path | str) -> str | None:
    """
    Return ``KERNEL_VERSION_STRING`` for an ELF inside its build tree.

    Reads ``<build>/zephyr/include/generated/zephyr/version.h``, looked up
    beside the ELF and a few directories up. Returns ``None`` for an ELF
    outside a build tree.
    """
    elf = Path(elf_path)
    try:
        elf = elf.resolve()
    except OSError:
        return None

    for base in (elf.parent, *list(elf.parents)[:_VERSION_SEARCH_DEPTH]):
        header = base / _VERSION_HEADER
        try:
            if not header.is_file():
                continue
            match = _VERSION_PATTERN.search(header.read_text())
        except OSError:
            continue

        if match:
            return match.group(1)

    return None
