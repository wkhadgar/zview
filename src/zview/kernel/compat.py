# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
Zephyr kernel offset resolution and Kconfig compatibility probes.

Field offsets are read from DWARF at runtime. This module holds the resolvers,
the Kconfig probes and the Zephyr version lookup. Each group's module holds the
candidate member paths of its own fields.

``resolve_offset`` tries candidate paths in order, summing the steps of a path,
so a field reached through a sub-struct such as ``k_mem_slab.info.num_used`` is
one entry. ``has_kconfig`` reads the ``CONFIG_*`` absolute symbols from the ELF
symbol table. ``detect_zephyr_version`` reads ``KERNEL_VERSION_STRING`` from the
build tree.

The groups' tables match the struct definitions in Zephyr v3.0.0, v3.3.0,
v3.7.0, v4.0.0 and v4.4.0. Where a field has more than one candidate, the
comment names the release that differs.
"""

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from zview.backend.elf_inspector import ElfInspector

# A member path whose per-step offsets are summed: (("k_mem_slab", "info"),
# ("k_mem_slab_info", "num_used")) resolves a field inside a sub-struct.
MemberPath = Sequence[tuple[str, str]]
FieldTable = dict[str, tuple[MemberPath, ...]]

CONFIG_WAITQ_SIMPLE = "CONFIG_WAITQ_SIMPLE"
CONFIG_WAITQ_SCALABLE = "CONFIG_WAITQ_SCALABLE"
CONFIG_INIT_STACKS = "CONFIG_INIT_STACKS"

WaitQFlavor = Literal["simple", "scalable", "unknown"]

_VERSION_HEADER = Path("include") / "generated" / "zephyr" / "version.h"
_VERSION_PATTERN = re.compile(r'#define\s+KERNEL_VERSION_STRING\s+"([^"]+)"')
_VERSION_SEARCH_DEPTH = 4


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
