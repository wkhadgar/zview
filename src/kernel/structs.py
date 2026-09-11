# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Struct geometry helpers shared by the object walkers."""

from backend.elf_inspector import ElfInspector


def struct_words(elf: ElfInspector, struct_name: str) -> int:
    """Word count covering a struct, rounded up so a trailing partial word is read."""
    return (elf.get_struct_size(struct_name) + 3) // 4
