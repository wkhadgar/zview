# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from backend.elf_inspector import ElfInspector


@pytest.fixture
def elf_path():
    base_dir = Path(__file__).parent
    return base_dir / "fixtures" / "zephyr.elf"


@pytest.fixture
def parser(elf_path):
    parser = ElfInspector(str(elf_path))
    yield parser


def test_symbol_address_lookup(parser):
    addr = parser.get_symbol_info("_kernel", "address")
    assert len(addr) == 1
    assert isinstance(addr[0], int)
    assert addr[0] > 0


def test_struct_member_offsets(parser):
    ts_offset = parser.get_struct_member_offset("k_thread", "base")
    assert ts_offset == 0

    val_offset = parser.get_struct_member_offset("k_thread", "init_data")
    assert val_offset == 100


def test_struct_size(parser):
    size = parser.get_struct_size("k_thread")
    assert size == 192


def test_find_variable_by_struct_type(parser):
    vars_found = parser.find_struct_variable_names("k_thread")

    assert vars_found is not None
    assert "z_main_thread" in vars_found


def test_struct_not_found(parser):
    with pytest.raises(LookupError):
        parser.get_struct_size("k_struct")


def test_find_struct_variable_names_order_is_deterministic(elf_path):
    """Two fresh ``ElfInspector`` instances on the same ELF return identical order."""
    first = ElfInspector(str(elf_path)).find_struct_variable_names("k_thread")
    second = ElfInspector(str(elf_path)).find_struct_variable_names("k_thread")
    assert first is not None
    assert first == second
