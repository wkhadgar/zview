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


def _inspector(
    variables: list[str],
    located: dict[str, list[int]] | None = None,
    declarations: list[str] | None = None,
    symbols: dict[str, list[tuple[int, int, bool]]] | None = None,
) -> ElfInspector:
    """
    ``ElfInspector`` over hand-written DWARF and symbol data, with no ELF on disk.

    ``symbols`` maps a name to the ``(address, size, is a data object)`` row of
    each symbol table entry carrying it.
    """
    inspector = ElfInspector.__new__(ElfInspector)
    inspector._struct_sizes = {"k_sem": 16}
    inspector._struct_variables = {"k_sem": variables}
    inspector._struct_addresses = {"k_sem": located or {}}
    inspector._struct_declarations = {"k_sem": declarations or []}

    symbols = symbols or {}
    inspector._symbols_address = {n: [a for a, _, _ in rows] for n, rows in symbols.items()}
    inspector._symbols_size = {n: [s for _, s, _ in rows] for n, rows in symbols.items()}
    inspector._symbols_is_object = {n: [o for _, _, o in rows] for n, rows in symbols.items()}
    return inspector


def test_a_same_named_pointer_is_not_an_instance():
    """Zephyr's Bluetooth controller has this pair, in ull.c and hci_driver.c."""
    inspector = _inspector(
        ["sem_recv"],
        located={"sem_recv": [0x20003110]},
        symbols={"sem_recv": [(0x20002D34, 4, True), (0x20003110, 16, True)]},
    )

    assert inspector.find_struct_instances("k_sem") == {"sem_recv": [0x20003110]}


def test_a_function_local_is_not_an_instance():
    """A local's location names a frame, so it has no address to read."""
    inspector = _inspector(["sync_sem"])

    assert inspector.find_struct_instances("k_sem") == {}


def test_a_name_known_only_by_declaration_comes_from_the_symbol_table():
    """``k_sys_work_q`` is declared in every user of it and located in none."""
    inspector = _inspector(
        ["sys_sem"],
        declarations=["sys_sem"],
        symbols={"sys_sem": [(0x20000100, 16, True)]},
    )

    assert inspector.find_struct_instances("k_sem") == {"sys_sem": [0x20000100]}


def test_a_declared_name_whose_symbol_is_the_wrong_size_is_not_an_instance():
    inspector = _inspector(
        ["sem_recv"],
        declarations=["sem_recv"],
        symbols={"sem_recv": [(0x20002D34, 4, True)]},
    )

    assert inspector.find_struct_instances("k_sem") == {}


def test_a_declared_name_that_is_not_a_data_symbol_is_not_an_instance():
    inspector = _inspector(
        ["sem"],
        declarations=["sem"],
        symbols={"sem": [(0x8001234, 16, False)]},
    )

    assert inspector.find_struct_instances("k_sem") == {}


def test_a_located_instance_carries_its_dwarf_address(parser):
    assert parser.find_struct_instances("k_heap") == {
        "my_kernel_heap": parser.get_symbol_info("my_kernel_heap", "address")
    }


def test_a_declared_instance_still_resolves(parser):
    """``z_main_thread`` has no DWARF location in a Zephyr build."""
    instances = parser.find_struct_instances("k_thread")

    assert instances["z_main_thread"] == parser.get_symbol_info("z_main_thread", "address")


def test_an_object_carries_the_site_it_is_declared_at(parser):
    address = parser.find_struct_instances("k_heap")["my_kernel_heap"][0]

    path, line = parser.decl_site(address)

    assert path.endswith("samples/basic/sys_heap/src/main.c")
    assert line == 21


def test_a_function_carries_the_site_it_is_defined_at(parser):
    """A Thumb pointer's low bit is not part of the address."""
    address = parser.get_symbol_info("main", "address")[0]

    assert parser.function_site(address) == parser.function_site(address | 1)
    path, line = parser.function_site(address)
    assert path.endswith("samples/basic/sys_heap/src/main.c")
    assert line == 83


def test_an_address_with_no_site_resolves_to_nothing(parser):
    """A function the linker dropped keeps a DIE at zero, which is not a site."""
    assert parser.decl_site(0x1) is None
    assert parser.function_site(0) is None
    assert parser.function_site(0x1) is None


def test_a_site_survives_the_cache(elf_path):
    """The sites ride in the cached scan, so a second session needs no rescan."""
    first = ElfInspector(str(elf_path))
    address = first.find_struct_instances("k_heap")["my_kernel_heap"][0]

    assert ElfInspector(str(elf_path)).decl_site(address) == first.decl_site(address)


@pytest.fixture
def extern_sites():
    """
    Host build with version 5 line tables, from three files:

    s.h:2  ``extern struct k_sem same_line_sem;``
    b.c:2  ``struct k_sem same_line_sem;``, the same line number as its declaration
    a.c:2  ``extern struct k_sem same_file_sem;``, then a.c:4 defines it
    a.c:6  ``main``
    """
    return ElfInspector(str(Path(__file__).parent / "fixtures" / "extern_sites.elf"))


def test_a_version_5_line_table_resolves_a_function(extern_sites):
    path, line = extern_sites.function_site(extern_sites.get_symbol_info("main", "address")[0])

    assert (path, line) == ("/src/a.c", 6)
