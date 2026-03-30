# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection


class ElfInspector:
    def __init__(self, elf_path: str):
        self._path = elf_path

        self._symbols_address: dict[str, list[int]] = {}
        self._symbols_size: dict[str, list[int]] = {}
        self._struct_sizes: dict[str, int] = {}
        self._struct_members: dict[str, dict[str, int]] = {}
        self._struct_variables: dict[str, list[str]] = {}

        self._elfclass = 0
        self._little_endian = True

        print("Loading ELF. This may take a while...", end="", flush=True)
        self._perform_single_pass_scan()
        print(" OK.")

    @property
    def elf(self):
        """Mock object to satisfy ZScraper's dependency on self.elf.little_endian."""

        class MockElf:
            little_endian = self._little_endian
            elfclass = self._elfclass

        return MockElf()

    def _perform_single_pass_scan(self):
        """
        Executes a strictly O(N) sweep of the ELF and DWARF tree.
        Extracts all necessary data into local dictionaries.
        """
        try:
            with open(self._path, "rb") as file:
                elf = ELFFile(file)
                self._little_endian = elf.little_endian
                self._elfclass = elf.elfclass

                symtab = elf.get_section_by_name(".symtab")
                if symtab and isinstance(symtab, SymbolTableSection):
                    for sym in symtab.iter_symbols():
                        name = sym.name
                        if not name:
                            continue
                        self._symbols_address.setdefault(name, []).append(sym.entry["st_value"])
                        self._symbols_size.setdefault(name, []).append(sym.entry["st_size"])

                if not elf.has_dwarf_info():
                    raise ValueError("ELF file lacks DWARF debug information.")

                dwarf = elf.get_dwarf_info()
                offset_to_struct = {}
                pending_vars = []

                for CU in dwarf.iter_CUs():
                    cu_offset = CU.cu_offset
                    for die in CU.iter_DIEs():
                        if die.tag == "DW_TAG_structure_type":
                            name_attr = die.attributes.get("DW_AT_name")
                            if not name_attr:
                                continue

                            struct_name = name_attr.value.decode(errors="ignore")
                            offset_to_struct[die.offset] = struct_name

                            # Skip DWARF forward declarations that lack members
                            if die.attributes.get("DW_AT_declaration"):
                                continue

                            size_attr = die.attributes.get("DW_AT_byte_size")
                            if size_attr:
                                self._struct_sizes[struct_name] = size_attr.value

                            # Initialize the dictionary safely without overwriting existing members
                            self._struct_members.setdefault(struct_name, {})

                            for child in die.iter_children():
                                if child.tag == "DW_TAG_member":
                                    m_name_attr = child.attributes.get("DW_AT_name")
                                    loc_attr = child.attributes.get("DW_AT_data_member_location")
                                    if m_name_attr and loc_attr:
                                        m_name = m_name_attr.value.decode(errors="ignore")
                                        loc = (
                                            loc_attr.value[1]
                                            if loc_attr.form == "DW_FORM_exprloc"
                                            else loc_attr.value
                                        )
                                        self._struct_members[struct_name][m_name] = loc

                        elif die.tag == "DW_TAG_variable":
                            name_attr = die.attributes.get("DW_AT_name")
                            type_attr = die.attributes.get("DW_AT_type")
                            if name_attr and type_attr:
                                var_name = name_attr.value.decode(errors="ignore")
                                # DWARF type references are relative to the CU start
                                type_offset = type_attr.value + cu_offset
                                pending_vars.append((var_name, type_offset))

                for var_name, type_offset in pending_vars:
                    if type_offset in offset_to_struct:
                        struct_name = offset_to_struct[type_offset]
                        self._struct_variables.setdefault(struct_name, []).append(var_name)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"ELF file not found at '{self._path}'") from e

    def get_symbol_info(self, symbol_name: str, info: Literal["address", "size"]) -> list[int]:
        if info == "address":
            if symbol_name not in self._symbols_address:
                raise LookupError(f"Symbol '{symbol_name}' address not found.")
            return self._symbols_address[symbol_name]
        elif info == "size":
            if symbol_name not in self._symbols_size:
                raise LookupError(f"Symbol '{symbol_name}' size not found.")
            return self._symbols_size[symbol_name]
        else:
            raise ValueError(f"Invalid info type: {info}")

    def get_struct_member_offset(self, struct_name: str, member_name: str) -> int:
        if struct_name not in self._struct_members:
            raise LookupError(f"Struct '{struct_name}' not found.")
        if member_name not in self._struct_members[struct_name]:
            raise LookupError(f"Member '{member_name}' not found in struct '{struct_name}'.")
        return self._struct_members[struct_name][member_name]

    def get_struct_size(self, struct_name: str) -> int:
        if struct_name not in self._struct_sizes:
            raise LookupError(f"Size info for struct '{struct_name}' not found.")
        return self._struct_sizes[struct_name]

    def find_struct_variable_names(self, struct_name: str) -> list[str] | None:
        if struct_name not in self._struct_variables:
            return None
        return list(set(self._struct_variables[struct_name]))

    def close(self):
        pass
