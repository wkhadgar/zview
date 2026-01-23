# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from typing import Literal

from elftools.dwarf.die import DIE
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection


class ElfInspector:
    def __init__(self, elf_path: str):
        self._symbol_cache: defaultdict[str, defaultdict[str, list[int]]] = defaultdict(
            defaultdict
        )
        self._struct_member_offset_cache: defaultdict[str, defaultdict[str, int]] = (
            defaultdict(defaultdict)
        )

        self.file = self._open_elf_file(elf_path)
        self.elf = ELFFile(self.file)
        self.dwarf = self._get_dwarf_info()

    @staticmethod
    def _open_elf_file(path: str):
        try:
            return open(path, "rb")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"ELF file not found at '{path}'") from e

    def _get_dwarf_info(self):
        if not self.elf.has_dwarf_info():
            raise ValueError("ELF file lacks DWARF debug information.")
        return self.elf.get_dwarf_info()

    def get_symbol_info(
        self, symbol_name: str, info: Literal["address", "size"]
    ) -> list[int]:
        if (
            symbol_name in self._symbol_cache
            and info in self._symbol_cache[symbol_name]
        ):
            return self._symbol_cache[symbol_name][info]

        symtab = self.elf.get_section_by_name(".symtab")
        if symtab is None or not isinstance(symtab, SymbolTableSection):
            raise LookupError("'.symtab' section not found.")

        symbols = symtab.get_symbol_by_name(symbol_name)
        if symbols is None:
            raise LookupError(f"Symbol '{symbol_name}' not found.")
        elif len(symbols) == 1:
            match info:
                case "address":
                    value = [symbols[0].entry["st_value"]]
                case "size":
                    value = [symbols[0].entry["st_size"]]
        else:
            match info:
                case "address":
                    value = [symbol.entry["st_value"] for symbol in symbols]
                case "size":
                    value = [symbol.entry["st_size"] for symbol in symbols]

        self._symbol_cache[symbol_name][info] = value

        return value

    def get_struct_member_offset(self, struct_name: str, member_name: str) -> int:
        if (
            struct_name in self._struct_member_offset_cache
            and member_name in self._struct_member_offset_cache[struct_name]
        ):
            return self._struct_member_offset_cache[struct_name][member_name]

        struct_dies = self._find_struct_dies(struct_name)
        if struct_dies is None:
            raise LookupError(f"Struct '{struct_name}' not found.")

        member_dies = []
        for struct_die in struct_dies:
            member_dies.append(self._find_member_die(struct_die, member_name))
        if len(member_dies) == 0 or not any(member_dies):
            raise LookupError(
                f"Member '{member_name}' not found in struct '{struct_name}'."
            )

        member_dies[:] = [die for die in member_dies if die is not None]
        member_die = member_dies[0]

        offset = self._extract_member_offset(member_die)
        self._struct_member_offset_cache[struct_name][member_name] = offset

        return offset

    def get_struct_size(self, struct_name: str) -> int:
        struct_dies = self._find_struct_dies(struct_name)
        if struct_dies is None:
            raise LookupError(f"Struct '{struct_name}' not found.")

        if not struct_dies[0].attributes["DW_AT_byte_size"]:
            raise ValueError(f"Struct '{struct_name}' has no size information.")

        return getattr(struct_dies[0].attributes["DW_AT_byte_size"], "value")

    def _find_struct_dies(self, struct_name: str) -> list[DIE] | None:
        dies = []
        for CU in self.dwarf.iter_CUs():
            for die in CU.iter_DIEs():
                if die.tag == "DW_TAG_structure_type":
                    if struct_name == "*" and die.attributes.get("DW_AT_name"):
                        dies.append(die)
                    if (
                        die.attributes.get("DW_AT_name")
                        and die.attributes["DW_AT_name"].value.decode(errors="ignore")
                        == struct_name
                    ):
                        dies.append(die)

        return dies if len(dies) else None

    def find_struct_variable_names(self, struct_name: str) -> list[str] | None:
        struct_dies = self._find_struct_dies(struct_name)
        if struct_dies is None:
            raise LookupError(f"Struct '{struct_name}' not found.")

        struct_variables = []
        for CU_ in self.dwarf.iter_CUs():
            for die_ in CU_.iter_DIEs():
                if die_.tag == "DW_TAG_variable":
                    die_type_offset = die_.attributes.get("DW_AT_type")
                    if die_type_offset is None:
                        continue
                    if (die_type_offset.value + die_.cu.cu_offset) in [
                        st_die.offset for st_die in struct_dies
                    ]:
                        die_name = die_.attributes.get("DW_AT_name").value
                        struct_variables.append(die_name.decode("utf-8"))

        found_variables = list(set(struct_variables))
        return found_variables if len(found_variables) else None

    @staticmethod
    def _find_member_die(struct_die: DIE, member_name: str) -> DIE | None:
        for child in struct_die.iter_children():
            if child.tag == "DW_TAG_member" and child.attributes.get("DW_AT_name"):
                child_member = child.attributes["DW_AT_name"].value.decode(
                    errors="ignore"
                )
                if child_member == member_name:
                    return child
        return None

    @staticmethod
    def _extract_member_offset(die: DIE) -> int:
        loc = die.attributes.get("DW_AT_data_member_location")
        if loc is None:
            raise ValueError("Member offset not specified.")
        return loc.value[1] if loc.form == "DW_FORM_exprloc" else loc.value

    def close(self):
        self.file.close()
