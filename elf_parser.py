from typing import Literal

from elftools.dwarf.die import DIE
from elftools.elf.elffile import ELFFile
from collections import defaultdict


class ZephyrSymbolParser:
    def __init__(self, elf_path: str):
        self._symbol_cache: defaultdict[str:defaultdict[str:int]] = defaultdict(defaultdict)
        self._struct_member_offset_cache: defaultdict[str:defaultdict[str:int]] = defaultdict(defaultdict)

        self.file = self._open_elf_file(elf_path)
        self.elf = ELFFile(self.file)
        self.dwarf = self._get_dwarf_info()

    @staticmethod
    def _open_elf_file(path: str):
        try:
            return open(path, "rb")
        except FileNotFoundError as e:
            raise RuntimeError(f"ELF file not found at '{path}'") from e

    def _get_dwarf_info(self):
        if not self.elf.has_dwarf_info():
            raise RuntimeError("ELF file lacks DWARF debug information.")
        return self.elf.get_dwarf_info()

    def get_symbol_info(self, symbol_name: str, info: Literal["address", "size"]) -> int:
        if symbol_name in self._symbol_cache and info in self._symbol_cache[symbol_name]:
            return self._symbol_cache[symbol_name][info]

        symtab = self.elf.get_section_by_name(".symtab")
        if symtab is None:
            raise RuntimeError("'.symtab' section not found.")

        symbols = symtab.get_symbol_by_name(symbol_name)
        if not symbols:
            raise RuntimeError(f"Symbol '{symbol_name}' not found.")

        match info:
            case "address":
                value = symbols[0].entry["st_value"]
            case "size":
                value = symbols[0].entry["st_size"]
            case _:
                raise Exception(f"'{info}' is not a valid information.")

        self._symbol_cache[symbol_name][info] = value

        return value

    def get_struct_member_offset(self, struct_name: str, member_name: str) -> int:
        if struct_name in self._struct_member_offset_cache and member_name in self._struct_member_offset_cache[
            struct_name]:
            return self._struct_member_offset_cache[struct_name][member_name]

        struct_die = self._find_struct_die(struct_name)
        if not struct_die:
            raise RuntimeError(f"Struct '{struct_name}' not found.")

        member_die = self._find_member_die(struct_die, member_name)
        if not member_die:
            raise RuntimeError(f"Member '{member_name}' not found in struct '{struct_name}'.")

        offset = self._extract_member_offset(member_die)
        self._struct_member_offset_cache[struct_name][member_name] = offset

        return offset

    def get_struct_size(self, struct_name: str) -> int:
        struct_die = self._find_struct_die(struct_name)
        if not struct_die:
            raise RuntimeError(f"Struct '{struct_name}' not found.")


        if not struct_die.attributes["DW_AT_byte_size"]:
            raise RuntimeError(f"Struct '{struct_name}' has no size information.")

        return getattr(struct_die.attributes["DW_AT_byte_size"], "value")

    def _find_struct_die(self, struct_name: str) -> DIE | None:
        for CU in self.dwarf.iter_CUs():
            for die in CU.iter_DIEs():
                if (die.tag == "DW_TAG_structure_type"
                        and die.attributes.get("DW_AT_name")
                        and die.attributes["DW_AT_name"].value.decode(errors="ignore") == struct_name):
                    return die

        return None

    @staticmethod
    def _find_member_die(struct_die: DIE, member_name: str) -> DIE | None:
        for child in struct_die.iter_children():
            if (child.tag == "DW_TAG_member" and
                    child.attributes.get("DW_AT_name") and
                    child.attributes["DW_AT_name"].value.decode(errors="ignore") == member_name):
                return child
        return None

    @staticmethod
    def _extract_member_offset(die: DIE) -> int:
        loc = die.attributes.get("DW_AT_data_member_location")
        if loc is None:
            raise RuntimeError("Member offset not specified.")
        return loc.value[1] if loc.form == "DW_FORM_exprloc" else loc.value

    def close(self):
        self.file.close()
