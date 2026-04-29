# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
ElfInspector performs a sweep of an ELF+DWARF file, extracting
symbol addresses, sizes, struct layouts, and variable-to-struct mappings into
plain Python dicts. The result is cached to disk so that subsequent sessions
on the same unmodified ELF skip the scan entirely.
"""

import hashlib
import hmac
import io
import marshal
import os
import sys
from pathlib import Path
from typing import Literal

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

# HMAC key derived from the user's home directory: machine-specific, makes
# tampered cache files detectable across machines.
_CACHE_HMAC_KEY = hashlib.sha256(str(Path.home()).encode()).digest()
_HMAC_SIZE = 32  # SHA-256 digest length in bytes


class ElfInspector:
    """
    Parses an ELF+DWARF file and exposes symbol and struct metadata.

    On first use, performs a single scan and persists the result to a
    cache directory. On subsequent uses with the same unmodified ELF,
    the cache is loaded directly, skipping the scan entirely.
    """

    def __init__(self, elf_path: str):
        self._path = Path(elf_path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"ELF file not found at '{self._path}'")

        self._cache_dir = self._resolve_os_cache_dir()
        self._cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)

        path_hash = hashlib.md5(str(self._path).encode()).hexdigest()
        self._cache_file = self._cache_dir / f"{path_hash}.bin"

        # Clean up any stale temp file left by a previously crashed write.
        self._cache_file.with_suffix(".tmp").unlink(missing_ok=True)

        self._symbols_address: dict[str, list[int]] = {}
        self._symbols_size: dict[str, list[int]] = {}
        self._struct_sizes: dict[str, int] = {}
        self._struct_members: dict[str, dict[str, int]] = {}
        self._struct_variables: dict[str, list[str]] = {}
        self._elfclass = 0
        self._little_endian = True

        if not self._load_cache():
            print("Loading ELF. This may take a while...", end="", flush=True, file=sys.stderr)
            self._perform_single_pass_scan()
            self._save_cache()
            print(" OK.", file=sys.stderr)
        else:
            print(f"Loaded cached ELF @ {self._cache_file}", file=sys.stderr)

    @staticmethod
    def _resolve_os_cache_dir() -> Path:
        """Returns the OS-appropriate cache directory."""
        if sys.platform == "win32":
            return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "zview"
        elif sys.platform == "darwin":
            return Path.home() / "Library" / "Caches" / "zview"
        else:
            return Path.home() / ".cache" / "zview"

    def _get_validation_metadata(self) -> tuple[float, int, int]:
        """
        Returns metadata used to validate the on-disk cache.

        The tuple (mtime, file size, sys.hexversion) uniquely identifies a
        particular build of a particular ELF on a particular Python version.
        Any change to the ELF or the interpreter invalidates the cache.
        """
        stats = self._path.stat()
        return (stats.st_mtime, stats.st_size, sys.hexversion)

    def _load_cache(self) -> bool:
        """
        Attempts to load and validate the on-disk cache.

        Returns True if the cache was valid and all internal dicts were
        restored, False if the cache is missing, stale, corrupted, or tampered.

        Security: the raw bytes are HMAC-SHA256 verified before marshal.load()
        is called. A maliciously crafted cache file will fail the signature
        check and be rejected without ever touching deserialization. This is a
        stretch, but better safe than sorry.
        """
        if not self._cache_file.exists():
            return False

        current_metadata = self._get_validation_metadata()

        try:
            raw = self._cache_file.read_bytes()

            if len(raw) < _HMAC_SIZE:
                return False

            stored_sig = raw[:_HMAC_SIZE]
            payload = raw[_HMAC_SIZE:]

            expected_sig = hmac.new(_CACHE_HMAC_KEY, payload, hashlib.sha256).digest()
            if not hmac.compare_digest(stored_sig, expected_sig):
                # Payload has been tampered with or belongs to a different machine.
                return False

            buf = io.BytesIO(payload)

            # nosec - marshal payload is HMAC-SHA256 verified above.
            cached_metadata = marshal.load(buf)
            if cached_metadata != current_metadata:
                return False

            # nosec - marshal payload is HMAC-SHA256 verified above.
            data = marshal.load(buf)

            self._little_endian = data["elf_le"]
            self._elfclass = data["elf_class"]
            self._symbols_address = data["sym_addr"]
            self._symbols_size = data["sym_size"]
            self._struct_sizes = data["struct_sizes"]
            self._struct_members = data["struct_members"]
            self._struct_variables = data["struct_vars"]
            return True

        except (EOFError, ValueError, TypeError, KeyError, OSError):
            return False

    def _save_cache(self):
        """
        Persists the parsed ELF data to disk as an HMAC-signed marshal payload.

        Write is atomic: payload is assembled in memory, prefixed with its
        HMAC-SHA256 signature, written to a .tmp file, then renamed into place.
        On POSIX, rename is atomic. On Windows, Path.replace() is the closest
        equivalent. A crash at any point leaves no corrupt cache file behind.
        """
        metadata = self._get_validation_metadata()
        data = {
            "elf_le": self._little_endian,
            "elf_class": self._elfclass,
            "sym_addr": self._symbols_address,
            "sym_size": self._symbols_size,
            "struct_sizes": self._struct_sizes,
            "struct_members": self._struct_members,
            "struct_vars": self._struct_variables,
        }

        tmp = self._cache_file.with_suffix(".tmp")
        try:
            buf = io.BytesIO()
            marshal.dump(metadata, buf)
            marshal.dump(data, buf)
            payload = buf.getvalue()

            sig = hmac.new(_CACHE_HMAC_KEY, payload, hashlib.sha256).digest()

            with open(tmp, "wb") as f:
                f.write(sig)
                f.write(payload)

            tmp.replace(self._cache_file)

        except OSError:
            tmp.unlink(missing_ok=True)

    def _perform_single_pass_scan(self):
        """
        Executes a sweep of the ELF and DWARF tree.

        Populates all internal dicts in a single pass with the `.symtab` and DWARF DIEs
        """
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
            offset_to_struct: dict[int, str] = {}
            pending_vars: list[tuple[str, int]] = []

            for CU in dwarf.iter_CUs():
                cu_offset = CU.cu_offset
                for die in CU.iter_DIEs():
                    if die.tag == "DW_TAG_structure_type":
                        name_attr = die.attributes.get("DW_AT_name")
                        if not name_attr:
                            continue

                        struct_name = name_attr.value.decode(errors="ignore")
                        offset_to_struct[die.offset] = struct_name

                        if die.attributes.get("DW_AT_declaration"):
                            continue

                        size_attr = die.attributes.get("DW_AT_byte_size")
                        if size_attr:
                            self._struct_sizes[struct_name] = size_attr.value

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
                            # DWARF type references are relative to the CU start.
                            type_offset = type_attr.value + cu_offset
                            pending_vars.append((var_name, type_offset))

            # Variable-to-struct resolution is deferred to a second loop over
            # pending_vars because DWARF type references may point forward in the
            # stream relative to the variable DIE's position.
            for var_name, type_offset in pending_vars:
                if type_offset in offset_to_struct:
                    struct_name = offset_to_struct[type_offset]
                    self._struct_variables.setdefault(struct_name, []).append(var_name)

    def get_symbol_info(self, symbol_name: str, info: Literal["address", "size"]) -> list[int]:
        """
        Returns the address or size list for a named ELF symbol.

        Raises LookupError if the symbol is not found.
        Raises ValueError for an unrecognized info type.
        """
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

    def get_symbol_name_at(self, addr: int) -> str | None:
        """Reverse symbol lookup. Returns the symbol whose address matches ``addr``."""
        cache = getattr(self, "_address_to_symbol", None)
        if cache is None:
            cache = {}
            for name, addrs in self._symbols_address.items():
                for a in addrs:
                    cache.setdefault(a, name)
            self._address_to_symbol = cache
        return cache.get(addr)

    def get_struct_member_offset(self, struct_name: str, member_name: str) -> int:
        """
        Returns the byte offset of a named member within a named struct.

        Raises LookupError if the struct or member is not found in the DWARF info.
        """
        if struct_name not in self._struct_members:
            raise LookupError(f"Struct '{struct_name}' not found.")
        if member_name not in self._struct_members[struct_name]:
            raise LookupError(f"Member '{member_name}' not found in struct '{struct_name}'.")
        return self._struct_members[struct_name][member_name]

    def get_struct_size(self, struct_name: str) -> int:
        """
        Returns the total byte size of a named struct.

        Raises LookupError if the struct is not found in the DWARF info.
        """
        if struct_name not in self._struct_sizes:
            raise LookupError(f"Size info for struct '{struct_name}' not found.")
        return self._struct_sizes[struct_name]

    def find_struct_variable_names(self, struct_name: str) -> list[str] | None:
        """
        Returns all global variable names whose type is the named struct,
        or None if no such variables were found in the DWARF info.

        The returned list is deduplicated (a symbol appearing in multiple
        compilation units is reported only once) and preserves DWARF
        discovery order: critical for downstream consumers (ZScraper's
        polling loop, recording/replay lockstep) that depend on a stable
        iteration order across processes.
        """
        if struct_name not in self._struct_variables:
            return None
        return list(dict.fromkeys(self._struct_variables[struct_name]))
