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

_CACHE_SCHEMA_VERSION = 6

# DWARF location opcodes: an object's own address, and the frame and register
# forms a function local takes.
_DW_OP_ADDR = 0x03
_DW_OP_FBREG = 0x91
_DW_OP_REGISTER_RANGE = range(0x50, 0x90)


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
        self._symbols_is_object: dict[str, list[bool]] = {}
        self._struct_sizes: dict[str, int] = {}
        self._struct_members: dict[str, dict[str, int]] = {}
        self._struct_variables: dict[str, list[str]] = {}
        self._struct_addresses: dict[str, dict[str, list[int]]] = {}
        self._struct_declarations: dict[str, list[str]] = {}
        self._decl_sites: dict[int, tuple[int, int, int]] = {}
        self._function_sites: dict[int, tuple[int, int, int]] = {}
        self._resolved_files: dict[tuple[int, int], str] = {}
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
        return (stats.st_mtime, stats.st_size, sys.hexversion, _CACHE_SCHEMA_VERSION)

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
            self._symbols_is_object = data["sym_is_obj"]
            self._struct_sizes = data["struct_sizes"]
            self._struct_members = data["struct_members"]
            self._struct_variables = data["struct_vars"]
            self._struct_addresses = data["struct_addrs"]
            self._struct_declarations = data["struct_decls"]
            self._decl_sites = data["decl_sites"]
            self._function_sites = data["function_sites"]
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
            "sym_is_obj": self._symbols_is_object,
            "struct_sizes": self._struct_sizes,
            "struct_members": self._struct_members,
            "struct_vars": self._struct_variables,
            "struct_addrs": self._struct_addresses,
            "struct_decls": self._struct_declarations,
            "decl_sites": self._decl_sites,
            "function_sites": self._function_sites,
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

    def _collect_members(self, parent_die, struct_name: str, base_offset: int) -> None:
        """Walk ``parent_die`` recording each ``DW_TAG_member``.

        Anonymous members whose type is itself a struct or union are flattened:
        their inner members are folded into ``struct_name`` at the parent's
        offset. This is what makes ``_thread_base.prio`` resolvable when ``prio``
        sits inside an anonymous union+struct in Zephyr's headers.
        """
        for child in parent_die.iter_children():
            if child.tag != "DW_TAG_member":
                continue
            loc_attr = child.attributes.get("DW_AT_data_member_location")
            if loc_attr is None:
                # Union members elide the location attribute (offset is 0).
                child_loc = 0
            else:
                child_loc = (
                    loc_attr.value[1] if loc_attr.form == "DW_FORM_exprloc" else loc_attr.value
                )
            full_loc = base_offset + child_loc

            m_name_attr = child.attributes.get("DW_AT_name")
            if m_name_attr:
                m_name = m_name_attr.value.decode(errors="ignore")
                self._struct_members[struct_name][m_name] = full_loc
                continue

            # Anonymous member: descend into its type if it's a struct/union.
            if "DW_AT_type" not in child.attributes:
                continue
            try:
                type_die = child.get_DIE_from_attribute("DW_AT_type")
            except Exception:
                continue
            if type_die.tag in ("DW_TAG_structure_type", "DW_TAG_union_type"):
                self._collect_members(type_die, struct_name, full_loc)

    def _variable_address(self, die, address_size: int) -> tuple[int | None, bool]:
        """
        ``(address, is static)`` from a variable DIE's ``DW_AT_location``.

        A frame or register location is not static. A location that is absent,
        at zero, or in a form this does not decode, is static with an unknown
        address.
        """
        location = die.attributes.get("DW_AT_location")
        if location is None or not isinstance(location.value, list) or not location.value:
            return None, True

        expression = bytes(location.value)
        opcode = expression[0]
        if opcode == _DW_OP_ADDR and len(expression) == 1 + address_size:
            order = "little" if self._little_endian else "big"
            # The linker leaves the location of an object it dropped at zero.
            return int.from_bytes(expression[1:], order) or None, True
        if opcode == _DW_OP_FBREG or opcode in _DW_OP_REGISTER_RANGE:
            return None, False

        return None, True

    @staticmethod
    def _declaration_site(die, cu_offset: int, declaration=None) -> tuple[int, int, int] | None:
        """
        ``(file index, line, CU offset)`` of a DIE's declaration, if it carries one.

        A definition leaves out the file or the line it shares with its
        ``declaration``, which then supplies it.
        """
        file_attr = die.attributes.get("DW_AT_decl_file")
        line_attr = die.attributes.get("DW_AT_decl_line")
        if declaration is not None:
            if file_attr is None:
                file_attr = declaration.attributes.get("DW_AT_decl_file")
                cu_offset = declaration.cu.cu_offset
            if line_attr is None:
                line_attr = declaration.attributes.get("DW_AT_decl_line")

        if file_attr is None or line_attr is None:
            return None
        return file_attr.value, line_attr.value, cu_offset

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
                    self._symbols_is_object.setdefault(name, []).append(
                        sym.entry["st_info"]["type"] == "STT_OBJECT"
                        and sym.entry["st_shndx"] not in ("SHN_UNDEF", "SHN_ABS")
                    )

            if not elf.has_dwarf_info():
                raise ValueError("ELF file lacks DWARF debug information.")

            dwarf = elf.get_dwarf_info()
            offset_to_struct: dict[int, str] = {}
            pending_vars: list[tuple[str, int, int | None, bool]] = []

            for CU in dwarf.iter_CUs():
                cu_offset = CU.cu_offset
                address_size = CU.header["address_size"]
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
                        self._collect_members(die, struct_name, base_offset=0)

                    elif die.tag == "DW_TAG_variable":
                        # The definition of an object declared extern keeps
                        # its address, and its name and type stay on the
                        # declaration it points to.
                        named = die
                        if (
                            "DW_AT_name" not in die.attributes
                            and "DW_AT_specification" in die.attributes
                        ):
                            named = die.get_DIE_from_attribute("DW_AT_specification")

                        name_attr = named.attributes.get("DW_AT_name")
                        type_attr = named.attributes.get("DW_AT_type")
                        if name_attr and type_attr:
                            var_name = name_attr.value.decode(errors="ignore")
                            # DWARF type references are relative to the CU start.
                            type_offset = type_attr.value + named.cu.cu_offset
                            address, is_static = self._variable_address(die, address_size)
                            pending_vars.append((var_name, type_offset, address, is_static))
                            if address is not None:
                                declaration = named if named is not die else None
                                site = self._declaration_site(die, cu_offset, declaration)
                                if site is not None:
                                    self._decl_sites[address] = site

                    elif die.tag == "DW_TAG_subprogram":
                        low_pc = die.attributes.get("DW_AT_low_pc")
                        site = self._declaration_site(die, cu_offset)
                        # A function the linker dropped keeps a DIE at zero.
                        if low_pc and low_pc.value and site is not None:
                            # Thumb code pointers carry bit 0 set.
                            self._function_sites[low_pc.value & ~1] = site

            # Variable-to-struct resolution is deferred to a second loop over
            # pending_vars because DWARF type references may point forward in the
            # stream relative to the variable DIE's position.
            for var_name, type_offset, address, is_static in pending_vars:
                if type_offset in offset_to_struct:
                    struct_name = offset_to_struct[type_offset]
                    self._struct_variables.setdefault(struct_name, []).append(var_name)

                    if address is not None:
                        addresses = self._struct_addresses.setdefault(struct_name, {}).setdefault(
                            var_name, []
                        )
                        if address not in addresses:
                            addresses.append(address)
                    elif is_static:
                        self._struct_declarations.setdefault(struct_name, []).append(var_name)

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
        """Reverse symbol lookup. Returns the symbol whose address matches ``addr``.

        Keys are stored with the Thumb bit cleared (Cortex-M function symbols
        carry an odd ``st_value`` which would otherwise miss every lookup).
        ARM mapping symbols (``$t``/``$a``/``$d``/``$x``) are skipped so they
        don't shadow real function symbols at the same address.
        """
        cache = getattr(self, "_address_to_symbol", None)
        if cache is None:
            cache = {}
            for name, addrs in self._symbols_address.items():
                if name.startswith("$"):
                    continue
                for a in addrs:
                    cache.setdefault(a & ~1, name)
            self._address_to_symbol = cache
        return cache.get(addr & ~1)

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

    def decl_site(self, address: int) -> tuple[str, int] | None:
        """``(path, line)`` where the object at ``address`` is declared, if known."""
        return self._resolve_site(self._decl_sites.get(address))

    def function_site(self, address: int) -> tuple[str, int] | None:
        """``(path, line)`` where the function at ``address`` is defined, if known."""
        if not address:
            return None
        return self._resolve_site(self._function_sites.get(address & ~1))

    def _resolve_site(self, site: tuple[int, int, int] | None) -> tuple[str, int] | None:
        """Turn a stored ``(file index, line, CU offset)`` into a path and a line."""
        if site is None:
            return None

        file_index, line, cu_offset = site
        key = (cu_offset, file_index)
        if key not in self._resolved_files:
            path = self._read_file_name(cu_offset, file_index)
            if path is None:
                return None
            self._resolved_files[key] = path

        return self._resolved_files[key], line

    def _read_file_name(self, cu_offset: int, file_index: int) -> str | None:
        """
        The path of a CU's file table entry, read from the ELF on demand.

        A line program at version 5 indexes its file table from zero, an older
        one from one, where directory zero is the compilation directory.
        """
        try:
            with open(self._path, "rb") as file:
                dwarf = ELFFile(file).get_dwarf_info()
                cu = next((c for c in dwarf.iter_CUs() if c.cu_offset == cu_offset), None)
                if cu is None:
                    return None

                header = dwarf.line_program_for_CU(cu).header
                version_5 = header["version"] >= 5
                entries = header["file_entry"]
                index = file_index - (0 if version_5 else 1)
                if not 0 <= index < len(entries):
                    return None

                entry = entries[index]
                name = entry.name.decode(errors="ignore")
                if name.startswith("/"):
                    return name

                directory = self._directory_name(cu, header, entry.dir_index, version_5)
                return f"{directory}/{name}" if directory else name
        except (OSError, KeyError, AttributeError, StopIteration):
            return None

    @staticmethod
    def _directory_name(cu, header, dir_index: int, version_5: bool) -> str | None:
        """The directory of a file table entry, by the indexing its version uses."""
        directories = header["include_directory"]
        if not version_5 and dir_index == 0:
            comp_dir = cu.get_top_DIE().attributes.get("DW_AT_comp_dir")
            return comp_dir.value.decode(errors="ignore") if comp_dir else None

        index = dir_index if version_5 else dir_index - 1
        if not 0 <= index < len(directories):
            return None

        entry = directories[index]
        raw = entry if isinstance(entry, bytes) else entry.name
        return raw.decode(errors="ignore")

    def find_struct_instances(self, struct_name: str) -> dict[str, list[int]]:
        """
        Map ``{name: addresses}`` for every global instance of the named struct.

        A variable's own DWARF location answers first, which separates
        same-named statics across translation units. A name known only through
        a declaration comes from the symbol table instead, where only a defined
        data symbol sized like the struct counts.
        A function local is not an instance. Order follows DWARF discovery.
        """
        located = self._struct_addresses.get(struct_name, {})
        declared = set(self._struct_declarations.get(struct_name, []))

        found: dict[str, list[int]] = {}
        for name in self.find_struct_variable_names(struct_name) or []:
            if name in located:
                found[name] = list(located[name])
            elif name in declared:
                addresses = self._object_symbols(name, struct_name)
                if addresses:
                    found[name] = addresses

        return found

    def _object_symbols(self, name: str, struct_name: str) -> list[int]:
        """Addresses of the defined data symbols named ``name`` sized like the struct."""
        if struct_name not in self._struct_sizes:
            return []
        struct_size = self._struct_sizes[struct_name]

        return [
            address
            for address, size, is_object in zip(
                self._symbols_address.get(name, []),
                self._symbols_size.get(name, []),
                self._symbols_is_object.get(name, []),
                strict=True,
            )
            if is_object and size in (0, struct_size)
        ]
