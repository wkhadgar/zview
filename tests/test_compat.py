# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from kernel import compat


class FakeElf:
    """ElfInspector stand-in with a fixed member table and symbol set."""

    def __init__(self, members: dict[tuple[str, str], int], symbols: tuple[str, ...] = ()):
        self._members = members
        self._symbols = set(symbols)

    def get_struct_member_offset(self, struct_name: str, member: str) -> int:
        try:
            return self._members[(struct_name, member)]
        except KeyError as e:
            raise LookupError(f"{struct_name}.{member}") from e

    def get_symbol_info(self, symbol_name: str, info: str) -> list[int]:
        if symbol_name not in self._symbols:
            raise LookupError(symbol_name)
        return [1]


def test_resolve_offset_takes_the_first_matching_candidate():
    elf = FakeElf({("k_msgq", "used_msgs"): 24, ("k_msgq", "used"): 99})

    offset = compat.resolve_offset(elf, ((("k_msgq", "used_msgs"),), (("k_msgq", "used"),)))

    assert offset == 24


def test_resolve_offset_falls_through_to_a_later_candidate():
    """A renamed member resolves through the next candidate."""
    elf = FakeElf({("k_msgq", "used"): 99})

    offset = compat.resolve_offset(elf, ((("k_msgq", "used_msgs"),), (("k_msgq", "used"),)))

    assert offset == 99


def test_resolve_offset_sums_a_nested_path():
    """k_mem_slab moved its counters into an ``info`` sub-struct."""
    elf = FakeElf({("k_mem_slab", "info"): 16, ("k_mem_slab_info", "num_used"): 8})

    offset = compat.resolve_offset(
        elf, ((("k_mem_slab", "info"), ("k_mem_slab_info", "num_used")),)
    )

    assert offset == 24


def test_partially_resolvable_path_is_rejected():
    """A path with a missing step does not resolve."""
    elf = FakeElf({("k_mem_slab", "info"): 16})

    offset = compat.resolve_offset(
        elf, ((("k_mem_slab", "info"), ("k_mem_slab_info", "num_used")),)
    )

    assert offset is None


def test_resolve_offset_without_any_candidate():
    assert compat.resolve_offset(FakeElf({}), ((("k_sem", "count"),),)) is None


def test_resolve_fields_is_all_or_nothing():
    """A group with a missing member resolves to None."""
    elf = FakeElf({("k_sem", "wait_q"): 0, ("k_sem", "count"): 8})

    assert compat.resolve_fields(elf, compat.SEMAPHORE_FIELDS) is None


def test_resolve_fields_returns_the_whole_group():
    elf = FakeElf({("k_sem", "wait_q"): 0, ("k_sem", "count"): 8, ("k_sem", "limit"): 12})

    assert compat.resolve_fields(elf, compat.SEMAPHORE_FIELDS) == {
        "sem_wait_q": 0,
        "sem_count": 8,
        "sem_limit": 12,
    }


def test_has_kconfig_reads_the_absolute_symbols():
    """Zephyr emits every enabled CONFIG_* as an absolute symbol."""
    elf = FakeElf({}, symbols=("CONFIG_THREAD_NAME",))

    assert compat.has_kconfig(elf, "CONFIG_THREAD_NAME")
    assert not compat.has_kconfig(elf, "CONFIG_THREAD_RUNTIME_STATS")


def test_waitq_flavor_simple():
    elf = FakeElf({}, symbols=(compat.CONFIG_WAITQ_SIMPLE,))

    assert compat.waitq_flavor(elf) == "simple"


def test_waitq_flavor_scalable():
    elf = FakeElf({}, symbols=(compat.CONFIG_WAITQ_SCALABLE,))

    assert compat.waitq_flavor(elf) == "scalable"


def test_waitq_flavor_unknown_is_not_walkable():
    """A build advertising neither option reports unknown."""
    assert compat.waitq_flavor(FakeElf({})) == "unknown"


def _write_version_header(build_dir: Path, version: str) -> Path:
    generated = build_dir / "include" / "generated" / "zephyr"
    generated.mkdir(parents=True)
    (generated / "version.h").write_text(
        "#define ZEPHYR_VERSION_CODE 263168\n"
        f'#define KERNEL_VERSION_STRING           "{version}"\n'
    )
    return build_dir


def test_detect_zephyr_version_beside_the_elf(tmp_path: Path):
    build = _write_version_header(tmp_path / "zephyr", "4.4.99")
    elf = build / "zephyr.elf"
    elf.write_bytes(b"")

    assert compat.detect_zephyr_version(elf) == "4.4.99"


def test_detect_zephyr_version_from_a_nested_elf(tmp_path: Path):
    """The header sits a few directories up from some build layouts."""
    build = _write_version_header(tmp_path / "build" / "zephyr", "4.3.0")
    nested = build / "sub" / "dir"
    nested.mkdir(parents=True)
    elf = nested / "zephyr.elf"
    elf.write_bytes(b"")

    assert compat.detect_zephyr_version(elf) == "4.3.0"


def test_detect_zephyr_version_of_a_relocated_elf(tmp_path: Path):
    """A bare ELF outside its build tree has no version to report."""
    elf = tmp_path / "zephyr.elf"
    elf.write_bytes(b"")

    assert compat.detect_zephyr_version(elf) is None


def test_detect_zephyr_version_of_a_missing_path(tmp_path: Path):
    assert compat.detect_zephyr_version(tmp_path / "nope" / "zephyr.elf") is None


def test_mem_slab_resolves_the_sub_struct_layout():
    """v3.7.0 and later keep the counters in a k_mem_slab_info sub-struct."""
    elf = FakeElf(
        {
            ("k_mem_slab", "wait_q"): 0,
            ("k_mem_slab", "info"): 16,
            ("k_mem_slab_info", "num_blocks"): 0,
            ("k_mem_slab_info", "block_size"): 4,
            ("k_mem_slab_info", "num_used"): 8,
        }
    )

    assert compat.resolve_fields(elf, compat.MEM_SLAB_FIELDS) == {
        "mem_slab_wait_q": 0,
        "mem_slab_num_blocks": 16,
        "mem_slab_block_size": 20,
        "mem_slab_num_used": 24,
    }


def test_mem_slab_falls_back_to_the_flat_layout():
    """v3.0 and v3.3 carry the counters directly on k_mem_slab."""
    elf = FakeElf(
        {
            ("k_mem_slab", "wait_q"): 0,
            ("k_mem_slab", "num_blocks"): 16,
            ("k_mem_slab", "block_size"): 20,
            ("k_mem_slab", "num_used"): 24,
        }
    )

    assert compat.resolve_fields(elf, compat.MEM_SLAB_FIELDS) == {
        "mem_slab_wait_q": 0,
        "mem_slab_num_blocks": 16,
        "mem_slab_block_size": 20,
        "mem_slab_num_used": 24,
    }


def test_usage_group_needs_the_kernel_wide_counter():
    """z_kernel.usage only exists from v3.7.0, so older trees get no CPU stats."""
    elf = FakeElf(
        {
            ("k_thread", "base"): 0,
            ("_thread_base", "usage"): 48,
            ("k_cycle_stats", "total"): 0,
        }
    )

    assert compat.resolve_fields(elf, compat.USAGE_FIELDS) is None


def test_heap_group_needs_max_allocated_bytes():
    """z_heap.max_allocated_bytes arrived in v3.3.0; a v3.0 tree reports no heaps."""
    elf = FakeElf(
        {
            ("z_heap", "free_bytes"): 16,
            ("z_heap", "allocated_bytes"): 20,
            ("z_heap", "end_chunk"): 8,
        }
    )

    assert compat.resolve_fields(elf, compat.HEAP_FIELDS) is None


def test_thread_entry_accepts_either_struct_spelling():
    """Vendor trees have shipped __thread_entry and _thread_entry."""
    double = FakeElf({("k_thread", "entry"): 112, ("__thread_entry", "pEntry"): 0})
    single = FakeElf({("k_thread", "entry"): 112, ("_thread_entry", "pEntry"): 0})

    for elf in (double, single):
        assert compat.resolve_optional_fields(elf, compat.THREAD_META_FIELDS) == {
            "thread_entry": 112
        }


def test_resolve_optional_fields_keeps_what_it_finds():
    """Metadata members resolve independently of each other."""
    elf = FakeElf({("k_thread", "base"): 0, ("_thread_base", "prio"): 14})

    assert compat.resolve_optional_fields(elf, compat.THREAD_META_FIELDS) == {"thread_priority": 14}


def test_resolve_optional_fields_of_nothing():
    assert compat.resolve_optional_fields(FakeElf({}), compat.THREAD_META_FIELDS) == {}


def test_kernel_object_registry_is_coherent():
    """Every registered object names a struct and the fields needed to read it."""
    assert set(compat.KERNEL_OBJECTS) == {
        "semaphores",
        "mutexes",
        "msgqs",
        "events",
        "mem_slabs",
        "work",
    }

    labels = [spec.label for spec in compat.KERNEL_OBJECTS.values()]
    structs = [spec.struct for spec in compat.KERNEL_OBJECTS.values()]
    assert len(set(labels)) == len(labels)
    assert len(set(structs)) == len(structs)

    for name, spec in compat.KERNEL_OBJECTS.items():
        assert spec.fields, name
        # Every field must offer at least one candidate path to resolve.
        for field, candidates in spec.fields.items():
            assert candidates, f"{name}.{field}"
