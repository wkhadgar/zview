# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

import pytest

from kernel.layout import KernelLayout
from kernel.mem_slabs import walk_mem_slabs


class FakeScraper:
    """Minimal AbstractScraper stand-in over a word-addressed memory map."""

    def __init__(self, words: dict[int, int], endianess: str = "<"):
        self._words = words
        self.endianess = endianess

    def read32(self, at: int, amount: int = 1):
        try:
            return tuple(self._words[at + 4 * i] for i in range(amount))
        except KeyError as e:
            raise AssertionError(f"unexpected read32 @ 0x{at:X}") from e


class FakeElf:
    """Minimal ElfInspector stand-in; a traced build carries one word more."""

    def __init__(self, size: int = 28):
        self._size = size

    def get_struct_size(self, name: str) -> int:
        if name != "k_mem_slab":
            raise LookupError(name)
        return self._size


# Synthetic 28-byte k_mem_slab: wait_q (head, tail), buffer, free_list, then
# the info members num_blocks, block_size, num_used.
LAYOUT = KernelLayout(
    threads_head=0,
    thread_next=0,
    stack_start=0,
    stack_size=0,
    mem_slab_wait_q=0,
    mem_slab_num_blocks=16,
    mem_slab_block_size=20,
    mem_slab_num_used=24,
    thread_qnode=0,
)

# The same slab on a build that tracks peak utilisation: one more word.
TRACED_LAYOUT = replace(LAYOUT, mem_slab_max_used=28)


def _slab_words(
    address: int,
    *,
    blocks: int,
    size: int,
    used: int,
    head: int | None = None,
    peak: int | None = None,
) -> dict[int, int]:
    """One slab's struct words; an empty wait queue points its head at itself."""
    words = {
        address + 0: address if head is None else head,
        address + 4: address,
        address + 8: 0,
        address + 12: 0,
        address + 16: blocks,
        address + 20: size,
        address + 24: used,
    }
    if peak is not None:
        words[address + 28] = peak
    return words


def test_reads_the_counters_of_every_slab():
    words = _slab_words(0x1000, blocks=16, size=256, used=6)
    words |= _slab_words(0x2000, blocks=4, size=64, used=0)

    slabs = walk_mem_slabs(
        FakeScraper(words), FakeElf(), {"frame_slab": [0x1000], "tiny": [0x2000]}, LAYOUT
    )

    assert (slabs["frame_slab"].num_used, slabs["frame_slab"].num_blocks) == (6, 16)
    assert slabs["frame_slab"].block_size == 256
    assert slabs["frame_slab"].total_bytes == 4096
    assert slabs["tiny"].num_used == 0


def test_peak_is_unknown_unless_the_build_tracks_it():
    """The member only exists under CONFIG_MEM_SLAB_TRACE_MAX_UTILIZATION."""
    words = _slab_words(0x1000, blocks=8, size=32, used=3)

    slabs = walk_mem_slabs(FakeScraper(words), FakeElf(), {"s": [0x1000]}, LAYOUT)

    assert slabs["s"].max_used is None


def test_peak_is_read_where_the_layout_resolved_it():
    words = _slab_words(0x1000, blocks=8, size=32, used=3, peak=7)

    slabs = walk_mem_slabs(FakeScraper(words), FakeElf(32), {"s": [0x1000]}, TRACED_LAYOUT)

    assert slabs["s"].max_used == 7


def test_fill_and_exhaustion_come_from_the_counters():
    words = _slab_words(0x1000, blocks=8, size=32, used=2)
    words |= _slab_words(0x2000, blocks=8, size=32, used=8)

    slabs = walk_mem_slabs(FakeScraper(words), FakeElf(), {"a": [0x1000], "b": [0x2000]}, LAYOUT)

    assert slabs["a"].fill_percent == 25.0
    assert not slabs["a"].is_exhausted
    assert slabs["b"].is_exhausted


def test_a_zero_block_slab_does_not_divide_by_zero():
    words = _slab_words(0x1000, blocks=0, size=32, used=0)

    slabs = walk_mem_slabs(FakeScraper(words), FakeElf(), {"s": [0x1000]}, LAYOUT)

    assert slabs["s"].fill_percent == 0.0
    assert not slabs["s"].is_exhausted


def test_a_thread_waiting_for_a_block_is_named():
    address, thread = 0x1000, 0x8000
    words = _slab_words(address, blocks=2, size=16, used=2, head=thread)
    words[thread] = address

    slabs = walk_mem_slabs(
        FakeScraper(words), FakeElf(), {"s": [address]}, LAYOUT, {thread: "frame_worker"}
    )

    assert slabs["s"].waiters == ("frame_worker",)


def test_an_unwalkable_layout_leaves_the_waiters_unknown():
    words = _slab_words(0x1000, blocks=2, size=16, used=1)

    slabs = walk_mem_slabs(
        FakeScraper(words), FakeElf(), {"s": [0x1000]}, LAYOUT, walk_waiters=False
    )

    assert slabs["s"].waiters is None


def test_instances_of_one_symbol_are_labelled_apart():
    words = _slab_words(0x1000, blocks=2, size=16, used=1)
    words |= _slab_words(0x2000, blocks=2, size=16, used=2)

    slabs = walk_mem_slabs(FakeScraper(words), FakeElf(), {"pool": [0x1000, 0x2000]}, LAYOUT)

    assert len(slabs) == 2
    assert all("pool" in label for label in slabs)


def test_a_read_failure_is_not_swallowed():
    with pytest.raises(AssertionError):
        walk_mem_slabs(FakeScraper({}), FakeElf(), {"s": [0x9999]}, LAYOUT)
