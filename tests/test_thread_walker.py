# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kernel.threads import walk_static_threads, walk_thread_list


class FakeScraper:
    """Minimal AbstractScraper stand-in for hermetic walk_thread_list tests."""

    def __init__(self, memory: dict[int, tuple[int, ...]], endianess: str = "<"):
        self._memory = memory
        self.endianess = endianess
        self._is_connected = True
        self.batch_count = 0

    @property
    def is_connected(self):
        return self._is_connected

    def connect(self):
        self._is_connected = True

    def begin_batch(self):
        self.batch_count += 1

    def end_batch(self):
        pass

    def read32(self, addr: int, amount: int = 1):
        if addr not in self._memory:
            raise AssertionError(f"unexpected read32 @ 0x{addr:X}")
        return self._memory[addr][:amount]


class FakeElf:
    """Minimal ElfInspector stand-in."""

    def __init__(self, struct_size: int = 32):
        self._size = struct_size

    def get_struct_size(self, name: str):
        if name != "k_thread":
            raise LookupError(name)
        return self._size


# Byte offsets for a synthetic 32-byte k_thread struct used across all tests.
# Layout (word indices):
#   0: base (ignored by walker)
#   1: stack_start
#   2: stack_size
#   3: next_thread
#   4-7: name (16 bytes)
from kernel.layout import KernelLayout  # noqa: E402

LAYOUT = KernelLayout(
    threads_head=0,  # unused by walker (head address comes as arg)
    thread_next=12,
    stack_start=4,
    stack_size=8,
    thread_name=16,
)
HEAD_ADDR = 0x500


def _name_to_words(name: str, slots: int = 4, endian: str = "little") -> tuple[int, ...]:
    """Pack ``name`` into ``slots`` 32-bit words, NUL-padded."""
    data = name.encode() + b"\x00" * (slots * 4 - len(name))
    return tuple(int.from_bytes(data[i : i + 4], endian) for i in range(0, slots * 4, 4))


def _make_thread(
    stack_start: int,
    stack_size: int,
    next_thread: int,
    name: str,
    endian: str = "little",
) -> tuple[int, ...]:
    return (0, stack_start, stack_size, next_thread, *_name_to_words(name, endian=endian))


def test_zero_head_address_returns_empty():
    scraper = FakeScraper({})
    threads = walk_thread_list(scraper, FakeElf(), 0, LAYOUT, "little", has_names=True)
    assert threads == {}
    assert scraper.batch_count == 1


def test_zero_head_pointer_returns_empty():
    scraper = FakeScraper({HEAD_ADDR: (0,)})
    threads = walk_thread_list(scraper, FakeElf(), HEAD_ADDR, LAYOUT, "little", has_names=True)
    assert threads == {}


def test_single_thread_walk():
    t1 = 0x1000
    scraper = FakeScraper(
        {
            HEAD_ADDR: (t1,),
            t1: _make_thread(0x2000, 512, 0, "main"),
        }
    )

    threads = walk_thread_list(scraper, FakeElf(), HEAD_ADDR, LAYOUT, "little", has_names=True)

    assert list(threads) == ["main"]
    t = threads["main"]
    assert t.address == t1
    assert t.stack_start == 0x2000
    assert t.stack_size == 512
    assert t.name == "main"
    assert t.runtime is None


def test_linked_list_multiple_threads():
    t1, t2, t3 = 0x1000, 0x1020, 0x1040
    scraper = FakeScraper(
        {
            HEAD_ADDR: (t1,),
            t1: _make_thread(0x2000, 512, t2, "main"),
            t2: _make_thread(0x3000, 1024, t3, "idle"),
            t3: _make_thread(0x4000, 768, 0, "worker"),
        }
    )

    threads = walk_thread_list(scraper, FakeElf(), HEAD_ADDR, LAYOUT, "little", has_names=True)

    assert set(threads) == {"main", "idle", "worker"}
    assert threads["main"].stack_size == 512
    assert threads["idle"].stack_size == 1024
    assert threads["worker"].stack_size == 768


def test_a_build_without_stack_info_walks_threads_with_no_geometry():
    """The struct words that would hold the geometry are something else there."""
    from dataclasses import replace

    t1 = 0x1000
    scraper = FakeScraper({HEAD_ADDR: (t1,), t1: _make_thread(0x2000, 512, 0, "main")})
    layout = replace(LAYOUT, stack_start=None, stack_size=None)

    (thread,) = walk_thread_list(scraper, FakeElf(), HEAD_ADDR, layout, "little", True).values()

    assert (thread.name, thread.stack_start, thread.stack_size) == ("main", None, None)


def test_no_names_uses_address_fallback():
    t1 = 0x1000
    scraper = FakeScraper(
        {
            HEAD_ADDR: (t1,),
            # name slots irrelevant when has_names=False
            t1: (0, 0x2000, 512, 0, 0, 0, 0, 0),
        }
    )

    threads = walk_thread_list(scraper, FakeElf(), HEAD_ADDR, LAYOUT, "little", has_names=False)

    assert list(threads) == [f"thread @ 0x{t1:X}"]


def test_max_threads_caps_walk():
    memory: dict[int, tuple[int, ...]] = {HEAD_ADDR: (0x1000,)}
    count = 10
    for i in range(count):
        addr = 0x1000 + i * 0x20
        nxt = 0x1000 + (i + 1) * 0x20 if i < count - 1 else 0
        memory[addr] = _make_thread(0x4000 + i * 0x100, 256, nxt, f"t{i}")

    scraper = FakeScraper(memory)
    threads = walk_thread_list(
        scraper, FakeElf(), HEAD_ADDR, LAYOUT, "little", has_names=True, max_threads=3
    )

    assert set(threads) == {"t0", "t1", "t2"}


def test_head_read_failure_raises_runtime_error():
    # HEAD_ADDR absent from memory -> read32 raises AssertionError from FakeScraper.
    scraper = FakeScraper({})
    with pytest.raises(RuntimeError, match="Unable to read kernel thread list"):
        walk_thread_list(scraper, FakeElf(), HEAD_ADDR, LAYOUT, "little", has_names=True)


def test_mid_walk_read_failure_raises():
    # Head resolves to a thread address that is not present in memory.
    t1 = 0x1000
    scraper = FakeScraper({HEAD_ADDR: (t1,)})
    with pytest.raises(Exception, match="Error reading thread struct at 0x1000"):
        walk_thread_list(scraper, FakeElf(), HEAD_ADDR, LAYOUT, "little", has_names=True)


def test_big_endian_name_decoding():
    t1 = 0x1000
    scraper = FakeScraper(
        {
            HEAD_ADDR: (t1,),
            t1: _make_thread(0x2000, 512, 0, "main", endian="big"),
        },
        endianess=">",
    )

    threads = walk_thread_list(scraper, FakeElf(), HEAD_ADDR, LAYOUT, "big", has_names=True)

    assert "main" in threads


# The same struct with base.thread_state in byte 1 of word 0.
STATIC_LAYOUT = KernelLayout(thread_state=1, stack_start=4, stack_size=8, thread_name=16)
_DUMMY, _DEAD = 1 << 0, 1 << 3


def _static_thread(state: int, stack_start: int, stack_size: int, name: str = "") -> tuple:
    return (state << 8, stack_start, stack_size, 0, *_name_to_words(name))


def _walk_static(memory: dict, objects: dict[str, list[int]], has_names=False, max_threads=64):
    return walk_static_threads(
        FakeScraper(memory), FakeElf(), objects, STATIC_LAYOUT, "little", has_names, max_threads
    )


def test_static_threads_are_named_after_their_objects():
    """K_THREAD_DEFINE's object carries the thread id, the name THREAD_NAME would give."""
    threads = _walk_static(
        {0x1000: _static_thread(0, 0x2000, 512), 0x1020: _static_thread(0, 0x3000, 256)},
        {"_k_thread_obj_worker": [0x1000], "z_idle_threads[0]": [0x1020]},
    )

    assert {n: (t.address, t.stack_size) for n, t in threads.items()} == {
        "worker": (0x1000, 512),
        "z_idle_threads[0]": (0x1020, 256),
    }


def test_objects_that_are_not_live_threads_are_left_out():
    """The monitor list never holds these, so a static walk must not show them."""
    threads = _walk_static(
        {
            0x1000: _static_thread(0, 0x2000, 512),
            0x1020: _static_thread(_DUMMY, 0x3000, 256),
            0x1040: _static_thread(_DEAD, 0x4000, 256),
            0x1060: (0,) * 8,
        },
        {"live": [0x1000], "_thread_dummy": [0x1020], "exited": [0x1040], "unstarted": [0x1060]},
    )

    assert list(threads) == ["live"]


def test_a_built_in_name_wins_and_a_missing_one_falls_back_to_the_object():
    threads = _walk_static(
        {
            0x1000: _static_thread(0, 0x2000, 512, "shell"),
            0x1020: _static_thread(0, 0x3000, 256),
            0x1040: _static_thread(0, 0x4000, 256, "shell"),
        },
        {"shell_ctx.thread": [0x1000], "_k_thread_obj_blinky": [0x1020], "other.thread": [0x1040]},
        has_names=True,
    )

    assert list(threads) == ["shell", "blinky", "shell @ 0x1040"]


def test_the_static_walk_stops_at_max_threads():
    memory = {0x1000 + 0x20 * i: _static_thread(0, 0x2000, 64) for i in range(4)}
    objects = {f"t{i}": [0x1000 + 0x20 * i] for i in range(4)}

    assert list(_walk_static(memory, objects, max_threads=2)) == ["t0", "t1"]
