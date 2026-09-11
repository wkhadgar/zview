# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from kernel.layout import KernelLayout
from kernel.mutexes import walk_mutexes


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
    """Minimal ElfInspector stand-in."""

    def get_struct_size(self, name: str) -> int:
        if name != "k_mutex":
            raise LookupError(name)
        return 16


# Synthetic 16-byte k_mutex: wait_q (head, tail), owner, lock_count.
LAYOUT = KernelLayout(
    threads_head=0,
    thread_next=0,
    stack_start=0,
    stack_size=0,
    mutex_wait_q=0,
    mutex_owner=8,
    mutex_lock_count=12,
    thread_qnode=0,
)


def _mutex_words(
    address: int,
    owner: int,
    lock_count: int,
    wait_head: int | None = None,
) -> dict[int, int]:
    """Lay out one k_mutex. An empty wait queue points at itself."""
    return {
        address: address if wait_head is None else wait_head,
        address + 4: address,
        address + 8: owner,
        address + 12: lock_count,
    }


def test_free_mutex():
    mutex = 0x2000
    scraper = FakeScraper(_mutex_words(mutex, owner=0, lock_count=0))

    info = walk_mutexes(scraper, FakeElf(), {"uart_mutex": [mutex]}, LAYOUT)["uart_mutex"]

    assert not info.is_locked
    assert info.owner_address == 0
    assert info.owner_name is None
    assert info.lock_count == 0
    assert info.waiters == ()


def test_locked_mutex_resolves_its_owner():
    mutex, owner = 0x2000, 0x3000
    scraper = FakeScraper(_mutex_words(mutex, owner=owner, lock_count=1))

    info = walk_mutexes(scraper, FakeElf(), {"spi_bus": [mutex]}, LAYOUT, {owner: "spi_task"})[
        "spi_bus"
    ]

    assert info.is_locked
    assert info.owner_name == "spi_task"
    assert info.lock_count == 1


def test_recursive_lock_depth_is_reported():
    mutex, owner = 0x2000, 0x3000
    scraper = FakeScraper(_mutex_words(mutex, owner=owner, lock_count=3))

    info = walk_mutexes(scraper, FakeElf(), {"m": [mutex]}, LAYOUT, {owner: "worker"})["m"]

    assert info.lock_count == 3


def test_unknown_owner_keeps_the_address():
    """An owner absent from the thread table keeps its address, unnamed."""
    mutex, owner = 0x2000, 0x3000
    scraper = FakeScraper(_mutex_words(mutex, owner=owner, lock_count=1))

    info = walk_mutexes(scraper, FakeElf(), {"m": [mutex]}, LAYOUT, {})["m"]

    assert info.is_locked
    assert info.owner_address == owner
    assert info.owner_name is None


def test_contended_mutex_lists_its_wait_queue():
    """A locked mutex reports its owner and its queued threads."""
    mutex, owner, first, second = 0x2000, 0x3000, 0x3100, 0x3200
    words = _mutex_words(mutex, owner=owner, lock_count=1, wait_head=first)
    words |= {first: second, second: mutex}

    info = walk_mutexes(
        FakeScraper(words),
        FakeElf(),
        {"spi_bus": [mutex]},
        LAYOUT,
        {owner: "spi_task", first: "sensor_task", second: "logger_task"},
    )["spi_bus"]

    assert info.owner_name == "spi_task"
    assert info.waiters == ("sensor_task", "logger_task")


def test_walk_waiters_disabled_reports_unknown():
    mutex, owner = 0x2000, 0x3000
    scraper = FakeScraper(_mutex_words(mutex, owner=owner, lock_count=1))

    info = walk_mutexes(
        scraper, FakeElf(), {"m": [mutex]}, LAYOUT, {owner: "t"}, walk_waiters=False
    )["m"]

    assert info.waiters is None
    assert info.owner_name == "t"


def test_same_symbol_at_several_addresses_gets_unique_labels():
    first, second = 0x2000, 0x2100
    words = _mutex_words(first, owner=0, lock_count=0) | _mutex_words(second, owner=0, lock_count=0)

    mutexes = walk_mutexes(FakeScraper(words), FakeElf(), {"locks": [first, second]}, LAYOUT)

    assert list(mutexes) == ["locks@0x2000", "locks@0x2100"]


def test_no_instances_returns_empty():
    assert walk_mutexes(FakeScraper({}), FakeElf(), {}, LAYOUT) == {}
