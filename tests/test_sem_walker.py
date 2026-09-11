# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from kernel.layout import KernelLayout
from kernel.semaphores import walk_semaphores


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
        if name != "k_sem":
            raise LookupError(name)
        return 16


# Synthetic 16-byte k_sem: wait_q (head, tail), count, limit.
LAYOUT = KernelLayout(
    threads_head=0,
    thread_next=0,
    stack_start=0,
    stack_size=0,
    sem_wait_q=0,
    sem_count=8,
    sem_limit=12,
    thread_qnode=0,
)


def _sem_words(
    address: int, count: int, limit: int, wait_head: int | None = None
) -> dict[int, int]:
    """Lay out one k_sem. An empty wait queue points at itself."""
    return {
        address: address if wait_head is None else wait_head,
        address + 4: address,
        address + 8: count,
        address + 12: limit,
    }


def test_single_semaphore_without_waiters():
    sem = 0x2000
    scraper = FakeScraper(_sem_words(sem, count=0, limit=4))

    semaphores = walk_semaphores(scraper, FakeElf(), {"data_ready": [sem]}, LAYOUT)

    assert list(semaphores) == ["data_ready"]
    info = semaphores["data_ready"]
    assert (info.address, info.count, info.limit) == (sem, 0, 4)
    # Empty queue: (), not None.
    assert info.waiters == ()


def test_semaphore_at_limit():
    sem = 0x2000
    scraper = FakeScraper(_sem_words(sem, count=4, limit=4))

    info = walk_semaphores(scraper, FakeElf(), {"conn_pool": [sem]}, LAYOUT)["conn_pool"]

    assert (info.count, info.limit) == (4, 4)


def test_semaphore_with_waiters_resolves_thread_names():
    sem, first, second = 0x2000, 0x3000, 0x3100
    words = _sem_words(sem, count=0, limit=1, wait_head=first)
    words |= {first: second, second: sem}

    info = walk_semaphores(
        FakeScraper(words),
        FakeElf(),
        {"data_ready": [sem]},
        LAYOUT,
        {first: "sensor_task", second: "logger_task"},
    )["data_ready"]

    assert info.waiters == ("sensor_task", "logger_task")


def test_unknown_waiter_is_labelled_by_address():
    sem, waiter = 0x2000, 0x3000
    words = _sem_words(sem, count=0, limit=1, wait_head=waiter)
    words |= {waiter: sem}

    info = walk_semaphores(FakeScraper(words), FakeElf(), {"s": [sem]}, LAYOUT, {})["s"]

    assert info.waiters == ("thread @ 0x3000",)


def test_walk_waiters_disabled_reports_unknown():
    """``walk_waiters=False`` reports waiters as unknown."""
    sem = 0x2000
    scraper = FakeScraper(_sem_words(sem, count=1, limit=1))

    info = walk_semaphores(scraper, FakeElf(), {"s": [sem]}, LAYOUT, {}, walk_waiters=False)["s"]

    assert info.waiters is None
    assert info.count == 1


def test_missing_qnode_offset_reports_unknown():
    """A layout without ``thread_qnode`` reports waiters as unknown."""
    sem = 0x2000
    layout = KernelLayout(
        threads_head=0,
        thread_next=0,
        stack_start=0,
        stack_size=0,
        sem_wait_q=0,
        sem_count=8,
        sem_limit=12,
    )

    info = walk_semaphores(FakeScraper(_sem_words(sem, 1, 1)), FakeElf(), {"s": [sem]}, layout)["s"]

    assert info.waiters is None


def test_multiple_semaphores():
    first, second = 0x2000, 0x2100
    words = _sem_words(first, count=1, limit=2) | _sem_words(second, count=0, limit=8)

    semaphores = walk_semaphores(
        FakeScraper(words), FakeElf(), {"a": [first], "b": [second]}, LAYOUT
    )

    assert [(s.name, s.count, s.limit) for s in semaphores.values()] == [("a", 1, 2), ("b", 0, 8)]


def test_same_symbol_at_several_addresses_gets_unique_labels():
    """One symbol with several addresses yields one label per instance."""
    first, second = 0x2000, 0x2100
    words = _sem_words(first, count=1, limit=1) | _sem_words(second, count=0, limit=1)

    semaphores = walk_semaphores(FakeScraper(words), FakeElf(), {"pool": [first, second]}, LAYOUT)

    assert list(semaphores) == ["pool@0x2000", "pool@0x2100"]


def test_no_instances_returns_empty():
    assert walk_semaphores(FakeScraper({}), FakeElf(), {}, LAYOUT) == {}
