# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kernel.wait_queues import resolve_waiter_names, walk_wait_queue


class FakeScraper:
    """Minimal AbstractScraper stand-in over a word-addressed memory map."""

    def __init__(self, words: dict[int, int], endianess: str = "<"):
        self._words = words
        self.endianess = endianess
        self.reads: list[int] = []

    def read32(self, at: int, amount: int = 1):
        self.reads.append(at)
        try:
            return tuple(self._words[at + 4 * i] for i in range(amount))
        except KeyError as e:
            raise AssertionError(f"unexpected read32 @ 0x{at:X}") from e


# A wait queue lives at LIST_ADDR; queued threads link through qnode_dlist,
# which sits at offset 0 of k_thread on a simple-waitq build.
LIST_ADDR = 0x1000
QNODE_OFFSET = 0


def test_null_list_address_returns_empty():
    scraper = FakeScraper({})
    assert walk_wait_queue(scraper, 0, QNODE_OFFSET) == []
    assert scraper.reads == []


def test_empty_queue_is_self_referential():
    """An empty sys_dlist_t points at itself."""
    scraper = FakeScraper({LIST_ADDR: LIST_ADDR})
    assert walk_wait_queue(scraper, LIST_ADDR, QNODE_OFFSET) == []


def test_null_head_returns_empty():
    scraper = FakeScraper({LIST_ADDR: 0})
    assert walk_wait_queue(scraper, LIST_ADDR, QNODE_OFFSET) == []


def test_single_waiter():
    thread = 0x3000
    scraper = FakeScraper({LIST_ADDR: thread, thread: LIST_ADDR})

    assert walk_wait_queue(scraper, LIST_ADDR, QNODE_OFFSET) == [thread]


def test_multiple_waiters_keep_queue_order():
    """Waiters are returned in queue order."""
    first, second, third = 0x3000, 0x3100, 0x3200
    scraper = FakeScraper(
        {
            LIST_ADDR: first,
            first: second,
            second: third,
            third: LIST_ADDR,
        }
    )

    assert walk_wait_queue(scraper, LIST_ADDR, QNODE_OFFSET) == [first, second, third]


def test_qnode_offset_is_subtracted():
    """The thread address is the node address minus the qnode offset."""
    thread, qnode_offset = 0x3000, 0x18
    node = thread + qnode_offset
    scraper = FakeScraper({LIST_ADDR: node, node: LIST_ADDR})

    assert walk_wait_queue(scraper, LIST_ADDR, qnode_offset) == [thread]


def test_head_argument_skips_the_list_read():
    """Passing ``head`` skips reading the list address."""
    thread = 0x3000
    scraper = FakeScraper({thread: LIST_ADDR})

    assert walk_wait_queue(scraper, LIST_ADDR, QNODE_OFFSET, head=thread) == [thread]
    assert LIST_ADDR not in scraper.reads


@pytest.mark.parametrize("loop_target", ("self", "first"))
def test_cycles_terminate(loop_target):
    """Traversal stops when a node repeats."""
    first, second = 0x3000, 0x3100
    target = first if loop_target == "first" else second
    scraper = FakeScraper({LIST_ADDR: first, first: second, second: target})

    waiters = walk_wait_queue(scraper, LIST_ADDR, QNODE_OFFSET)

    assert waiters == [first, second]


def test_max_waiters_caps_traversal():
    """Traversal stops at ``max_waiters``."""
    # 0x3000 -> 0x3100 -> 0x3200 -> ... each node pointing at the next.
    words = {LIST_ADDR: 0x3000}
    for index in range(10):
        words[0x3000 + index * 0x100] = 0x3000 + (index + 1) * 0x100
    words[0x3000 + 10 * 0x100] = LIST_ADDR

    scraper = FakeScraper(words)

    assert len(walk_wait_queue(scraper, LIST_ADDR, QNODE_OFFSET, max_waiters=4)) == 4


def test_resolve_waiter_names_uses_the_thread_table():
    assert resolve_waiter_names([0x3000, 0x3100], {0x3000: "sensor", 0x3100: "logger"}) == (
        "sensor",
        "logger",
    )


def test_resolve_waiter_names_falls_back_to_the_address():
    """An unknown address renders as ``thread @ 0x...``."""
    assert resolve_waiter_names([0x3000], {}) == ("thread @ 0x3000",)
    assert resolve_waiter_names([0x3000], None) == ("thread @ 0x3000",)


def test_resolve_waiter_names_of_empty_queue():
    assert resolve_waiter_names([], {0x3000: "sensor"}) == ()
