# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from kernel.layout import KernelLayout
from kernel.msgqs import walk_msgqs


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
        if name != "k_msgq":
            raise LookupError(name)
        return 24


# Synthetic 24-byte k_msgq: wait_q (head, tail), msg_size, max_msgs,
# used_msgs, and a trailing word standing in for the buffer pointer.
LAYOUT = KernelLayout(
    threads_head=0,
    thread_next=0,
    stack_start=0,
    stack_size=0,
    msgq_wait_q=0,
    msgq_msg_size=8,
    msgq_max_msgs=12,
    msgq_used_msgs=16,
    thread_qnode=0,
)


def _msgq_words(
    address: int, *, used: int, capacity: int, msg_size: int, head: int | None = None
) -> dict[int, int]:
    """One queue's struct words; an empty wait queue points its head at itself."""
    return {
        address + 0: address if head is None else head,
        address + 4: address,
        address + 8: msg_size,
        address + 12: capacity,
        address + 16: used,
        address + 20: 0,
    }


def test_reads_the_counters_of_every_queue():
    words = _msgq_words(0x1000, used=3, capacity=8, msg_size=16)
    words |= _msgq_words(0x2000, used=0, capacity=4, msg_size=32)

    queues = walk_msgqs(
        FakeScraper(words), FakeElf(), {"sensor_q": [0x1000], "log_q": [0x2000]}, LAYOUT
    )

    assert (queues["sensor_q"].used_msgs, queues["sensor_q"].max_msgs) == (3, 8)
    assert queues["sensor_q"].msg_size == 16
    assert (queues["log_q"].used_msgs, queues["log_q"].max_msgs) == (0, 4)
    assert queues["log_q"].address == 0x2000


def test_an_empty_wait_queue_reads_as_empty_not_unknown():
    words = _msgq_words(0x1000, used=1, capacity=2, msg_size=4)

    queues = walk_msgqs(FakeScraper(words), FakeElf(), {"q": [0x1000]}, LAYOUT)

    assert queues["q"].waiters == ()


def test_a_queued_thread_is_named():
    address, thread = 0x1000, 0x8000
    words = _msgq_words(address, used=0, capacity=4, msg_size=8, head=thread)
    words[thread] = address

    queues = walk_msgqs(FakeScraper(words), FakeElf(), {"q": [address]}, LAYOUT, {thread: "reader"})

    assert queues["q"].waiters == ("reader",)


def test_an_unwalkable_layout_leaves_the_waiters_unknown():
    """``None`` is not ``()``: a scalable wait queue carries no information here."""
    words = _msgq_words(0x1000, used=1, capacity=4, msg_size=8)

    queues = walk_msgqs(FakeScraper(words), FakeElf(), {"q": [0x1000]}, LAYOUT, walk_waiters=False)

    assert queues["q"].waiters is None


def test_instances_of_one_symbol_are_labelled_apart():
    words = _msgq_words(0x1000, used=1, capacity=4, msg_size=8)
    words |= _msgq_words(0x2000, used=2, capacity=4, msg_size=8)

    queues = walk_msgqs(FakeScraper(words), FakeElf(), {"pool": [0x1000, 0x2000]}, LAYOUT)

    assert len(queues) == 2
    assert all("pool" in label for label in queues)


def test_a_read_failure_is_not_swallowed():
    """A failing read is the caller's to report, not something to skip past."""
    with pytest.raises(AssertionError):
        walk_msgqs(FakeScraper({}), FakeElf(), {"q": [0x9999]}, LAYOUT)


def test_fill_and_full_come_from_the_counters():
    words = _msgq_words(0x1000, used=2, capacity=8, msg_size=4)
    words |= _msgq_words(0x2000, used=4, capacity=4, msg_size=4)

    queues = walk_msgqs(FakeScraper(words), FakeElf(), {"a": [0x1000], "b": [0x2000]}, LAYOUT)

    assert queues["a"].fill_percent == 25.0
    assert not queues["a"].is_full
    assert queues["b"].is_full


def test_a_zero_capacity_queue_does_not_divide_by_zero():
    words = _msgq_words(0x1000, used=0, capacity=0, msg_size=4)

    queues = walk_msgqs(FakeScraper(words), FakeElf(), {"q": [0x1000]}, LAYOUT)

    assert queues["q"].fill_percent == 0.0
    assert not queues["q"].is_full
