# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses

from zview.backend.base import WorkqInfo
from zview.kernel.layout import KernelLayout
from zview.kernel.session import Session
from zview.kernel.workqs import Workqs


class FakeScraper:
    """Minimal AbstractScraper stand-in over a word-addressed memory map."""

    def __init__(self, words: dict[int, int], endianess: str = "<"):
        self._words = words
        self.endianess = endianess
        self.reads: list[tuple[int, int]] = []

    def read32(self, at: int, amount: int = 1):
        self.reads.append((at, amount))
        try:
            return tuple(self._words[at + 4 * i] for i in range(amount))
        except KeyError as e:
            raise AssertionError(f"unexpected read32 @ 0x{at:X}") from e


class FakeElf:
    """Reverse symbol lookup over a fixed table."""

    def __init__(self, symbols: dict[int, str] | None = None):
        self._symbols = symbols or {}

    def get_symbol_name_at(self, addr: int) -> str | None:
        return self._symbols.get(addr)


# Offsets as they sit on the target, past the deprecated embedded thread.
QUEUE = 0x5000
LAYOUT = KernelLayout(
    threads_head=0,
    thread_next=0,
    stack_start=0,
    stack_size=0,
    workq_thread_id=208,
    workq_pending=212,
    workq_drainq=228,
    workq_flags=236,
    work_node=0,
    work_handler=4,
    thread_qnode=0,
)

STARTED, BUSY, DRAIN, PLUGGED = 1, 2, 4, 8
HANDLER_A, HANDLER_B = 0x1100, 0x1200
SYMBOLS = {HANDLER_A: "flush_handler", HANDLER_B: "retry_handler"}


def _queue_words(*, head: int, flags: int, thread_id: int = 0x9000) -> dict[int, int]:
    """An empty drain queue points its head at itself."""
    return {
        QUEUE + 208: thread_id,
        QUEUE + 212: head,
        QUEUE + 216: head,
        QUEUE + 220: 0,
        QUEUE + 224: 0,
        QUEUE + 228: QUEUE + 228,
        QUEUE + 232: QUEUE + 228,
        QUEUE + 236: flags,
    }


def _item(address: int, *, handler: int, nxt: int) -> dict[int, int]:
    return {address: nxt, address + 4: handler}


def _session(scraper, elf, layout, thread_names=None, *, walk_waiters=True) -> Session:
    """The session a walker reads with, from the arguments it used to take."""
    return Session(
        scraper,
        elf,
        layout,
        waitq_flavor="simple" if walk_waiters else "scalable",
        rows={"threads": {name: [address] for address, name in (thread_names or {}).items()}},
    )


def test_an_idle_queue_has_no_items():
    scraper = FakeScraper(_queue_words(head=0, flags=STARTED))

    queue = Workqs().walk(_session(scraper, FakeElf(), LAYOUT), {"sys_workq": [QUEUE]})["sys_workq"]

    assert queue.pending == ()
    assert queue.depth == 0
    assert not queue.is_pending_truncated
    assert queue.is_started and not queue.is_busy


def test_queued_items_are_named_by_their_handler():
    first, second = 0x6000, 0x6100
    words = {
        **_queue_words(head=first, flags=STARTED | BUSY),
        **_item(first, handler=HANDLER_A, nxt=second),
        **_item(second, handler=HANDLER_B, nxt=0),
    }

    queue = Workqs().walk(
        _session(FakeScraper(words), FakeElf(SYMBOLS), LAYOUT), {"bench_workq": [QUEUE]}
    )["bench_workq"]

    assert queue.pending == ("flush_handler", "retry_handler")
    assert queue.depth == 2
    assert queue.is_busy


def test_items_sharing_a_handler_are_told_apart_by_their_own_symbol():
    first, second = 0x6000, 0x6100
    words = {
        **_queue_words(head=first, flags=STARTED | BUSY),
        **_item(first, handler=HANDLER_A, nxt=second),
        **_item(second, handler=HANDLER_A, nxt=0),
    }
    elf = FakeElf({**SYMBOLS, first: "tick_work", second: "trail_work"})

    queue = Workqs().walk(_session(FakeScraper(words), elf, LAYOUT), {"bench_workq": [QUEUE]})[
        "bench_workq"
    ]

    assert queue.pending == ("tick_work (flush_handler)", "trail_work (flush_handler)")


def test_a_handler_without_a_symbol_falls_back_to_its_address():
    item = 0x6000
    words = {**_queue_words(head=item, flags=STARTED), **_item(item, handler=0x1234, nxt=0)}

    queue = Workqs().walk(_session(FakeScraper(words), FakeElf(), LAYOUT), {"q": [QUEUE]})["q"]

    assert queue.pending == ("0x1234",)


def test_a_queue_running_its_only_item_reads_busy_and_empty():
    """BUSY is the only sign of work in flight."""
    scraper = FakeScraper(_queue_words(head=0, flags=STARTED | BUSY))

    queue = Workqs().walk(_session(scraper, FakeElf(), LAYOUT), {"q": [QUEUE]})["q"]

    assert queue.depth == 0
    assert queue.is_busy


def test_a_list_longer_than_the_cap_reports_a_floor():
    items = [0x6000 + 0x100 * i for i in range(6)]
    words = dict(_queue_words(head=items[0], flags=STARTED))
    for address, nxt in zip(items, items[1:] + [0], strict=True):
        words |= _item(address, handler=HANDLER_A, nxt=nxt)

    queue = Workqs().walk(
        _session(FakeScraper(words), FakeElf(SYMBOLS), LAYOUT), {"q": [QUEUE]}, max_pending=4
    )["q"]

    assert queue.depth == 4
    assert queue.is_pending_truncated


def test_a_list_that_loops_mid_walk_terminates():
    """A relinked node can point back into the list."""
    first, second = 0x6000, 0x6100
    words = {
        **_queue_words(head=first, flags=STARTED),
        **_item(first, handler=HANDLER_A, nxt=second),
        **_item(second, handler=HANDLER_B, nxt=first),
    }

    queue = Workqs().walk(_session(FakeScraper(words), FakeElf(SYMBOLS), LAYOUT), {"q": [QUEUE]})[
        "q"
    ]

    assert queue.pending == ("flush_handler", "retry_handler")
    assert queue.is_pending_truncated


def test_a_declared_queue_that_never_started_says_so():
    scraper = FakeScraper(_queue_words(head=0, flags=0, thread_id=0))

    queue = Workqs().walk(_session(scraper, FakeElf(), LAYOUT), {"q": [QUEUE]})["q"]

    assert not queue.is_started
    assert queue.thread_name is None


def test_the_animating_thread_is_named():
    scraper = FakeScraper(_queue_words(head=0, flags=STARTED))

    queue = Workqs().walk(
        _session(scraper, FakeElf(), LAYOUT, {0x9000: "bench_workq_thread"}), {"q": [QUEUE]}
    )["q"]

    assert queue.thread_name == "bench_workq_thread"


def test_a_release_before_thread_id_names_the_thread_embedded_in_the_queue():
    """k_work_q.thread_id arrived in v4.2.0, and the thread itself is still not read."""
    scraper = FakeScraper(_queue_words(head=0, flags=STARTED))
    layout = dataclasses.replace(LAYOUT, workq_thread_id=None, workq_thread=0)

    queue = Workqs().walk(
        _session(scraper, FakeElf(), layout, {QUEUE: "sysworkq"}), {"q": [QUEUE]}
    )["q"]

    assert queue.thread_name == "sysworkq"
    assert min(at for at, _ in scraper.reads) == QUEUE + 212


def test_draining_threads_are_reported_as_waiters():
    waiter = 0x7000
    words = _queue_words(head=0, flags=STARTED | DRAIN)
    words[QUEUE + 228] = waiter
    words[waiter] = QUEUE + 228

    queue = Workqs().walk(
        _session(FakeScraper(words), FakeElf(), LAYOUT, {waiter: "drain_waiter"}), {"q": [QUEUE]}
    )["q"]

    assert queue.waiters == ("drain_waiter",)
    assert queue.is_draining


def test_an_unwalkable_layout_leaves_the_waiters_unknown():
    """``None`` is not an empty drain queue: nothing was walked."""
    scraper = FakeScraper(_queue_words(head=0, flags=STARTED))

    queue = Workqs().walk(_session(scraper, FakeElf(), LAYOUT, walk_waiters=False), {"q": [QUEUE]})[
        "q"
    ]

    assert queue.waiters is None


def test_the_deprecated_thread_is_never_read():
    """208 of a k_work_q's 240 bytes are the deprecated embedded thread."""
    scraper = FakeScraper(_queue_words(head=0, flags=STARTED))

    Workqs().walk(_session(scraper, FakeElf(), LAYOUT), {"q": [QUEUE]})

    span_read = next(r for r in scraper.reads if r[1] > 1)
    assert span_read == (QUEUE + 208, 8)


def test_the_state_words_follow_the_bits():
    queue = WorkqInfo(name="q", address=QUEUE, flags=STARTED | BUSY | PLUGGED)

    assert queue.states == ("started", "busy", "plugged")
    assert WorkqInfo(name="q", address=QUEUE, flags=0).states == ()
