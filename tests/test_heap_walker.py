# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest

from zview.backend.base import ProbeReadError
from zview.kernel.heaps import walk_heap_waiters, walk_heaps
from zview.kernel.session import Session
from zview.orchestrator import ZScraper


class FakeScraper:
    """Minimal AbstractScraper stand-in for hermetic heap-walker tests."""

    def __init__(self, end_chunk: int, heap_bytes: bytes, endianess: str = "<"):
        self.end_chunk = end_chunk
        self.heap_bytes = heap_bytes
        self.endianess = endianess

    def read32(self, at: int, amount: int = 1):
        return (self.end_chunk,)

    def read_bytes(self, at: int, amount: int) -> bytes:
        return self.heap_bytes[:amount]


def _make_walker(scraper: FakeScraper) -> ZScraper:
    """Bypass ``ZScraper.__init__`` and seed only what the walker reads."""
    from zview.kernel.layout import KernelLayout

    walker = object.__new__(ZScraper)
    walker._m_scraper = scraper
    walker._layout = KernelLayout(
        threads_head=0,
        thread_next=0,
        stack_start=0,
        stack_size=0,
        heap_end_chunk=0,
    )
    return walker


def _chunk0_header_16(first_chunk_id: int, endian: str = "<") -> bytes:
    buf = bytearray(8)
    struct.pack_into(f"{endian}H", buf, 2, (first_chunk_id << 1) | 1)
    return bytes(buf)


def _chunk_body_16(size_units: int, used: bool, endian: str = "<") -> bytes:
    buf = bytearray(8)
    struct.pack_into(f"{endian}H", buf, 2, (size_units << 1) | (1 if used else 0))
    return bytes(buf)


def _chunk0_header_32(first_chunk_id: int, endian: str = "<") -> bytes:
    buf = bytearray(8)
    struct.pack_into(f"{endian}I", buf, 4, (first_chunk_id << 1) | 1)
    return bytes(buf)


def _chunk_body_32(size_units: int, used: bool, endian: str = "<") -> bytes:
    buf = bytearray(8)
    struct.pack_into(f"{endian}I", buf, 4, (size_units << 1) | (1 if used else 0))
    return bytes(buf)


def test_null_address_returns_empty():
    walker = _make_walker(FakeScraper(end_chunk=0, heap_bytes=b""))
    assert walker.get_heap_fragmentation(0) == []


def test_end_chunk_zero_returns_empty():
    walker = _make_walker(FakeScraper(end_chunk=0, heap_bytes=b""))
    assert walker.get_heap_fragmentation(0x1000) == []


def test_16bit_walk():
    image = (
        _chunk0_header_16(first_chunk_id=1)
        + _chunk_body_16(size_units=2, used=True)  # c=1
        + b"\x00" * 8  # c=2 (inside chunk 1)
        + _chunk_body_16(size_units=2, used=False)  # c=3
        + b"\x00" * 8  # c=4 (inside chunk 3)
    )
    walker = _make_walker(FakeScraper(end_chunk=5, heap_bytes=image))

    assert walker.get_heap_fragmentation(0x1000) == [
        {"used": True, "size": 16},
        {"used": False, "size": 16},
    ]


def test_32bit_walk():
    image = (
        _chunk0_header_32(first_chunk_id=1)
        + _chunk_body_32(size_units=2, used=True)
        + b"\x00" * 8
        + _chunk_body_32(size_units=2, used=False)
        + b"\x00" * 8
    )
    walker = _make_walker(FakeScraper(end_chunk=5, heap_bytes=image))

    assert walker.get_heap_fragmentation(0x1000) == [
        {"used": True, "size": 16},
        {"used": False, "size": 16},
    ]


def test_oversize_heap_raises():
    # end_chunk * 8 = 40 MB, over the 32 MB sanity ceiling.
    walker = _make_walker(FakeScraper(end_chunk=5_000_000, heap_bytes=b""))
    with pytest.raises(ValueError, match="sanity"):
        walker.get_heap_fragmentation(0x1000)


def test_corrupted_chunk0_raises():
    # All zeros: both 16-bit and 32-bit interpretations fail bounds+used check.
    walker = _make_walker(FakeScraper(end_chunk=3, heap_bytes=bytes(24)))
    with pytest.raises(ValueError, match="Corrupted chunk0"):
        walker.get_heap_fragmentation(0x1000)


def test_zero_size_mid_walk_raises():
    image = (
        _chunk0_header_16(first_chunk_id=1)
        + _chunk_body_16(size_units=0, used=True)  # triggers infinite-loop guard
        + b"\x00" * 8
    )
    walker = _make_walker(FakeScraper(end_chunk=3, heap_bytes=image))
    with pytest.raises(RuntimeError, match="Infinite loop"):
        walker.get_heap_fragmentation(0x1000)


def test_big_endian_16bit_walk():
    image = (
        _chunk0_header_16(first_chunk_id=1, endian=">")
        + _chunk_body_16(size_units=2, used=True, endian=">")
        + b"\x00" * 8
    )
    walker = _make_walker(FakeScraper(end_chunk=3, heap_bytes=image, endianess=">"))

    assert walker.get_heap_fragmentation(0x1000) == [
        {"used": True, "size": 16},
    ]


class FakeWaitQueueScraper:
    """Word-addressed stand-in for the ``k_heap`` wait queue reads."""

    def __init__(self, words: dict[int, int]):
        self._words = words
        self.endianess = "<"

    def read32(self, at: int, amount: int = 1):
        try:
            return tuple(self._words[at + 4 * i] for i in range(amount))
        except KeyError as e:
            raise AssertionError(f"unexpected read32 @ 0x{at:X}") from e


# k_heap: a sys_heap at offset 0, the wait queue behind it.
_HEAP = 0x8000
_WAIT_Q = 12
_QNODE = 0


def test_a_quiet_heap_has_no_waiters():
    scraper = FakeWaitQueueScraper({_HEAP + _WAIT_Q: _HEAP + _WAIT_Q})

    assert walk_heap_waiters(scraper, _HEAP, _WAIT_Q, _QNODE) == ()


def test_threads_waiting_for_memory_are_named():
    first, second = 0x3000, 0x3100
    scraper = FakeWaitQueueScraper(
        {
            _HEAP + _WAIT_Q: first,
            first: second,
            second: _HEAP + _WAIT_Q,
        }
    )

    waiters = walk_heap_waiters(
        scraper, _HEAP, _WAIT_Q, _QNODE, {first: "worker", second: "logger"}
    )

    assert waiters == ("worker", "logger")


def test_an_unnamed_waiter_falls_back_to_its_address():
    thread = 0x3000
    scraper = FakeWaitQueueScraper({_HEAP + _WAIT_Q: thread, thread: _HEAP + _WAIT_Q})

    assert walk_heap_waiters(scraper, _HEAP, _WAIT_Q, _QNODE) == ("thread @ 0x3000",)


class FakeHeapMemory:
    """Word-addressed stand-in covering the reads one heap poll makes."""

    def __init__(self, words: dict[int, int]):
        self._words = words
        self.endianess = "<"
        self.is_connected = True

    def read32(self, at: int, amount: int = 1):
        try:
            return tuple(self._words[at + 4 * i] for i in range(amount))
        except KeyError as e:
            raise AssertionError(f"unexpected read32 @ 0x{at:X}") from e


def _heap_session(words: dict[int, int], *, wait_q: int | None, flavor: str = "simple") -> Session:
    """A session over one heap and nothing else."""
    from zview.kernel.layout import KernelLayout

    layout = KernelLayout(
        heap_free_bytes=4,
        heap_allocated_bytes=8,
        heap_max_allocated_bytes=12,
        heap_end_chunk=0,
        heap_wait_q=wait_q,
        thread_qnode=0,
    )
    return Session(
        FakeHeapMemory(words), None, layout, waitq_flavor=flavor, parts=frozenset({"heap_waiters"})
    )


def _poll(session: Session) -> list:
    """One frame of the single ``bench_heap``."""
    return list(walk_heaps(session, {"bench_heap": [_HEAP]}).values())


_SYS_HEAP = 0x9000
# The k_heap's first word points at the z_heap the stats live in.
_STATS = {_HEAP: _SYS_HEAP, _SYS_HEAP + 4: 1536, _SYS_HEAP + 8: 2560, _SYS_HEAP + 12: 2816}


def test_a_polled_heap_reports_the_threads_waiting_for_memory():
    thread = 0x3000
    session = _heap_session(
        {**_STATS, _HEAP + _WAIT_Q: thread, thread: _HEAP + _WAIT_Q},
        wait_q=_WAIT_Q,
    )
    session.thread_names = {thread: "worker"}

    heaps = _poll(session)

    assert heaps[0].waiters == ("worker",)
    assert heaps[0].free_bytes == 1536


def test_a_heap_without_a_resolved_queue_reports_no_waiter_list():
    """``waiters`` is None, not empty: nothing was walked."""
    session = _heap_session(_STATS, wait_q=None)

    assert _poll(session)[0].waiters is None


def test_a_recording_without_a_chunk_map_stops_asking_for_one(monkeypatch):
    """A drift leaves the replay cursor untouched, so the heap is still reported."""
    from zview.backend.replay import ReplayMismatch

    def drift(*_args):
        raise ReplayMismatch("Replay drift at index 19: expected end_batch, got read_bytes.")

    quiet_queue = {_HEAP + _WAIT_Q: _HEAP + _WAIT_Q}
    session = _heap_session({**_STATS, **quiet_queue}, wait_q=_WAIT_Q)
    session.state["heaps"] = {"all_chunks": True}
    monkeypatch.setattr("zview.kernel.heaps.walk_heap_fragmentation", drift)

    heap = _poll(session)[0]

    assert heap.chunks is None
    assert heap.free_bytes == 1536
    assert not session.state["heaps"]["all_chunks"]
    assert session.errors == []


class LiveHeapRefusingChunkMap(FakeHeapMemory):
    """A live heap whose chunk map read the stub refuses, as QEMU's did at 2 KB."""

    is_live = True

    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def begin_batch(self) -> None:
        pass

    def end_batch(self) -> None:
        pass

    def read_bytes(self, at: int, amount: int) -> bytes:
        raise ProbeReadError(f"GDB error at 0x{at:X}: E22")


def test_a_chunk_map_the_live_stub_refused_replays_frame_after_frame(tmp_path):
    """Replay meets the refusal where the live session did, so later frames still line up."""
    from zview.backend.recording import RecordingScraper
    from zview.backend.replay import ReplayScraper

    # 264 chunks of 8 bytes: the 2112 byte read of the talk's recording.
    words = {**_STATS, _HEAP + _WAIT_Q: _HEAP + _WAIT_Q, _SYS_HEAP: 264}
    path = tmp_path / "refused.ndjson.gz"

    live = _heap_session(words, wait_q=_WAIT_Q)
    live.state["heaps"] = {"all_chunks": True}
    with RecordingScraper(LiveHeapRefusingChunkMap(words), path) as recorder:
        live.scraper = recorder
        for _ in range(3):
            recorder.begin_batch()
            _poll(live)
            recorder.end_batch()

    replayed = _heap_session(words, wait_q=_WAIT_Q)
    replayed.state["heaps"] = {"all_chunks": True}
    with ReplayScraper(path, honor_timing=False) as replay:
        replayed.scraper = replay
        for _ in range(3):
            replay.begin_batch()
            assert _poll(replayed)[0].free_bytes == 1536
            replay.end_batch()

    assert (
        replayed.errors == ["Error reading sparsity for bench_heap: GDB error at 0x9000: E22"] * 3
    )


def test_a_recording_that_never_walked_the_queue_reports_no_waiter_list():
    """Replay matches a read sequence, so a feature the recording lacks must not be read."""
    session = _heap_session(_STATS, wait_q=_WAIT_Q)
    session.parts = frozenset()

    assert _poll(session)[0].waiters is None


def test_a_scalable_wait_queue_is_left_unwalked():
    session = _heap_session(_STATS, wait_q=_WAIT_Q, flavor="scalable")

    assert _poll(session)[0].waiters is None
