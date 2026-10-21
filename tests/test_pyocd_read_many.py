# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from zview.backend.pyocd import PyOCDScraper


class TransferFault(Exception):
    pass


class QueueingTarget:
    """
    pyOCD target stand-in. A ``now=False`` read waits in a queue until a result
    is asked for, and a block read sends the queue first, as pyOCD's DAP layer does.

    A fault fails the way pyOCD's does: one faulting word aborts its whole
    packet, reported at the packet's first word, and a faulting block read
    leaves every result still waiting broken.
    """

    def __init__(self, words: dict[int, int], faults: tuple[int, ...] = ()):
        self.words = words
        self.faults = faults
        self.sent: list[tuple[str, int]] = []
        self._queue: list[int] = []
        self._failed: dict[int, Exception] = {}

    def _send(self) -> None:
        if not self._queue:
            return
        self.sent.append(("packet", len(self._queue)))
        if any(addr in self.faults for addr in self._queue):
            error = TransferFault(f"fault at 0x{self._queue[0]:X}")
            self._failed.update(dict.fromkeys(self._queue, error))
        self._queue = []

    def read_memory(self, addr: int, transfer_size: int = 32, now: bool = True):
        assert transfer_size == 32 and not now
        self._queue.append(addr)

        def result() -> int:
            self._send()
            if addr in self._failed:
                raise self._failed[addr]
            return self.words[addr]

        return result

    def _block(self, addr: int, size: int) -> None:
        self._send()
        self.sent.append(("block", addr))
        if any(addr + i in self.faults for i in range(size)):
            self._failed = dict.fromkeys(self.words, AssertionError())
            raise TransferFault(f"fault at 0x{addr:X}")

    def read_memory_block32(self, addr: int, size: int):
        self._block(addr, size * 4)
        return [self.words[addr + 4 * i] for i in range(size)]

    def read_memory_block8(self, addr: int, size: int):
        self._block(addr, size)
        return b"".join(self.words[addr + i].to_bytes(4, "little") for i in range(0, size, 4))


def _scraper(target: QueueingTarget, endianess: str = "<") -> PyOCDScraper:
    scraper = PyOCDScraper(target_mcu=None)
    scraper.target = target
    scraper.endianess = endianess
    return scraper


# A switch stamp, two 64-bit counters, the stamp again: the usage snapshot.
_SNAPSHOT_WORDS = {0x100: 7, 0x2010: 0x2, 0x2014: 0x1, 0x3010: 0x5, 0x3014: 0x0}
_SNAPSHOT = (
    ("read32", 0x100, 1),
    ("read64", 0x2010, 1),
    ("read64", 0x3010, 1),
    ("read32", 0x100, 1),
)


def test_small_reads_reach_the_probe_together():
    target = QueueingTarget(_SNAPSHOT_WORDS)

    results = list(_scraper(target).read_many(_SNAPSHOT))

    assert results == [(7,), (0x1_0000_0002,), (5,), (7,)]
    assert target.sent == [("packet", 6)]


def test_a_long_read_is_a_block_read_in_its_place():
    """The words queued before it reach the probe first, so the order holds."""
    words = {**_SNAPSHOT_WORDS, **{0x4000 + 4 * i: i for i in range(64)}}
    target = QueueingTarget(words)

    results = list(
        _scraper(target).read_many(
            (("read32", 0x100, 1), ("read32", 0x4000, 64), ("read64", 0x2010, 1))
        )
    )

    assert [tuple(r) for r in results] == [(7,), tuple(range(64)), (0x1_0000_0002,)]
    assert target.sent == [("packet", 1), ("block", 0x4000), ("packet", 2)]


def test_a_big_endian_target_puts_the_high_word_first():
    target = QueueingTarget({0x2010: 0x1, 0x2014: 0x2})

    assert list(_scraper(target, ">").read_many((("read64", 0x2010, 1),))) == [(0x1_0000_0002,)]


def test_a_failing_word_raises_in_its_place_after_the_results_before_it():
    target = QueueingTarget(_SNAPSHOT_WORDS, faults=(0x3014,))
    results = _scraper(target).read_many(_SNAPSHOT)

    assert tuple(next(results)) == (7,)
    assert tuple(next(results)) == (0x1_0000_0002,)
    with pytest.raises(TransferFault, match="0x3010"):
        next(results)


def test_a_failing_block_read_raises_its_own_fault_after_the_results_before_it():
    """A faulting block read leaves the queued results broken, not only its own."""
    words = {**_SNAPSHOT_WORDS, **{0x4000 + 4 * i: i for i in range(64)}}
    target = QueueingTarget(words, faults=(0x4000,))
    results = _scraper(target).read_many(
        (("read32", 0x100, 1), ("read64", 0x2010, 1), ("read32", 0x4000, 64), ("read32", 0x100, 1))
    )

    assert tuple(next(results)) == (7,)
    assert tuple(next(results)) == (0x1_0000_0002,)
    with pytest.raises(TransferFault, match="0x4000"):
        next(results)
