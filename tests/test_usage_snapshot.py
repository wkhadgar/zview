# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from zview.backend.base import AbstractScraper
from zview.kernel.usage import read_usage_snapshot

_USAGE = 0x40
_STAMP = 0x100
_IDLE, _WORKER = 0x1000, 0x2000


class RunningTarget:
    """Counters and a switch stamp that move while they are read, as on a running core."""

    def __init__(self, switches_after: tuple[int, ...] = ()):
        self.words = {_STAMP: 500, _IDLE + _USAGE: 9000, _WORKER + _USAGE: 300}
        self.reads: list[int] = []
        self.batches = 0
        self._switches_after = switches_after

    def read_many(self, reads):
        self.batches += 1
        yield from AbstractScraper.read_many(self, reads)

    def _served(self, at: int) -> int:
        value = self.words[at]
        self.reads.append(at)
        if len(self.reads) in self._switches_after:
            # The worker switches out: its counter and the stamp move together.
            self.words[_WORKER + _USAGE] += 40
            self.words[_STAMP] += 40
        return value

    def read32(self, at: int, amount: int = 1):
        return (self._served(at),)

    def read64(self, at: int, amount: int = 1):
        return (self._served(at),)


def test_a_quiet_target_holds_as_of_one_instant():
    target = RunningTarget()

    counters, coherent = read_usage_snapshot(target, (_IDLE, _WORKER), _USAGE, (_STAMP,))

    assert counters == {_IDLE: 9000, _WORKER: 300}
    assert coherent is True
    assert target.reads == [_STAMP, _IDLE + _USAGE, _WORKER + _USAGE, _STAMP]


def test_each_pass_is_one_batch():
    """A backend that queues reads sends a whole pass together."""
    quiet, switching = RunningTarget(), RunningTarget(switches_after=(2,))

    read_usage_snapshot(quiet, (_IDLE, _WORKER), _USAGE, (_STAMP,), attempts=2)
    read_usage_snapshot(switching, (_IDLE, _WORKER), _USAGE, (_STAMP,), attempts=2)

    assert (quiet.batches, switching.batches) == (1, 2)


def test_a_switch_between_the_counter_reads_is_kept_and_said():
    """The pass saw the idle counter before the switch and the worker's after it."""
    target = RunningTarget(switches_after=(2,))

    counters, coherent = read_usage_snapshot(target, (_IDLE, _WORKER), _USAGE, (_STAMP,))

    assert counters == {_IDLE: 9000, _WORKER: 340}
    assert coherent is False
    assert len(target.reads) == 4


def test_a_recording_from_before_the_single_pass_read_a_moved_pass_again():
    target = RunningTarget(switches_after=(2,))

    counters, coherent = read_usage_snapshot(
        target, (_IDLE, _WORKER), _USAGE, (_STAMP,), attempts=2
    )

    assert counters == {_IDLE: 9000, _WORKER: 340}
    assert coherent is True
    assert len(target.reads) == 8


def test_the_last_pass_is_kept_when_the_target_switches_again():
    target = RunningTarget(switches_after=(2, 6))

    counters, coherent = read_usage_snapshot(
        target, (_IDLE, _WORKER), _USAGE, (_STAMP,), attempts=2
    )

    # The second pass read the worker after its second switch.
    assert counters == {_IDLE: 9000, _WORKER: 380}
    assert coherent is False


def test_a_build_without_the_stamps_reads_the_counters_once_and_cannot_tell():
    target = RunningTarget()

    counters, coherent = read_usage_snapshot(target, (_IDLE, _WORKER), _USAGE, (), attempts=2)

    assert counters == {_IDLE: 9000, _WORKER: 300}
    assert coherent is None
    assert target.reads == [_IDLE + _USAGE, _WORKER + _USAGE]
