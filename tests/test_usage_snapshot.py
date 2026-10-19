# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from zview.kernel.usage import read_usage_snapshot

_USAGE = 0x40
_STAMP = 0x100
_IDLE, _WORKER = 0x1000, 0x2000


class RunningTarget:
    """Counters and a switch stamp that move while they are read, as on a running core."""

    def __init__(self, switches_after: tuple[int, ...] = ()):
        self.words = {_STAMP: 500, _IDLE + _USAGE: 9000, _WORKER + _USAGE: 300}
        self.reads: list[int] = []
        self._switches_after = switches_after

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


def test_a_quiet_target_is_read_in_one_pass():
    target = RunningTarget()

    counters = read_usage_snapshot(target, (_IDLE, _WORKER), _USAGE, (_STAMP,))

    assert counters == {_IDLE: 9000, _WORKER: 300}
    assert target.reads == [_STAMP, _IDLE + _USAGE, _WORKER + _USAGE, _STAMP]


def test_a_switch_between_the_counter_reads_is_read_again():
    """The first pass saw the idle counter before the switch and the worker's after it."""
    target = RunningTarget(switches_after=(2,))

    counters = read_usage_snapshot(target, (_IDLE, _WORKER), _USAGE, (_STAMP,))

    assert counters == {_IDLE: 9000, _WORKER: 340}
    assert len(target.reads) == 8


def test_the_second_pass_is_kept_when_the_target_switches_again():
    target = RunningTarget(switches_after=(2, 6))

    counters = read_usage_snapshot(target, (_IDLE, _WORKER), _USAGE, (_STAMP,))

    # The second pass read the worker after its second switch.
    assert counters == {_IDLE: 9000, _WORKER: 380}
    assert len(target.reads) == 8


def test_a_build_without_the_stamps_reads_the_counters_once():
    target = RunningTarget()

    counters = read_usage_snapshot(target, (_IDLE, _WORKER), _USAGE, ())

    assert counters == {_IDLE: 9000, _WORKER: 300}
    assert target.reads == [_IDLE + _USAGE, _WORKER + _USAGE]
