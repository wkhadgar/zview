# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as Info

from zview.frames import HISTORY_LENGTH, Frame, FrameHistory


def test_a_group_a_frame_leaves_out_keeps_its_last_infos():
    """The objects view reads no threads, and the thread list must not empty."""
    frames = FrameHistory()
    main = Info(address=0x1000, name="main")

    frames.add(Frame({"threads": [main], "mutexes": []}))
    frames.add(Frame({"mutexes": [Info(address=0x2000, name="lock")]}))

    assert frames.latest("threads") == [main]
    assert [m.name for m in frames.latest("mutexes")] == ["lock"]


def test_each_object_keeps_its_own_history_oldest_first():
    frames = FrameHistory()
    for count in (1, 2, 3):
        frames.add(
            Frame({"semaphores": [Info(address=0x10, count=count), Info(address=0x20, count=0)]})
        )

    assert [s.count for s in frames.of("semaphores", 0x10)] == [1, 2, 3]
    assert [s.count for s in frames.of("semaphores", 0x20)] == [0, 0, 0]
    assert frames.of("mutexes", 0x10) == []


def test_a_history_keeps_its_last_frames_only():
    frames = FrameHistory()
    for count in range(HISTORY_LENGTH + 5):
        frames.add(Frame({"semaphores": [Info(address=0x10, count=count)]}))

    counts = [s.count for s in frames.of("semaphores", 0x10)]
    assert counts == list(range(5, HISTORY_LENGTH + 5))
