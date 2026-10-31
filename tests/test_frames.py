# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace as Info

from zview.frames import HISTORY_LENGTH, Frame, FrameHistory
from zview.selection import Selection


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
            Frame(
                {
                    "semaphores": [
                        Info(address=0x10, name="gate", count=count),
                        Info(address=0x20, name="slots", count=0),
                    ]
                }
            )
        )

    assert [s.count for s in frames.of("semaphores", 0x10)] == [1, 2, 3]
    assert [s.count for s in frames.of("semaphores", 0x20)] == [0, 0, 0]
    assert frames.of("mutexes", 0x10) == []


def test_a_history_keeps_its_last_frames_only():
    frames = FrameHistory()
    for count in range(HISTORY_LENGTH + 5):
        frames.add(Frame({"semaphores": [Info(address=0x10, name="gate", count=count)]}))

    counts = [s.count for s in frames.of("semaphores", 0x10)]
    assert counts == list(range(5, HISTORY_LENGTH + 5))


def test_a_selection_changed_after_a_frame_applies_to_it():
    """A replay that ended takes no other frame."""
    frames = FrameHistory()
    frames.add(
        Frame(
            {
                "threads": [
                    Info(address=0x1000, name="main"),
                    Info(address=0x2000, name="worker"),
                    Info(address=0x3000, name="logging"),
                ]
            }
        )
    )
    selection = Selection(exclude=((None, "logging"),))

    assert [t.name for t in frames.latest("threads", selection)] == ["main", "worker"]


def test_a_group_a_frame_leaves_out_keeps_the_time_it_was_last_read():
    """The objects view reads no threads, so their rows grow old while the mutexes stay new."""
    frames = FrameHistory()
    frames.add(Frame({"threads": [], "mutexes": []}, at=100.0))
    frames.add(Frame({"mutexes": []}, at=101.0))

    assert (frames.latest_at("threads"), frames.latest_at("mutexes")) == (100.0, 101.0)
    assert frames.latest_at("heaps") is None
