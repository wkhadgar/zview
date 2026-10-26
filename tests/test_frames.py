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


def test_a_muted_row_stays_as_last_read_while_the_others_move_on():
    frames = FrameHistory()
    muted = Selection(muted=frozenset({("threads", "worker")}))

    frames.add(
        Frame(
            {
                "threads": [
                    Info(address=0x1000, name="main", count=1),
                    Info(address=0x2000, name="worker", count=1),
                ]
            }
        )
    )
    frames.add(Frame({"threads": [Info(address=0x1000, name="main", count=2)]}))

    shown = frames.latest("threads", muted)
    assert [(t.name, t.count) for t in shown] == [("main", 2), ("worker", 1)]


def test_a_selection_changed_after_a_frame_applies_to_it():
    """A replay that ended takes no other frame, and a row muted since is listed once."""
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
    selection = Selection(exclude=((None, "logging"),), muted=frozenset({("threads", "worker")}))

    assert [t.name for t in frames.latest("threads", selection)] == ["main", "worker"]


def test_a_muted_row_the_patterns_leave_out_is_not_listed():
    frames = FrameHistory()
    frames.add(Frame({"threads": [Info(address=0x2000, name="worker")]}))
    muted = frozenset({("threads", "worker")})

    assert frames.latest("threads", Selection(include=((None, "main"),), muted=muted)) == []
    assert frames.latest("threads", Selection(groups=frozenset({"heaps"}), muted=muted)) == []
