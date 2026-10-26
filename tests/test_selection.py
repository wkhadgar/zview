# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from zview.selection import Selection


def test_every_group_is_read_by_default_even_one_added_later():
    assert all(Selection().reads(group, "anything") for group in ("threads", "timers"))


def test_a_group_left_out_reads_none_of_its_rows():
    selection = Selection(groups=frozenset({"threads"}))

    assert selection.reads("threads", "main")
    assert not selection.reads("semaphores", "main")


def test_include_patterns_narrow_and_exclude_patterns_win():
    selection = Selection(
        include=((None, "Philosopher*"), (None, "logging")), exclude=((None, "Philosopher 5"),)
    )

    assert selection.reads("threads", "Philosopher 0")
    assert selection.reads("threads", "logging")
    assert not selection.reads("threads", "Philosopher 5")
    assert not selection.reads("threads", "idle")


def test_a_scoped_include_narrows_only_its_own_group():
    """``threads:gate*`` keeps the gate threads without emptying the semaphores."""
    selection = Selection(include=(("threads", "gate*"),), exclude=((None, "*_old"),))

    assert selection.reads("threads", "gate_keeper")
    assert not selection.reads("threads", "main")
    assert selection.reads("semaphores", "bench_slots")
    assert not selection.reads("semaphores", "bench_old")


def test_a_pattern_line_reads_back_as_the_patterns_it_writes():
    """Commas split it, since a thread name such as ``Philosopher 5`` holds a space."""
    selection = Selection().with_pattern_line(" threads:gate*, lock,, !Philosopher 5 ")

    assert selection.include == (("threads", "gate*"), (None, "lock"))
    assert selection.exclude == ((None, "Philosopher 5"),)
    assert selection.pattern_line() == "threads:gate*, lock, !Philosopher 5"


def test_a_frame_keeps_only_the_selected_groups_and_rows():
    from types import SimpleNamespace as Row

    frame = {
        "threads": [Row(name="main"), Row(name="logging")],
        "semaphores": [Row(name="gate")],
        "mutexes": [Row(name="lock")],
    }
    selection = Selection(groups=frozenset({"threads", "mutexes"}), exclude=((None, "logging"),))

    kept = selection.keep(frame)

    assert {group: [r.name for r in rows] for group, rows in kept.items()} == {
        "threads": ["main"],
        "mutexes": ["lock"],
    }


def test_only_the_patterns_that_name_no_read_row_are_reported():
    """A typo would empty a group silently, and a group left out matches no pattern."""
    selection = Selection(
        groups=frozenset({"threads", "semaphores"}),
        include=(("threads", "gate*"), ("threads", "gtae*"), (None, "lock")),
        exclude=((None, "*_old"),),
        muted=frozenset({("semaphores", "bench_gate"), ("threads", "gone")}),
    )
    names = {
        "threads": ["gate_keeper", "main"],
        "semaphores": ["bench_gate"],
        "mutexes": ["lock"],
    }

    assert selection.unmatched(names) == (
        "Warning: include pattern threads:gtae* matches no row",
        "Warning: include pattern lock matches no row",
        "Warning: exclude pattern *_old matches no row",
        "Warning: muted row threads:gone matches no row",
    )


def test_a_muted_row_is_matched_by_its_exact_name():
    """The brackets of an array element would read as a glob character class."""
    selection = Selection(muted=frozenset({("threads", "z_idle_threads[0]")}))

    assert not selection.reads("threads", "z_idle_threads[0]")
    assert selection.reads("threads", "z_idle_threads0")
    assert selection.reads("mutexes", "z_idle_threads[0]")
