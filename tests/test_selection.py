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


def test_a_muted_row_is_matched_by_its_exact_name():
    """The brackets of an array element would read as a glob character class."""
    selection = Selection(muted=frozenset({("threads", "z_idle_threads[0]")}))

    assert not selection.reads("threads", "z_idle_threads[0]")
    assert selection.reads("threads", "z_idle_threads0")
    assert selection.reads("mutexes", "z_idle_threads[0]")
