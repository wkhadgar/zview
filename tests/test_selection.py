# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import pytest

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


def test_a_groups_own_includes_replace_the_ones_for_every_group():
    """``threads:gate*`` keeps the gate threads, and every exclude still applies."""
    selection = Selection().with_pattern_line("*, threads:gate*, !*_old")

    assert selection.reads("threads", "gate_keeper")
    assert not selection.reads("threads", "main")
    assert not selection.reads("threads", "gate_old")
    assert selection.reads("semaphores", "bench_slots")
    assert not selection.reads("semaphores", "bench_old")


def test_a_line_reads_only_the_groups_its_patterns_reach():
    """Nothing is implied beside a pattern, and a group that excludes ``*`` is off."""
    groups = ("threads", "heaps", "msgqs")
    lines = ("*", "heaps:*", "heaps:!*", "*, heaps:!*, msgqs:!*", "heaps:*, msgqs:*")

    read = {
        line: [group for group in groups if Selection().with_pattern_line(line).reads_group(group)]
        for line in lines
    }

    assert read == {
        "*": ["threads", "heaps", "msgqs"],
        "heaps:*": ["heaps"],
        "heaps:!*": [],
        "*, heaps:!*, msgqs:!*": ["threads"],
        "heaps:*, msgqs:*": ["heaps", "msgqs"],
    }


def test_a_pattern_line_reads_back_as_the_patterns_it_writes():
    """Commas split it, since a thread name such as ``Philosopher 5`` holds a space."""
    selection = Selection().with_pattern_line(
        " threads:gate*, lock,, !Philosopher 5, heaps:!min_* "
    )

    assert selection.include == (("threads", "gate*"), (None, "lock"))
    assert selection.exclude == ((None, "Philosopher 5"), ("heaps", "min_*"))
    assert selection.pattern_line() == "threads:gate*, lock, !Philosopher 5, heaps:!min_*"


def test_an_exclude_within_one_group_takes_the_bang_after_the_group():
    with pytest.raises(ValueError, match=r"Write heaps:!\* to exclude within one group"):
        Selection().with_pattern_line("*, !heaps:*")


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
    )


def test_a_pattern_that_turns_its_group_off_is_not_reported():
    selection = Selection().with_pattern_line("*, heaps:!*")

    assert selection.unmatched({"threads": ["main"], "heaps": ["min_heap"]}) == ()


def test_a_line_that_reads_no_group_says_so():
    """An emptied / line, or one of excludes only, would blank every view silently."""
    names = {"threads": ["idle", "main"]}

    for line in ("", "!idle"):
        notices = Selection().with_pattern_line(line).unmatched(names)
        assert notices == ("Warning: the patterns read no group",)


def test_a_header_without_includes_reads_every_row_by_name():
    selection = Selection.from_header({"groups": None})

    assert selection.include == ((None, "*"),)
    assert selection.reads("threads", "main")


def test_an_excluded_row_is_matched_by_its_exact_name():
    """The brackets of an array element would read as a glob character class."""
    selection = Selection().excluding({("threads", "z_idle_threads[0]")})

    assert not selection.reads("threads", "z_idle_threads[0]")
    assert selection.reads("threads", "z_idle_threads0")
    assert selection.reads("mutexes", "z_idle_threads[0]")
