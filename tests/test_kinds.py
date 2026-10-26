# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from zview.backend.base import ThreadInfo, ThreadRuntime
from zview.frontend.tui.views.kernel_object_list import KernelObjectListView
from zview.kernel.groups import GROUPS
from zview.kinds import BY_KEY, KINDS
from zview.kinds.kind import Kind, Level, State


def test_every_group_has_a_kind_in_the_order_a_frame_reads_them():
    """A group added without a kind would read rows nothing shows or dumps."""
    assert [kind.key for kind in KINDS] == [group.key for group in GROUPS]


def test_every_kind_the_objects_view_lists_names_and_states_its_rows():
    listed = [kind for kind in KINDS if KernelObjectListView.OBJECTS.reads_group(kind.key)]

    assert listed
    for kind in listed:
        assert kind.label and kind.title, kind.key
        assert type(kind).state is not Kind.state, kind.key


def test_a_thread_reads_at_a_glance_from_its_measured_share():
    """A build that counts no usage has no share to show."""

    def thread(cpu: float | None, active: bool) -> ThreadInfo:
        runtime = ThreadRuntime(cpu, cpu, active, None, None)
        return ThreadInfo(0x1000, 0, 512, "main", runtime)

    threads = BY_KEY["threads"]

    assert threads.state(thread(12.5, True)) == State(12.5, "running", Level.BUSY)
    assert threads.state(thread(0.0, False)) == State(0.0, "waiting", Level.QUIET)
    assert threads.state(thread(None, False)) is None


def test_a_row_in_json_keeps_every_field_beside_its_line_and_glance():
    """A thread's scheduler ``state`` field must survive its glance."""
    runtime = ThreadRuntime(40.0, 25.0, True, 64, 12.5)
    info = ThreadInfo(0x1000, 0, 512, "main", runtime, state=4)

    row = BY_KEY["threads"].to_json(info)

    assert row["state"] == 4
    assert row["runtime"]["cpu"] == 40.0
    assert row["line"] == BY_KEY["threads"].dump(info)
    assert row["glance"] == {"fill": 25.0, "text": "running", "level": "busy"}
    assert "glance" not in BY_KEY["threads"].to_json(ThreadInfo(0x1000, 0, 512, "main", None))
