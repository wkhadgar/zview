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
