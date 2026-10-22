# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from zview.frontend.tui.views.kernel_object_list import KernelObjectListView
from zview.kernel.groups import GROUPS
from zview.kinds import KINDS
from zview.kinds.kind import Kind


def test_every_group_has_a_kind_in_the_order_a_frame_reads_them():
    """A group added without a kind would read rows nothing shows or dumps."""
    assert [kind.key for kind in KINDS] == [group.key for group in GROUPS]


def test_every_kind_the_objects_view_lists_names_and_states_its_rows():
    listed = [kind for kind in KINDS if KernelObjectListView.OBJECTS.reads_group(kind.key)]

    assert listed
    for kind in listed:
        assert kind.label and kind.title, kind.key
        assert type(kind).state is not Kind.state, kind.key
