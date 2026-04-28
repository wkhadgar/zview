# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Pure-string coverage for TUI footer rendering."""

from unittest.mock import MagicMock

import pytest

from frontend.tui.views.base import BaseStateView, Keybind, ZViewTUIAttributes
from frontend.tui.views.fatal_error import FatalErrorView
from frontend.tui.views.heap_detail import HeapDetailView
from frontend.tui.views.heap_list import HeapListView
from frontend.tui.views.thread_detail import ThreadDetailView
from frontend.tui.views.thread_list import ThreadListView


@pytest.fixture
def theme() -> ZViewTUIAttributes:
    return ZViewTUIAttributes.create_mono()


@pytest.fixture
def controller() -> MagicMock:
    c = MagicMock()
    c.scraper.has_heaps = True
    return c


def test_thread_list_footer_truncates_at_max_with_ellipsis(controller, theme):
    """5 view bindings + ``Help: ?`` -> top 4 shown, ``...`` then ``Help: ?``."""
    view = ThreadListView(controller, theme)
    hint = view._footer_hint().rstrip()
    parts = [p.strip() for p in hint.split("|")]
    assert parts[:4] == ["Detail: <Enter>", "Sort: s", "Heaps: h", "Invert: i"]
    assert parts[-2:] == ["...", "Help: ?"]


def test_thread_list_footer_no_ellipsis_when_no_heaps(controller, theme):
    """Without heaps the binding count drops to 4; no ``...`` should appear."""
    controller.scraper.has_heaps = False
    view = ThreadListView(controller, theme)
    hint = view._footer_hint()
    assert "..." not in hint


def test_heap_list_footer_at_cap_no_ellipsis(controller, theme):
    """4 view bindings exactly fill the cap — no overflow."""
    view = HeapListView(controller, theme)
    hint = view._footer_hint()
    assert "..." not in hint
    assert hint.endswith("Help: ? ")


def test_thread_detail_footer_minimal(controller, theme):
    view = ThreadDetailView(controller, theme)
    assert view._footer_hint() == "Back: <Enter> | Help: ? "


def test_heap_detail_footer_minimal(controller, theme):
    view = HeapDetailView(controller, theme)
    assert view._footer_hint() == "Back: <Enter> | Help: ? "


def test_fatal_error_footer_is_help_gateway_only(controller, theme):
    """A view with zero bindings still advertises the help gateway."""
    view = FatalErrorView(controller, theme)
    assert view._footer_hint() == "Help: ? "


def test_footer_hint_uses_label_for_compactness():
    """``label`` populates the footer; ``help_text`` does not leak into it."""

    class _StubView(BaseStateView):
        def keybindings(self):
            return [Keybind("x", "Short", "An overly verbose description")]

        def render(self, *_a, **_kw):
            pass

        def handle_input(self, *_a, **_kw):
            return None

    view = _StubView(MagicMock(), ZViewTUIAttributes.create_mono())
    hint = view._footer_hint()
    assert "Short: x" in hint
    assert "verbose" not in hint
