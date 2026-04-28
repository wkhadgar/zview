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
    """5 view bindings -> ``Help: ?`` prepended, top 3 shown, trailing ``…``."""
    view = ThreadListView(controller, theme)
    hint = view._footer_hint().rstrip()
    parts = [p.strip() for p in hint.split("|")]
    assert parts[0] == "Help: ?"
    assert parts[1:4] == ["Detail: <Enter>", "Heaps: h", "Refresh: r"]
    assert parts[-1] == "…"


def test_heap_list_footer_overflow(controller, theme):
    """4 view bindings overflow the cap of 3 -> trailing ``…``."""
    view = HeapListView(controller, theme)
    hint = view._footer_hint().rstrip()
    parts = [p.strip() for p in hint.split("|")]
    assert parts[0] == "Help: ?"
    assert parts[-1] == "…"


def test_thread_detail_footer_minimal(controller, theme):
    view = ThreadDetailView(controller, theme)
    assert view._footer_hint() == "Help: ? | Back: <Enter> "


def test_heap_detail_footer_minimal(controller, theme):
    view = HeapDetailView(controller, theme)
    assert view._footer_hint() == "Help: ? | Back: <Enter> "


def test_fatal_error_footer_is_help_gateway_only(controller, theme):
    """A view with zero bindings still advertises the help gateway."""
    view = FatalErrorView(controller, theme)
    assert view._footer_hint() == "Help: ? "


def test_fit_str_pads_when_shorter():
    from frontend.tui.widgets import _fit_str

    assert _fit_str("ab", 6) == "  ab  "
    assert _fit_str("ab", 6, align="<") == "ab    "
    assert _fit_str("ab", 6, align=">") == "    ab"


def test_fit_str_truncates_when_longer():
    """Hard-clip the formatted string at exactly ``width`` — never spill over."""
    from frontend.tui.widgets import _fit_str

    assert _fit_str("1234567890", 4) == "1234"
    assert len(_fit_str("9999999 / 99999999", 8)) == 8


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
