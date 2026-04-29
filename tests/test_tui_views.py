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
    """5 view bindings -> ``Help: ?`` prepended, top 3 shown, trailing overflow indicator."""
    view = ThreadListView(controller, theme)
    hint = view._footer_hint().rstrip()
    parts = [p.strip() for p in hint.split("|")]
    assert parts[0] == "Help: ?"
    assert parts[1:4] == ["Detail: <Enter>", "Heaps: h", "Refresh: r"]
    assert parts[-1] == "…"


def test_heap_list_footer_overflow(controller, theme):
    """4 view bindings overflow the cap of 3 -> trailing overflow indicator."""
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
    """Hard-clip the formatted string at exactly ``width``; never spill over."""
    from frontend.tui.widgets import _fit_str

    assert _fit_str("1234567890", 4) == "1234"
    assert len(_fit_str("9999999 / 99999999", 8)) == 8


def test_truncate_str_returns_unchanged_when_shorter():
    from frontend.tui.widgets import _truncate_str

    assert _truncate_str("hi", 5) == "hi"


def test_truncate_str_returns_unchanged_when_exact():
    """Off-by-one guard: name length equal to column width must NOT be truncated."""
    from frontend.tui.widgets import _truncate_str

    assert _truncate_str("hello", 5) == "hello"


def test_truncate_str_clips_with_ellipsis_when_longer():
    from frontend.tui.widgets import _truncate_str

    assert _truncate_str("hello world", 8) == "hello..."
    assert len(_truncate_str("a_very_long_name_here", 10)) == 10


class _Named:
    """Tiny stand-in for ThreadInfo/HeapInfo with just a ``name``."""

    def __init__(self, name: str):
        self.name = name


_THREAD_SCHEMA = [25, 7, 7, 27, 14]  # floor sum + 4 seps = 84


def test_compute_flex_widths_returns_floor_when_terminal_at_floor():
    from frontend.tui.views.base import compute_flex_widths

    # floor=84, reserved col bumps that to 85 before any growth happens
    assert compute_flex_widths(_THREAD_SCHEMA, 84, []) == _THREAD_SCHEMA
    assert compute_flex_widths(_THREAD_SCHEMA, 85, []) == _THREAD_SCHEMA


def test_compute_flex_widths_distributes_extra_to_bar_when_no_name_overflow():
    """No name needs more than the floor -> all extra to bar."""
    from frontend.tui.views.base import compute_flex_widths

    items = [_Named("main"), _Named("idle")]  # both well under 25
    widths = compute_flex_widths(_THREAD_SCHEMA, 100, items)
    assert widths == [25, 7, 7, 27 + (100 - 84 - 1), 14]


def test_compute_flex_widths_grows_name_only_as_needed():
    """Long name absorbs the extra it needs; remainder goes to bar."""
    from frontend.tui.views.base import compute_flex_widths

    items = [_Named("x" * 36)]
    # extra (after reserve) = 95 - 84 - 1 = 10; name needs 11; takes all 10, bar floor.
    widths = compute_flex_widths(_THREAD_SCHEMA, 95, items)
    assert widths == [35, 7, 7, 27, 14]


def test_compute_flex_widths_caps_name_at_actual_need():
    """Once name is satisfied, leftover goes to bar; name never exceeds longest item."""
    from frontend.tui.views.base import compute_flex_widths

    items = [_Named("x" * 36)]
    # extra = 130 - 84 - 1 = 45; name takes 11 (36 - 25), bar takes the remaining 34
    widths = compute_flex_widths(_THREAD_SCHEMA, 130, items)
    assert widths == [36, 7, 7, 27 + 34, 14]


def test_compute_flex_widths_never_shrinks_below_floor():
    """Bar must never drop below schema floor regardless of input."""
    from frontend.tui.views.base import compute_flex_widths

    # Empty items, terminal == floor: no growth, no shrink.
    widths = compute_flex_widths(_THREAD_SCHEMA, 84, [])
    assert widths[3] >= _THREAD_SCHEMA[3]
    # Long name, tight terminal: name swallows everything; bar must stay at floor.
    widths = compute_flex_widths(_THREAD_SCHEMA, 90, [_Named("x" * 50)])
    assert widths[3] >= _THREAD_SCHEMA[3]


def test_compute_flex_widths_empty_items_is_safe():
    """No data -> max_name defaults to 0; all extra to bar."""
    from frontend.tui.views.base import compute_flex_widths

    widths = compute_flex_widths(_THREAD_SCHEMA, 100, [])
    assert widths[0] == _THREAD_SCHEMA[0]
    assert widths[3] > _THREAD_SCHEMA[3]


# --- handle_input state machines -------------------------------------------------


import curses  # noqa: E402

from backend.base import HeapInfo, ThreadInfo  # noqa: E402
from frontend.tui.views.base import SpecialCode, ZViewState  # noqa: E402


def _make_thread(name: str, addr: int = 0x1000) -> ThreadInfo:
    return ThreadInfo(address=addr, stack_start=0, stack_size=1024, name=name, runtime=None)


def _make_heap(name: str, addr: int = 0x2000) -> HeapInfo:
    return HeapInfo(
        name=name,
        address=addr,
        free_bytes=100,
        allocated_bytes=50,
        max_allocated_bytes=80,
        usage_percent=33.0,
        chunks=None,
    )


def test_thread_list_handle_input_arrow_keys_move_cursor(controller, theme):
    view = ThreadListView(controller, theme)
    view.cursor = 5
    view.handle_input(curses.KEY_DOWN)
    assert view.cursor == 6
    view.handle_input(curses.KEY_UP)
    assert view.cursor == 5


def test_thread_list_handle_input_sort_cycles_index(controller, theme):
    view = ThreadListView(controller, theme)
    initial = view._current_sort_idx
    view.handle_input(SpecialCode.SORT)
    assert view._current_sort_idx == (initial + 1) % len(view._sort_keys)


def test_thread_list_handle_input_invert_flips_order(controller, theme):
    view = ThreadListView(controller, theme)
    initial = view._invert_sorting
    view.handle_input(SpecialCode.INVERSE)
    assert view._invert_sorting != initial


def test_thread_list_handle_input_heaps_transitions_when_supported(controller, theme):
    view = ThreadListView(controller, theme)
    assert view.handle_input(SpecialCode.HEAPS) is ZViewState.HEAP_LIST_VIEW


def test_thread_list_handle_input_enter_with_no_threads_stays_put(controller, theme):
    """ENTER on an empty list must not crash or transition."""
    controller.threads_data = []
    view = ThreadListView(controller, theme)
    assert view.handle_input(SpecialCode.NEWLINE) is None


def test_thread_list_handle_input_enter_selects_and_transitions(controller, theme):
    controller.threads_data = [_make_thread("main"), _make_thread("idle")]
    view = ThreadListView(controller, theme)
    view.cursor = 1
    state = view.handle_input(SpecialCode.NEWLINE)
    assert state is ZViewState.THREAD_DETAIL_VIEW
    assert controller.detailing_thread in {"main", "idle"}


def test_thread_list_handle_input_quit_sets_running_false(controller, theme):
    controller.running = True
    view = ThreadListView(controller, theme)
    view.handle_input(SpecialCode.QUIT)
    assert controller.running is False


def test_heap_list_handle_input_arrow_keys_move_cursor(controller, theme):
    view = HeapListView(controller, theme)
    view.cursor = 2
    view.handle_input(curses.KEY_DOWN)
    assert view.cursor == 3
    view.handle_input(curses.KEY_UP)
    assert view.cursor == 2


def test_heap_list_handle_input_sort_and_invert(controller, theme):
    view = HeapListView(controller, theme)
    initial_sort = view._current_sort_idx
    initial_inv = view._invert_sorting
    view.handle_input(SpecialCode.SORT)
    view.handle_input(SpecialCode.INVERSE)
    assert view._current_sort_idx != initial_sort
    assert view._invert_sorting != initial_inv


def test_heap_list_handle_input_h_returns_to_threads(controller, theme):
    view = HeapListView(controller, theme)
    assert view.handle_input(SpecialCode.HEAPS) is ZViewState.THREAD_LIST_VIEW


def test_heap_list_handle_input_enter_with_no_heaps_stays_put(controller, theme):
    controller.heaps_data = []
    view = HeapListView(controller, theme)
    assert view.handle_input(SpecialCode.NEWLINE) is None


def test_heap_list_handle_input_enter_selects_and_transitions(controller, theme):
    controller.heaps_data = [_make_heap("h1", 0x1000), _make_heap("h2", 0x2000)]
    view = HeapListView(controller, theme)
    view.cursor = 0
    state = view.handle_input(SpecialCode.NEWLINE)
    assert state is ZViewState.HEAPS_DETAIL_VIEW
    assert controller.detailing_heap_address in {0x1000, 0x2000}


def test_thread_detail_handle_input_enter_returns_to_list(controller, theme):
    view = ThreadDetailView(controller, theme)
    assert view.handle_input(SpecialCode.NEWLINE) is ZViewState.THREAD_LIST_VIEW


def test_thread_detail_handle_input_quit(controller, theme):
    controller.running = True
    view = ThreadDetailView(controller, theme)
    view.handle_input(SpecialCode.QUIT)
    assert controller.running is False


def test_heap_detail_handle_input_enter_returns_to_list(controller, theme):
    view = HeapDetailView(controller, theme)
    assert view.handle_input(SpecialCode.NEWLINE) is ZViewState.HEAP_LIST_VIEW


def test_heap_detail_handle_input_quit(controller, theme):
    controller.running = True
    view = HeapDetailView(controller, theme)
    view.handle_input(SpecialCode.QUIT)
    assert controller.running is False


def test_fatal_error_handle_input_quit(controller, theme):
    controller.running = True
    view = FatalErrorView(controller, theme)
    view.handle_input(SpecialCode.QUIT)
    assert controller.running is False


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
