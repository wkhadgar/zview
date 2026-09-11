# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the message log behind the ``m`` overlay."""

import curses
import queue
import re
from collections import deque
from unittest.mock import MagicMock

import pytest

from frontend.tui.views.base import Keybind, SpecialCode, ZViewState, ZViewTUIAttributes
from frontend.tui.widgets import PopupRow, TUIPopup
from frontend.zview_tui import _MESSAGE_LOG_SIZE, LogEntry, ZView


class _StrictWin:
    """Window stand-in that rejects the writes curses rejects."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.writes: list[tuple[int, int, str]] = []

    def addstr(self, y, x, text, attr=0):
        del attr
        if y < 0 or x < 0 or y >= self.height or x + len(text) > self.width:
            raise curses.error("addwstr() returned ERR")
        if y == self.height - 1 and x + len(text) == self.width:
            raise curses.error("addwstr() returned ERR")
        self.writes.append((y, x, text))

    def attron(self, attr):
        del attr

    def attroff(self, attr):
        del attr

    def getmaxyx(self):
        return self.height, self.width


class _RecordingWin:
    """Minimal curses window stand-in recording the writes it receives."""

    def __init__(self):
        self.writes: list[tuple[int, int, str]] = []

    def addstr(self, y, x, text, attr=0):
        del attr
        self.writes.append((y, x, text))

    def attron(self, attr):
        del attr

    def attroff(self, attr):
        del attr

    def getmaxyx(self):
        return 20, 60


@pytest.fixture
def app() -> ZView:
    """``ZView`` instance without going through curses init."""
    a = ZView.__new__(ZView)
    a.min_dimensions = (14, 85)
    a.threads_data = []
    a.heaps_data = []
    a.status_message = ""
    a.messages = deque(maxlen=_MESSAGE_LOG_SIZE)
    a.update_count = 0
    a.idle_thread = None
    a.data_queue = queue.Queue()
    a.scraper = MagicMock()
    a.scraper.idle_threads_address = 0xDEAD
    a.state = ZViewState.THREAD_LIST_VIEW
    a.stdscr = MagicMock()
    a._theme = ZViewTUIAttributes.create_mono()
    a._log_popup = TUIPopup(" Messages ", 0, 0, {}, keep_tail=True)
    a._help_popup = TUIPopup(" Help ", 0, 0, {})
    a._overlay = None
    return a


@pytest.fixture
def crowded_bindings() -> list[PopupRow]:
    """More rows, and wider ones, than a minimum terminal can hold."""
    return [
        PopupRow(f"  key{i}", f"a help line long enough to crowd the popup {i}") for i in range(40)
    ]


def test_report_sets_the_status_row_and_logs(app):
    app.report("Refreshing thread list...")

    assert app.status_message == "Refreshing thread list..."
    assert [e.text for e in app.messages] == ["Refreshing thread list..."]
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d", app.messages[0].time)


def test_heartbeat_does_not_reach_the_log(app):
    """``Running...`` is liveness, not a message, so it stays off the log."""
    app.report("Initializing...")
    for _ in range(5):
        app.process_data({"threads": []})

    assert app.status_message.startswith("Running")
    assert [e.text for e in app.messages] == ["Initializing..."]


def test_repeats_collapse_into_a_count(app):
    """A per-poll error must not push everything else out of the log."""
    app.report("Refreshing thread list...")
    for _ in range(12):
        app.process_data({"error": "read timeout"})

    assert [e.text for e in app.messages] == [
        "Refreshing thread list...",
        "Error: read timeout",
    ]
    assert app.messages[-1].count == 12
    assert app.messages[-1].line() == "Error: read timeout (x12)"


def test_a_collapsed_time_is_marked_as_the_first_of_several(app):
    """The time of a collapsed entry is the first of many, not the only one."""
    app.report("Error: read timeout")
    once = app._message_rows()[-1].label

    app.report("Error: read timeout")
    collapsed = app._message_rows()[-1].label

    assert not once.endswith("+")
    assert collapsed == f"{once}+"


def test_distinct_messages_are_kept_apart(app):
    app.report("first")
    app.report("second")
    app.report("first")

    assert [e.text for e in app.messages] == ["first", "second", "first"]


def test_multiline_messages_are_flattened(app):
    """The status row renders newlines; a log row is one line."""
    app.report("TARGET LOST\n\nprobe went away")

    assert app.messages[-1].text == "TARGET LOST probe went away"


def test_log_is_bounded(app):
    for i in range(_MESSAGE_LOG_SIZE * 2):
        app.report(f"message {i}")

    assert len(app.messages) == _MESSAGE_LOG_SIZE
    assert app.messages[-1].text == f"message {_MESSAGE_LOG_SIZE * 2 - 1}"


def test_rows_are_oldest_first(app):
    for i in range(30):
        app.report(f"message {i}")

    rows = app._message_rows()

    assert [row.text for row in rows] == [f"message {i}" for i in range(30)]


def test_rows_say_so_when_nothing_was_reported(app):
    assert app._message_rows() == [PopupRow("--:--:--.-", "Nothing reported yet.")]


def test_rows_carry_the_level_of_each_message(app):
    app.report("Refreshing thread list...")
    app.report("Warning: MSGQ_LIST_VIEW is not yet implemented.")
    app.report("Error: read timeout")

    assert [row.level for row in app._message_rows()] == ["info", "warning", "error"]


def test_m_opens_the_log_and_any_key_dismisses_it(app):
    app.stdscr.getch.return_value = SpecialCode.MESSAGES
    app.process_events()
    assert app._overlay == "messages"

    app.stdscr.getch.return_value = ord("x")
    app.process_events()
    assert app._overlay is None


def test_help_and_the_log_do_not_stack(app):
    """One overlay at a time: the open one swallows the next key."""
    app.stdscr.getch.return_value = SpecialCode.HELP
    app.process_events()
    assert app._overlay == "help"

    app.stdscr.getch.return_value = SpecialCode.MESSAGES
    app.process_events()
    assert app._overlay is None


def test_log_entry_line_omits_the_count_of_a_single_report():
    assert LogEntry("12:00:00", "once").line() == "once"


def test_the_view_is_redrawn_under_an_open_overlay(app):
    """Both layers are drawn every frame, so an open overlay cannot freeze one."""
    app.views = {app.state: MagicMock()}
    app._overlay = "messages"

    for _ in range(3):
        app.draw_tui(40, 120)

    assert app.views[app.state].render.call_count == 3


def test_popup_clips_to_the_terminal_instead_of_vanishing():
    """A long message must not silently cost the whole popup."""
    win = _RecordingWin()
    rows = [PopupRow(f"12:00:{i:02d}", "read timeout at 0x20000100 " * 20) for i in range(40)]

    TUIPopup(" Messages ", 0, 0, {}, keep_tail=True).draw(win, 20, 60, rows)

    assert win.writes, "the popup drew nothing"
    assert max(x + len(text) for _, x, text in win.writes) <= 60
    assert max(y for y, _, _ in win.writes) < 20


def test_long_text_wraps_instead_of_being_cut(app):
    """A narrow popup keeps the whole message."""
    reason = (
        "Error: Target lost after 3 retries: Error reading global CPU cycles: "
        "SWD/JTAG communication failure, check the probe wiring and power"
    )
    app.report(reason)

    win = _StrictWin(40, 104)
    app._log_popup.draw(win, 40, 104, app._message_rows())

    drawn = " ".join(text.strip() for _, _, text in win.writes if text.strip())
    for word in reason.split():
        assert word in drawn


def test_wrapped_lines_carry_the_label_once(app):
    """Continuation lines are unlabelled, so one entry reads as one entry."""
    window = TUIPopup(" Messages ", 0, 0, {}, keep_tail=True)
    row = PopupRow("12:00:00.0", "a message far wider than the column it is drawn in")

    wrapped = window._wrap([row], 20)

    assert len(wrapped) > 1
    assert wrapped[0].label == "12:00:00.0"
    assert [line.label for line in wrapped[1:]] == [""] * (len(wrapped) - 1)
    assert all(len(line.text) <= 20 for line in wrapped)


def test_wrapping_does_not_push_the_popup_past_the_terminal(app):
    """Wrapped lines are what has to fit the height, not the entries."""
    for i in range(_MESSAGE_LOG_SIZE):
        app.report(f"Error: read timeout at 0x{i:08X}, {'and a very long tail ' * 6}")

    win = _StrictWin(14, 85)
    app._log_popup.draw(win, 14, 85, app._message_rows())

    assert win.writes
    assert max(y for y, _, _ in win.writes) < 13
    assert max(x + len(text) for _, x, text in win.writes) < 85


def test_help_rows_are_grouped_under_headings(app):
    """Global bindings first, then the ones the current view adds."""
    app.views = {app.state: MagicMock()}
    app.views[app.state].keybindings.return_value = [Keybind("s", "Sort", "Cycle sort keys")]

    rows = app._help_rows()

    assert [row.label for row in rows if row.heading] == ["Global", "This view"]
    assert rows[-1].text == "Cycle sort keys"


def test_a_message_burst_fits_the_smallest_terminal(app):
    """A full log stays inside the terminal at every supported size."""
    for i in range(_MESSAGE_LOG_SIZE * 2):
        app.report(f"Error: read timeout at 0x{i:08X} on a very long bus name")

    for height, width in ((14, 85), (24, 85), (57, 209)):
        win = _StrictWin(height, width)
        app._log_popup.draw(win, height, width, app._message_rows())

        assert win.writes, f"drew nothing at {width}x{height}"
        assert max(y for y, _, _ in win.writes) < height - 1
        assert max(x + len(text) for _, x, text in win.writes) < width


def test_the_log_popup_keeps_the_newest_entries_that_fit(app):
    """What is dropped is the top of the log, not the message just reported."""
    for i in range(40):
        app.report(f"message {i}")

    win = _StrictWin(14, 85)
    app._log_popup.draw(win, 14, 85, app._message_rows())

    assert any("message 39" in text for _, _, text in win.writes)
    assert not any("message 0 " in text for _, _, text in win.writes)


def test_the_log_popup_gives_up_on_a_terminal_too_narrow_to_frame(app):
    """Too small to frame is not drawn at all, rather than drawn broken."""
    app.report("Error: read timeout")

    win = _StrictWin(14, 20)
    app._log_popup.draw(win, 14, 20, app._message_rows())

    assert win.writes == []


def test_help_popup_fits_the_smallest_terminal(app, crowded_bindings):
    """The help popup grows with the view bindings and is clamped the same way."""
    win = _StrictWin(14, 85)

    app._help_popup.draw(win, 14, 85, crowded_bindings)

    assert win.writes
    assert max(y for y, _, _ in win.writes) < 13
    assert max(x + len(text) for _, x, text in win.writes) < 85


def test_each_level_is_drawn_in_its_own_color():
    """The level is what a log is read by, so it carries the color."""
    win = _AttrWin()
    window = TUIPopup(" Messages ", frame_attr=1, label_attr=2, level_attrs={"info": 3, "error": 4})

    window.draw(
        win,
        24,
        85,
        [
            PopupRow("12:00:00", "Refreshing thread list...", "info"),
            PopupRow("12:00:01", "Error: gone", "error"),
        ],
    )

    painted = {text.strip(): attr for _, _, text, attr in win.writes if text.strip()}
    assert painted["Refreshing thread list..."] == 3
    assert painted["Error: gone"] == 4
    assert painted["12:00:00"] == 2


class _AttrWin(_StrictWin):
    """``_StrictWin`` that also keeps the attribute of every write."""

    def __init__(self):
        super().__init__(24, 85)
        self.writes: list[tuple[int, int, str, int]] = []

    def addstr(self, y, x, text, attr=0):
        if y >= self.height or x + len(text) > self.width:
            raise curses.error("addwstr() returned ERR")
        self.writes.append((y, x, text, attr))
