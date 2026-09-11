# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the message log behind the ``m`` overlay."""

import queue
from collections import deque
from unittest.mock import MagicMock

import pytest

from frontend.tui.views.base import SpecialCode, ZViewState, ZViewTUIAttributes
from frontend.tui.widgets import TUITooltip
from frontend.zview_tui import _MESSAGE_LOG_SIZE, LogEntry, ZView


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
    a._overlay = None
    return a


def test_report_sets_the_status_row_and_logs(app):
    app.report("Refreshing thread list...")

    assert app.status_message == "Refreshing thread list..."
    assert [e.text for e in app.messages] == ["Refreshing thread list..."]
    assert app.messages[0].time


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


def test_rows_are_oldest_first_and_capped_by_height(app):
    for i in range(30):
        app.report(f"message {i}")

    rows = app._message_rows(14)

    assert len(rows) == 8
    assert rows[0][1] == "message 22"
    assert rows[-1][1] == "message 29"


def test_rows_say_so_when_nothing_was_reported(app):
    assert app._message_rows(24) == [("", "Nothing reported yet.")]


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
    """
    An overlay that latched its drawing left the screen frozen, so a resize
    or a repaint showed a stale frame. Both layers are drawn every frame.
    """
    app.views = {app.state: MagicMock()}
    app._overlay = "messages"

    for _ in range(3):
        app.draw_tui(40, 120)

    assert app.views[app.state].render.call_count == 3


def test_popup_clips_to_the_terminal_instead_of_vanishing():
    """A long message must not silently cost the whole popup."""
    win = _RecordingWin()
    entries = [(f"12:00:{i:02d}", "read timeout at 0x20000100 " * 20) for i in range(40)]

    TUITooltip([("", entries)], 0, " Messages ").draw(win, 20, 60)

    assert win.writes, "the popup drew nothing"
    assert max(x + len(text) for _, x, text in win.writes) <= 60
    assert max(y for y, _, _ in win.writes) < 20


def test_untitled_section_contributes_no_heading_row():
    """The log popup titles itself in its box, so its section adds no heading."""
    titled = TUITooltip([("Global", [("?", "Help")])], 0)
    untitled = TUITooltip([("", [("12:00:00", "something happened")])], 0, " Messages ")

    assert len(titled._build_rows()) == 2
    assert len(untitled._build_rows()) == 1
