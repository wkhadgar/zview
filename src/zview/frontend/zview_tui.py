# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses
import dataclasses
import io
import queue
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from zview.core import Core
from zview.frames import Fatal, Frame, FrameHistory, Notice, ReplayEnd
from zview.frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
)
from zview.frontend.tui.views.fatal_error import FatalErrorView
from zview.frontend.tui.views.heap_detail import HeapDetailView
from zview.frontend.tui.views.kernel_object_detail import ObjectDetailView
from zview.frontend.tui.views.kernel_object_list import KernelObjectListView
from zview.frontend.tui.views.mem_slab_detail import MemSlabDetailView
from zview.frontend.tui.views.msgq_detail import MsgqDetailView
from zview.frontend.tui.views.mutex_detail import MutexDetailView
from zview.frontend.tui.views.semaphore_detail import SemaphoreDetailView
from zview.frontend.tui.views.thread_detail import ThreadDetailView
from zview.frontend.tui.views.thread_list import ThreadListView
from zview.frontend.tui.views.workq_detail import WorkqDetailView
from zview.frontend.tui.widgets import PopupRow, TUILineEdit, TUIPopup
from zview.kernel.threads import USAGE_WINDOW_MAX
from zview.orchestrator import ZScraper
from zview.selection import Selection

_GLOBAL_KEYBINDINGS: list[Keybind] = [
    Keybind("?", "Help", "Toggle this help overlay"),
    Keybind("m", "Messages", "Toggle the message log"),
    Keybind(
        "/",
        "Patterns",
        "Edit the row patterns: GLOB, GROUP:GLOB, !GLOB, GROUP:!GLOB, comma separated",
    ),
    Keybind("R", "Reload", "Re-parse the ELF and reattach the probe (full cycle)"),
    Keybind("q", "Quit", "Exit ZView"),
]

_MESSAGE_LOG_SIZE = 64

# Message levels, by how a reported message opens.
_ERROR_PREFIXES = ("Error", "Unable", "TARGET LOST", "Reconnection failed")
_WARNING_PREFIXES = ("Warning",)


class _StagedWindow:
    """
    Window proxy whose ``refresh`` stages the frame rather than displaying it.

    Views refresh at the end of their own render, which puts the frame on the
    screen before anything drawn on top of it.
    """

    def __init__(self, window: curses.window):
        self._window = window

    def __getattr__(self, name: str) -> Any:
        return getattr(self._window, name)

    def refresh(self) -> None:
        self._window.noutrefresh()


@dataclass
class LogEntry:
    """One reported message: the time it was first seen, its text, its repeats."""

    time: str
    text: str
    level: str = "info"
    count: int = 1

    def line(self) -> str:
        return self.text if self.count == 1 else f"{self.text} (x{self.count})"

    def stamp(self) -> str:
        """The time, marked when it is the first of several rather than the only one."""
        return self.time if self.count == 1 else f"{self.time}+"


class ZView:
    """
    A curses-based application for viewing Zephyr RTOS thread runtime information.

    This class manages the curses UI, and draws the frames and events of the
    core it is attached to.
    """

    def __init__(self, scraper: ZScraper, stdscr, selection: Selection):
        """
        Initializes the ZView application.

        Args:
            stdscr: The main curses window object provided by curses.wrapper.
            selection: What the lists show, and all a live target reads.
        """
        self.min_dimensions = (14, 85)
        self.stdscr: curses.window = stdscr
        self.scraper: ZScraper = scraper
        self.running = True
        self.frames = FrameHistory()
        # What the lists show: the launch's selection, then what the keys change.
        self.select(selection)
        self.status_message: str = ""
        # One entry per reported message, newest last.
        self.messages: deque[LogEntry] = deque(maxlen=_MESSAGE_LOG_SIZE)
        self.core = Core(scraper)
        self.data_queue = self.core.events
        self.update_count = 0

        self.state: ZViewState = ZViewState.THREAD_LIST_VIEW

        self.detailing_thread: str | None = None
        # The group key and address of the object the detail view shows.
        self.detailing: tuple[str, int] | None = None
        # Wait queues are walkable only in the dlist (simple) flavor.
        self.waiters_unknown: bool = scraper.waitq_flavor != "simple"
        # Name of the open overlay ("help" or "messages"), or None.
        self._overlay: str | None = None
        # The pattern line being edited, or None.
        self._prompt: TUILineEdit | None = None

        theme = self._init_curses()
        self._theme = theme

        levels = {
            "info": theme.INACTIVE,
            "warning": theme.PROGRESS_BAR_MEDIUM,
            "error": theme.ERROR,
        }
        self._log_popup = TUIPopup(
            " Messages ",
            theme.ACTIVE,
            theme.INACTIVE | curses.A_DIM,
            levels,
            keep_tail=True,
        )
        # Keys carry their own color: the frame and the headings are already
        # cyan, and the key is what the reader is looking for.
        self._help_popup = TUIPopup(" Help ", theme.ACTIVE, theme.PROGRESS_BAR_LOW, levels)

        self.views: dict[ZViewState, BaseStateView] = {
            ZViewState.FATAL_ERROR: FatalErrorView(self, theme),
            ZViewState.THREAD_LIST_VIEW: ThreadListView(self, theme),
            ZViewState.THREAD_DETAIL_VIEW: ThreadDetailView(self, theme),
            ZViewState.KERNEL_OBJECT_LIST_VIEW: KernelObjectListView(self, theme),
            ZViewState.OBJECT_DETAIL_VIEW: ObjectDetailView(
                self,
                theme,
                [
                    view(self, theme)
                    for view in (
                        HeapDetailView,
                        SemaphoreDetailView,
                        MutexDetailView,
                        MsgqDetailView,
                        MemSlabDetailView,
                        WorkqDetailView,
                    )
                ],
            ),
        }

        # The opening view is the thread list. A replay keeps polling whatever
        # its recording holds.
        if self.scraper.is_live:
            self.scraper.view = self.views[self.state].reads()

    def _init_curses(self) -> ZViewTUIAttributes:
        """
        Initializes curses settings and defines color pairs used in the UI.
        """
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        # How long a lone Esc waits for the rest of a key sequence, in milliseconds.
        curses.set_escdelay(25)
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)

        if not curses.has_colors():
            return ZViewTUIAttributes.create_mono()
        else:
            curses.start_color()

            # Text keeps the terminal's own background. ANSI black is only the
            # same color on a terminal whose background is ANSI black, and
            # anywhere else it paints a block behind the text.
            text_background = curses.COLOR_BLACK
            with contextlib.suppress(curses.error):
                curses.use_default_colors()
                text_background = -1

            # Active thread name
            curses.init_pair(1, curses.COLOR_CYAN, text_background)
            # Inactive thread name
            curses.init_pair(2, curses.COLOR_WHITE, text_background)
            # Progress bar: low usage
            curses.init_pair(3, curses.COLOR_GREEN, text_background)
            # Progress bar: medium usage
            curses.init_pair(4, curses.COLOR_YELLOW, text_background)
            # Progress bar: high usage
            curses.init_pair(5, curses.COLOR_RED, text_background)
            # Header/Footer background
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)
            # Error message text
            curses.init_pair(7, curses.COLOR_RED, text_background)
            # Cursor selection
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_WHITE)
            # Graph A
            curses.init_pair(9, curses.COLOR_MAGENTA, text_background)
            # Graph B
            curses.init_pair(10, curses.COLOR_CYAN, text_background)

            return ZViewTUIAttributes(
                curses.color_pair(1),
                curses.color_pair(2),
                curses.color_pair(3),
                curses.color_pair(4),
                curses.color_pair(5),
                curses.color_pair(6),
                curses.color_pair(7),
                curses.color_pair(8),
                curses.color_pair(9),
                curses.color_pair(10),
            )

    def report(self, message: str) -> None:
        """
        Put a message on the status row and into the message log.

        A message repeated back to back bumps the count of the last entry
        instead of adding another.
        """
        self.status_message = message

        text = " ".join(message.split())
        if self.messages and self.messages[-1].text == text:
            self.messages[-1].count += 1
            return

        level = "info"
        if text.startswith(_ERROR_PREFIXES):
            level = "error"
        elif text.startswith(_WARNING_PREFIXES):
            level = "warning"

        # One decimal: enough to order messages inside a poll.
        self.messages.append(LogEntry(datetime.now().strftime("%H:%M:%S.%f")[:-5], text, level))

    def purge_queue(self):
        with self.data_queue.mutex:
            self.data_queue.queue.clear()

    def attempt_reconnect(self):
        """Reload the ELF, reattach the probe and restart the data pipeline."""
        if not self.scraper.is_live:
            self.report("Reconnect is not available in replay mode.")
            return

        self.report("Reloading the ELF and reconnecting...")
        self.core.stop()
        self.purge_queue()
        # A rebuilt ELF can reassign the addresses the rows are kept by.
        self.frames = FrameHistory()
        # A muted row has no last info left to show, so it is read again.
        self.select(dataclasses.replace(self.selection, muted=frozenset()))

        try:
            self._reload_and_verify()
            self.scraper.discover()
            self.scraper.reset_runtime_state()
            self.core.start(self.core.period)
            self.transition_to(ZViewState.THREAD_LIST_VIEW)
            self.stdscr.clear()
        except Exception as e:
            self.process_data(Fatal(f"Reconnection failed: {e}"))

    def _reload_and_verify(self) -> None:
        """Reattach the target, with its stderr going to the log."""
        overlay = self._overlay
        self._overlay = "messages"
        self.draw_frame()

        captured = io.StringIO()
        try:
            with contextlib.redirect_stderr(captured):
                self.scraper.reattach()
        finally:
            for line in captured.getvalue().splitlines():
                self.report(line)
            self._overlay = overlay

        for notice in self.scraper.absent_features():
            self.report(notice)

    def draw_frame(self) -> None:
        """Draw one frame, dropping it if the screen shrank under its writes."""
        height, width = self.stdscr.getmaxyx()
        with contextlib.suppress(curses.error):
            self.draw_tui(height, width)

    def draw_tui(self, height, width):
        if height < self.min_dimensions[0] or width < self.min_dimensions[1]:
            self.stdscr.erase()

            msgs = [
                "Terminal is too small.",
                "Please resize your terminal to at least "
                f"{self.min_dimensions[1]}x{self.min_dimensions[0]}",
                f"Current: {width}x{height}",
            ]

            mid_y = height // 2
            start_y = mid_y - 1

            for i, msg in enumerate(msgs):
                if 0 <= start_y + i < height:
                    centered_line = f"{msg:^{width}}"[: width - 1]
                    self.stdscr.addstr(start_y + i, 0, centered_line)
            return

        # Under an overlay or the pattern line the view stages its frame, for
        # one update per frame.
        staged = self._overlay is not None or self._prompt is not None
        target = _StagedWindow(self.stdscr) if staged else self.stdscr
        self.views[self.state].render(target, height, width)

        if self._overlay:
            self._draw_overlay(height, width)
        if self._prompt is not None:
            self._prompt.draw(self.stdscr, height - 2, width, "/ ")
        if staged:
            self.stdscr.refresh()

    def _draw_overlay(self, height: int, width: int) -> None:
        popup, rows = {
            "messages": (self._log_popup, self._message_rows),
            "help": (self._help_popup, self._help_rows),
        }[self._overlay]
        popup.draw(self.stdscr, height, width, rows())

    def _message_rows(self) -> list[PopupRow]:
        """The log, oldest first."""
        if not self.messages:
            return [PopupRow("--:--:--.-", "Nothing reported yet.")]

        return [PopupRow(entry.stamp(), entry.line(), entry.level) for entry in self.messages]

    def _help_rows(self) -> list[PopupRow]:
        """The global bindings, then the ones the current view adds."""
        rows = [PopupRow("Global", heading=True)]
        rows += [PopupRow(f"  {b.key}", b.help_text) for b in _GLOBAL_KEYBINDINGS]

        view_bindings = self.views[self.state].keybindings()
        if view_bindings:
            rows.append(PopupRow(""))
            rows.append(PopupRow("This view", heading=True))
            rows += [PopupRow(f"  {b.key}", b.help_text) for b in view_bindings]

        return rows

    def select(self, selection: Selection) -> None:
        """Show what ``selection`` reads, and read only that from a live target."""
        self.selection = selection
        if self.scraper.is_live:
            self.scraper.selection = selection

    def toggle_mute(self, group: str, name: str) -> None:
        """Mute the row, or unmute it if it is muted."""
        muted = self.selection.muted ^ {(group, name)}
        self.select(dataclasses.replace(self.selection, muted=muted))

    def unmute(self, group: str, name: str) -> None:
        """Read the row again, if it is muted, as the detail view opening on it tracks it."""
        if (group, name) in self.selection.muted:
            self.toggle_mute(group, name)

    @property
    def usage_window(self) -> int:
        """
        The polls the CPU and load shares are summed over, 1 on a recording
        from before the usage snapshot, whose shares are of each poll alone.
        """
        summed = "usage_snapshot" in self.scraper.parts
        return self.scraper.groups["threads"].usage_window if summed else 1

    def scale_usage_window(self, wider: bool) -> None:
        """Double the polls the CPU and load shares are summed over, or halve them."""
        threads = self.scraper.groups["threads"]
        window = threads.usage_window * 2 if wider else threads.usage_window // 2
        threads.usage_window = max(1, min(USAGE_WINDOW_MAX, window))

    def _edit_prompt(self, key: int) -> None:
        """Type into the pattern line: Enter selects it, Esc drops it."""
        match key:
            case SpecialCode.ESCAPE:
                self._prompt = None
            case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                line, self._prompt = self._prompt.text, None
                self._select_patterns(line)
            case _:
                self._prompt.edit(key)

    def _select_patterns(self, line: str) -> None:
        """
        Select the patterns of ``line``, unless it is malformed or names a group
        the session lacks.
        """
        try:
            selection = self.selection.with_pattern_line(line)
            self.scraper.check_selection(selection)
        except ValueError as e:
            self.report(f"Error: {e}")
            return

        self.select(selection)
        for notice in self.scraper.unmatched(selection):
            self.report(notice)

    def has_kernel_objects(self) -> bool:
        """True while a group the objects view and the selection read is available."""
        return any(
            available
            and KernelObjectListView.OBJECTS.reads_group(key)
            and self.selection.reads_group(key)
            for key, available in self.scraper.available.items()
        )

    def transition_to(self, new_state: ZViewState):
        """Open ``new_state``'s view, reading on a live target what the view declares."""
        if new_state not in self.views:
            self.report(f"Warning: {new_state.name} is not yet implemented.")
            return

        view = self.views[new_state]
        if not view.can_open():
            return
        if row := view.row():
            self.unmute(*row)

        # Replay backends cannot absorb polling-shape mutations without drifting
        # against the recording. Views still transition; the display filters
        # from the full frame on its own.
        if self.scraper.is_live:
            self.views[self.state].closed()
            self.scraper.view = view.reads()
            view.opened()
        self.purge_queue()

        self.state = new_state

    def process_events(self):
        key = self.stdscr.getch()
        if key == -1:
            return

        if key == curses.KEY_RESIZE:
            # A size change, not a keypress: it dismisses nothing.
            self.stdscr.clear()
            return

        if self._prompt is not None:
            self._edit_prompt(key)
            return

        if self._overlay:
            self._overlay = None
            self.stdscr.clear()
            return

        if key == SpecialCode.HELP:
            self._overlay = "help"
            return

        if key == SpecialCode.MESSAGES:
            self._overlay = "messages"
            return

        if key == SpecialCode.RECONNECT:
            self.attempt_reconnect()
            return

        if key == SpecialCode.PATTERNS:
            self._prompt = TUILineEdit(self.selection.pattern_line())
            return

        new_state = self.views[self.state].handle_input(key)
        if new_state and new_state != self.state:
            self.transition_to(new_state)

    def process_data(self, data: Frame | Notice | Fatal | ReplayEnd):
        match data:
            case Fatal(text):
                self.state = ZViewState.FATAL_ERROR
                self.report(f"TARGET LOST\n\n{text}")
            case ReplayEnd():
                self.report("Recording ended; replay complete.")
            case Notice(text):
                self.report(f"Error: {text}")
            case Frame():
                self.status_message = f"Running{'.' * (self.update_count % 4)}"
                self.update_count += 1
                # A replay's frame holds the muted rows too.
                self.frames.add(data, muted=self.selection.muted)

    def run(self, inspection_period):
        """
        The main application loop.

        This loop starts the core's poll, takes its events, updates the UI,
        and processes user input (e.g., 'q' to quit).
        """
        self.report("Initializing...")

        try:
            self.scraper.discover()
        except RuntimeError as e:
            self.report(f"Unable to update available threads [{e}]")
        for notice in self.scraper.unmatched(self.selection):
            self.report(notice)

        self.core.start(inspection_period)

        while self.running:
            # Drain the queue completely on every frame
            while not self.data_queue.empty():
                with contextlib.suppress(queue.Empty):
                    data = self.data_queue.get_nowait()
                    self.process_data(data)

            self.draw_frame()

            self.process_events()

            time.sleep(0.01)


def tui_run(stdscr, scraper: ZScraper, inspection_period, selection: Selection):
    """
    The entry point for the curses application.

    This function is intended to be wrapped by `curses.wrapper` to handle
    curses library initialization and cleanup.

    Args:
        :param stdscr: Standard screen window object provided by `curses.wrapper`.
        :param scraper: ZScraper instance for data gathering.
        :param inspection_period: Period for inspection, in seconds.
        :param selection: What the lists show, and all a live target reads.
    """
    app = ZView(scraper, stdscr, selection)
    for notice in scraper.absent_features():
        app.report(notice)

    try:
        app.run(inspection_period)
    finally:
        app.core.stop()
