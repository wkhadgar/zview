# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime

from backend.base import (
    HeapInfo,
    MsgqInfo,
    MutexInfo,
    MutexState,
    SemaphoreInfo,
    ThreadInfo,
)
from frontend.tui.views.base import (
    Any,
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
)
from frontend.tui.views.fatal_error import FatalErrorView
from frontend.tui.views.heap_detail import HeapDetailView
from frontend.tui.views.heap_list import HeapListView
from frontend.tui.views.kernel_object_list import KernelObjectListView
from frontend.tui.views.msgq_detail import MsgqDetailView
from frontend.tui.views.mutex_detail import MutexDetailView
from frontend.tui.views.semaphore_detail import SemaphoreDetailView
from frontend.tui.views.thread_detail import ThreadDetailView
from frontend.tui.views.thread_list import ThreadListView
from frontend.tui.widgets import PopupRow, TUIPopup
from orchestrator import ZScraper

_GLOBAL_KEYBINDINGS: list[Keybind] = [
    Keybind("?", "Help", "Toggle this help overlay"),
    Keybind("m", "Messages", "Toggle the message log"),
    Keybind("R", "Reconnect", "Disconnect probe and reattach (full cycle)"),
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

    This class manages the curses UI, starts a background thread for data polling,
    and updates the display with real-time thread statistics from a connected MCU.
    """

    def __init__(self, scraper: ZScraper, stdscr):
        """
        Initializes the ZView application.

        Args:
            stdscr: The main curses window object provided by curses.wrapper.
        """
        self.min_dimensions = (14, 85)
        self.stdscr: curses.window = stdscr
        self.scraper: ZScraper = scraper
        self.running = True
        self.threads_data: list[ThreadInfo] = []
        self.heaps_data: list[HeapInfo] = []
        self.semaphores_data: list[SemaphoreInfo] = []
        self.msgqs_data: list[MsgqInfo] = []
        self.mutexes_data: list[MutexInfo] = []
        # One sample per frame per object, for the detail views.
        self.mutex_history: dict[int, deque[MutexState]] = {}
        self.sem_history: dict[int, deque[int]] = {}
        self.msgq_history: dict[int, deque[int]] = {}
        self.status_message: str = ""
        # One entry per reported message, newest last.
        self.messages: deque[LogEntry] = deque(maxlen=_MESSAGE_LOG_SIZE)
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.update_count = 0

        self.state: ZViewState = ZViewState.THREAD_LIST_VIEW

        self.detailing_thread: str | None = None
        self.detailing_heap_address: int | None = None
        self.detailing_mutex_address: int | None = None
        self.detailing_semaphore_address: int | None = None
        self.detailing_msgq_address: int | None = None
        # Wait queues are walkable only in the dlist (simple) flavor.
        self.waiters_unknown: bool = scraper.waitq_flavor != "simple"
        self.idle_thread: ThreadInfo | None = None
        # Name of the open overlay ("help" or "messages"), or None.
        self._overlay: str | None = None

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
            ZViewState.HEAP_LIST_VIEW: HeapListView(self, theme),
            ZViewState.HEAPS_DETAIL_VIEW: HeapDetailView(self, theme),
            ZViewState.KERNEL_OBJECT_LIST_VIEW: KernelObjectListView(self, theme),
            ZViewState.MUTEX_DETAIL_VIEW: MutexDetailView(self, theme),
            ZViewState.SEMAPHORE_DETAIL_VIEW: SemaphoreDetailView(self, theme),
            ZViewState.MSGQ_DETAIL_VIEW: MsgqDetailView(self, theme),
        }

        # The opening view is the thread list. A replay keeps polling whatever
        # its recording holds.
        if self.scraper._m_scraper.is_live:
            self.scraper.poll_kernel_objects = False

    def _init_curses(self) -> ZViewTUIAttributes:
        """
        Initializes curses settings and defines color pairs used in the UI.
        """
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
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
        """Executes hardware reconnection and data pipeline reset."""
        if not self.scraper._m_scraper.is_live:
            self.report("Reconnect is not available in replay mode.")
            return

        self.report("Attempting to reconnect...")
        self.scraper.finish_polling_thread()
        self.scraper._m_scraper.disconnect()
        self.purge_queue()

        self.stop_event.clear()

        try:
            self.scraper.update_available_threads()
            self.scraper.reset_thread_pool()
            self.scraper.reset_runtime_state()
            self.scraper.start_polling_thread(
                self.data_queue, self.stop_event, self.scraper.inspection_period
            )
            self.transition_to(ZViewState.THREAD_LIST_VIEW)
            self.stdscr.clear()
        except Exception as e:
            self.process_data({"fatal_error": f"Reconnection failed: {e}"})

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

        # Under an overlay the view stages its frame, for one update per frame.
        target = _StagedWindow(self.stdscr) if self._overlay else self.stdscr
        self.views[self.state].render(target, height, width)

        if self._overlay:
            self._draw_overlay(height, width)
            self.stdscr.refresh()

    def _draw_overlay(self, height: int, width: int) -> None:
        if self._overlay == "messages":
            self._log_popup.draw(self.stdscr, height, width, self._message_rows())
            return

        self._help_popup.draw(self.stdscr, height, width, self._help_rows())

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

    def transition_to(self, new_state: ZViewState):
        """Centralized state transition and data pipeline management."""
        if new_state not in self.views:
            self.report(f"Warning: {new_state.name} is not yet implemented.")
            return

        # Replay backends cannot absorb polling-shape mutations without drifting
        # against the recording. Views still transition; the display filters
        # from the full frame on its own.
        live = self.scraper._m_scraper.is_live

        match new_state:
            case ZViewState.THREAD_LIST_VIEW:
                if live:
                    self.scraper.thread_pool = list(self.scraper.all_threads.values())
                    self.scraper.poll_kernel_objects = False
                self.purge_queue()

            case ZViewState.THREAD_DETAIL_VIEW:
                if self.detailing_thread is None:
                    return

                target_thread = self.scraper.all_threads.get(self.detailing_thread)
                if target_thread is None:
                    return

                if live:
                    new_pool = [target_thread]
                    idle_t = next(
                        (
                            t
                            for t in self.scraper.all_threads.values()
                            if t.address == self.scraper.idle_threads_address
                        ),
                        None,
                    )
                    if idle_t and idle_t.address != new_pool[0].address:
                        new_pool.append(idle_t)
                    self.scraper.thread_pool = new_pool

                self.purge_queue()

            case ZViewState.HEAP_LIST_VIEW:
                if live:
                    self.scraper.extra_info_heap_address = None
                    self.scraper.thread_pool = []
                    self.scraper.poll_kernel_objects = False
                self.purge_queue()

            case (
                ZViewState.KERNEL_OBJECT_LIST_VIEW
                | ZViewState.MUTEX_DETAIL_VIEW
                | ZViewState.SEMAPHORE_DETAIL_VIEW
                | ZViewState.MSGQ_DETAIL_VIEW
            ):
                if live:
                    # This view reads only the primitives.
                    self.scraper.extra_info_heap_address = None
                    self.scraper.thread_pool = []
                    self.scraper.poll_kernel_objects = True
                self.purge_queue()

            case ZViewState.HEAPS_DETAIL_VIEW:
                if live:
                    self.scraper.extra_info_heap_address = self.detailing_heap_address
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

        new_state = self.views[self.state].handle_input(key)
        if new_state and new_state != self.state:
            self.transition_to(new_state)

    def process_data(self, data):
        if data.get("fatal_error"):
            self.state = ZViewState.FATAL_ERROR
            self.report(f"TARGET LOST\n\n{data['fatal_error']}")
            return

        if data.get("replay_complete"):
            self.report("Recording ended; replay complete.")
            return

        if data.get("error"):
            self.report(f"Error: {data['error']}")
        else:
            self.status_message = f"Running{'.' * (self.update_count % 4)}"
            self.update_count += 1
            threads_data: list[ThreadInfo] = data.get("threads", [])
            heaps_data: list[HeapInfo] = data.get("heaps", [])

            if len(threads_data):
                idle_thread = next(
                    (t for t in threads_data if t.address == self.scraper.idle_threads_address),
                    None,
                )
                if idle_thread:
                    self.idle_thread = idle_thread
                    threads_data.remove(idle_thread)
                self.threads_data = threads_data
            if len(heaps_data):
                self.heaps_data = heaps_data

            if "semaphores" in data:
                self.semaphores_data = data["semaphores"]
                self._record_counts(self.semaphores_data)
            if "mutexes" in data:
                self.mutexes_data = data["mutexes"]
                self._record_contention(self.mutexes_data)
            if "msgqs" in data:
                self.msgqs_data = data["msgqs"]
                self._record_depths(self.msgqs_data)

    def _record_counts(self, semaphores: list[SemaphoreInfo]) -> None:
        """Append one count sample per semaphore, per frame."""
        for sem in semaphores:
            history = self.sem_history.setdefault(sem.address, deque(maxlen=256))
            history.append(sem.count)

    def _record_depths(self, msgqs: list[MsgqInfo]) -> None:
        """Append one depth sample per queue, per frame."""
        for msgq in msgqs:
            history = self.msgq_history.setdefault(msgq.address, deque(maxlen=256))
            history.append(msgq.used_msgs)

    def _record_contention(self, mutexes: list[MutexInfo]) -> None:
        """
        Append one lock state sample per mutex, per frame.

        Samples are per frame, not per lock operation.
        """
        for mutex in mutexes:
            history = self.mutex_history.setdefault(mutex.address, deque(maxlen=256))
            history.append(mutex.state)

    def run(self, inspection_period):
        """
        The main application loop.

        This loop continuously checks for new data from the polling thread,
        updates the UI, and processes user input (e.g., 'q' to quit).
        """
        self.report("Initializing...")

        try:
            self.scraper.update_available_threads()
        except RuntimeError as e:
            self.report(f"Unable to update available threads [{e}]")

        self.scraper.reset_thread_pool()
        self.scraper.start_polling_thread(self.data_queue, self.stop_event, inspection_period)

        while self.running:
            # Drain the queue completely on every frame
            while not self.data_queue.empty():
                with contextlib.suppress(queue.Empty):
                    data = self.data_queue.get_nowait()
                    self.process_data(data)

            self.draw_frame()

            self.process_events()

            time.sleep(0.01)


def tui_run(stdscr, scraper: ZScraper, inspection_period):
    """
    The entry point for the curses application.

    This function is intended to be wrapped by `curses.wrapper` to handle
    curses library initialization and cleanup.

    Args:
        :param stdscr: Standard screen window object provided by `curses.wrapper`.
        :param scraper: ZScraper instance for data gathering.
        :param inspection_period: Period for inspection, in seconds.
    """
    app = ZView(scraper, stdscr)

    try:
        app.run(inspection_period)
    finally:
        app.scraper.finish_polling_thread()
