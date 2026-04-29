# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses
import queue
import threading
import time

from backend.base import HeapInfo, ThreadInfo
from frontend.tui.views.base import (
    BaseStateView,
    Keybind,
    SpecialCode,
    ZViewState,
    ZViewTUIAttributes,
)
from frontend.tui.views.fatal_error import FatalErrorView
from frontend.tui.views.heap_detail import HeapDetailView
from frontend.tui.views.heap_list import HeapListView
from frontend.tui.views.thread_detail import ThreadDetailView
from frontend.tui.views.thread_list import ThreadListView
from frontend.tui.widgets import TUITooltip
from orchestrator import ZScraper

_GLOBAL_KEYBINDINGS: list[Keybind] = [
    Keybind("?", "Help", "Toggle this help overlay"),
    Keybind("R", "Reconnect", "Disconnect probe and reattach (full cycle)"),
    Keybind("q", "Quit", "Exit ZView"),
]


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
        self.status_message: str = ""
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.update_count = 0

        self.state: ZViewState = ZViewState.THREAD_LIST_VIEW

        self.detailing_thread: str | None = None
        self.detailing_heap_address: int | None = None
        self.idle_thread: ThreadInfo | None = None
        self._help_open: bool = False

        theme = self._init_curses()
        self._theme = theme

        self.views: dict[ZViewState, BaseStateView] = {
            ZViewState.FATAL_ERROR: FatalErrorView(self, theme),
            ZViewState.THREAD_LIST_VIEW: ThreadListView(self, theme),
            ZViewState.THREAD_DETAIL_VIEW: ThreadDetailView(self, theme),
            ZViewState.HEAP_LIST_VIEW: HeapListView(self, theme),
            ZViewState.HEAPS_DETAIL_VIEW: HeapDetailView(self, theme),
        }

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
            # Active thread name
            curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)
            # Inactive thread name
            curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)
            # Progress bar: low usage
            curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
            # Progress bar: medium usage
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
            # Progress bar: high usage
            curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)
            # Header/Footer background
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)
            # Error message text
            curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)
            # Cursor selection
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_WHITE)
            # Graph A
            curses.init_pair(9, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
            # Graph B
            curses.init_pair(10, curses.COLOR_CYAN, curses.COLOR_BLACK)

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

    def purge_queue(self):
        with self.data_queue.mutex:
            self.data_queue.queue.clear()

    def attempt_reconnect(self):
        """Executes hardware reconnection and data pipeline reset."""
        if not self.scraper._m_scraper.is_live:
            self.status_message = "Reconnect is not available in replay mode."
            return

        self.status_message = "Attempting to reconnect..."
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

        self.views[self.state].render(self.stdscr, height, width)

        if self._help_open:
            sections: list[tuple[str, list[tuple[str, str]]]] = [
                ("Global", [(b.key, b.help_text) for b in _GLOBAL_KEYBINDINGS]),
            ]
            view_bindings = self.views[self.state].keybindings()
            if view_bindings:
                sections.append(
                    ("This view", [(b.key, b.help_text) for b in view_bindings]),
                )
            TUITooltip(sections, self._theme.HEADER_FOOTER).draw(self.stdscr, height, width)

    def transition_to(self, new_state: ZViewState):
        """Centralized state transition and data pipeline management."""
        if new_state not in self.views:
            self.status_message = f"Warning: {new_state.name} is not yet implemented."
            return

        # Replay backends cannot absorb polling-shape mutations without drifting
        # against the recording. Views still transition; the display filters
        # from the full frame on its own.
        live = self.scraper._m_scraper.is_live

        match new_state:
            case ZViewState.THREAD_LIST_VIEW:
                if live:
                    self.scraper.thread_pool = list(self.scraper.all_threads.values())
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

        if self._help_open:
            self._help_open = False
            self.stdscr.clear()
            return

        if key == SpecialCode.HELP:
            self._help_open = True
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
            self.status_message = f"TARGET LOST\n\n{data['fatal_error']}"
            return

        if data.get("replay_complete"):
            self.status_message = "Recording ended; replay complete."
            return

        if data.get("error"):
            self.status_message = f"Error: {data['error']}"
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

    def run(self, inspection_period):
        """
        The main application loop.

        This loop continuously checks for new data from the polling thread,
        updates the UI, and processes user input (e.g., 'q' to quit).
        """
        self.status_message = "Initializing..."

        try:
            self.scraper.update_available_threads()
        except RuntimeError as e:
            self.status_message = f"Unable to update available threads [{e}]"

        self.scraper.reset_thread_pool()
        self.scraper.start_polling_thread(self.data_queue, self.stop_event, inspection_period)

        while self.running:
            # Drain the queue completely on every frame
            while not self.data_queue.empty():
                with contextlib.suppress(queue.Empty):
                    data = self.data_queue.get_nowait()
                    self.process_data(data)

            h, w = self.stdscr.getmaxyx()

            self.draw_tui(h, w)

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
