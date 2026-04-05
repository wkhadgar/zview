# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses
import enum
import queue
import threading
import time
from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

from backend.z_scraper import (
    HeapInfo,
    ThreadInfo,
    ThreadRuntime,
    ZScraper,
)
from frontend.tui_widgets import TUIBox, TUIGraph, TUIHeapInfo, TUIThreadInfo


@dataclass
class ZViewTUIScheme:
    col_widths: dict[str, int]


class ZViewState(enum.Enum):
    FATAL_ERROR = 1
    THREAD_LIST_VIEW = 2
    THREAD_DETAIL_VIEW = 3
    HEAP_LIST_VIEW = 4
    HEAPS_DETAIL_VIEW = 5


class SpecialCode:
    QUIT = ord("q")
    NEWLINE = ord("\n")
    RETURN = ord("\r")
    SORT = ord("s")
    INVERSE = ord("i")
    HEAPS = ord("h")
    RECONNECT = ord("r")


class BaseStateView:
    def __init__(self, controller: Any, frame_attribute: int, error_attribute: int):
        """
        The controller reference allows the view to access global state
        (like colors or the max threads limit) without owning it.
        """
        self.controller = controller
        self.cursor: int = 0
        self._frame_attr: int = frame_attribute
        self._error_attr: int = error_attribute

    def _render_status(
        self,
        stdscr: curses.window,
        width: int,
        y: int,
    ):
        is_error = self.controller.status_message.startswith("Error")
        attr = stdscr.getbkgd()
        attr = attr & ~0xFF if isinstance(attr, int) else attr[0]
        status_row = y
        stdscr.addstr(
            status_row,
            0,
            self.controller.status_message[:width],
            (attr | self._error_attr) if is_error else attr,
        )

    def _render_frame(
        self,
        stdscr: curses.window,
        footer_hint: str,
        height: int,
        width: int,
    ):
        header_text = "ZView - Zephyr RTOS Runtime Viewer"
        footer_text = {
            ZViewState.THREAD_DETAIL_VIEW: "Quit: q | All threads: <Enter> ",
            ZViewState.HEAPS_DETAIL_VIEW: "Quit: q | All heaps: <Enter> ",
        }

        stdscr.move(0, 0)
        stdscr.clrtoeol()
        stdscr.addstr(0, 0, f"{header_text:^{width}}", self._frame_attr)
        footer_row = height - 1
        stdscr.move(footer_row, 0)
        stdscr.clrtoeol()
        with contextlib.suppress(curses.error):
            # This is needed since curses try to advance the cursor to the next
            # position, wich is outside the terminal, we safely ignore this.
            stdscr.addstr(footer_row, 0, f"{footer_hint:>{width}}", self._frame_attr)

    @abstractmethod
    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        """
        Draw the state to the provided curses window.
        Must be implemented by all subclasses.
        """
        pass

    @abstractmethod
    def handle_input(self, key: int) -> ZViewState | None:
        """
        Process navigation and state-specific inputs.

        Returns:
            A state enum (e.g., ZViewState.THREAD_DETAIL) to request a
            context switch from the Router, or None if the state remains unchanged.
        """
        pass


class FatalErrorView(BaseStateView):
    def __init__(
        self,
        controller: Any,
        frame_attr: int,
        error_attribute: int,
    ):
        super().__init__(controller, frame_attr, error_attribute)
        self._attr_error: int = error_attribute

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()

        self._render_frame(stdscr, "Quit: q | Reconnect: r ", height, width)

        stdscr.attron(self._attr_error)
        msg_lines = self.controller.status_message.split('\n')
        start_y = (height // 2) - (len(msg_lines) // 2)

        for i, line in enumerate(msg_lines):
            if 0 <= start_y + i < height - 2:
                clean_line = line[: width - 2]
                x_pos = max(0, (width // 2) - (len(clean_line) // 2))
                stdscr.addstr(start_y + i, x_pos, clean_line)

        stdscr.attroff(self._attr_error)
        stdscr.refresh()

    def handle_input(self, key: int) -> ZViewState | None:
        if key == SpecialCode.QUIT:
            self.controller.running = False

        elif key == SpecialCode.RECONNECT:
            self.controller.attempt_reconnect()

        return None


class ThreadListView(BaseStateView):
    def __init__(
        self, controller: Any, tui_thread_info: TUIThreadInfo, frame_attr: int, error_attr: int
    ):
        super().__init__(controller, frame_attr, error_attr)

        self._current_sort_idx = 0
        self._invert_sorting = False
        self._scheme = {
            "Thread": 30,
            "CPU %": 8,
            "Load %": 8,
            "Stack Usage % (Watermark)": 32,
            "Watermark Bytes": 18,
        }
        self._collumns: list[str] = list(self._scheme.keys())
        self._sort_keys = [
            lambda t: t.name,
            lambda t: t.runtime.cpu if t.runtime else -1,
            lambda t: t.runtime.cpu_normalized if t.runtime else -1,
            lambda t: t.runtime.stack_watermark_percent if t.runtime else -1,
            lambda t: t.runtime.stack_watermark if t.runtime else -1,
        ]

        self.top_line: int = 0

        self._tui_thread_info: TUIThreadInfo = tui_thread_info

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        """
        Draws the thread data table and its general informations.
        """

        stdscr.erase()

        self._render_frame(
            stdscr,
            "Quit: q | Sort: s | Invert: i | Details: <Enter> " + "| Heaps: h "
            if self.controller.scraper.has_heaps
            else "",
            height,
            width,
        )

        max_table_rows = height - 6
        total_threads = len(self.controller.threads_data)
        start_num = self.top_line + 1 if total_threads > 0 else 0
        end_num = min(self.top_line + max_table_rows, total_threads)
        thread_column_width = self._scheme[self._collumns[0]]

        self.cursor = max(min(total_threads - 1, self.cursor), 0)
        if self.cursor >= self.top_line + max_table_rows:
            self.top_line = self.cursor - max_table_rows + 1
        elif self.cursor < self.top_line:
            self.top_line = self.cursor

        order_symbol = " ▼" if self._invert_sorting else " ▲"
        sorting_header = self._collumns[self._current_sort_idx]

        curr_x = 0
        for col_header, h_width in self._scheme.items():
            if curr_x >= width:
                break

            if col_header == sorting_header:
                col_header += order_symbol

            txt = f"{col_header:^{h_width}}"[: width - curr_x]
            stdscr.addstr(1, curr_x, txt)
            curr_x += h_width + 1

        scroll_indicator = f" Threads: {start_num}-{end_num} of {total_threads} "
        stdscr.addstr(height - 1, 0, scroll_indicator[:width], self._frame_attr)

        table_start = 4

        stack_size_sum = sum(t.stack_size for t in self.controller.threads_data)
        stack_watermark_sum = sum(
            t.runtime.stack_watermark if t.runtime else 0 for t in self.controller.threads_data
        )
        is_any_thread_active = any(
            t.runtime.active if t.runtime else False for t in self.controller.threads_data
        )
        aggregate_stack_usage_pct = (
            (stack_watermark_sum / stack_size_sum * 100) if stack_size_sum > 0 else 0.0
        )
        aggregate_stack_usage_pct = (
            (stack_watermark_sum / stack_size_sum * 100) if stack_size_sum > 0 else 0.0
        )
        aggregate_load = sum(
            t.runtime.cpu for t in self.controller.threads_data if t.runtime and t.runtime.cpu > 0
        )
        aggregate_cpu = sum(
            t.runtime.cpu_normalized
            for t in self.controller.threads_data
            if t.runtime and t.runtime.cpu_normalized > 0
        )

        all_threads_info = ThreadInfo(
            address=0,
            stack_start=0,
            stack_size=stack_size_sum,
            name="All Threads".center(thread_column_width - 1),
            runtime=ThreadRuntime(
                cpu=aggregate_load,
                cpu_normalized=aggregate_cpu,
                active=is_any_thread_active,
                stack_watermark=stack_watermark_sum,
                stack_watermark_percent=aggregate_stack_usage_pct,
            ),
        )

        self._tui_thread_info.draw(stdscr, 2, 0, all_threads_info, False)
        stdscr.hline(3, 0, curses.ACS_S3, width)

        key_func = self._sort_keys[self._current_sort_idx]

        sorted_threads = sorted(
            self.controller.threads_data, key=key_func, reverse=self._invert_sorting
        )

        for idx, thread in enumerate(
            sorted_threads[self.top_line : self.top_line + max_table_rows]
        ):
            target_y = table_start + idx

            if target_y >= height - 2:
                break

            absolute_idx = self.top_line + idx
            self._tui_thread_info.draw(
                stdscr,
                target_y,
                0,
                thread,
                selected=(absolute_idx == self.cursor),
            )

        self._render_status(stdscr, width, height - 2)

        stdscr.refresh()

    def handle_input(self, key: int) -> ZViewState | None:
        match key:
            case curses.KEY_DOWN:
                self.cursor += 1
            case curses.KEY_UP:
                self.cursor -= 1
            case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                if not self.controller.threads_data:
                    return None

                key_func = self._sort_keys[self._current_sort_idx]
                sorted_threads = sorted(
                    self.controller.threads_data, key=key_func, reverse=self._invert_sorting
                )

                self.controller.detailing_thread = sorted_threads[self.cursor].name

                return ZViewState.THREAD_DETAIL_VIEW

            case SpecialCode.SORT:
                self._current_sort_idx = (self._current_sort_idx + 1) % len(self._sort_keys)

            case SpecialCode.INVERSE:
                self._invert_sorting = not self._invert_sorting

            case SpecialCode.HEAPS:
                if not self.controller.scraper.has_heaps:
                    return None

                return ZViewState.HEAP_LIST_VIEW

            case SpecialCode.QUIT:
                self.controller.running = False

            case _:
                return None


class ThreadDetailView(ThreadListView):
    def __init__(
        self,
        controller: Any,
        tui_thread_info: TUIThreadInfo,
        frame_attr: int,
        error_attr: int,
        graph_a_attr: int,
        graph_b_attr: int,
    ):
        super().__init__(controller, tui_thread_info, frame_attr, error_attr)
        self._cpu_graph: TUIGraph = TUIGraph(
            "CPU %", "Thread cycles / Cycles", (0, 100), graph_b_attr
        )
        self._load_graph: TUIGraph = TUIGraph(
            "Load %", "Thread cycles / Non-idle cycles", (0, 100), graph_a_attr
        )

        self._current_thread_name: str | None = None
        self._usages: dict[str, list[int]] = {"cpu": [], "load": []}

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        stdscr.erase()

        self._render_frame(stdscr, "Quit: q | All threads: <Enter> ", height, width)

        curr_x = 0
        for col_header, h_width in self._scheme.items():
            if curr_x >= width:
                break

            txt = f"{col_header:^{h_width}}"[: width - curr_x]
            stdscr.addstr(1, curr_x, txt)
            curr_x += h_width + 1

        thread = next(
            (t for t in self.controller.threads_data if t.name == self.controller.detailing_thread),
            None,
        )
        if not thread or thread.runtime is None:
            self._render_status(stdscr, width, height - 2)
            stdscr.refresh()
            return

        # Reset history if switching to a new thread
        if self._current_thread_name != thread.name:
            self._current_thread_name = thread.name
            self._usages = {"cpu": [], "load": []}

        self._tui_thread_info.draw(stdscr, 2, 0, thread)

        self._usages["cpu"].append(int(thread.runtime.cpu_normalized))
        self._usages["load"].append(int(thread.runtime.cpu))

        graph_height = max(self.controller.min_dimensions[0] - 6, height - 7)
        graph_width = width // 2

        if len(self._usages["load"]) > graph_width - 2:
            self._usages["load"].pop(0)
            self._usages["cpu"].pop(0)

        y = 4

        self._cpu_graph.draw(
            stdscr,
            y,
            0,
            graph_height,
            graph_width,
            points=self._usages["cpu"],
        )

        self._load_graph.draw(
            stdscr,
            y,
            graph_width,
            graph_height,
            graph_width,
            points=self._usages["load"],
        )

        self._render_status(stdscr, width, height - 2)

        stdscr.refresh()

    def handle_input(self, key: int) -> ZViewState | None:
        if key in (curses.KEY_ENTER, SpecialCode.NEWLINE, SpecialCode.RETURN):
            return ZViewState.THREAD_LIST_VIEW
        elif key == SpecialCode.QUIT:
            self.controller.running = False
        return None


class HeapListView(BaseStateView):
    def __init__(
        self, controller: Any, tui_heap_info: TUIHeapInfo, frame_attr: int, error_attribute: int
    ):
        super().__init__(controller, frame_attr, error_attribute)

        self._current_sort_idx = 0
        self._invert_sorting = False

        self._scheme = {
            "Heap": 30,
            "Free B": 8,
            "Used B": 8,
            "Heap Usage %": 32,
            "Watermark Bytes": 18,
        }
        self._columns: list[str] = list(self._scheme.keys())
        self._sort_keys = [
            lambda h: h.name,
            lambda h: h.free_bytes,
            lambda h: h.allocated_bytes,
            lambda h: h.max_allocated_bytes,
            lambda h: (
                (h.allocated_bytes / (h.allocated_bytes + h.free_bytes))
                if (h.allocated_bytes + h.free_bytes) > 0
                else 0
            ),
        ]

        self.top_line: int = 0
        self._tui_heap_info: TUIHeapInfo = tui_heap_info

    def render(self, stdscr: curses.window, height: int, width: int) -> None:
        """
        Draws the heap data table and its aggregate information.
        """
        stdscr.erase()

        self._render_frame(stdscr, "Quit: q | Threads: h | Details: <Enter> ", height, width)

        max_table_rows = height - 6
        total_heaps = len(self.controller.heaps_data)
        start_num = self.top_line + 1 if total_heaps > 0 else 0
        end_num = min(self.top_line + max_table_rows, total_heaps)
        heap_column_width = self._scheme[self._columns[0]]

        self.cursor = max(min(total_heaps - 1, self.cursor), 0)
        if self.cursor >= self.top_line + max_table_rows:
            self.top_line = self.cursor - max_table_rows + 1
        elif self.cursor < self.top_line:
            self.top_line = self.cursor

        order_symbol = " ▼" if self._invert_sorting else " ▲"
        sorting_header = self._columns[self._current_sort_idx]

        curr_x = 0
        for col_header, h_width in self._scheme.items():
            if curr_x >= width:
                break

            if col_header == sorting_header:
                col_header += order_symbol

            txt = f"{col_header:^{h_width}}"[: width - curr_x]
            stdscr.addstr(1, curr_x, txt)
            curr_x += h_width + 1

        scroll_indicator = f" Heaps: {start_num}-{end_num} of {total_heaps} "
        stdscr.addstr(height - 1, 0, scroll_indicator[:width], self._frame_attr)

        table_start = 4

        free_bytes_sum = sum(h.free_bytes for h in self.controller.heaps_data)
        allocated_bytes_sum = sum(h.allocated_bytes for h in self.controller.heaps_data)
        max_allocated_bytes_sum = sum(h.max_allocated_bytes for h in self.controller.heaps_data)
        total_heap_bytes = free_bytes_sum + allocated_bytes_sum

        aggregate_usage_pct = (
            (allocated_bytes_sum / total_heap_bytes * 100) if total_heap_bytes > 0 else 0.0
        )

        all_heaps_info = HeapInfo(
            name="All Heaps".center(heap_column_width - 1),
            address=0,
            free_bytes=free_bytes_sum,
            allocated_bytes=allocated_bytes_sum,
            max_allocated_bytes=max_allocated_bytes_sum,
            usage_percent=aggregate_usage_pct,
            chunks=None,
        )

        self._tui_heap_info.draw(stdscr, 2, 0, all_heaps_info, False)
        stdscr.hline(3, 0, curses.ACS_S3, width)

        key_func = self._sort_keys[self._current_sort_idx]
        sorted_heaps = sorted(
            self.controller.heaps_data, key=key_func, reverse=self._invert_sorting
        )

        for idx, heap in enumerate(sorted_heaps[self.top_line : self.top_line + max_table_rows]):
            target_y = table_start + idx

            if target_y >= height - 2:
                break

            absolute_idx = self.top_line + idx
            self._tui_heap_info.draw(
                stdscr,
                target_y,
                0,
                heap,
                selected=(absolute_idx == self.cursor),
            )

        self._render_status(stdscr, width, height - 2)

        stdscr.refresh()

    def handle_input(self, key: int) -> ZViewState | None:
        match key:
            case curses.KEY_DOWN:
                self.cursor += 1
            case curses.KEY_UP:
                self.cursor -= 1
            case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                if not self.controller.heaps_data:
                    return None

                key_func = self._sort_keys[self._current_sort_idx]
                sorted_heaps = sorted(
                    self.controller.heaps_data, key=key_func, reverse=self._invert_sorting
                )

                self.controller.detailing_heap_address = sorted_heaps[self.cursor].address
                return ZViewState.HEAPS_DETAIL_VIEW

            case SpecialCode.SORT:
                self._current_sort_idx = (self._current_sort_idx + 1) % len(self._sort_keys)

            case SpecialCode.INVERSE:
                self._invert_sorting = not self._invert_sorting

            case SpecialCode.HEAPS:
                return ZViewState.THREAD_LIST_VIEW

            case SpecialCode.QUIT:
                self.controller.running = False

            case _:
                return None


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
        self.min_dimensions = (14, 86)
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
        self.idle_thread: ThreadInfo | None = None

        self._init_curses()
        bar_attributes = (
            self.ATTR_PROGRESS_BAR_LOW,
            self.ATTR_PROGRESS_BAR_MEDIUM,
            self.ATTR_PROGRESS_BAR_HIGH,
        )

        self.views: dict[ZViewState, BaseStateView] = {
            ZViewState.FATAL_ERROR: FatalErrorView(
                self,
                self.ATTR_HEADER_FOOTER,
                self.ATTR_ERROR,
            ),
            ZViewState.THREAD_LIST_VIEW: ThreadListView(
                self,
                TUIThreadInfo(
                    self.ATTR_CURSOR,
                    self.ATTR_ACTIVE_THREAD,
                    self.ATTR_INACTIVE_THREAD,
                    bar_attributes,
                ),
                self.ATTR_HEADER_FOOTER,
                self.ATTR_ERROR,
            ),
            ZViewState.THREAD_DETAIL_VIEW: ThreadDetailView(
                self,
                TUIThreadInfo(
                    self.ATTR_CURSOR,
                    self.ATTR_ACTIVE_THREAD,
                    self.ATTR_INACTIVE_THREAD,
                    bar_attributes,
                ),
                self.ATTR_HEADER_FOOTER,
                self.ATTR_ERROR,
                self.ATTR_GRAPH_A,
                self.ATTR_GRAPH_B,
            ),
            ZViewState.HEAP_LIST_VIEW: HeapListView(
                self,
                TUIHeapInfo(
                    self.ATTR_CURSOR,
                    self.ATTR_ACTIVE_THREAD,
                    bar_attributes,
                ),
                self.ATTR_HEADER_FOOTER,
                self.ATTR_ERROR,
            ),
        }

    def _init_curses(self):
        """
        Initializes curses settings and defines color pairs used in the UI.
        """
        curses.curs_set(0)
        curses.noecho()
        curses.cbreak()
        self.stdscr.keypad(True)
        self.stdscr.nodelay(True)

        if curses.has_colors():
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

            self.ATTR_ACTIVE_THREAD = curses.color_pair(1)
            self.ATTR_INACTIVE_THREAD = curses.color_pair(2)
            self.ATTR_PROGRESS_BAR_LOW = curses.color_pair(3)
            self.ATTR_PROGRESS_BAR_MEDIUM = curses.color_pair(4)
            self.ATTR_PROGRESS_BAR_HIGH = curses.color_pair(5)
            self.ATTR_HEADER_FOOTER = curses.color_pair(6)
            self.ATTR_ERROR = curses.color_pair(7)
            self.ATTR_CURSOR = curses.color_pair(8)
            self.ATTR_GRAPH_A = curses.color_pair(9)
            self.ATTR_GRAPH_B = curses.color_pair(10)

    def purge_queue(self):
        with self.data_queue.mutex:
            self.data_queue.queue.clear()

    def attempt_reconnect(self):
        """Executes hardware reconnection and data pipeline reset."""
        self.status_message = "Attempting to reconnect..."
        self.scraper.finish_polling_thread()
        self.scraper._m_scraper.disconnect()
        self.purge_queue()

        self.stop_event.clear()

        try:
            self.scraper.update_available_threads()
            self.scraper.reset_thread_pool()
            self.scraper.start_polling_thread(
                self.data_queue, self.stop_event, self.scraper.inspection_period
            )
            self.transition_to(ZViewState.THREAD_LIST_VIEW)
            self.stdscr.clear()
        except Exception as e:
            self.process_data({"fatal_error": f"Reconnection failed: {e}"})

    def get_sparsity_map(self, chunks: list[dict], width: int, height: int) -> list[str]:
        """
        Compresses a linear map of physical heap chunks into a 2D terminal grid.

        Args:
            chunks: List of dicts, e.g., [{"used": True, "size": 32}, {"used": False, "size": 128}]
            width: The exact integer number of columns available for rendering.
            height: The exact integer number of rows available for rendering.

        Returns:
            A list of strings, where each string is exactly `width` characters long.
        """
        total_chars = width * height
        if not chunks or total_chars <= 0:
            return []

        total_bytes = sum(chunk["size"] for chunk in chunks)
        if total_bytes == 0:
            return []

        bytes_per_char = total_bytes / total_chars

        output = []
        chunk_idx = 0

        chunk_rem = float(chunks[0]["size"])
        chunk_is_used = chunks[0]["used"]

        for _ in range(total_chars):
            bucket_used = 0.0
            bucket_rem = bytes_per_char

            while bucket_rem > 0 and chunk_idx < len(chunks):
                take = min(chunk_rem, bucket_rem)

                if chunk_is_used:
                    bucket_used += take

                chunk_rem -= take
                bucket_rem -= take

                if chunk_rem <= 0:
                    chunk_idx += 1
                    if chunk_idx < len(chunks):
                        chunk_rem = float(chunks[chunk_idx]["size"])
                        chunk_is_used = chunks[chunk_idx]["used"]

            ratio = bucket_used / bytes_per_char

            if ratio == 0:
                output.append(" ")
            elif ratio <= 0.33:
                output.append("░")
            elif ratio <= 0.66:
                output.append("▒")
            elif ratio <= 0.99:
                output.append("▓")
            else:
                output.append("█")

        return ["".join(output[i : i + width]) for i in range(0, len(output), width)]

    def _get_fragmentation_metrics(self, chunks: list[dict]) -> dict:
        if not chunks:
            return {}

        total_chunks = len(chunks)
        allocated_chunks = sum(1 for c in chunks if c["used"])
        free_bytes = sum(c["size"] for c in chunks if not c["used"])
        largest_free = max((c["size"] for c in chunks if not c["used"]), default=0)
        ratio = (1 - largest_free / free_bytes) * 100 if free_bytes > 0 else 0.0

        return {
            "Largest free": (largest_free, "bytes"),
            "Frag ratio": (ratio, "percent"),
            "Chunks": (f"{allocated_chunks}/{total_chunks}", "raw"),
        }

    def _get_heap_details_footer(self, metrics: dict):
        if not metrics:
            return

        def fmt(value, hint):
            if hint == "bytes":
                if value >= 1024:
                    return f"{value / 1024:.1f} KB"
                return f"{value} B"
            if hint == "percent":
                return f"{value:.1f}%"
            return str(value)

        parts = [f"{k}: {fmt(v, h)}" for k, (v, h) in metrics.items()]
        return " · ".join(parts)

    def _draw_heaps_detail_view(self, height: int, width: int):
        tui_heap_info = TUIHeapInfo(
            self.ATTR_CURSOR,
            self.ATTR_ACTIVE_THREAD,
            (
                self.ATTR_PROGRESS_BAR_LOW,
                self.ATTR_PROGRESS_BAR_MEDIUM,
                self.ATTR_PROGRESS_BAR_HIGH,
            ),
        )

        for heap in self.heaps_data:
            if heap.address != self.scraper.extra_info_heap_address or not heap.chunks:
                continue

            tui_heap_info.draw(self.stdscr, 2, 0, heap)

            start_y = 5
            start_x = 1

            map_height = height - start_y - 4
            map_width = width - start_x - 1

            if map_height <= 0 or map_width <= 0:
                break

            sparsity_matrix = self.get_sparsity_map(heap.chunks, map_width, map_height)
            metrics = self._get_fragmentation_metrics(heap.chunks)
            desc = self._get_heap_details_footer(metrics)
            TUIBox(
                f"Fragmentation Map ({heap.name})",
                desc if desc else "",
                self.ATTR_GRAPH_B,
            ).draw(
                self.stdscr,
                start_y - 1,
                start_x - 1,
                map_height + 2,
                map_width + 2,
            )

            for i, row_str in enumerate(sparsity_matrix):
                with contextlib.suppress(curses.error):
                    self.stdscr.addstr(start_y + i, start_x, row_str, self.ATTR_GRAPH_A)

            break

        self.stdscr.refresh()

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

    def transition_to(self, new_state: ZViewState):
        """Centralized state transition and data pipeline management."""
        if new_state not in self.views:
            self.status_message = f"Warning: {new_state.name} is not yet implemented."
            return

        match new_state:
            case ZViewState.THREAD_LIST_VIEW:
                self.scraper.thread_pool = list(self.scraper.all_threads.values())
                self.purge_queue()

            case ZViewState.THREAD_DETAIL_VIEW:
                if self.detailing_thread is None:
                    return

                target_thread = self.scraper.all_threads.get(self.detailing_thread)
                if target_thread:
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
                self.scraper.thread_pool = []
                self.purge_queue()

        self.state = new_state

    def process_events(self):
        key = self.stdscr.getch()

        new_state = self.views[self.state].handle_input(key)
        if new_state:
            self.transition_to(new_state)

        return

        match key:
            case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                match self.state:
                    case ZViewState.HEAPS_DETAIL:
                        self.state = ZViewState.HEAPS_VIEW
                        self.scraper.extra_info_heap_address = None
                        self.purge_queue()

        return

    def process_data(self, data):
        if data.get("fatal_error"):
            self.state = ZViewState.FATAL_ERROR
            self.status_message = f"TARGET LOST\n\n{data['fatal_error']}"
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
