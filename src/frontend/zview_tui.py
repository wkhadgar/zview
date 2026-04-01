# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import contextlib
import curses
import enum
import queue
import threading
import time
from dataclasses import dataclass

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
    DEFAULT_VIEW = 2
    THREAD_DETAIL = 3
    HEAPS_VIEW = 4
    HEAPS_DETAIL = 5


class SpecialCode:
    QUIT = ord("q")
    NEWLINE = ord("\n")
    RETURN = ord("\r")
    SORT = ord("s")
    INVERSE = ord("i")
    HEAPS = ord("h")
    RECONNECT = ord("r")


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
        self.stdscr = stdscr
        self.scraper: ZScraper = scraper
        self.running = True
        self.threads_data: list[ThreadInfo] = []
        self.heaps_data: list[HeapInfo] = []
        self.status_message: str = ""
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.update_count = 0

        self.state: ZViewState = ZViewState.DEFAULT_VIEW
        self.ui: dict[ZViewState, ZViewTUIScheme] = {}

        self.cursor: dict[ZViewState, int] = {ZViewState.DEFAULT_VIEW: 0, ZViewState.HEAPS_VIEW: 0}
        self.top_line: int = 0

        self.sort_keys: dict[ZViewState, list] = {
            ZViewState.DEFAULT_VIEW: [
                lambda t: t.name,
                lambda t: t.runtime.cpu if t.runtime else -1,
                lambda t: t.runtime.cpu_normalized if t.runtime else -1,
                lambda t: t.runtime.stack_watermark_percent if t.runtime else -1,
                lambda t: t.runtime.stack_watermark if t.runtime else -1,
            ],
            ZViewState.HEAPS_VIEW: [
                lambda h: h.name,
                lambda h: h.free_bytes,
                lambda h: h.allocated_bytes,
                lambda h: h.allocated_bytes / (h.allocated_bytes + h.free_bytes),
                lambda h: h.max_allocated_bytes,
            ],
            ZViewState.FATAL_ERROR: [],
        }

        self.current_sort: dict[ZViewState, int] = {
            ZViewState.DEFAULT_VIEW: 0,
            ZViewState.HEAPS_VIEW: 0,
            ZViewState.FATAL_ERROR: 0,
        }
        self.invert_sorting: bool = False
        self.detailing_thread: str | None = None
        self.detailing_thread_usages = {}
        self.idle_thread: ThreadInfo | None = None

        self._init_curses()
        self._set_ui_schemes()

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

    def _set_ui_schemes(self):  # TODO: Remove this logic in favor of view state classes
        thread_basic_info_scheme = {
            "Thread": 30,
            "CPU %": 8,
            "Load %": 8,
            "Stack Usage % (Watermark)": 32,
            "Watermark Bytes": 18,
        }
        heaps_info_scheme = {
            "Heap": 30,
            "Free B": 8,
            "Used B": 8,
            "Heap Usage %": 32,
            "Watermark Bytes": 18,
        }

        thread_scheme = ZViewTUIScheme(thread_basic_info_scheme)
        heap_scheme = ZViewTUIScheme(heaps_info_scheme)

        self.ui[ZViewState.DEFAULT_VIEW] = thread_scheme
        self.ui[ZViewState.THREAD_DETAIL] = thread_scheme
        self.ui[ZViewState.HEAPS_VIEW] = heap_scheme
        self.ui[ZViewState.HEAPS_DETAIL] = heap_scheme
        self.ui[ZViewState.FATAL_ERROR] = ZViewTUIScheme({})

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

    def _base_draw(self, height, width):
        self.stdscr.erase()

        header_text = "ZView - Zephyr RTOS Runtime Viewer"
        footer_text = {
            ZViewState.DEFAULT_VIEW: "Quit: q | Sort: s | Invert: i | Details: <Enter> ",
            ZViewState.THREAD_DETAIL: "Quit: q | All threads: <Enter> ",
            ZViewState.HEAPS_VIEW: "Quit: q | Threads: h | Details: <Enter> ",
            ZViewState.HEAPS_DETAIL: "Quit: q | All heaps: <Enter> ",
        }

        if self.scraper.has_heaps:
            footer_text[ZViewState.DEFAULT_VIEW] += "| Heaps: h "

        self.stdscr.attron(self.ATTR_HEADER_FOOTER)
        self.stdscr.move(0, 0)
        self.stdscr.clrtoeol()
        self.stdscr.addstr(0, 0, f"{header_text:^{width}}")

        footer_row = height - 1
        self.stdscr.move(footer_row, 0)
        self.stdscr.clrtoeol()
        with contextlib.suppress(curses.error):
            # This is needed since curses try to advance the cursor to the next
            # position, wich is outside the terminal, we safely ignore this.
            self.stdscr.addstr(footer_row, 0, f"{footer_text[self.state]:>{width}}")
        self.stdscr.attroff(self.ATTR_HEADER_FOOTER)

        is_error = self.status_message.startswith("Error")

        if is_error:
            self.stdscr.attron(self.ATTR_ERROR)

        status_row = footer_row - 1
        self.stdscr.addstr(status_row, 0, self.status_message[:width])

        if is_error:
            self.stdscr.attroff(self.ATTR_ERROR)

        if height <= 5:  # Realistic minimum height check
            self.stdscr.addstr(2, 0, "Terminal too small.")
            return

        ui_cfg = self.ui[self.state]
        curr_x = 0
        col_headers = list(ui_cfg.col_widths.keys())

        if self.state in (ZViewState.DEFAULT_VIEW, ZViewState.HEAPS_VIEW):
            order_symbol = " ▼" if self.invert_sorting else " ▲"
            sorting_header = col_headers[self.current_sort[self.state]]
        else:
            sorting_header = ""
            order_symbol = ""

        for col_header, h_width in ui_cfg.col_widths.items():
            if curr_x >= width:
                break

            if col_header == sorting_header:
                col_header += order_symbol

            txt = f"{col_header:^{h_width}}"[: width - curr_x]
            self.stdscr.addstr(1, curr_x, txt)
            curr_x += h_width + 1

    def _draw_fatal_error_view(self, height: int, width: int):
        self.stdscr.erase()
        self.stdscr.attron(self.ATTR_HEADER_FOOTER)
        self.stdscr.addstr(0, 0, f"{'ZView - Zephyr RTOS Runtime Viewer':^{width - 1}}")
        self.stdscr.addstr(height - 1, 0, f"{' Quit: q | Reconnect: r ':>{width - 1}}")
        self.stdscr.attroff(self.ATTR_HEADER_FOOTER)

        self.stdscr.attron(self.ATTR_ERROR)
        msg_lines = self.status_message.split('\n')

        start_y = (height // 2) - (len(msg_lines) // 2)
        for i, line in enumerate(msg_lines):
            if 0 <= start_y + i < height - 2:
                clean_line = line[: width - 2]
                self.stdscr.addstr(start_y + i, (width // 2) - (len(clean_line) // 2), clean_line)
        self.stdscr.attroff(self.ATTR_ERROR)
        self.stdscr.refresh()

    def _draw_default_view(self, height, width):
        """
        Draws the thread data table and its general informations.
        """
        max_table_rows = height - 6
        total_threads = len(self.threads_data)
        start_num = self.top_line + 1 if total_threads > 0 else 0
        end_num = min(self.top_line + max_table_rows, total_threads)
        thread_column_width = list(self.ui[self.state].col_widths.values())[0]

        scroll_indicator = f" Threads: {start_num}-{end_num} of {total_threads} "

        self.stdscr.attron(self.ATTR_HEADER_FOOTER)
        self.stdscr.addstr(height - 1, 0, scroll_indicator[:width])
        self.stdscr.attroff(self.ATTR_HEADER_FOOTER)

        thread_info_printer = TUIThreadInfo(
            self.ATTR_CURSOR,
            self.ATTR_ACTIVE_THREAD,
            self.ATTR_INACTIVE_THREAD,
            (
                self.ATTR_PROGRESS_BAR_LOW,
                self.ATTR_PROGRESS_BAR_MEDIUM,
                self.ATTR_PROGRESS_BAR_HIGH,
            ),
        )

        table_start = 4

        stack_size_sum = sum(t.stack_size for t in self.threads_data)
        stack_watermark_sum = sum(
            t.runtime.stack_watermark if t.runtime else 0 for t in self.threads_data
        )
        is_any_thread_active = any(
            t.runtime.active if t.runtime else False for t in self.threads_data
        )

        aggregate_stack_usage_pct = (
            (stack_watermark_sum / stack_size_sum * 100) if stack_size_sum > 0 else 0.0
        )

        aggregate_stack_usage_pct = (
            (stack_watermark_sum / stack_size_sum * 100) if stack_size_sum > 0 else 0.0
        )

        # Dynamically calculate true system load and CPU usage
        aggregate_load = sum(
            t.runtime.cpu for t in self.threads_data if t.runtime and t.runtime.cpu > 0
        )
        aggregate_cpu = sum(
            t.runtime.cpu_normalized
            for t in self.threads_data
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

        thread_info_printer.draw(self.stdscr, 2, 0, all_threads_info, False)
        self.stdscr.hline(3, 0, curses.ACS_S3, width)

        key_func = self.sort_keys[ZViewState.DEFAULT_VIEW][
            self.current_sort[ZViewState.DEFAULT_VIEW]
        ]

        sorted_threads = sorted(self.threads_data, key=key_func, reverse=self.invert_sorting)

        for idx, thread in enumerate(
            sorted_threads[self.top_line : self.top_line + max_table_rows]
        ):
            target_y = table_start + idx

            if target_y >= height - 2:
                break

            absolute_idx = self.top_line + idx
            thread_info_printer.draw(
                self.stdscr,
                target_y,
                0,
                thread,
                selected=(absolute_idx == self.cursor[ZViewState.DEFAULT_VIEW]),
            )

        self.stdscr.refresh()

    def _draw_thread_detail_view(self, h, w, y=2):
        """
        Draws a single thread details, and its recent CPU usage as a graph.
        """
        thread = next((t for t in self.threads_data if t.name == self.detailing_thread), None)

        if not thread or thread.runtime is None:
            return

        thread_info_printer = TUIThreadInfo(
            self.ATTR_CURSOR,
            self.ATTR_ACTIVE_THREAD,
            self.ATTR_INACTIVE_THREAD,
            (
                self.ATTR_PROGRESS_BAR_LOW,
                self.ATTR_PROGRESS_BAR_MEDIUM,
                self.ATTR_PROGRESS_BAR_HIGH,
            ),
        )

        thread_info_printer.draw(self.stdscr, y, 0, thread)

        if not self.detailing_thread_usages.get(thread.name):
            self.detailing_thread_usages[thread.name] = {"cpu": [], "load": []}

        self.detailing_thread_usages[thread.name]["cpu"].append(int(thread.runtime.cpu_normalized))
        self.detailing_thread_usages[thread.name]["load"].append(int(thread.runtime.cpu))

        graph_height = max(self.min_dimensions[0] - 6, h - 7)
        graph_width = w // 2

        if len(self.detailing_thread_usages[thread.name]["load"]) > graph_width - 2:
            self.detailing_thread_usages[thread.name]["load"].pop(0)
            self.detailing_thread_usages[thread.name]["cpu"].pop(0)

        y += 2

        TUIGraph(graph_height, graph_width, "CPU %", "Thread cycles / Cycles", (0, 100)).draw(
            self.stdscr,
            y,
            0,
            self.ATTR_GRAPH_B,
            points=self.detailing_thread_usages[thread.name]["cpu"],
        )

        TUIGraph(
            graph_height, graph_width, "Load %", "Thread cycles / Non-idle cycles", (0, 100)
        ).draw(
            self.stdscr,
            y,
            graph_width,
            self.ATTR_GRAPH_A,
            points=self.detailing_thread_usages[thread.name]["load"],
        )

        self.stdscr.refresh()

    def _draw_heaps_view(self, height, width):
        max_table_rows = height - 6
        total_heaps = len(self.heaps_data)
        start_num = self.top_line + 1 if total_heaps > 0 else 0
        end_num = min(self.top_line + max_table_rows, total_heaps)
        heaps_column_width = list(self.ui[self.state].col_widths.values())[0]

        scroll_indicator = f" Heaps: {start_num}-{end_num} of {total_heaps} "

        self.stdscr.attron(self.ATTR_HEADER_FOOTER)
        self.stdscr.addstr(height - 1, 0, scroll_indicator[:width])
        self.stdscr.attroff(self.ATTR_HEADER_FOOTER)

        table_start = 4

        free_bytes_sum = sum(h.free_bytes for h in self.heaps_data)
        allocated_bytes_sum = sum(h.allocated_bytes for h in self.heaps_data)
        max_allocated_bytes_sum = sum(h.max_allocated_bytes for h in self.heaps_data)

        total_heap_bytes = free_bytes_sum + allocated_bytes_sum
        aggregate_usage_pct = (
            (allocated_bytes_sum / total_heap_bytes * 100) if total_heap_bytes > 0 else 0.0
        )

        all_heaps_info = HeapInfo(
            name="All Heaps".center(heaps_column_width),
            address=0,
            free_bytes=free_bytes_sum,
            allocated_bytes=allocated_bytes_sum,
            max_allocated_bytes=max_allocated_bytes_sum,
            usage_percent=aggregate_usage_pct,
            chunks=None,
        )

        tui_heap_info = TUIHeapInfo(
            self.ATTR_CURSOR,
            self.ATTR_ACTIVE_THREAD,
            (
                self.ATTR_PROGRESS_BAR_LOW,
                self.ATTR_PROGRESS_BAR_MEDIUM,
                self.ATTR_PROGRESS_BAR_HIGH,
            ),
        )

        tui_heap_info.draw(self.stdscr, 2, 0, all_heaps_info)
        self.stdscr.hline(3, 0, curses.ACS_S3, width)

        key_func = self.sort_keys[ZViewState.HEAPS_VIEW][self.current_sort[ZViewState.HEAPS_VIEW]]

        sorted_heaps = sorted(self.heaps_data, key=key_func, reverse=self.invert_sorting)

        for idx, heap in enumerate(sorted_heaps[self.top_line : self.top_line + max_table_rows]):
            target_y = table_start + idx

            if target_y >= height - 2:
                break

            absolute_idx = self.top_line + idx
            tui_heap_info.draw(
                self.stdscr,
                target_y,
                0,
                heap,
                selected=(absolute_idx == self.cursor[ZViewState.HEAPS_VIEW]),
            )

        self.stdscr.refresh()

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
                map_height + 2,
                map_width + 2,
                f"Fragmentation Map ({heap.name})",
                desc if desc else "",
            ).draw(self.stdscr, start_y - 1, start_x - 1, self.ATTR_GRAPH_B)

            for i, row_str in enumerate(sparsity_matrix):
                with contextlib.suppress(curses.error):
                    self.stdscr.addstr(start_y + i, start_x, row_str, self.ATTR_GRAPH_A)

            break

        self.stdscr.refresh()

    def draw_state(self, state: ZViewState, height: int, width: int):
        if state == ZViewState.FATAL_ERROR:
            self._draw_fatal_error_view(height, width)
            return

        self._base_draw(height, width)
        match state:
            case ZViewState.DEFAULT_VIEW:
                self._draw_default_view(height, width)
            case ZViewState.THREAD_DETAIL:
                self._draw_thread_detail_view(height, width)
            case ZViewState.HEAPS_VIEW:
                self._draw_heaps_view(height, width)
            case ZViewState.HEAPS_DETAIL:
                self._draw_heaps_detail_view(height, width)

    def draw_terminal_size_warning(self, height: int, width: int):
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

    def draw_tui(self, height, width):
        if height < self.min_dimensions[0] or width < self.min_dimensions[1]:
            self.draw_terminal_size_warning(height, width)
        else:
            self.draw_state(self.state, height, width)

    def process_events(self, height, inspection_period):
        key = self.stdscr.getch()

        if self.state == ZViewState.FATAL_ERROR:
            if key == SpecialCode.QUIT:
                self.running = False
            elif key == SpecialCode.RECONNECT:
                self.status_message = "Attempting to reconnect..."
                self.scraper.finish_polling_thread()
                self.scraper._m_scraper.disconnect()
                self.purge_queue()
                self.state = ZViewState.DEFAULT_VIEW

                self.stop_event.clear()

                # Re-initialize the connection cleanly
                try:
                    self.scraper.update_available_threads()
                    self.scraper.reset_thread_pool()
                    self.scraper.start_polling_thread(
                        self.data_queue, self.stop_event, inspection_period
                    )
                except Exception as e:
                    self.process_data({"fatal_error": f"Reconnection failed: {e}"})
            return  # Block all other input during fatal error

        if self.state in (ZViewState.DEFAULT_VIEW, ZViewState.HEAPS_VIEW):
            max_table_size = len(
                self.threads_data if self.state is ZViewState.DEFAULT_VIEW else self.heaps_data
            )

            match key:
                case curses.KEY_DOWN:
                    if max_table_size > 0:
                        self.cursor[self.state] = min(
                            max_table_size - 1, self.cursor[self.state] + 1
                        )
                        if self.cursor[self.state] >= self.top_line + (height - 6):
                            self.top_line = self.cursor[self.state] - (height - 7)
                            return
                case curses.KEY_UP:
                    if max_table_size > 0:
                        self.cursor[self.state] = max(0, self.cursor[self.state] - 1)
                        if self.cursor[self.state] < self.top_line:
                            self.top_line = self.cursor[self.state]
                            return

        match key:
            case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                match self.state:
                    case ZViewState.DEFAULT_VIEW:
                        if not self.threads_data:
                            return

                        key_func = self.sort_keys[ZViewState.DEFAULT_VIEW][
                            self.current_sort[ZViewState.DEFAULT_VIEW]
                        ]
                        sorted_threads = sorted(
                            self.threads_data, key=key_func, reverse=self.invert_sorting
                        )

                        self.state = ZViewState.THREAD_DETAIL
                        self.detailing_thread = sorted_threads[
                            self.cursor[ZViewState.DEFAULT_VIEW]
                        ].name

                        new_pool = [self.scraper.all_threads[self.detailing_thread]]
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

                    case ZViewState.THREAD_DETAIL:
                        self.state = ZViewState.DEFAULT_VIEW
                        self.scraper.thread_pool = list(self.scraper.all_threads.values())
                        self.purge_queue()

                    case ZViewState.HEAPS_VIEW:
                        if not self.heaps_data:
                            return

                        key_func = self.sort_keys[ZViewState.HEAPS_VIEW][
                            self.current_sort[ZViewState.HEAPS_VIEW]
                        ]
                        sorted_heaps = sorted(
                            self.heaps_data, key=key_func, reverse=self.invert_sorting
                        )

                        self.state = ZViewState.HEAPS_DETAIL
                        self.scraper.extra_info_heap_address = sorted_heaps[
                            self.cursor[ZViewState.HEAPS_VIEW]
                        ].address
                        self.purge_queue()

                    case ZViewState.HEAPS_DETAIL:
                        self.state = ZViewState.HEAPS_VIEW
                        self.scraper.extra_info_heap_address = None
                        self.purge_queue()

            case SpecialCode.SORT:
                if self.state not in (ZViewState.DEFAULT_VIEW, ZViewState.HEAPS_VIEW):
                    pass

                self.current_sort[self.state] = (self.current_sort[self.state] + 1) % len(
                    self.sort_keys[self.state]
                )
            case SpecialCode.INVERSE:
                if self.state not in (ZViewState.DEFAULT_VIEW, ZViewState.HEAPS_VIEW):
                    pass

                self.invert_sorting = not self.invert_sorting
            case SpecialCode.HEAPS:
                if not self.scraper.has_heaps:
                    return

                if self.state is ZViewState.HEAPS_VIEW:
                    self.scraper.reset_thread_pool()
                    self.state = ZViewState.DEFAULT_VIEW
                elif self.state is ZViewState.DEFAULT_VIEW:
                    self.scraper.thread_pool = []
                    self.state = ZViewState.HEAPS_VIEW

                self.purge_queue()
            case SpecialCode.QUIT:
                self.running = False
            case _:
                return

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

            self.process_events(h, inspection_period)

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
