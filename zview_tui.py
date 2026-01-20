# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0


import argparse
import curses
import enum
import threading
import queue
import time
from dataclasses import dataclass
from typing import List

from backend.z_scraper import ZScraper, ThreadInfo, PyOCDScraper, JLinkScraper, HeapInfo
from backend.runner_parser import RunnerParser


@dataclass
class ZViewTUIScheme:
    col_headers: list[str]
    col_widths: list[int]


class ZViewState(enum.Enum):
    DEFAULT_VIEW = 1
    THREAD_DETAIL = 2
    HEAPS_DETAIL = 3


class SpecialCode:
    QUIT = ord("q")
    NEWLINE = ord("\n")
    RETURN = ord("\r")
    SORT = ord("s")
    INVERSE = ord("i")
    HEAPS = ord("h")


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
        self.min_dimensions = (21, 86)
        self.stdscr = stdscr
        self.scraper = scraper
        self.running = True
        self.threads_data: List[ThreadInfo] = []
        self.heaps_data: List[HeapInfo] = []
        self.status_message: str = ""
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.state: ZViewState = ZViewState.DEFAULT_VIEW
        self.ui: dict[ZViewState, ZViewTUIScheme] = {}

        self.cursor = 0
        self.top_line = 0

        self.sorting_options = ["name", "cpu", "watermark_p", "watermark_b"]
        self.sort_keys = {
            "name": lambda t: t.name,
            "cpu": lambda t: t.runtime.cpu,
            "watermark_p": lambda t: t.runtime.stack_watermark / t.stack_size,
            "watermark_b": lambda t: t.runtime.stack_watermark,
        }

        self._sort_by: str = self.sorting_options[0]
        self._invert_sorting: bool = False
        self.detailing_thread: str | None = None
        self.detailing_thread_usages = {}

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
            curses.init_pair(
                1, curses.COLOR_CYAN, curses.COLOR_BLACK
            )  # Active thread name
            curses.init_pair(
                2, curses.COLOR_WHITE, curses.COLOR_BLACK
            )  # Inactive thread name
            curses.init_pair(
                3, curses.COLOR_GREEN, curses.COLOR_BLACK
            )  # Progress bar: low usage
            curses.init_pair(
                4, curses.COLOR_YELLOW, curses.COLOR_BLACK
            )  # Progress bar: medium usage
            curses.init_pair(
                5, curses.COLOR_RED, curses.COLOR_BLACK
            )  # Progress bar: high usage
            curses.init_pair(
                6, curses.COLOR_WHITE, curses.COLOR_BLUE
            )  # Header/Footer background
            curses.init_pair(
                7, curses.COLOR_RED, curses.COLOR_BLACK
            )  # Error message text
            curses.init_pair(
                8, curses.COLOR_BLACK, curses.COLOR_WHITE
            )  # Cursor selection
            curses.init_pair(9, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Graphs

            self.ATTR_ACTIVE_THREAD = curses.color_pair(1)
            self.ATTR_INACTIVE_THREAD = curses.color_pair(2)
            self.ATTR_PROGRESS_BAR_LOW = curses.color_pair(3)
            self.ATTR_PROGRESS_BAR_MEDIUM = curses.color_pair(4)
            self.ATTR_PROGRESS_BAR_HIGH = curses.color_pair(5)
            self.ATTR_HEADER_FOOTER = curses.color_pair(6)
            self.ATTR_ERROR = curses.color_pair(7)
            self.ATTR_CURSOR = curses.color_pair(8)
            self.ATTR_GRAPH = curses.color_pair(9)

    @property
    def sort_by(self):
        return self._sort_by

    @sort_by.setter
    def sort_by(self, sorting: str):
        if sorting not in self.sorting_options:
            raise NotImplementedError(
                f"Sort by '{sorting}' is not available. Valid options are: {[f'{op}' for op in self.sorting_options]}"
            )

        self._sort_by = sorting

    @property
    def invert_sorting(self):
        return self._invert_sorting

    @invert_sorting.setter
    def invert_sorting(self, invert: bool):
        self._invert_sorting = invert

    def _set_ui_schemes(self):
        thread_basic_info_scheme = {
            "Thread": 30,
            "CPU %": 6,
            "Stack Usage (Watermark)": 30,
            "Watermark (Bytes)": 16,
        }
        heaps_info_scheme = {
            "Heap": 30,
            "Free Bytes": 12,
            "Allocated Bytes": 16,
            "Watermark (Bytes)": 16,
        }

        thread_scheme = ZViewTUIScheme(
            list(thread_basic_info_scheme.keys()),
            list(thread_basic_info_scheme.values()),
        )
        heap_scheme = ZViewTUIScheme(
            list(heaps_info_scheme.keys()), list(heaps_info_scheme.values())
        )

        self.ui[ZViewState.DEFAULT_VIEW] = thread_scheme
        self.ui[ZViewState.THREAD_DETAIL] = thread_scheme
        self.ui[ZViewState.HEAPS_DETAIL] = heap_scheme

    def _draw_graph(self, y, x, h, w, points, title="", maximum=100):
        horizontal_limit = "─" * (w - 2)
        self.stdscr.addstr(y, x, "┌" + horizontal_limit + "┐")
        self.stdscr.addstr(y + h, x, "└" + horizontal_limit + "┘")
        self.stdscr.addstr(y, 1, title)

        blocks = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]
        x_idx = len(points)
        for x_step in range(w):
            for y_step in range(h - 1):
                y_pos = y + y_step + 1
                x_pos = x + w - x_step - 1
                if x_step == 0 or x_step == w - 1:
                    self.stdscr.addstr(y_pos, x_pos, "│")
                else:
                    full_blocks = int((points[x_idx] / maximum) * (h - 2))
                    last_block = int(
                        (((points[x_idx] / maximum) * (h - 2)) - full_blocks)
                        * (len(blocks) - 1)
                    )
                    self.stdscr.addstr(
                        y_pos,
                        x_pos,
                        " " if y_step < ((h - 1) - full_blocks) else blocks[-1],
                    )
                    self.stdscr.addstr(
                        y_pos,
                        x_pos,
                        blocks[last_block] if y_step == ((h - 2) - full_blocks) else "",
                    )
            x_idx -= 1 if x_idx else 0

        self.stdscr.addstr(y + 1, x + w - (len(f"{maximum}")), str(maximum))
        self.stdscr.addstr(y + h - 1, x + w - 1, "0")

    def _draw_progress_bar(
        self,
        y,
        x,
        width: int,
        percentage: float,
        medium_threshold: float,
        high_threshold: float,
    ):
        if percentage > high_threshold:
            bar_color_attr = self.ATTR_PROGRESS_BAR_HIGH
        elif percentage > medium_threshold:
            bar_color_attr = self.ATTR_PROGRESS_BAR_MEDIUM
        else:
            bar_color_attr = self.ATTR_PROGRESS_BAR_LOW
        bar_width = width - 2
        completed_chars = int(bar_width * (percentage / 100))
        remaining_chars = bar_width - completed_chars
        bar_str = "│" + "█" * completed_chars + "-" * remaining_chars + "│"
        self.stdscr.attron(bar_color_attr)
        self.stdscr.addstr(y, x, bar_str)

        percent_display = f"{percentage:.1f}%"
        overlap = percent_display.center(len(bar_str))[:completed_chars].strip()
        self.stdscr.addstr(
            y, x + (width // 2) - (len(percent_display) // 2), percent_display
        )
        self.stdscr.attron(curses.A_REVERSE)
        self.stdscr.addstr(y, x + (width // 2) - (len(percent_display) // 2), overlap)
        self.stdscr.attroff(curses.A_REVERSE)
        self.stdscr.attroff(bar_color_attr)

    def _base_draw(self, height, width):
        self.stdscr.erase()

        header_text = "ZView - Zephyr RTOS Runtime Viewer"
        footer_text = "Quit: q | Sort: s | Invert: i | Heaps: h "

        self.stdscr.attron(self.ATTR_HEADER_FOOTER)
        self.stdscr.move(0, 0)
        self.stdscr.clrtoeol()
        self.stdscr.addstr(0, 0, f"{header_text:^{width}}")

        self.stdscr.move(height - 2, 0)
        self.stdscr.clrtoeol()
        self.stdscr.addstr(height - 2, 0, f"{footer_text:>{width}}")
        self.stdscr.attroff(self.ATTR_HEADER_FOOTER)

        status_row = height - 3
        is_error = self.status_message.startswith("Error")

        if is_error:
            self.stdscr.attron(self.ATTR_ERROR)

        self.stdscr.addstr(status_row, 0, self.status_message[:width])

        if is_error:
            self.stdscr.attroff(self.ATTR_ERROR)

        if height <= 5:  # Realistic minimum height check
            self.stdscr.addstr(2, 0, "Terminal too small.")
            return

        ui_cfg = self.ui[self.state]
        curr_x = 0
        for h, w in zip(ui_cfg.col_headers, ui_cfg.col_widths):
            if curr_x >= width:
                break

            txt = f"{h:^{w}}"[: width - curr_x]
            self.stdscr.addstr(2, curr_x, txt)
            curr_x += w + 1

    def _draw_thread_info(self, y, thread_info: ThreadInfo, selected: bool = False):
        col_pos = 0

        # Widths
        scheme = self.ui[self.state]
        thread_name_width = scheme.col_widths[0]
        cpu_usage_width = scheme.col_widths[1]
        stack_usage_width = scheme.col_widths[2]
        stack_bytes_width = scheme.col_widths[3]

        # Thread name
        thread_name_display = thread_info.name[:thread_name_width].ljust(
            thread_name_width
        )
        if len(thread_info.name) > thread_name_width:
            thread_name_display = thread_name_display[:-3] + "..."
        elif thread_info.name == "idle":
            thread_name_display = (
                thread_name_display[:4] + "*" + thread_name_display[4:-1]
            )

        thread_name_attr = (
            self.ATTR_CURSOR
            if selected
            else (
                self.ATTR_ACTIVE_THREAD
                if thread_info.runtime.active
                else self.ATTR_INACTIVE_THREAD
            )
        )
        self.stdscr.attron(thread_name_attr)
        self.stdscr.addstr(y, col_pos, thread_name_display)
        self.stdscr.attroff(thread_name_attr)
        col_pos += thread_name_width + 1

        # Thread CPUs
        cpu_display = f"{round(thread_info.runtime.cpu, 1)}%".ljust(cpu_usage_width)
        self.stdscr.addstr(y, col_pos, cpu_display)
        col_pos += cpu_usage_width + 1

        # Thread Watermark Progress Bar
        usage_ratio = (
            (thread_info.runtime.stack_watermark / thread_info.stack_size)
            if thread_info.stack_size > 0
            else 0
        )
        self._draw_progress_bar(
            y, col_pos, stack_usage_width, usage_ratio * 100, 70, 90
        )
        col_pos += stack_usage_width + 1

        # Thread Watermark Bytes
        watermark_bytes_display = (
            f"{thread_info.runtime.stack_watermark} / {thread_info.stack_size}".ljust(
                stack_bytes_width
            )
        )
        self.stdscr.addstr(y, col_pos, watermark_bytes_display)

    def _draw_heap_info(self, y, heap_info: HeapInfo, selected: bool = False):
        col_pos = 0

        # Widths
        scheme = self.ui[self.state]
        heap_name_width = scheme.col_widths[0]
        free_bytes_width = scheme.col_widths[1]
        allocated_bytes_width = scheme.col_widths[2]
        watermark_width = scheme.col_widths[3]

        # Heap name
        heap_name_display = heap_info.name[:heap_name_width].ljust(heap_name_width)
        if len(heap_info.name) > heap_name_width:
            heap_name_display = heap_name_display[:-3] + "..."

        heap_name_attr = self.ATTR_CURSOR if selected else self.ATTR_ACTIVE_THREAD
        self.stdscr.attron(heap_name_attr)
        self.stdscr.addstr(y, col_pos, heap_name_display)
        self.stdscr.attroff(heap_name_attr)
        col_pos += heap_name_width + 1

        # Free bytes
        free_bytes_display = f"{heap_info.free_bytes}".ljust(free_bytes_width)
        self.stdscr.addstr(y, col_pos, free_bytes_display)
        col_pos += free_bytes_width + 1

        heap_size = heap_info.allocated_bytes + heap_info.free_bytes
        # Heap Usage Progress Bar
        usage_ratio = (
            (heap_info.allocated_bytes / heap_size)
            if heap_info.allocated_bytes > 0
            else 0
        )
        self._draw_progress_bar(
            y, col_pos, allocated_bytes_width, usage_ratio * 100, 70, 90
        )
        col_pos += allocated_bytes_width + 1

        # Heap Watermark Bytes
        watermark_bytes_display = (
            f"{heap_info.max_allocated_bytes} / {heap_size}".ljust(watermark_width)
        )
        self.stdscr.addstr(y, col_pos, watermark_bytes_display)

    def _draw_default_view(self, height, width):
        """
        Draws all UI elements, including header, footer, status bar, and the thread data table.
        The UI is redrawn completely on each update cycle.
        """
        max_table_rows = height - 6
        total_threads = len(self.threads_data)
        start_num = self.top_line + 1 if total_threads > 0 else 0
        end_num = min(self.top_line + max_table_rows, total_threads)

        scroll_indicator = f" Threads: {start_num}-{end_num} of {total_threads} "

        self.stdscr.attron(self.ATTR_HEADER_FOOTER)
        self.stdscr.addstr(height - 2, 0, scroll_indicator[:width])
        self.stdscr.attroff(self.ATTR_HEADER_FOOTER)

        table_start = 3

        key_func = self.sort_keys.get(self._sort_by, self.sort_keys["name"])

        sorted_threads = sorted(
            self.threads_data, key=key_func, reverse=self._invert_sorting
        )

        for idx, thread in enumerate(
            sorted_threads[self.top_line : self.top_line + max_table_rows]
        ):
            target_y = table_start + idx

            if target_y >= height - 3:
                break

            absolute_idx = self.top_line + idx
            self._draw_thread_info(
                target_y, thread, selected=(absolute_idx == self.cursor)
            )

        self.stdscr.refresh()

    def _draw_thread_detail_view(self):
        """
        Draws all UI elements, including header, footer, status bar, and the thread data table.
        The UI is redrawn completely on each update cycle.
        """
        current_row_y = 3
        data_amount = sum(self.ui[self.state].col_widths)
        for thread in self.threads_data:
            if thread.name != self.detailing_thread:
                continue

            self._draw_thread_info(current_row_y, thread)

            if not self.detailing_thread_usages.get(thread.name):
                self.detailing_thread_usages[thread.name] = []
            self.detailing_thread_usages[thread.name].append(
                round(thread.runtime.cpu, 1)
            )

            if len(self.detailing_thread_usages[thread.name]) > data_amount:
                self.detailing_thread_usages[thread.name] = (
                    self.detailing_thread_usages[thread.name][1:]
                )

            current_row_y += 2
            self.stdscr.attron(self.ATTR_GRAPH)
            self._draw_graph(
                current_row_y,
                0,
                12,
                data_amount,
                self.detailing_thread_usages[thread.name],
                title="CPU%",
            )
            self.stdscr.attroff(self.ATTR_GRAPH)

        self.stdscr.refresh()

    def _draw_heaps_view(self, height):
        table_start_row = 2
        current_row_y = table_start_row + 1
        table_height = height - 1
        for idx, heap in enumerate(self.heaps_data):
            if current_row_y >= table_start_row + table_height:
                break

            self._draw_heap_info(current_row_y, heap, selected=idx == self.cursor)

            current_row_y += 1

        self.stdscr.refresh()

    def run(self, inspection_period):
        """
        The main application loop.

        This loop continuously checks for new data from the polling thread,
        updates the UI, and processes user input (e.g., 'q' to quit).
        """
        self.status_message = "Initializing..."
        self.scraper.start_polling_thread(
            self.data_queue, self.stop_event, inspection_period
        )
        self.scraper.thread_pool = list(self.scraper.all_threads.values())
        current_sort = 0
        while self.running:
            try:
                data = self.data_queue.get_nowait()
                if "error" in data:
                    self.status_message = f"Error: {data['error']}"
                else:
                    self.status_message = "Running..."
                    threads_data = data.get("threads")
                    heaps_data = data.get("heaps")

                    if threads_data is not None:
                        self.threads_data = threads_data
                    if heaps_data is not None:
                        self.heaps_data = heaps_data
            except queue.Empty:
                pass

            match self.stdscr.getch():
                case curses.KEY_DOWN:
                    self.cursor = min(len(self.threads_data) - 1, self.cursor + 1)
                case curses.KEY_UP:
                    self.cursor = max(0, self.cursor - 1)
                case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                    match self.state:
                        case ZViewState.DEFAULT_VIEW:
                            key_func = self.sort_keys.get(
                                self._sort_by, self.sort_keys["name"]
                            )

                            sorted_threads = sorted(
                                self.threads_data,
                                key=key_func,
                                reverse=self._invert_sorting,
                            )

                            self.detailing_thread = sorted_threads[self.cursor].name
                            self.scraper.thread_pool = [
                                self.scraper.all_threads[self.detailing_thread]
                            ]
                            self.state = ZViewState.THREAD_DETAIL
                            self.cursor = 0
                        case ZViewState.THREAD_DETAIL:
                            self.scraper.thread_pool = list(
                                self.scraper.all_threads.values()
                            )
                            self.state = ZViewState.DEFAULT_VIEW
                        case _:
                            self.scraper.thread_pool = []
                            pass
                case SpecialCode.SORT:
                    current_sort = (current_sort + 1) % len(self.sorting_options)
                    self.sort_by = self.sorting_options[current_sort]
                case SpecialCode.INVERSE:
                    self.invert_sorting = not self.invert_sorting
                case SpecialCode.HEAPS:
                    if not self.scraper.has_heaps:
                        pass
                    elif self.state == ZViewState.HEAPS_DETAIL:
                        self.state = ZViewState.DEFAULT_VIEW
                    else:
                        self.state = ZViewState.HEAPS_DETAIL
                case SpecialCode.QUIT:
                    self.running = False
                case _:
                    pass

            h, w = self.stdscr.getmaxyx()

            if h < self.min_dimensions[0] or w < self.min_dimensions[1]:
                self.stdscr.erase()

                msgs = [
                    "Terminal is too small.",
                    f"Please resize your terminal to at least {self.min_dimensions[1]}x{self.min_dimensions[0]}",
                    f"Current: {w}x{h}",
                ]

                mid_y = h // 2
                start_y = mid_y - 1

                for i, msg in enumerate(msgs):
                    if 0 <= start_y + i < h:
                        centered_line = f"{msg:^{w}}"[: w - 1]
                        self.stdscr.addstr(start_y + i, 0, centered_line)
            else:
                self._base_draw(h, w)

                match self.state:
                    case ZViewState.DEFAULT_VIEW:
                        max_visible_rows = h - 6

                        if self.cursor < self.top_line:
                            self.top_line = self.cursor
                        elif self.cursor >= self.top_line + max_visible_rows:
                            self.top_line = self.cursor - max_visible_rows + 1

                        self._draw_default_view(h, w)
                    case ZViewState.THREAD_DETAIL:
                        self._draw_thread_detail_view()
                    case ZViewState.HEAPS_DETAIL:
                        self._draw_heaps_view(h)

                time.sleep(inspection_period)


def curses_main(stdscr, scraper: ZScraper, inspection_period):
    """
    The entry point for the curses application.

    This function is intended to be wrapped by `curses.wrapper` to handle
    curses library initialization and cleanup.

    Args:
        :param inspection_period: Period for inspection, in seconds.
        :param stdscr: The standard screen window object provided by `curses.wrapper`.
    """
    app = ZView(scraper, stdscr)
    try:
        app.run(inspection_period)
    finally:
        app.scraper.finish_polling_thread()


def main():
    arg_parser = argparse.ArgumentParser(
        description="ZView - A real-time thread viewer for Zephyr RTOS."
    )
    arg_parser.add_argument(
        "-e",
        "--elf-file",
        required=True,
        help="Path to the application's .elf firmware file.",
    )
    arg_parser.add_argument(
        "--runners_yaml",
        default=None,
        help="Path to the generated runners YAML",
    )
    arg_parser.add_argument(
        "--runner",
        choices=["jlink", "pyocd"],
        help="Runner to start analysis with.",
    )
    arg_parser.add_argument(
        "--period",
        default=0.025,
        required=False,
        type=float,
        help="Minimum period to update system information.",
    )
    arg_parser.add_argument(
        "-n",
        "--has_thread_names",
        action="store_true",
        help="Target was built with thread names enabled.",
    )
    arg_parser.add_argument(
        "-r",
        "--has_thread_runtime_stats",
        action="store_true",
        help="Target was built with thread runtime statistics enabled.",
    )
    arg_parser.add_argument(
        "-m",
        "--has_heap_stats",
        action="store_true",
        help="Target was built with heap statistics enabled.",
    )
    arg_parser.add_argument(
        "--max_threads",
        type=int,
        default=32,
        help="Maximum number of threads for the target.",
    )
    arg_parser.add_argument(
        "--thread_name_size",
        type=int,
        default=32,
        help="Maximum thread name size, if available.",
    )
    args = arg_parser.parse_args()

    if not args.has_thread_names:
        print("NO thread names will be shown (CONFIG_THREAD_NAME=n)")
    if not args.has_thread_runtime_stats:
        print("NO cpu stats will be shown (CONFIG_THREAD_RUNTIME_STATS=n)")
    if not args.has_heap_stats:
        print("NO heap stats will be shown (CONFIG_SYS_HEAP_RUNTIME_STATS=n)")

    runner = args.runner
    target_mcu = None
    if args.runners_yaml:
        runner_parser = RunnerParser(args.runners_yaml)
        runner, target_mcu = runner_parser.get_config(preferred_runner=runner)
        if runner != "jlink":
            runner, target_mcu = runner_parser.get_config(preferred_runner="pyocd")

    with (
        PyOCDScraper(target_mcu) if runner != "jlink" else JLinkScraper(target_mcu)
    ) as meta_scraper:
        z_scraper = ZScraper(
            meta_scraper, args.elf_file, args.max_threads, args.thread_name_size
        )
        curses.wrapper(curses_main, z_scraper, args.period)
