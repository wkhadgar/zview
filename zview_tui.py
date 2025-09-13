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

from backend.z_scraper import ZScraper, ThreadInfo, PyOCDScraper, JLinkScraper


@dataclass
class ZViewUI:
    col_headers: list[str]
    col_widths: list[int]


class ZViewState(enum.Enum):
    DEFAULT_VIEW = 1
    THREAD_DETAIL = 2


class SpecialCode:
    QUIT = ord("q")
    NEWLINE = ord("\n")
    RETURN = ord("\r")
    SORT = ord("s")
    INVERSE = ord("i")


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
        self.status_message: str = ""
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()

        self.state: ZViewState = ZViewState.DEFAULT_VIEW
        self.ui: dict[ZViewState, ZViewUI] = {}

        self.cursor = 0

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
            curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Active thread name
            curses.init_pair(2, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Inactive thread name
            curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Progress bar: low usage
            curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Progress bar: medium usage
            curses.init_pair(5, curses.COLOR_RED, curses.COLOR_BLACK)  # Progress bar: high usage
            curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Header/Footer background
            curses.init_pair(7, curses.COLOR_RED, curses.COLOR_BLACK)  # Error message text
            curses.init_pair(8, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Cursor selection
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

    def _set_ui_schemes(self):
        thread_basic_info_scheme = {"Thread": 30, "CPU %": 6, "Stack Usage (Watermark)": 30, "Watermark (Bytes)": 16}
        self.ui[ZViewState.DEFAULT_VIEW] = ZViewUI(list(thread_basic_info_scheme.keys()),
                                                   list(thread_basic_info_scheme.values()))
        self.ui[ZViewState.THREAD_DETAIL] = ZViewUI(list(thread_basic_info_scheme.keys()),
                                                    list(thread_basic_info_scheme.values()))

    def _draw_graph(self, y, x, h, w, points, title="", maximum=100, length=100):
        horizontal_limit = ("─" * (w - 2))
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
                    last_block = int((((points[x_idx] / maximum) * (h - 2)) - full_blocks) * (len(blocks) - 1))
                    self.stdscr.addstr(y_pos, x_pos,
                                       " " if y_step < ((h - 1) - full_blocks) else blocks[-1])
                    self.stdscr.addstr(y_pos, x_pos,
                                       blocks[last_block] if y_step == ((h - 2) - full_blocks) else "")
            x_idx -= 1 if x_idx else 0

        self.stdscr.addstr(y + 1, x + w - (len(f"{maximum}")), str(maximum))
        self.stdscr.addstr(y + h - 1, x + w - 1, "0")

    def _draw_progress_bar(self, y, x, width: int, percentage: float, medium_threshold: float, high_threshold: float):
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

        percent_display = f"{percentage:.1f}%" if percentage > 0 else "N/A"
        overlap = percent_display.center(len(bar_str))[:completed_chars].strip()
        self.stdscr.addstr(y, x + (width // 2) - (len(percent_display) // 2), percent_display)
        self.stdscr.attron(curses.A_REVERSE)
        self.stdscr.addstr(y, x + (width // 2) - (len(percent_display) // 2), overlap)
        self.stdscr.attroff(curses.A_REVERSE)
        self.stdscr.attroff(bar_color_attr)

    def _base_draw(self, scr_size: tuple[int, int]):
        self.stdscr.clear()

        header_text = "ZView - Zephyr RTOS Runtime Viewer"
        self.stdscr.attron(self.ATTR_HEADER_FOOTER)
        self.stdscr.addstr(0, 0, header_text.center(scr_size[1]))
        self.stdscr.attroff(self.ATTR_HEADER_FOOTER)

        footer_text = "Press 'q' to quit"
        self.stdscr.attron(self.ATTR_HEADER_FOOTER)
        self.stdscr.addstr(scr_size[0] - 2, 0, footer_text.rjust(scr_size[1]))
        self.stdscr.attroff(self.ATTR_HEADER_FOOTER)

        status_row = scr_size[0] - 3
        if self.status_message.startswith("Error"):
            self.stdscr.attron(self.ATTR_ERROR)
        self.stdscr.addstr(status_row, 0, self.status_message.ljust(scr_size[1])[:scr_size[1]])
        if self.status_message.startswith("Error"):
            self.stdscr.attroff(self.ATTR_ERROR)

        if (scr_size[0] - 1) <= 0:
            self.stdscr.addstr(2, 0, "Window too small to display table. Resize terminal.")
            self.stdscr.refresh()
            return

        current_col_x = 0
        for header, width in zip(self.ui[self.state].col_headers, self.ui[self.state].col_widths):
            display_header = header.center(width)
            self.stdscr.addstr(2, current_col_x, display_header[:scr_size[1] - current_col_x])
            current_col_x += width + 1

    def _draw_thread_info(self, y, thread_info: ThreadInfo, selected: bool = False):
        col_pos = 0

        # Widths
        scheme = self.ui[self.state]
        thread_name_width = scheme.col_widths[0]
        cpu_usage_width = scheme.col_widths[1]
        stack_usage_width = scheme.col_widths[2]
        stack_bytes_width = scheme.col_widths[3]

        # Thread name
        thread_name_display = thread_info.name[:thread_name_width].ljust(thread_name_width)
        if len(thread_info.name) > thread_name_width:
            thread_name_display = thread_name_display[:-3] + "..."
        elif thread_info.name == "idle":
            thread_name_display = thread_name_display[:4] + "*" + thread_name_display[4:-1]

        thread_name_attr = self.ATTR_CURSOR if selected else (
            self.ATTR_ACTIVE_THREAD if thread_info.runtime.active else self.ATTR_INACTIVE_THREAD)
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
                thread_info.runtime.stack_watermark / thread_info.stack_size) if thread_info.stack_size > 0 else 0
        self._draw_progress_bar(y, col_pos, stack_usage_width, usage_ratio * 100, 70, 90)
        col_pos += stack_usage_width + 1

        # Thread Watermark Bytes
        watermark_bytes_display = f"{thread_info.runtime.stack_watermark} / {thread_info.stack_size}".ljust(
            stack_bytes_width)
        self.stdscr.addstr(y, col_pos, watermark_bytes_display)

    def _draw_default_view(self):
        """
        Draws all UI elements, including header, footer, status bar, and the thread data table.
        The UI is redrawn completely on each update cycle.
        """
        height, width = self.stdscr.getmaxyx()

        self._base_draw((height, width))

        table_start_row = 2
        current_row_y = table_start_row + 1
        table_height = height - 1
        for idx, thread in enumerate(self.threads_data):
            if current_row_y >= table_start_row + table_height:
                break

            self._draw_thread_info(current_row_y, thread, selected=idx == self.cursor)

            current_row_y += 1

        self.stdscr.refresh()

    def _draw_thread_detail(self):
        """
        Draws all UI elements, including header, footer, status bar, and the thread data table.
        The UI is redrawn completely on each update cycle.
        """
        height, width = self.stdscr.getmaxyx()

        self._base_draw((height, width))

        current_row_y = 3
        data_amount = sum(self.ui[self.state].col_widths)
        for thread in self.threads_data:
            if thread.name != self.detailing_thread:
                continue

            self._draw_thread_info(current_row_y, thread)

            if not self.detailing_thread_usages.get(thread.name):
                self.detailing_thread_usages[thread.name] = []
            self.detailing_thread_usages[thread.name].append(round(thread.runtime.cpu, 1))

            if len(self.detailing_thread_usages[thread.name]) > data_amount:
                self.detailing_thread_usages[thread.name] = self.detailing_thread_usages[thread.name][1:]

            current_row_y += 2
            self.stdscr.attron(self.ATTR_GRAPH)
            self._draw_graph(current_row_y, 0, 12, data_amount, self.detailing_thread_usages[thread.name], title="CPU%")
            self.stdscr.attroff(self.ATTR_GRAPH)

        self.stdscr.refresh()

    def run(self, inspection_period):
        """
        The main application loop.

        This loop continuously checks for new data from the polling thread,
        updates the UI, and processes user input (e.g., 'q' to quit).
        """
        self.status_message = f"Initializing..."
        self.scraper.start_polling_thread(self.data_queue, self.stop_event, inspection_period)
        self.scraper.thread_pool = self.scraper.all_threads.values()
        sortings = ["name", "cpu", "watermark_p", "watermark_b"]
        current_sort = 0
        while self.running:
            try:
                data = self.data_queue.get_nowait()
                if "threads" in data:
                    self.threads_data = data["threads"]
                    self.status_message = f"Running..."
                elif "error" in data:
                    self.status_message = f"Error: {data['error']}"
            except queue.Empty:
                pass

            match self.stdscr.getch():
                case curses.KEY_DOWN:
                    self.cursor += self.cursor < len(self.threads_data) - 1
                case curses.KEY_UP:
                    self.cursor -= self.cursor > 0
                case curses.KEY_ENTER | SpecialCode.NEWLINE | SpecialCode.RETURN:
                    self.state = ZViewState.THREAD_DETAIL if self.state is ZViewState.DEFAULT_VIEW else ZViewState.DEFAULT_VIEW
                    self.detailing_thread = self.threads_data[self.cursor].name
                    self.cursor = 0
                case SpecialCode.SORT:
                    current_sort = (current_sort + 1) % len(sortings)
                    self.scraper.sort_by = sortings[current_sort]
                case SpecialCode.INVERSE:
                    self.scraper.invert_sorting = not self.scraper.invert_sorting
                case SpecialCode.QUIT:
                    self.running = False
                case _:
                    pass

            current_dims = self.stdscr.getmaxyx()
            if current_dims[0] < self.min_dimensions[0] or current_dims[1] < self.min_dimensions[1]:
                self.stdscr.clear()
                msg0 = f"Terminal is too small."
                msg1 = f"Please resize your terminal to at least {self.min_dimensions[1]}x{self.min_dimensions[0]}."
                msg2 = f"Current dimensions {current_dims[1]}x{current_dims[0]}."
                self.stdscr.addstr((current_dims[0] // 2) - 1, (current_dims[1] - len(msg0)) // 2, msg0)
                self.stdscr.addstr(current_dims[0] // 2, (current_dims[1] - len(msg1)) // 2, msg1)
                self.stdscr.addstr((current_dims[0] // 2) + 1, (current_dims[1] - len(msg2)) // 2, msg2)
            else:
                match self.state:
                    case ZViewState.DEFAULT_VIEW:
                        self.scraper.thread_pool = list(self.scraper.all_threads.values())
                        self._draw_default_view()
                    case ZViewState.THREAD_DETAIL:
                        self.scraper.thread_pool = [self.scraper.all_threads[self.detailing_thread]]
                        self._draw_thread_detail()
                        pass

                time.sleep(inspection_period)


def main(stdscr, inspection_period):
    """
    The entry point for the curses application.

    This function is intended to be wrapped by `curses.wrapper` to handle
    curses library initialization and cleanup.

    Args:
        :param inspection_period: Period for inspection, in seconds.
        :param stdscr: The standard screen window object provided by `curses.wrapper`.
    """
    app = ZView(z_scraper, stdscr)
    try:
        app.run(inspection_period)
    finally:
        app.scraper.finish_polling_thread()


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="ZView - A real-time thread viewer for Zephyr RTOS.")
    arg_parser.add_argument("--mcu", required=False, default=None, help="Override target MCU (e.g., 'STM32F407VG').")
    arg_parser.add_argument("-e", "--elf-file", required=True, help="Path to the application's .elf firmware file.")
    arg_parser.add_argument("--runner", default="jlink", choices=["jlink", "pyocd"],
                            help="Runner to start analysis with.")
    arg_parser.add_argument("-p", "--period", default=0.025, required=False, type=float,
                            help="Minimum period to update system information.")
    args = arg_parser.parse_args()

    with (PyOCDScraper(args.mcu) if args.runner == "pyocd" else JLinkScraper(args.mcu)) as meta_scraper:
        z_scraper = ZScraper(meta_scraper, args.elf_file)
        curses.wrapper(main, args.period)
