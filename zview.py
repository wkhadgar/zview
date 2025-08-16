# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0


import argparse
import curses
import enum
import threading
import queue
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

from mcu_scraper import *
from elf_parser import ZephyrSymbolParser


@dataclass
class ThreadInfo:
    """
    Data class to hold information about a single Zephyr RTOS thread.
    """
    name: str
    cpu: float
    active: bool
    stack_size: int
    stack_watermark: int


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


class ZView:
    """
    A curses-based application for viewing Zephyr RTOS thread runtime information.

    This class manages the curses UI, starts a background thread for data polling,
    and updates the display with real-time thread statistics from a connected MCU.
    """

    def __init__(self, scraper: AbstractScraper, stdscr, elf_path: str, target_mcu: Optional[str] = None,
                 inspection_period: float = 0.2,
                 ):
        """
        Initializes the ZView application.

        Args:
            stdscr: The main curses window object provided by curses.wrapper.
            elf_path: Path to the Zephyr application's ELF file for symbol parsing.
            target_mcu: Optional J-Link target MCU name (e.g., "STM32F407VG").
            inspection_period: Period for system information gathering and update.
        """
        self.inspection_period = inspection_period
        self.stdscr = stdscr
        self.elf_path = elf_path
        self.target_mcu = target_mcu
        self.running = True
        self.threads_data: List[ThreadInfo] = []
        self.status_message: str = ""
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.polling_thread: Optional[threading.Thread] = None

        self.state: ZViewState = ZViewState.DEFAULT_VIEW
        self.ui: dict[ZViewState, ZViewUI] = {}

        self.cursor = 0

        self.detailing_thread: None | str = None
        self.detailing_thread_usages = {}

        self.min_dimensions = (21, 86)
        self._init_curses()
        self._set_ui_schemes()
        self._start_polling_thread(scraper=scraper)

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

        thread_name_attr = self.ATTR_CURSOR if selected else (
            self.ATTR_ACTIVE_THREAD if thread_info.active else self.ATTR_INACTIVE_THREAD)
        self.stdscr.attron(thread_name_attr)
        self.stdscr.addstr(y, col_pos, thread_name_display)
        self.stdscr.attroff(thread_name_attr)
        col_pos += thread_name_width + 1

        # Thread CPUs
        cpu_display = f"{round(thread_info.cpu, 1)}%".ljust(cpu_usage_width)
        self.stdscr.addstr(y, col_pos, cpu_display)
        col_pos += cpu_usage_width + 1

        # Thread Watermark Progress Bar
        usage_ratio = (thread_info.stack_watermark / thread_info.stack_size) if thread_info.stack_size > 0 else 0
        self._draw_progress_bar(y, col_pos, stack_usage_width, usage_ratio * 100, 70, 90)
        col_pos += stack_usage_width + 1

        # Thread Watermark Bytes
        watermark_bytes_display = f"{thread_info.stack_watermark} / {thread_info.stack_size}".ljust(stack_bytes_width)
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
            self.detailing_thread_usages[thread.name].append(round(thread.cpu, 1))

            if len(self.detailing_thread_usages[thread.name]) > data_amount:
                self.detailing_thread_usages[thread.name] = self.detailing_thread_usages[thread.name][1:]

            current_row_y += 2
            self.stdscr.attron(self.ATTR_GRAPH)
            self._draw_graph(current_row_y, 0, 12, data_amount, self.detailing_thread_usages[thread.name], title="CPU%")
            self.stdscr.attroff(self.ATTR_GRAPH)

        self.stdscr.refresh()

    def _start_polling_thread(self, scraper: AbstractScraper):
        """
        Starts a separate daemon thread to continuously poll data from the MCU.
        """
        self.polling_thread = threading.Thread(
            target=self._poll_threads_worker,
            args=(scraper, self.data_queue, self.stop_event, self.elf_path, self.inspection_period)
        )
        self.polling_thread.daemon = True
        self.polling_thread.start()

    @staticmethod
    def _calculate_dynamic_watermark(scraper: AbstractScraper, stack_start: int, stack_size: int) -> int:
        """
        Reads a thread's stack memory and scans for the 0xAA fill pattern
        to determine the current stack watermark (lowest point of stack usage).

        Args:
            scraper: The JLinkScraper instance for reading MCU memory.
            stack_start: The starting address of the thread's stack.
            stack_size: The total size of the thread's stack in bytes.

        Returns:
            The calculated stack watermark in bytes, indicating the minimum
            amount of stack space that has not been used.
        """
        if stack_size == 0:
            return 0

        watermark = stack_size
        stack_words = scraper.read32(stack_start, stack_size // 4)

        for word_index, word in enumerate(stack_words):
            if word == 0xAA_AA_AA_AA:
                watermark -= 4
            else:
                word_bytes = word.to_bytes(4, 'little')
                for byte_index, byte_value in enumerate(word_bytes):
                    if byte_value != 0xAA:
                        watermark -= (4 - byte_index)
                        break
                break

        return watermark

    def _poll_threads_worker(self, scraper: AbstractScraper, data_queue: queue.Queue, stop_event: threading.Event,
                             elf_path: str, inspection_period=0.2):
        """
        Worker function executed by the polling thread.

        Connects to the MCU via J-Link, parses Zephyr kernel symbols from the ELF file,
        and continuously reads and processes thread runtime data, pushing it to `data_queue`.

        Args:
            data_queue: A queue for sending processed thread data to the main UI thread.
            stop_event: An event to signal the polling thread to terminate.
            elf_path: Path to the Zephyr application's ELF file.
        """
        try:
            parser = ZephyrSymbolParser(elf_path)
            kernel_base = parser.get_symbol_info("_kernel", info="address")
            threads_offset = parser.get_struct_member_offset("z_kernel", "threads")
            kernel_cpu_offset = parser.get_struct_member_offset("z_kernel", "cpus")
            kernel_usage0_offset = parser.get_struct_member_offset("_cpu", "usage0")

            threads_addr = kernel_base + threads_offset
            kernel_usage_addr = kernel_base + kernel_cpu_offset + kernel_usage0_offset

            stack_struct_size = parser.get_struct_size("k_thread")
            stack_info_offset = parser.get_struct_member_offset("k_thread", "stack_info")
            base_offset = parser.get_struct_member_offset("k_thread", "base")
            offsets = {
                "next": parser.get_struct_member_offset("k_thread", "next_thread"),
                "name": parser.get_struct_member_offset("k_thread", "name"),
                "usage": base_offset + parser.get_struct_member_offset("_thread_base", "usage"),
                "stack_start": stack_info_offset + parser.get_struct_member_offset("_thread_stack_info", "start"),
                "stack_size": stack_info_offset + parser.get_struct_member_offset("_thread_stack_info", "size"),
                "stack_delta": stack_info_offset + parser.get_struct_member_offset("_thread_stack_info", "delta"),
            }

            self.static_threads_info = {}
            ptr = scraper.read32(threads_addr)[0] if threads_addr else 0

            words_to_read = stack_struct_size // 4
            next_ptr_word_idx = offsets["next"] // 4
            name_word_idx = offsets["name"] // 4
            stack_start_word_idx = offsets["stack_start"] // 4
            stack_size_word_idx = offsets["stack_size"] // 4

            for _ in range(50):
                if ptr == 0:
                    break

                try:
                    stack_struct_words = scraper.read32(ptr, words_to_read)
                    name_bytes_raw = b""
                    for i in range(8):
                        if (name_word_idx + i) < len(stack_struct_words):
                            name_bytes_raw += stack_struct_words[name_word_idx + i].to_bytes(4, "little")
                        else:
                            break

                    thread_name = (name_bytes_raw.partition(b"\0")[0]).decode(
                        errors='ignore') or f"thread_0x{ptr:X}"

                    self.static_threads_info[thread_name] = ({
                        "address": ptr,
                        "stack_start": stack_struct_words[stack_start_word_idx],
                        "stack_size": stack_struct_words[stack_size_word_idx],
                        "name": thread_name,
                        "usage_offset": offsets["usage"]
                    })

                    ptr = stack_struct_words[next_ptr_word_idx]
                except Exception as e:
                    data_queue.put({"error": f"Error parsing thread struct at 0x{ptr:X}: {e}"})
                    break

            # Main loop for dynamic data polling
            last_thread_cycles = {}  # Stores last read usage cycles for each thread
            last_cpu_cycles = {}  # Stores last read total CPU cycles for each thread (to mitigate I/O read delays)
            last_cpu_delta = 0

            while not stop_event.is_set():
                current_threads_data = []

                try:
                    current_cpu_cycles = scraper.read32(kernel_usage_addr)[0]

                    cpu_cycles_delta = current_cpu_cycles - last_cpu_cycles if last_cpu_cycles else 0
                    last_cpu_cycles = current_cpu_cycles

                    if cpu_cycles_delta > 0:
                        last_cpu_delta = cpu_cycles_delta
                    elif cpu_cycles_delta <= 0 < last_cpu_delta:
                        cpu_cycles_delta = last_cpu_delta

                    if self.state == ZViewState.DEFAULT_VIEW:
                        thread_pool = self.static_threads_info.values()
                    elif self.state == ZViewState.THREAD_DETAIL:
                        thread_pool = [self.static_threads_info[self.detailing_thread]]

                    for thread_info in thread_pool:
                        try:
                            thread_usage = scraper.read64(thread_info["address"] + thread_info["usage_offset"])[0]

                            thread_usage_delta = thread_usage - last_thread_cycles.get(thread_info["name"], 0)
                            last_thread_cycles[thread_info["name"]] = thread_usage

                            cpu_percent = 0.0
                            if cpu_cycles_delta > 0:
                                cpu_percent = (thread_usage_delta / cpu_cycles_delta) * 100
                            elif thread_usage_delta > 0:
                                cpu_percent = (thread_usage_delta / last_cpu_delta) * 100

                            is_active = thread_usage_delta > 0

                            watermark = ZView._calculate_dynamic_watermark(
                                scraper, thread_info["stack_start"], thread_info["stack_size"]
                            )

                            current_threads_data.append(
                                ThreadInfo(
                                    thread_info["name"],
                                    cpu_percent,
                                    is_active,
                                    thread_info["stack_size"],
                                    watermark
                                )
                            )
                        except Exception as e:
                            data_queue.put({"error": f"Error polling thread {thread_info['name']}: {e}"})
                            continue

                except Exception as e:
                    data_queue.put({"error": f"Error reading global CPU cycles: {e}"})
                    continue

                data_queue.put({"threads": sorted(current_threads_data, key=lambda t: t.name)})
                time.sleep(inspection_period)

        except Exception as e:
            data_queue.put({"error": f"Polling thread initialization error: {e}"})

    def run(self):
        """
        The main application loop.

        This loop continuously checks for new data from the polling thread,
        updates the UI, and processes user input (e.g., 'q' to quit).
        """
        self.status_message = f"Initializing..."
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
                    self._draw_default_view()
                case ZViewState.THREAD_DETAIL:
                    self._draw_thread_detail()
                    pass

            time.sleep(0.25)


def main(stdscr, parser_args: Namespace):
    """
    The entry point for the curses application.

    This function is intended to be wrapped by `curses.wrapper` to handle
    curses library initialization and cleanup.

    Args:
        :param stdscr: The standard screen window object provided by `curses.wrapper`.
        :param parser_args: Command-line arguments parsed by `argparse`.
    """
    app = ZView(mcu_scraper, stdscr, elf_path=parser_args.elf_file, target_mcu=parser_args.mcu,
                inspection_period=parser_args.period)
    try:
        app.run()
    finally:
        app.stop_event.set()
        if app.polling_thread and app.polling_thread.is_alive():
            app.polling_thread.join(timeout=1.0)
            if app.polling_thread.is_alive():
                print("Warning: Polling thread did not terminate gracefully.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="ZView - A real-time thread viewer for Zephyr RTOS.")
    arg_parser.add_argument("--mcu", required=False, default=None, help="Override target MCU (e.g., 'STM32F407VG').")
    arg_parser.add_argument("-e", "--elf-file", required=True, help="Path to the application's .elf firmware file.")
    arg_parser.add_argument("-p", "--period", default=0.2, required=False, type=float,
                            help="Minimum period to update system information.")
    args = arg_parser.parse_args()

    with PyOCDScraper(args.mcu) as mcu_scraper:
        curses.wrapper(main, args)
