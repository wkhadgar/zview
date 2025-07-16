import argparse
import curses
import threading
import queue
import time
from argparse import Namespace
from dataclasses import dataclass
from typing import List, Optional

from mcu_scraper import JLinkScraper
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


class ZView:
    """
    A curses-based application for viewing Zephyr RTOS thread runtime information.

    This class manages the curses UI, starts a background thread for data polling,
    and updates the display with real-time thread statistics from a connected MCU.
    """

    def __init__(self, stdscr, elf_path: str, target_mcu: Optional[str] = None):
        """
        Initializes the ZView application.

        Args:
            stdscr: The main curses window object provided by curses.wrapper.
            elf_path: Path to the Zephyr application's ELF file for symbol parsing.
            target_mcu: Optional J-Link target MCU name (e.g., "STM32F407VG").
        """
        self.stdscr = stdscr
        self.elf_path = elf_path
        self.target_mcu = target_mcu
        self.running = True
        self.threads_data: List[ThreadInfo] = []
        self.status_message: str = ""
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.polling_thread: Optional[threading.Thread] = None

        self._init_curses()
        self._start_polling_thread()

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

            self.ATTR_ACTIVE_THREAD = curses.color_pair(1)
            self.ATTR_INACTIVE_THREAD = curses.color_pair(2)
            self.ATTR_PROGRESS_BAR_LOW = curses.color_pair(3)
            self.ATTR_PROGRESS_BAR_MEDIUM = curses.color_pair(4)
            self.ATTR_PROGRESS_BAR_HIGH = curses.color_pair(5)
            self.ATTR_HEADER_FOOTER = curses.color_pair(6)
            self.ATTR_ERROR = curses.color_pair(7)

    def _start_polling_thread(self):
        """
        Starts a separate daemon thread to continuously poll data from the MCU.
        """
        self.polling_thread = threading.Thread(
            target=self._poll_threads_worker,
            args=(self.data_queue, self.stop_event, self.elf_path, self.target_mcu)
        )
        self.polling_thread.daemon = True
        self.polling_thread.start()

    @staticmethod
    def _calculate_dynamic_watermark(scraper: JLinkScraper, stack_start: int, stack_size: int) -> int:
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

    @staticmethod
    def _poll_threads_worker(data_queue: queue.Queue, stop_event: threading.Event, elf_path: str,
                             target_mcu: Optional[str]):
        """
        Worker function executed by the polling thread.

        Connects to the MCU via J-Link, parses Zephyr kernel symbols from the ELF file,
        and continuously reads and processes thread runtime data, pushing it to `data_queue`.

        Args:
            data_queue: A queue for sending processed thread data to the main UI thread.
            stop_event: An event to signal the polling thread to terminate.
            elf_path: Path to the Zephyr application's ELF file.
            target_mcu: Optional J-Link target MCU name.
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

            with (JLinkScraper(target_mcu=target_mcu) as scraper):
                threads_static_info = []
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

                        thread_name = (name_bytes_raw.partition(b'\0')[0]).decode(
                            errors='ignore') or f"thread_0x{ptr:X}"

                        threads_static_info.append({
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

                last_thread_cycles = {}
                last_cpu_cycles = 0

                while not stop_event.is_set():
                    current_threads_data = []

                    try:
                        for thread_info in threads_static_info:
                            current_cpu_cycles = scraper.read32(kernel_usage_addr)[0]
                            cpu_cycles_delta = current_cpu_cycles - last_cpu_cycles if last_cpu_cycles else 0
                            last_cpu_cycles = current_cpu_cycles
                            bkp_cpu_cycles_delta = cpu_cycles_delta if cpu_cycles_delta > 0 else 1

                            try:
                                thread_usage = scraper.read64(thread_info["address"] + thread_info["usage_offset"])[0]
                                thread_usage_delta = thread_usage - last_thread_cycles.get(thread_info["name"], 0)
                                last_thread_cycles[thread_info["name"]] = thread_usage

                                cpu_percent = 0.0
                                if cpu_cycles_delta > 0:
                                    cpu_percent = (thread_usage_delta / cpu_cycles_delta) * 100
                                elif thread_usage_delta > 0:
                                    cpu_percent = (thread_usage_delta / bkp_cpu_cycles_delta) * 100

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
                        current_threads_data = []

                    data_queue.put({"threads": sorted(current_threads_data, key=lambda t: t.name)})
                    time.sleep(0.2)

        except Exception as e:
            data_queue.put({"error": f"Polling thread initialization error: {e}"})

    def _draw_ui(self):
        """
        Draws all UI elements, including header, footer, status bar, and the thread data table.
        The UI is redrawn completely on each update cycle.
        """
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        header_text = "ZView - Zephyr RTOS Runtime Viewer"
        self.stdscr.attron(curses.color_pair(6))
        self.stdscr.addstr(0, 0, header_text.ljust(width))
        self.stdscr.attroff(curses.color_pair(6))

        footer_text = "Press 'q' to quit"
        self.stdscr.attron(curses.color_pair(6))
        self.stdscr.addstr(height - 2, 0, footer_text.ljust(width))
        self.stdscr.attroff(curses.color_pair(6))

        status_row = height - 3
        if self.status_message.startswith("Error"):
            self.stdscr.attron(curses.color_pair(7))
        self.stdscr.addstr(status_row, 0, self.status_message.ljust(width)[:width])
        if self.status_message.startswith("Error"):
            self.stdscr.attroff(curses.color_pair(7))

        table_start_row = 2
        table_height = height - table_start_row - 3
        if table_height <= 0:
            self.stdscr.addstr(table_start_row, 0, "Window too small to display table. Resize terminal.")
            self.stdscr.refresh()
            return

        headers = ["Thread", "CPU %", "Watermark %", "Stack Usage", "Watermark (Bytes)"]
        col_widths = [25, 10, 15, 30, 20]

        current_col_x = 0
        for i, header in enumerate(headers):
            display_header = header.ljust(col_widths[i])
            self.stdscr.addstr(table_start_row, current_col_x, display_header[:width - current_col_x])
            current_col_x += col_widths[i] + 1

        current_row_y = table_start_row + 1
        for thread in self.threads_data:
            if current_row_y >= table_start_row + table_height:
                break

            col_pos = 0

            # Thread names
            thread_name_display = thread.name[:col_widths[0]].ljust(col_widths[0])
            thread_name_attr = self.ATTR_ACTIVE_THREAD if thread.active else self.ATTR_INACTIVE_THREAD
            self.stdscr.attron(thread_name_attr)
            self.stdscr.addstr(current_row_y, col_pos, thread_name_display[:width - col_pos])
            self.stdscr.attroff(thread_name_attr)
            col_pos += col_widths[0] + 1

            # Thread CPUs
            cpu_display = f"{round(thread.cpu, 1)}%".ljust(col_widths[1])
            self.stdscr.addstr(current_row_y, col_pos, cpu_display[:width - col_pos])
            col_pos += col_widths[1] + 1

            # Thread Watermark %
            watermark_percent_display = f"{(100 * thread.stack_watermark) / thread.stack_size: .1f}%".ljust(
                col_widths[2]) if thread.stack_size > 0 else "N/A".ljust(col_widths[2])
            self.stdscr.addstr(current_row_y, col_pos, watermark_percent_display[:width - col_pos])
            col_pos += col_widths[2] + 1

            # Thread Watermark Progress Bar
            usage_percent = (thread.stack_watermark / thread.stack_size * 100) if thread.stack_size > 0 else 0
            if usage_percent > 90:
                bar_color_attr = self.ATTR_PROGRESS_BAR_HIGH
            elif usage_percent > 75:
                bar_color_attr = self.ATTR_PROGRESS_BAR_MEDIUM
            else:
                bar_color_attr = self.ATTR_PROGRESS_BAR_LOW
            bar_width = col_widths[3] - 2
            completed_chars = int(bar_width * (usage_percent / 100))
            remaining_chars = bar_width - completed_chars
            bar_str = "[" + "█" * completed_chars + "-" * remaining_chars + "]"
            self.stdscr.attron(bar_color_attr)
            self.stdscr.addstr(current_row_y, col_pos, bar_str[:width - col_pos])
            self.stdscr.attroff(bar_color_attr)
            col_pos += col_widths[3] + 1

            # Thread Watermark Bytes
            watermark_bytes_display = f"{thread.stack_watermark} / {thread.stack_size}".ljust(col_widths[4])
            self.stdscr.addstr(current_row_y, col_pos, watermark_bytes_display[:width - col_pos])

            current_row_y += 1

        self.stdscr.refresh()

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

            self._draw_ui()

            try:
                key = self.stdscr.getch()
                if key == ord('q'):
                    self.running = False
            except curses.error:
                pass

            time.sleep(0.05)


def main(stdscr, args: Namespace):
    """
    The entry point for the curses application.

    This function is intended to be wrapped by `curses.wrapper` to handle
    curses library initialization and cleanup.

    Args:
        stdscr: The standard screen window object provided by `curses.wrapper`.
        args: Command-line arguments parsed by `argparse`.
    """
    app = ZView(stdscr, elf_path=args.elf_file, target_mcu=args.mcu)
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
    arg_parser.add_argument("--elf-file", required=True, help="Path to the application's .elf firmware file.")
    args = arg_parser.parse_args()

    curses.wrapper(main, args)
