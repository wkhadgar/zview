import argparse
import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

import textual

from rich.progress_bar import ProgressBar
from rich.text import Text
from textual.app import App, ComposeResult
from textual.message import Message
from textual.widgets import DataTable, Header, Footer, Static
from elf_parser import ZephyrSymbolParser
from mcu_scraper import JLinkScraper


@dataclass
class ThreadInfo:
    """
    Data class to encapsulate information about a single Zephyr RTOS thread.

    Attributes:
        name: The name of the thread.
        cpu: The CPU usage percentage of the thread.
        active: Boolean indicating if the thread has had recent CPU activity.
        stack_size: The total allocated stack size for the thread in bytes.
        stack_watermark: The lowest point of stack usage (highest address) in bytes,
                         indicating the remaining unused stack space.
    """
    name: str
    cpu: float
    active: bool
    stack_size: int
    stack_watermark: int


class StatusFooter(Footer):
    """A custom footer that includes a status message area."""

    def __init__(self) -> None:
        super().__init__()
        self.status_widget = Static()

    def compose(self) -> ComposeResult:
        """Compose the footer with the status widget."""
        yield self.status_widget

    def update_status(self, text: Text) -> None:
        """Update the status message."""
        self.status_widget.update(text)


class ZViewApp(App):
    """
    A Textual-based application for real-time viewing of Zephyr RTOS thread statistics.

    This application connects to a target MCU via J-Link, parses ELF symbols,
    and displays dynamic thread information such as CPU usage and stack watermark.
    """
    TITLE = "ZView - Zephyr RTOS Runtime Viewer"
    BINDINGS = [("q", "quit", "Quit")]

    @dataclass
    class UpdateThreads(Message):
        """Message to trigger a UI update with new thread data."""
        threads: List[ThreadInfo]

    @dataclass
    class ShowError(Message):
        """Message to display an error in the status bar."""
        error: str

    def __init__(self, elf_path: str, target_mcu: Optional[str] = None):
        """
        Initializes the ZViewApp.

        Args:
            elf_path: Path to the Zephyr application's ELF file for symbol parsing.
            target_mcu: Optional J-Link target MCU name (e.g., "STM32F407VG").
        """
        super().__init__()
        self.target_mcu = target_mcu
        self.elf_path = elf_path
        # Store the last known CPU deltas for threads to handle zero total CPU cycles
        self._last_cpu_deltas = {}

    def compose(self) -> ComposeResult:
        """
        Composes the Textual UI layout.

        Yields:
            ComposeResult: The Textual widgets for the application.
        """
        yield Header()
        yield DataTable(id="threads_table")
        yield StatusFooter()

    def on_mount(self) -> None:
        """
        Called when the application is mounted.
        Starts the background worker thread for polling MCU data.
        """
        self.run_worker(self.__poll_threads, thread=True)

        # Initialize DataTable columns
        table = self.query_one(DataTable)

        columns = ["Thread", "CPU %", "Watermark %", "Stack Usage", "Watermark (Bytes)"]
        for i, col in enumerate(columns):
            table.add_column(col, key=str(i))

        status_view = self.query_one(StatusFooter)
        status_view.update_status(Text("Starting...", style="grey"))

    @textual.on(UpdateThreads)
    def on_update_threads(self, message: UpdateThreads) -> None:
        """
        Handles the UpdateThreads message to refresh the DataTable with new thread data.
        """
        table = self.query_one(DataTable)

        for thread in message.threads:
            usage_percent = (thread.stack_watermark / thread.stack_size * 100) if thread.stack_size > 0 else 0
            bar = ProgressBar(total=100, completed=usage_percent, width=30)
            bar.complete_style = "red" if usage_percent > 90 else "yellow" if usage_percent > 75 else "green"
            bar.finished_style = "red"

            new_row_data = [
                Text(thread.name, style="cyan" if thread.active else "grey"),
                f"{round(thread.cpu, 1):.1f}%",
                f"{(100 * thread.stack_watermark) / thread.stack_size:.1f}%",
                bar,
                f"{thread.stack_watermark} / {thread.stack_size}",
            ]

            if thread.name in table.rows:
                for column_key, cell_data in enumerate(new_row_data):
                    table.update_cell(thread.name, str(column_key), cell_data)
            else:
                table.add_row(*new_row_data, key=thread.name)

        status_view = self.query_one(StatusFooter)
        status_view.update_status(Text(f"Running", style="green"))

    @textual.on(ShowError)
    def on_show_error(self, message: ShowError) -> None:
        """
        Handles the ShowError message to display an error in the status bar.

        Args:
            message: The ShowError message containing the error string.
        """
        status_view = self.query_one(StatusFooter)
        status_view.update_status(Text(f"Error: {message.error}", style="red"))

    @staticmethod
    def __calculate_dynamic_watermark(scraper: JLinkScraper, stack_start: int, stack_size: int) -> int:
        """
        Reads a thread's stack memory and scans for the 0xAA fill pattern
        to determine the current stack watermark (lowest point of stack usage).

        Args:
            scraper: The JLinkScraper instance for reading MCU memory.
            stack_start: The starting address of the thread's stack.
            stack_size: The total size of the thread's stack in bytes.

        Returns:
            The calculated stack watermark in bytes, indicating the minimum
            amount of stack space that has not been used. Returns 0 on error
            or if the stack size is zero.
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

    async def __poll_threads(self) -> None:
        """
        Background worker function for polling thread data from the MCU.

        This function connects to the J-Link debugger, parses necessary
        Zephyr kernel symbols from the ELF file, and then enters a loop
        to continuously read and process dynamic thread information.
        Updates are sent to the main UI thread via messages.
        """
        try:
            parser = ZephyrSymbolParser(self.elf_path)
            kernel_base = parser.get_symbol_info("_kernel", info="address")
            threads_offset = parser.get_struct_member_offset("z_kernel", "threads")
            kernel_cpu_offset = parser.get_struct_member_offset("z_kernel", "cpus")
            kernel_usage0_offset = parser.get_struct_member_offset("_cpu", "usage0")

            # Calculate absolute addresses for key kernel structures
            threads_addr = kernel_base + threads_offset
            kernel_usage_addr = kernel_base + kernel_cpu_offset + kernel_usage0_offset

            # Get sizes and offsets for k_thread structure members
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

            with JLinkScraper(target_mcu=self.target_mcu) as scraper:
                threads_static_info = []
                # Read the initial pointer to the thread list head
                ptr = scraper.read32(threads_addr)[0] if threads_addr else 0

                # Calculate word indices for easier access
                words_to_read = stack_struct_size // 4
                next_ptr_word_idx = offsets["next"] // 4
                name_word_idx = offsets["name"] // 4
                stack_start_word_idx = offsets["stack_start"] // 4
                stack_size_word_idx = offsets["stack_size"] // 4

                # Iterate through the linked list of threads to get static info once
                for _ in range(50):  # Limit to 50 threads to prevent infinite loops on bad data
                    if ptr == 0:  # End of thread list
                        break

                    try:
                        # Read the entire k_thread structure
                        stack_struct_words = scraper.read32(ptr, words_to_read)

                        # Extract thread name (assuming 32 bytes and null-terminated)
                        name_bytes_raw = b""
                        for i in range(8):  # 32 bytes = 8 words
                            if (name_word_idx + i) < len(stack_struct_words):
                                name_bytes_raw += stack_struct_words[name_word_idx + i].to_bytes(4, "little")
                            else:
                                break  # Prevent index out of bounds if struct is truncated

                        thread_name = (name_bytes_raw.partition(b'\0')[0]).decode(
                            errors='ignore') or f"thread_0x{ptr:X}"

                        threads_static_info.append({
                            "address": ptr,
                            "stack_start": stack_struct_words[stack_start_word_idx],
                            "stack_size": stack_struct_words[stack_size_word_idx],
                            "name": thread_name,
                            "usage_offset": offsets["usage"]  # Store usage offset for dynamic reading
                        })

                        ptr = stack_struct_words[next_ptr_word_idx]  # Move to the next thread in the list
                    except Exception as e:
                        self.post_message(self.ShowError(f"Error parsing thread struct at 0x{ptr:X}: {e}"))
                        break

                # Main loop for dynamic data polling
                last_thread_cycles = {}  # Stores last read usage cycles for each thread
                last_cpu_cycles = {}  # Stores last read total CPU cycles for each thread (to mitigate I/O read delays)

                while self.is_running:
                    current_threads_data = []

                    try:
                        # Read current total CPU cycles
                        current_cpu_cycles = scraper.read32(kernel_usage_addr)[0]

                        # Calculate the delta for total CPU cycles
                        cpu_cycles_delta = current_cpu_cycles - last_cpu_cycles if last_cpu_cycles else 0
                        last_cpu_cycles = current_cpu_cycles

                        # If total CPU cycles didn't change, use the last known non-zero delta
                        # to avoid division by zero and provide a more stable CPU percentage.
                        if cpu_cycles_delta == 0 and self._last_cpu_deltas.get("total_cpu", 0) > 0:
                            cpu_cycles_delta = self._last_cpu_deltas["total_cpu"]
                        elif cpu_cycles_delta > 0:
                            self._last_cpu_deltas["total_cpu"] = cpu_cycles_delta

                        for thread_info in threads_static_info:
                            try:
                                # Read current thread usage cycles (64-bit)
                                thread_usage = scraper.read64(thread_info["address"] + thread_info["usage_offset"])[0]

                                # Calculate thread usage delta
                                thread_usage_delta = thread_usage - last_thread_cycles.get(thread_info["name"], 0)
                                last_thread_cycles[thread_info["name"]] = thread_usage

                                cpu_percent = 0.0
                                if cpu_cycles_delta > 0:
                                    # Calculate CPU percentage for the thread
                                    cpu_percent = (thread_usage_delta / cpu_cycles_delta) * 100
                                elif thread_usage_delta > 0:
                                    # If total CPU cycles didn't change but thread usage did,
                                    # it implies a very high CPU usage for this thread.
                                    cpu_percent = (thread_usage_delta / self._last_cpu_deltas["total_cpu"]) * 100

                                # Determine if the thread is "active" (had recent usage)
                                is_active = thread_usage_delta > 0

                                # Calculate stack watermark
                                watermark = ZViewApp.__calculate_dynamic_watermark(
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
                                self.post_message(self.ShowError(f"Error polling thread {thread_info['name']}: {e}"))
                                break
                    except Exception as e:
                        self.post_message(self.ShowError(f"Error reading global CPU cycles: {e}"))
                        # If global CPU read fails, clear data to avoid displaying stale info
                        current_threads_data = []

                    # Sort threads by name before sending to UI
                    if self.is_running:
                        self.post_message(self.UpdateThreads(sorted(current_threads_data, key=lambda t: t.name)))

                    await asyncio.sleep(0.2)  # Poll data every 200 milliseconds

        except Exception as e:
            # Catch any initialization errors in the polling thread
            self.post_message(self.ShowError(f"Polling thread initialization error: {e}"))


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="ZView - A real-time thread viewer for Zephyr RTOS.")
    arg_parser.add_argument("--mcu", required=False, default=None, help="Override target MCU.")
    arg_parser.add_argument("--elf-file", required=True, help="Path to the application's .elf firmware file.")
    args = arg_parser.parse_args()

    ZViewApp(elf_path=args.elf_file, target_mcu=args.mcu).run()
