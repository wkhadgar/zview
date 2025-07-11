import argparse
import asyncio
from dataclasses import dataclass
from typing import List

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
    name: str
    cpu: float
    active: bool
    stack_size: int
    stack_watermark: int


class ZViewApp(App):
    TITLE = "ZView - Zephyr RTOS Runtime Viewer"
    BINDINGS = [("q", "quit", "Quit")]

    @dataclass
    class UpdateThreads(Message):
        threads: List[ThreadInfo]

    @dataclass
    class ShowError(Message):
        error: str

    def __init__(self, elf_path: str, target_mcu: str | None = None):
        super().__init__()
        self.target_mcu: str | None = target_mcu
        self.elf_path = elf_path

    def compose(self) -> ComposeResult:
        yield Header()
        yield DataTable(id="threads_table")
        yield Static(id="status_view")
        yield Footer()

    def on_mount(self) -> None:
        self.run_worker(self.__poll_threads, thread=True)

    @textual.on(UpdateThreads)
    def on_update_threads(self, message: UpdateThreads) -> None:
        table = self.query_one(DataTable)
        prev_cursor_y = table.cursor_row
        prev_cursor_x = table.cursor_column
        table.clear()
        for thread in message.threads:
            usage_percent = (thread.stack_watermark / thread.stack_size * 100) if thread.stack_size > 0 else 0
            bar = ProgressBar(total=100, completed=usage_percent, width=30)
            bar.complete_style = "red" if usage_percent > 90 else "yellow" if usage_percent > 75 else "green"
            bar.finished_style = "red"
            table.add_row(
                Text(thread.name, style="cyan" if thread.active else "grey"),
                f"{thread.cpu:.1f}%",
                f"{(100 * thread.stack_watermark) / thread.stack_size: .1f}%",
                bar,
                f"{thread.stack_watermark} / {thread.stack_size}",
                key=thread.name
            )
        table.move_cursor(row=prev_cursor_y, column=prev_cursor_x)

    @staticmethod
    def __calculate_dynamic_watermark(scraper: JLinkScraper, stack_start: int, stack_size: int) -> int:
        """Reads a thread's stack and scans for the 0xAA pattern to find the watermark."""
        watermark = stack_size
        stack_bytes = scraper.read32(stack_start, stack_size // 4)

        for word in stack_bytes:
            if word == 0xAA_AA_AA_AA:
                watermark -= 4
            else:
                watermark -= word.to_bytes(4).count(b"\xaa")
                break

        return watermark

    async def __poll_threads(self) -> None:
        from mcu_scraper import JLinkScraper

        print("Polling threads")
        parser = ZephyrSymbolParser(self.elf_path)
        kernel_base = parser.get_symbol_info("_kernel", info="address")
        threads_offset = parser.get_struct_member_offset("z_kernel", "threads")
        kernel_usage_offset = parser.get_struct_member_offset("z_kernel", "usage")
        threads_addr = kernel_base + threads_offset
        kernel_usage_addr = kernel_base + kernel_usage_offset

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

        table = self.query_one(DataTable)

        with (JLinkScraper(target_mcu=self.target_mcu) as scraper):
            threads_info = []
            ptr = scraper.read32(threads_addr)[0]
            for _ in range(50):
                if ptr == 0:
                    break

                stack_struct_words = scraper.read32(ptr, stack_struct_size // 4)
                stack_name_words = stack_struct_words[offsets["name"] // 4: (offsets["name"] + 32) // 4]
                stack_name_bytes = b"".join((word.to_bytes(4, "little") for word in stack_name_words))
                thread_static_data = {"address": ptr,
                                      "stack_start": stack_struct_words[offsets["stack_start"] // 4],
                                      "stack_size": stack_struct_words[offsets["stack_size"] // 4],
                                      "name": (stack_name_bytes.partition(b'\0')[0]).decode(
                                          errors='ignore') or f"thread_0x{ptr:X}"}
                ptr = stack_struct_words[offsets["next"] // 4]

                threads_info.append(thread_static_data)

            ## Mainloop
            reading = False
            last_cycles = {}
            last_cpu_cycles = {}
            cpus = {}
            while self.is_running:
                message = []
                for thread_info in threads_info:
                    thread_usage = scraper.read64(thread_info["address"] + offsets["usage"])[0]
                    thread_usage_delta = thread_usage - last_cycles[thread_info["name"]] if last_cycles.get(
                        thread_info["name"]) else 0
                    last_cycles[thread_info["name"]] = thread_usage

                    cpu_usage = scraper.read64(kernel_usage_addr)[0]
                    cpu_usage_delta = cpu_usage - last_cpu_cycles[thread_info["name"]] if last_cpu_cycles.get(
                        thread_info["name"]) else 0.1
                    last_cpu_cycles[thread_info["name"]] = cpu_usage

                    cpus[thread_info["name"]] = thread_usage_delta / cpu_usage_delta

                # for thread_info in threads_info:
                    watermark = self.__calculate_dynamic_watermark(scraper, thread_info["stack_start"],
                                                                   thread_info["stack_size"])

                    message.append(
                        ThreadInfo(thread_info["name"], cpus[thread_info["name"]] * 100, thread_usage_delta > 0,
                                   thread_info["stack_size"],
                                   watermark))

                if self.is_running:
                    if not reading:
                        self.call_from_thread(table.add_columns, f"Thread", "CPU %", "Watermark %", "Stack Usage",
                                              "Watermark (Bytes)")
                        reading = True
                    if not self.post_message(self.UpdateThreads(sorted(message, key=lambda t: t.name))):
                        raise Exception("Unable to send message")

                await asyncio.sleep(0.2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="ZView - A real-time thread viewer for Zephyr RTOS.")
    arg_parser.add_argument("--mcu", required=False, default=None, help="Override target MCU.")
    arg_parser.add_argument("--elf-file", required=True, help="Path to the application's .elf firmware file.")
    args = arg_parser.parse_args()

    ZViewApp(elf_path=args.elf_file, target_mcu=args.mcu).run()
