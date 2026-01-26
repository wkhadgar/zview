# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0


import queue
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from pylink import JLink, JLinkException, JLinkInterfaces
from pyocd.core.helpers import ConnectHelper
from pyocd.core.session import Session
from pyocd.core.target import Target
from yaml import safe_load

from backend.elf_inspector import ElfInspector


class RunnerConfig:
    def __init__(self, runners_yaml_path):
        self.path = Path(runners_yaml_path)
        if not self.path.exists():
            raise FileNotFoundError(f"runners.yaml not found at {runners_yaml_path}")

        with open(self.path) as f:
            self.data = safe_load(f)

    def get_config(self, preferred_runner=None):
        """
        Returns a tuple of (runner_name, mcu_target)
        """
        runner = preferred_runner or self.data.get("flash-runner") or "jlink"

        args = self.data.get("args", {}).get(runner, [])

        mcu = None
        if runner == "jlink":
            mcu = self._find_arg(args, "--device")
        elif runner == "pyocd":
            mcu = self._find_arg(args, "--target")

        return runner, mcu

    def _find_arg(self, args_list, prefix):
        """Helper to find an argument starting with a specific prefix"""
        for arg in args_list:
            if arg.startswith(prefix):
                if "=" in arg:
                    return arg.split("=")[1]
                idx = args_list.index(arg)
                if idx + 1 < len(args_list):
                    return args_list[idx + 1]
        return None


@dataclass
class ThreadRuntime:
    """
    Data class to hold runtime information about the thread.
    """

    cpu: float
    active: bool
    stack_watermark: int


@dataclass
class ThreadInfo:
    """
    Data class to hold information about a single Zephyr RTOS thread.
    """

    address: int
    stack_start: int
    stack_size: int
    name: str
    runtime: ThreadRuntime | None


@dataclass
class HeapInfo:
    name: str
    address: int
    free_bytes: int
    allocated_bytes: int
    max_allocated_bytes: int


class AbstractScraper:
    def __init__(self, target_mcu: str | None):
        self._target_mcu: str | None = target_mcu
        self._is_connected: bool = False
        self.watermark_cache = {}

    def __enter__(self):
        self.connect()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            with open("zview.log", "w") as log:
                log.write(f"{exc_type}\n{exc_val}\n{exc_tb}")

        self.disconnect()

    @property
    def is_connected(self):
        return self._is_connected

    def connect(self):
        self._is_connected = True
        print("Connect was called")

    def disconnect(self):
        self._is_connected = False
        print("Disconnect was called")

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        print(f"Read {amount} bytes from {hex(at)}")
        return []

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        print(f"Read {amount} words from {hex(at)}")
        return []

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        print(f"Read {amount} double words from {hex(at)}")
        return []

    def calculate_dynamic_watermark(
        self,
        stack_start: int,
        stack_size: int,
        unused_pattern: int = 0xAA_AA_AA_AA,
        *,
        id,
    ) -> int:
        """
        Reads a stack memory and scans for the 0xAA fill pattern
        to determine the current stack watermark (lowest point of stack usage).

        Args:
            :param stack_start: The starting address of the thread's stack.
            :param stack_size: The total size of the thread's stack in bytes.
            :param unused_pattern: Unused stack fill word.
            :param id: Unique identification for the given thread.

        Returns:
            The calculated stack watermark in bytes, indicating the minimum
            amount of stack space that has not been used.
        """
        if stack_size == 0:
            return 0

        cache_watermark = self.watermark_cache.get(id, 0)
        watermark = stack_size - cache_watermark

        stack_words = self.read32(stack_start, (stack_size // 4) - (cache_watermark // 4))

        for word in stack_words:
            if word == unused_pattern:
                watermark -= 4
            else:
                while word:
                    word >>= 8
                    watermark -= 1
                break

        self.watermark_cache[id] = watermark + cache_watermark

        return self.watermark_cache[id]


class PyOCDScraper(AbstractScraper):
    def __init__(self, target_mcu: str | None):
        super().__init__(target_mcu)
        self.session: Session | None = ConnectHelper.session_with_chosen_probe(
            target_override=self._target_mcu, connect_mode="attach"
        )
        self.target: Target | None = None

    def connect(self):
        if self.session is None:
            raise Exception("Unable to create a PyOCD session.")

        try:
            self.session.open()
            self.target = self.session.target
        except Exception as e:
            raise Exception(f"\nUnable to connect with MCU [{self._target_mcu}].\n") from e

        self._is_connected = True

    def disconnect(self):
        if self.session is None:
            return

        self.session.close()
        self._is_connected = False

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        if self.target is None:
            raise Exception("No target available.")

        return self.target.read_memory_block8(at, amount)

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        if self.target is None:
            raise Exception("No target available.")

        return self.target.read_memory_block32(at, amount)

    def read64(self, at, amount: int = 1) -> Sequence[int]:
        words = self.read32(at, amount * 2)
        dwords = []
        for i in range(0, len(words), 2):
            dwords.append((words[i + 1] << 32) | words[i])

        return dwords


class JLinkScraper(AbstractScraper):
    def __init__(self, target_mcu: str | None):
        super().__init__(target_mcu)
        self.probe = JLink()

    def connect(self):
        try:
            self.probe.open()
        except JLinkException as e:
            raise Exception("\nUnable to connect to JLink") from e

        self.probe.set_tif(JLinkInterfaces.SWD)

        try:
            self.probe.connect(self._target_mcu)
        except JLinkException as e:
            self.probe.close()
            raise Exception(f"\nUnable to connect with [{self._target_mcu}]") from e
        self._is_connected = True

    def disconnect(self):
        self.probe.close()
        self._is_connected = False

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read8(at, amount)

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read32(at, amount)

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read64(at, amount)


class ZScraper:
    def __init__(
        self,
        meta_scraper: AbstractScraper,
        elf_path: str,
        max_threads: int,
        thread_name_size: int,
    ):
        self._all_threads_info: dict[str, ThreadInfo] = {}
        self._elf_inspector: ElfInspector = ElfInspector(elf_path)
        self._m_scraper: AbstractScraper = meta_scraper
        self._polling_thread: threading.Thread | None = None
        self._thread_pool: list[ThreadInfo] | None = None
        self._stop_event: Event | None = None

        self.has_heaps = True
        self.has_usage = True
        self.has_extra_info = True

        self._MAX_THREADS: int = max_threads
        self._MAX_THREAD_NAME_BYTES: int = thread_name_size

        self._offsets = {
            "kernel": {
                "cpu": self._elf_inspector.get_struct_member_offset("z_kernel", "cpus"),
                "threads": self._elf_inspector.get_struct_member_offset("z_kernel", "threads"),
            },
            "k_thread": {
                "base": self._elf_inspector.get_struct_member_offset("k_thread", "base"),
                "name": self._elf_inspector.get_struct_member_offset("k_thread", "name"),
                "stack_info": self._elf_inspector.get_struct_member_offset(
                    "k_thread", "stack_info"
                ),
                "next_thread": self._elf_inspector.get_struct_member_offset(
                    "k_thread", "next_thread"
                ),
            },
            "k_heap": {
                "heap": self._elf_inspector.get_struct_member_offset("k_heap", "heap"),
            },
            "sys_heap": {
                "heap": self._elf_inspector.get_struct_member_offset("sys_heap", "heap"),
            },
        }

        self._offsets["thread_info"] = {
            "stack_start": self._offsets["k_thread"]["stack_info"]
            + self._elf_inspector.get_struct_member_offset("_thread_stack_info", "start"),
            "stack_size": self._offsets["k_thread"]["stack_info"]
            + self._elf_inspector.get_struct_member_offset("_thread_stack_info", "size"),
            "stack_delta": self._offsets["k_thread"]["stack_info"]
            + self._elf_inspector.get_struct_member_offset("_thread_stack_info", "delta"),
        }

        # Known to be only one (at least, expected)
        self._kernel_base_address = self._elf_inspector.get_symbol_info("_kernel", info="address")[
            0
        ]
        self._threads_address = self._kernel_base_address + self._offsets["kernel"]["threads"]

        try:
            self._offsets["kernel"]["usage"] = self._elf_inspector.get_struct_member_offset(
                "z_kernel", "usage"
            )
            self._offsets["thread_info"].update(
                {
                    "usage_base": self._offsets["k_thread"]["base"]
                    + self._elf_inspector.get_struct_member_offset("_thread_base", "usage")
                }
            )
            self._cpu_usage_address = self._kernel_base_address + self._offsets["kernel"]["usage"]
        except LookupError:
            self.has_usage = False

        try:
            self._offsets["thread_info"]["usage"] = self._offsets["thread_info"][
                "usage_base"
            ] + self._elf_inspector.get_struct_member_offset("k_cycle_stats", "total")
            self._offsets["thread_info"]["longest_window"] = self._offsets["thread_info"][
                "usage_base"
            ] + self._elf_inspector.get_struct_member_offset("k_cycle_stats", "longest")

            self._offsets["thread_info"]["amount_windows"] = self._offsets["thread_info"][
                "usage_base"
            ] + self._elf_inspector.get_struct_member_offset("k_cycle_stats", "num_windows")
        except LookupError:
            self.has_extra_info = False

        try:
            self._offsets["heap_info"] = {
                "z_heap_base": self._offsets["k_heap"]["heap"] + self._offsets["sys_heap"]["heap"],
                "free_bytes": self._elf_inspector.get_struct_member_offset("z_heap", "free_bytes"),
                "allocated_bytes": self._elf_inspector.get_struct_member_offset(
                    "z_heap", "allocated_bytes"
                ),
                "max_allocated_bytes": self._elf_inspector.get_struct_member_offset(
                    "z_heap", "max_allocated_bytes"
                ),
            }
        except LookupError:
            self.has_heaps = False
            return

        all_heaps = self._elf_inspector.find_struct_variable_names("k_heap")

        if all_heaps is None or len(all_heaps) == 0:
            self.has_heaps = False
            return

        self._k_heap_addresses = {}
        for heap in all_heaps:
            self._k_heap_addresses[heap] = self._elf_inspector.get_symbol_info(heap, "address")

    def __enter__(self):
        self._m_scraper.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            with open("zview.log", "w") as log:
                log.write(f"{exc_type}\n{exc_val}\n{exc_tb}\n")

        self._m_scraper.disconnect()

    @property
    def all_threads(self):
        return self._all_threads_info

    @property
    def thread_pool(self):
        return self._thread_pool

    @thread_pool.setter
    def thread_pool(self, new: list[ThreadInfo]):
        self._thread_pool = new

    def read_variable(self, var_name: str) -> list[Sequence[int]]:
        if not self._m_scraper.is_connected:
            self._m_scraper.connect()

        var_addresses = self._elf_inspector.get_symbol_info(var_name, "address")
        var_sizes = self._elf_inspector.get_symbol_info(var_name, "size")

        return [
            self._m_scraper.read8(var_address, var_size)
            for var_address, var_size in zip(var_addresses, var_sizes, strict=True)
        ]

    def update_available_threads(self):
        try:
            if not self._m_scraper.is_connected:
                self._m_scraper.connect()

            thread_ptr = (
                self._m_scraper.read32(self._threads_address)[0] if self._threads_address else 0
            )
        except Exception as e:
            raise RuntimeError("Unable to read kernel thread list.") from e

        stack_struct_size = self._elf_inspector.get_struct_size("k_thread")
        words_to_read = stack_struct_size // 4
        next_ptr_word_idx = self._offsets["k_thread"]["next_thread"] // 4
        name_word_idx = self._offsets["k_thread"]["name"] // 4
        stack_start_word_idx = self._offsets["thread_info"]["stack_start"] // 4
        stack_size_word_idx = self._offsets["thread_info"]["stack_size"] // 4

        for _ in range(self._MAX_THREADS):
            if thread_ptr == 0:
                break

            try:
                stack_struct_words = self._m_scraper.read32(thread_ptr, words_to_read)
                name_bytes_raw = b""
                for i in range(self._MAX_THREAD_NAME_BYTES // 4):
                    if (name_word_idx + i) >= len(stack_struct_words):
                        break

                    name_bytes_raw += stack_struct_words[name_word_idx + i].to_bytes(4, "little")

                thread_name = (
                    name_bytes_raw.partition(b"\0")[0].decode(errors="ignore")
                    or f"thread_0x{thread_ptr:X}"
                )

                self._all_threads_info[thread_name] = ThreadInfo(
                    thread_ptr,
                    stack_struct_words[stack_start_word_idx],
                    stack_struct_words[stack_size_word_idx],
                    thread_name,
                    None,
                )

                thread_ptr = stack_struct_words[next_ptr_word_idx]
            except Exception as e:
                raise Exception(f"Error reading thread struct at 0x{thread_ptr:X}") from e

    def start_polling_thread(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        inspection_period: float,
    ):
        """
        Starts a separate daemon thread to continuously poll data from the MCU.
        """
        if self._polling_thread is not None:
            data_queue.put({"error": "Already started..."})
            return

        try:
            self.update_available_threads()
        except RuntimeError as e:
            data_queue.put({"error": f"{e}"})
            return

        self._polling_thread = threading.Thread(
            target=self._poll_thread_worker,
            args=(data_queue, stop_event, inspection_period),
            daemon=True,
        )
        self._stop_event = stop_event
        self._polling_thread.daemon = True
        self._polling_thread.start()

    def finish_polling_thread(self):
        if self._polling_thread is None or self._stop_event is None:
            return

        self._stop_event.set()
        if self._polling_thread.is_alive():
            self._polling_thread.join(timeout=1.0)
            if self._polling_thread.is_alive():
                print("Warning: Polling thread did not terminate gracefully.")

    def _poll_thread_worker(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        inspection_period: float,
    ):
        """
        Worker function executed by the polling thread.

        Connects to the MCU via J-Link, analyzes Zephyr kernel symbols from the ELF
        file, and continuously reads and processes runtime data, pushing it to
        `data_queue`.

        Args:
            data_queue: A queue for sending processed data to the main UI thread.
            stop_event: An event to signal the polling thread to terminate.
        """

        def get_thread_info():
            nonlocal last_cpu_delta
            nonlocal last_cpu_cycles
            nonlocal last_thread_cycles

            if self.has_usage:
                try:
                    current_cpu_cycles = self._m_scraper.read32(self._cpu_usage_address)[0]
                except Exception as e:
                    data_queue.put({"error": f"Error reading global CPU cycles: {e}"})
                    return

                cpu_cycles_delta = current_cpu_cycles - last_cpu_cycles
                last_cpu_cycles = current_cpu_cycles
                if cpu_cycles_delta > 0:
                    last_cpu_delta = cpu_cycles_delta
                elif cpu_cycles_delta <= 0:  # When current_cpu_cycles is 0 due to context resets
                    cpu_cycles_delta = last_cpu_delta
            else:
                cpu_cycles_delta = -1

            if self.thread_pool is None:
                return

            for thread_info in self.thread_pool:
                try:
                    thread_usage = (
                        self._m_scraper.read64(
                            thread_info.address + self._offsets["thread_info"]["usage"]
                        )[0]
                        if self.has_usage
                        else -1
                    )

                    watermark = self._m_scraper.calculate_dynamic_watermark(
                        thread_info.stack_start,
                        thread_info.stack_size,
                        id=thread_info.address,
                    )
                except Exception as e:
                    data_queue.put({"error": f"Error polling thread {thread_info}: {e}"})
                    continue

                thread_usage_delta = thread_usage - last_thread_cycles.get(thread_info.name, 0)
                last_thread_cycles[thread_info.name] = thread_usage

                if self.has_usage and thread_info.name == "idle":
                    cpu_cycles_delta += thread_usage_delta
                cpu_percent = 0.0
                if cpu_cycles_delta > 0:
                    cpu_percent = (thread_usage_delta / cpu_cycles_delta) * 100
                elif thread_usage_delta > 0:
                    cpu_percent = (thread_usage_delta / last_cpu_delta) * 100
                else:
                    cpu_percent = -1

                is_active = thread_usage_delta > 0

                if thread_info.runtime:
                    thread_info.runtime.cpu = min(cpu_percent, 100)
                    thread_info.runtime.active = is_active
                    thread_info.runtime.stack_watermark = watermark
                else:
                    thread_info.runtime = ThreadRuntime(cpu_percent, is_active, watermark)

                data_queue.put({"threads": self.thread_pool})

        def get_heap_info():
            heap_info = []
            for heap_name, heap_addresses in self._k_heap_addresses.items():
                for heap_address in heap_addresses:
                    heap_address = self._m_scraper.read32(heap_address)[0]
                    try:
                        free_bytes = self._m_scraper.read32(
                            heap_address + self._offsets["heap_info"]["free_bytes"]
                        )[0]
                        allocated_bytes = self._m_scraper.read32(
                            heap_address + self._offsets["heap_info"]["allocated_bytes"]
                        )[0]
                        max_allocated_bytes = self._m_scraper.read32(
                            heap_address + self._offsets["heap_info"]["max_allocated_bytes"]
                        )[0]
                    except Exception as e:
                        data_queue.put({"error": f"Error reading heap info: {e}"})
                        return

                    heap_info.append(
                        HeapInfo(
                            heap_name,
                            heap_address,
                            free_bytes,
                            allocated_bytes,
                            max_allocated_bytes,
                        )
                    )

                    data_queue.put({"heaps": heap_info})

        try:
            if not self._m_scraper.is_connected:
                self._m_scraper.connect()

            init_cpu_cycles = (
                self._m_scraper.read64(self._cpu_usage_address)[0] if self.has_usage else 0
            )
        except Exception as e:
            data_queue.put({"error": f"Polling thread initialization error: {e}"})
            return

        last_cpu_delta = 1000
        last_cpu_cycles = init_cpu_cycles
        last_thread_cycles = {}

        while not stop_event.is_set():
            get_thread_info()

            if self.has_heaps:
                get_heap_info()

            time.sleep(inspection_period)
