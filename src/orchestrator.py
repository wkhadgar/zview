# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
ZScraper coordinates an AbstractScraper backend, a DWARF-resolved ElfInspector,
and the kernel-object walkers, owning the polling thread that produces frames
of thread + heap state for the TUI.
"""

import contextlib
import queue
import threading
import time
from threading import Event
from typing import Literal

from backend.base import AbstractScraper, HeapInfo, ThreadInfo, ThreadRuntime
from backend.elf_inspector import ElfInspector
from kernel.heaps import walk_heap_fragmentation
from kernel.threads import walk_thread_list


class ZScraper:
    def __init__(
        self,
        meta_scraper: AbstractScraper,
        elf_path,
        max_threads: int = 64,
    ):
        self._all_threads_info: dict[str, ThreadInfo] = {}
        self._elf_inspector: ElfInspector = ElfInspector(elf_path)
        self._endianess: Literal["little", "big"] = (
            "little" if self._elf_inspector._little_endian else "big"
        )
        self._m_scraper: AbstractScraper = meta_scraper
        self._m_scraper.endianess = "<" if self._endianess == "little" else ">"
        self._polling_thread: threading.Thread | None = None
        self._thread_pool: list[ThreadInfo] | None = None
        self._stop_event: Event | None = None
        self.inspection_period = 0.2

        self.has_heaps: bool = True
        self.has_usage: bool = True
        self.has_names: bool = True

        self._MAX_THREADS: int = max_threads

        self._offsets = {
            "kernel": {
                "cpu": self._elf_inspector.get_struct_member_offset("z_kernel", "cpus"),
                "threads": self._elf_inspector.get_struct_member_offset("z_kernel", "threads"),
            },
            "k_thread": {
                "base": self._elf_inspector.get_struct_member_offset("k_thread", "base"),
                "stack_info": self._elf_inspector.get_struct_member_offset(
                    "k_thread", "stack_info"
                ),
                "next_thread": self._elf_inspector.get_struct_member_offset(
                    "k_thread", "next_thread"
                ),
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

        self.idle_threads_address = self._elf_inspector.get_symbol_info(
            "z_idle_threads", "address"
        )[0]

        try:
            self._offsets["k_thread"]["name"] = self._elf_inspector.get_struct_member_offset(
                "k_thread", "name"
            )
        except LookupError:
            self.has_names = False

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
            self._m_scraper.begin_batch()
            self.init_cpu_cycles = self._m_scraper.read64(self._cpu_usage_address)[0]
            self._m_scraper.end_batch()
            self.last_cpu_delta = self.init_cpu_cycles
            self.last_cpu_cycles = self.init_cpu_cycles
            self.last_thread_cycles = {}
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
                "z_heap_base": self._elf_inspector.get_struct_member_offset("k_heap", "heap")
                + self._elf_inspector.get_struct_member_offset("sys_heap", "heap"),
                "free_bytes": self._elf_inspector.get_struct_member_offset("z_heap", "free_bytes"),
                "allocated_bytes": self._elf_inspector.get_struct_member_offset(
                    "z_heap", "allocated_bytes"
                ),
                "max_allocated_bytes": self._elf_inspector.get_struct_member_offset(
                    "z_heap", "max_allocated_bytes"
                ),
                "end_chunk": self._elf_inspector.get_struct_member_offset("z_heap", "end_chunk"),
            }

        except LookupError:
            self.has_heaps = False
            return

        all_heaps = self._elf_inspector.find_struct_variable_names("k_heap")

        if all_heaps is None or len(all_heaps) == 0:
            self.has_heaps = False
            return

        self._k_heap_addresses = {}
        self.extra_info_heap_address: int | None = None
        for heap in all_heaps:
            self._k_heap_addresses[heap] = self._elf_inspector.get_symbol_info(heap, "address")

    def __enter__(self):
        self._m_scraper.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._m_scraper.__exit__(exc_type, exc_val, exc_tb)

    @property
    def all_threads(self):
        return self._all_threads_info

    @property
    def thread_pool(self):
        return self._thread_pool

    @thread_pool.setter
    def thread_pool(self, new: list[ThreadInfo]):
        self._thread_pool = new

    def read_variable(self, var_name: str) -> list:
        if not self._m_scraper.is_connected:
            self._m_scraper.connect()

        var_addresses = self._elf_inspector.get_symbol_info(var_name, "address")
        var_sizes = self._elf_inspector.get_symbol_info(var_name, "size")

        return [
            self._m_scraper.read8(var_address, var_size)
            for var_address, var_size in zip(var_addresses, var_sizes, strict=True)
        ]

    def update_available_threads(self):
        threads = walk_thread_list(
            self._m_scraper,
            self._elf_inspector,
            self._threads_address,
            self._offsets,
            self._endianess,
            self.has_names,
            self._MAX_THREADS,
        )
        self._all_threads_info.clear()
        self._all_threads_info.update(threads)

    def get_heap_fragmentation(self, z_heap_addr: int) -> list[dict]:
        return walk_heap_fragmentation(
            self._m_scraper,
            z_heap_addr,
            self._offsets["heap_info"]["end_chunk"],
        )

    def reset_thread_pool(self):
        self.thread_pool = list(self.all_threads.values())

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

        self.inspection_period = inspection_period
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
                print("Polling thread did not terminate gracefully.")

        self._polling_thread = None

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
        consecutive_errors = 0
        MAX_TOLERATED_ERRORS = 3

        try:
            if not self._m_scraper.is_connected:
                self._m_scraper.connect()
        except Exception as e:
            data_queue.put({"fatal_error": f"Failed to connect to target: {e}"})
            return

        while not stop_event.is_set():
            in_batch = False
            heap_info = []
            try:
                self._m_scraper.begin_batch()
                in_batch = True

                if self.has_usage:
                    try:
                        current_cpu_cycles = self._m_scraper.read32(self._cpu_usage_address)[0]
                    except Exception as e:
                        raise RuntimeError(f"Error reading global CPU cycles: {e}") from e

                    cpu_cycles_delta = current_cpu_cycles - self.last_cpu_cycles
                    self.last_cpu_cycles = current_cpu_cycles
                    if cpu_cycles_delta > 0:
                        self.last_cpu_delta = cpu_cycles_delta
                    elif (
                        cpu_cycles_delta <= 0
                    ):  # When current_cpu_cycles is 0 due to context resets
                        cpu_cycles_delta = self.last_cpu_delta
                else:
                    cpu_cycles_delta = -1

                if self.thread_pool is None:
                    raise RuntimeError("Thread pool is uninitialized.")

                polled_raw_data = []
                for thread_info in self.thread_pool:
                    if self.has_usage:
                        try:
                            thread_usage = self._m_scraper.read64(
                                thread_info.address + self._offsets["thread_info"]["usage"]
                            )[0]
                        except Exception as e:
                            raise RuntimeError(
                                f"Error polling thread usage {thread_info.name}: {e}"
                            ) from e

                        thread_usage_delta = thread_usage - self.last_thread_cycles.get(
                            thread_info.address, 0
                        )
                        self.last_thread_cycles[thread_info.address] = thread_usage

                        if thread_info.address == self.idle_threads_address:
                            cpu_cycles_delta += thread_usage_delta

                        is_active = thread_usage_delta > 0
                    else:
                        thread_usage_delta = 0
                        is_active = False

                    try:
                        watermark = self._m_scraper.calculate_dynamic_watermark(
                            thread_info.stack_start,
                            thread_info.stack_size,
                            thread_id=thread_info.address,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Error polling stack watermark for {thread_info.name}: {e}"
                        ) from e

                    stack_usage_pct = (
                        (watermark / thread_info.stack_size * 100)
                        if thread_info.stack_size > 0
                        else 0.0
                    )

                    polled_raw_data.append(
                        {
                            "info": thread_info,
                            "usage_delta": thread_usage_delta,
                            "is_active": is_active,
                            "watermark": watermark,
                            "stack_usage_pct": stack_usage_pct,
                        }
                    )

                idle_thread_data = next(
                    (d for d in polled_raw_data if d["info"].address == self.idle_threads_address),
                    None,
                )

                if idle_thread_data:
                    active_cycles_total = cpu_cycles_delta - idle_thread_data["usage_delta"]
                else:
                    active_cycles_total = cpu_cycles_delta

                final_threads = []
                for data in polled_raw_data:
                    # Absolute CPU % (t / total_system_time)
                    absolute_cpu = (
                        (data["usage_delta"] / cpu_cycles_delta * 100)
                        if cpu_cycles_delta > 0
                        else 0.0
                    )

                    # Relative Load % (t / total_active_time)
                    if data["info"].address == self.idle_threads_address:
                        load_pct = 0.0
                    elif active_cycles_total > 0:
                        load_pct = min((data["usage_delta"] / active_cycles_total) * 100.0, 100.0)
                    else:
                        load_pct = 0.0

                    final_runtime = ThreadRuntime(
                        cpu=load_pct,
                        cpu_normalized=absolute_cpu,
                        active=data["is_active"],
                        stack_watermark=data["watermark"],
                        stack_watermark_percent=data["stack_usage_pct"],
                    )
                    final_threads.append(
                        ThreadInfo(
                            address=data["info"].address,
                            stack_start=data["info"].stack_start,
                            stack_size=data["info"].stack_size,
                            name=data["info"].name,
                            runtime=final_runtime,
                        )
                    )

                if self.has_heaps:
                    for heap_name, heap_addresses in self._k_heap_addresses.items():
                        for heap_address in heap_addresses:
                            try:
                                heap_address_val = self._m_scraper.read32(heap_address)[0]
                                free_bytes = self._m_scraper.read32(
                                    heap_address_val + self._offsets["heap_info"]["free_bytes"]
                                )[0]
                                allocated_bytes = self._m_scraper.read32(
                                    heap_address_val + self._offsets["heap_info"]["allocated_bytes"]
                                )[0]
                                max_allocated_bytes = self._m_scraper.read32(
                                    heap_address_val
                                    + self._offsets["heap_info"]["max_allocated_bytes"]
                                )[0]
                            except Exception as e:
                                data_queue.put(
                                    {"error": f"Error reading heap info for {heap_name}: {e}"},
                                    block=False,
                                )
                                continue

                            chunks = None
                            if self.extra_info_heap_address == heap_address_val:
                                try:
                                    chunks = self.get_heap_fragmentation(heap_address_val)
                                except Exception as e:
                                    data_queue.put(
                                        {"error": f"Error reading sparsity for {heap_name}: {e}"},
                                        block=False,
                                    )

                            total_heap_size = allocated_bytes + free_bytes
                            heap_usage_pct = (
                                (allocated_bytes / total_heap_size * 100)
                                if total_heap_size > 0
                                else 0.0
                            )

                            heap_info.append(
                                HeapInfo(
                                    heap_name,
                                    heap_address_val,
                                    free_bytes,
                                    allocated_bytes,
                                    max_allocated_bytes,
                                    heap_usage_pct,
                                    chunks,
                                )
                            )

                if self.has_heaps:
                    data_queue.put({"threads": final_threads, "heaps": heap_info}, block=False)
                else:
                    data_queue.put({"threads": final_threads}, block=False)

                # Frame successful. Reset error counter.
                consecutive_errors = 0

            except queue.Full:
                pass
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= MAX_TOLERATED_ERRORS:
                    data_queue.put(
                        {"fatal_error": f"Target lost after {MAX_TOLERATED_ERRORS} retries: {e}"}
                    )
                    break  # Terminate loop cleanly
                else:
                    data_queue.put(
                        {
                            "error": f"Transient read fault "
                            f"({consecutive_errors}/{MAX_TOLERATED_ERRORS}): {e}"
                        },
                        block=False,
                    )

            finally:
                if in_batch:
                    with contextlib.suppress(Exception):
                        self._m_scraper.end_batch()

            time.sleep(inspection_period)
