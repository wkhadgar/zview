# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
ZScraper coordinates an AbstractScraper backend, a DWARF-resolved ElfInspector,
and the kernel-object walkers, owning the polling thread that produces frames
of thread + heap state for the TUI.
"""

import contextlib
import logging
import queue
import threading
import time
from threading import Event
from typing import Literal

from backend.base import AbstractScraper, HeapInfo, ThreadInfo, ThreadRuntime
from backend.elf_inspector import ElfInspector
from backend.replay import ReplayComplete
from kernel.heaps import walk_heap_fragmentation
from kernel.layout import KernelLayout
from kernel.threads import walk_thread_list

logger = logging.getLogger("zview.scraper")


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

        self._layout = self._resolve_layout()
        self._resolve_addresses()

        if self.has_usage:
            self._baseline_cpu_cycles()

        if self.has_heaps:
            self._discover_heap_addresses()

    def _resolve_layout(self) -> KernelLayout:
        """Resolve all DWARF-derived offsets, dropping optional features that don't apply."""
        elf = self._elf_inspector
        stack_info = elf.get_struct_member_offset("k_thread", "stack_info")

        fields: dict = {
            "threads_head": elf.get_struct_member_offset("z_kernel", "threads"),
            "thread_next": elf.get_struct_member_offset("k_thread", "next_thread"),
            "stack_start": stack_info + elf.get_struct_member_offset("_thread_stack_info", "start"),
            "stack_size": stack_info + elf.get_struct_member_offset("_thread_stack_info", "size"),
        }

        if (extras := self._try_resolve_thread_name()) is not None:
            fields.update(extras)
        else:
            self.has_names = False

        if (extras := self._try_resolve_usage()) is not None:
            fields.update(extras)
        else:
            self.has_usage = False

        if (extras := self._try_resolve_heaps()) is not None:
            fields.update(extras)
        else:
            self.has_heaps = False

        return KernelLayout(**fields)

    def _try_resolve_thread_name(self) -> dict | None:
        try:
            return {"thread_name": self._elf_inspector.get_struct_member_offset("k_thread", "name")}
        except LookupError:
            return None

    def _try_resolve_usage(self) -> dict | None:
        elf = self._elf_inspector
        try:
            cpu_usage = elf.get_struct_member_offset("z_kernel", "usage")
            usage_base = elf.get_struct_member_offset(
                "k_thread", "base"
            ) + elf.get_struct_member_offset("_thread_base", "usage")
            thread_usage = usage_base + elf.get_struct_member_offset("k_cycle_stats", "total")
        except LookupError:
            return None
        return {"cpu_usage": cpu_usage, "thread_usage": thread_usage}

    def _try_resolve_heaps(self) -> dict | None:
        elf = self._elf_inspector
        try:
            return {
                "heap_free_bytes": elf.get_struct_member_offset("z_heap", "free_bytes"),
                "heap_allocated_bytes": elf.get_struct_member_offset("z_heap", "allocated_bytes"),
                "heap_max_allocated_bytes": elf.get_struct_member_offset(
                    "z_heap", "max_allocated_bytes"
                ),
                "heap_end_chunk": elf.get_struct_member_offset("z_heap", "end_chunk"),
            }
        except LookupError:
            return None

    def _resolve_addresses(self) -> None:
        elf = self._elf_inspector
        self._kernel_base_address = elf.get_symbol_info("_kernel", "address")[0]
        self._threads_address = self._kernel_base_address + self._layout.threads_head
        self.idle_threads_address = elf.get_symbol_info("z_idle_threads", "address")[0]
        if self.has_usage:
            self._cpu_usage_address = self._kernel_base_address + self._layout.cpu_usage

    def _baseline_cpu_cycles(self) -> None:
        """Read the initial CPU cycle counter so the first polling delta has an anchor."""
        self._m_scraper.begin_batch()
        self.init_cpu_cycles = self._m_scraper.read64(self._cpu_usage_address)[0]
        self._m_scraper.end_batch()
        self.last_cpu_delta = self.init_cpu_cycles
        self.last_cpu_cycles = self.init_cpu_cycles
        self.last_thread_cycles: dict = {}

    def _discover_heap_addresses(self) -> None:
        """Locate every ``k_heap`` global and store its address(es). Disables heaps if none."""
        elf = self._elf_inspector
        names = elf.find_struct_variable_names("k_heap") or []
        if not names:
            self.has_heaps = False
            return
        self._k_heap_addresses = {n: elf.get_symbol_info(n, "address") for n in names}
        self.extra_info_heap_address: int | None = None

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
            self._layout,
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
            self._layout.heap_end_chunk,
        )

    def reset_thread_pool(self):
        self.thread_pool = list(self.all_threads.values())

    def reset_runtime_state(self):
        """
        Drop all per-thread caches and re-baseline CPU cycle counters.
        Call this whenever the target has been reset or reconnected so that
        stale watermark assumptions and cycle deltas do not leak across sessions.
        """
        self._m_scraper.watermark_cache.clear()

        if not self.has_usage:
            return

        self.last_thread_cycles.clear()
        self._m_scraper.begin_batch()
        try:
            self.init_cpu_cycles = self._m_scraper.read64(self._cpu_usage_address)[0]
        finally:
            self._m_scraper.end_batch()
        self.last_cpu_cycles = self.init_cpu_cycles
        self.last_cpu_delta = self.init_cpu_cycles

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
        self._polling_thread.start()

    def finish_polling_thread(self):
        if self._polling_thread is None or self._stop_event is None:
            return

        self._stop_event.set()
        if self._polling_thread.is_alive():
            self._polling_thread.join(timeout=1.0)
            if self._polling_thread.is_alive():
                logger.warning("Polling thread did not terminate gracefully.")

        self._polling_thread = None

    _MAX_TOLERATED_ERRORS = 3

    def _poll_thread_worker(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        inspection_period: float,
    ):
        """
        Polling thread entry point. Connects, then loops one frame at a time,
        delegating per-phase work to the helpers below. Tolerates up to
        ``_MAX_TOLERATED_ERRORS`` transient read faults before declaring the
        target lost; ends cleanly when a recording exhausts.
        """
        try:
            if not self._m_scraper.is_connected:
                self._m_scraper.connect()
        except Exception as e:
            data_queue.put({"fatal_error": f"Failed to connect to target: {e}"})
            return

        consecutive_errors = 0
        while not stop_event.is_set():
            in_batch = False
            try:
                self._m_scraper.begin_batch()
                in_batch = True

                cpu_cycles_delta = self._read_cpu_cycles_delta()
                polled_raw, cpu_cycles_delta = self._poll_threads(cpu_cycles_delta)
                final_threads = self._finalize_threads(polled_raw, cpu_cycles_delta)

                frame: dict = {"threads": final_threads}
                if self.has_heaps:
                    frame["heaps"] = self._poll_heaps(data_queue)
                data_queue.put(frame, block=False)

                consecutive_errors = 0
            except queue.Full:
                pass
            except ReplayComplete as e:
                data_queue.put({"replay_complete": str(e)})
                break
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= self._MAX_TOLERATED_ERRORS:
                    data_queue.put(
                        {
                            "fatal_error": f"Target lost after "
                            f"{self._MAX_TOLERATED_ERRORS} retries: {e}"
                        }
                    )
                    break
                data_queue.put(
                    {
                        "error": f"Transient read fault "
                        f"({consecutive_errors}/{self._MAX_TOLERATED_ERRORS}): {e}"
                    },
                    block=False,
                )
            finally:
                if in_batch:
                    with contextlib.suppress(Exception):
                        self._m_scraper.end_batch()

            time.sleep(inspection_period)

    def _read_cpu_cycles_delta(self) -> int:
        """Return the cycles elapsed since the last frame, or -1 when usage tracking is off."""
        if not self.has_usage:
            return -1

        try:
            current = self._m_scraper.read32(self._cpu_usage_address)[0]
        except Exception as e:
            raise RuntimeError(f"Error reading global CPU cycles: {e}") from e

        delta = current - self.last_cpu_cycles
        self.last_cpu_cycles = current
        if delta > 0:
            self.last_cpu_delta = delta
            return delta
        # current_cpu_cycles wrapped or reset; keep last positive delta.
        return self.last_cpu_delta

    def _poll_threads(self, cpu_cycles_delta: int) -> tuple[list[dict], int]:
        """
        Per-thread loop: read usage + watermark, accumulate raw polling state.
        The idle thread's usage delta is folded back into ``cpu_cycles_delta``,
        so the second return value may differ from the input.
        """
        if self.thread_pool is None:
            raise RuntimeError("Thread pool is uninitialized.")

        polled: list[dict] = []
        for thread in self.thread_pool:
            if self.has_usage:
                try:
                    thread_usage = self._m_scraper.read64(
                        thread.address + self._layout.thread_usage
                    )[0]
                except Exception as e:
                    raise RuntimeError(f"Error polling thread usage {thread.name}: {e}") from e

                usage_delta = thread_usage - self.last_thread_cycles.get(thread.address, 0)
                self.last_thread_cycles[thread.address] = thread_usage

                if thread.address == self.idle_threads_address:
                    cpu_cycles_delta += usage_delta

                is_active = usage_delta > 0
            else:
                usage_delta = 0
                is_active = False

            try:
                watermark = self._m_scraper.calculate_dynamic_watermark(
                    thread.stack_start,
                    thread.stack_size,
                    thread_id=thread.address,
                )
            except Exception as e:
                raise RuntimeError(f"Error polling stack watermark for {thread.name}: {e}") from e

            stack_usage_pct = (
                (watermark / thread.stack_size * 100) if thread.stack_size > 0 else 0.0
            )

            polled.append(
                {
                    "info": thread,
                    "usage_delta": usage_delta,
                    "is_active": is_active,
                    "watermark": watermark,
                    "stack_usage_pct": stack_usage_pct,
                }
            )

        return polled, cpu_cycles_delta

    def _finalize_threads(self, polled: list[dict], cpu_cycles_delta: int) -> list[ThreadInfo]:
        """Compute per-thread CPU% / relative load and emit the final ThreadInfo list."""
        idle = next(
            (d for d in polled if d["info"].address == self.idle_threads_address),
            None,
        )
        active_total = (cpu_cycles_delta - idle["usage_delta"]) if idle else cpu_cycles_delta

        final: list[ThreadInfo] = []
        for data in polled:
            absolute_cpu = (
                (data["usage_delta"] / cpu_cycles_delta * 100) if cpu_cycles_delta > 0 else 0.0
            )

            if data["info"].address == self.idle_threads_address:
                load_pct = 0.0
            elif active_total > 0:
                load_pct = min((data["usage_delta"] / active_total) * 100.0, 100.0)
            else:
                load_pct = 0.0

            runtime = ThreadRuntime(
                cpu=load_pct,
                cpu_normalized=absolute_cpu,
                active=data["is_active"],
                stack_watermark=data["watermark"],
                stack_watermark_percent=data["stack_usage_pct"],
            )
            final.append(
                ThreadInfo(
                    address=data["info"].address,
                    stack_start=data["info"].stack_start,
                    stack_size=data["info"].stack_size,
                    name=data["info"].name,
                    runtime=runtime,
                )
            )
        return final

    def _poll_heaps(self, data_queue: queue.Queue) -> list[HeapInfo]:
        """
        Read each k_heap's metadata and emit a HeapInfo list. Per-heap read
        failures and fragmentation-walk failures are reported as non-fatal
        ``error`` frames on ``data_queue`` and skip that heap.
        """
        heaps: list[HeapInfo] = []
        for heap_name, heap_addresses in self._k_heap_addresses.items():
            for heap_address in heap_addresses:
                try:
                    heap_struct = self._m_scraper.read32(heap_address)[0]
                    free = self._m_scraper.read32(heap_struct + self._layout.heap_free_bytes)[0]
                    allocated = self._m_scraper.read32(
                        heap_struct + self._layout.heap_allocated_bytes
                    )[0]
                    max_allocated = self._m_scraper.read32(
                        heap_struct + self._layout.heap_max_allocated_bytes
                    )[0]
                except Exception as e:
                    data_queue.put(
                        {"error": f"Error reading heap info for {heap_name}: {e}"},
                        block=False,
                    )
                    continue

                chunks = None
                if self.extra_info_heap_address == heap_struct:
                    try:
                        chunks = self.get_heap_fragmentation(heap_struct)
                    except Exception as e:
                        data_queue.put(
                            {"error": f"Error reading sparsity for {heap_name}: {e}"},
                            block=False,
                        )

                total = allocated + free
                usage_pct = (allocated / total * 100) if total > 0 else 0.0

                heaps.append(
                    HeapInfo(
                        heap_name,
                        heap_struct,
                        free,
                        allocated,
                        max_allocated,
                        usage_pct,
                        chunks,
                    )
                )
        return heaps
