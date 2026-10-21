# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""ZScraper: backend + ELF + kernel walkers + polling thread."""

import contextlib
import logging
import queue
import threading
import time
from threading import Event
from typing import Literal

from zview.backend.base import AbstractScraper, ThreadInfo, ThreadRuntime
from zview.backend.elf_inspector import ElfInspector
from zview.backend.replay import ReplayComplete, ReplayMismatch
from zview.kernel import compat
from zview.kernel.group import Group
from zview.kernel.groups import GROUPS
from zview.kernel.layout import KernelLayout
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.kernel.threads import walk_static_threads, walk_thread_list
from zview.kernel.usage import read_usage_snapshot
from zview.selection import Selection

logger = logging.getLogger("zview.scraper")


class _BrokenFrame(Exception):
    """Raised when a poll cycle observes a counter regression."""


class ZScraper:
    last_thread_cycles: dict[int, int]
    # What the session reads, and what the view on screen needs: a frame reads
    # what both allow. The TUI sets ``view`` on a live target only, dump and
    # record leave it reading everything.
    selection: Selection = Selection()
    view: Selection = Selection()

    def __init__(
        self,
        meta_scraper: AbstractScraper,
        elf_path,
        max_threads: int = 64,
    ):
        self._all_threads_info: dict[str, ThreadInfo] = {}
        self._m_scraper: AbstractScraper = meta_scraper
        self._polling_thread: threading.Thread | None = None
        self._thread_pool: list[ThreadInfo] | None = None
        self._stop_event: Event | None = None
        self.inspection_period = 0.2
        self._MAX_THREADS: int = max_threads

        self._load_elf(elf_path)

    def reload_elf(self) -> None:
        """Re-parse the ELF at the same path and rebuild everything derived from it."""
        self._load_elf(self.elf_path)

    def _load_elf(self, elf_path) -> None:
        """Parse ``elf_path`` and resolve the layout, addresses and features it provides."""
        self.elf_path = elf_path
        self._elf_inspector: ElfInspector = ElfInspector(elf_path)
        self._endianess: Literal["little", "big"] = (
            "little" if self._elf_inspector._little_endian else "big"
        )
        self._m_scraper.endianess = "<" if self._endianess == "little" else ">"

        self.has_usage: bool = True
        self.has_names: bool = True
        self.has_thread_list: bool = True
        self.has_stack_info: bool = True
        # One of each group per ELF, in the order a frame reads them, and per
        # group key whether it is read and the rows its discover found.
        self.groups: dict[str, Group] = {cls.key: cls() for cls in GROUPS}
        self.available: dict[str, bool] = {}
        self.rows: dict[str, dict] = {}
        # Every group's parts start on, and a replay keeps those its recording names.
        self.parts: set[str] = {part for group in self.groups.values() for part in group.parts}
        # The thread metadata bytes in one read, and the entry pointer read
        # once per thread. A recording taken before that holds one read per
        # field per frame, so replaying it keeps asking that way.
        self.merged_thread_meta: bool = True
        self._thread_entry_cache: dict[int, int] = {}
        # Every thread's usage counter read first, as of one instant. A
        # recording taken before that read the kernel-wide counter first and
        # each thread's own beside its stack.
        self.usage_snapshot: bool = True

        # Built Zephyr version (when the ELF is in its build tree) and the wait
        # queue layout.
        self.zephyr_version: str | None = compat.detect_zephyr_version(elf_path)
        self.waitq_flavor: compat.WaitQFlavor = compat.waitq_flavor(self._elf_inspector)
        # Without the fill pattern there is nothing for the watermark scan to find.
        self.has_stack_fill: bool = compat.has_kconfig(
            self._elf_inspector, compat.CONFIG_INIT_STACKS
        )

        self._layout = self._resolve_layout()
        self._resolve_addresses()

        self._discover_groups()
        self._restrict_features_to_recording()

        if not self._m_scraper.is_live:
            self.capture_all()

        self.reset_runtime_state()

    def _resolve_layout(self) -> KernelLayout:
        """Resolve DWARF-derived offsets. Sets ``has_*`` flags for missing features."""
        elf = self._elf_inspector

        fields: dict = {}

        for group, flag in (
            (compat.THREAD_FIELDS, "has_thread_list"),
            (compat.THREAD_STACK_FIELDS, "has_stack_info"),
            (compat.THREAD_NAME_FIELDS, "has_names"),
            (compat.USAGE_FIELDS, "has_usage"),
        ):
            if (extras := compat.resolve_fields(elf, group)) is not None:
                fields.update(extras)
            else:
                setattr(self, flag, False)

        for group in self.groups.values():
            required = [compat.resolve_fields(elf, table) for table in group.required]
            self.available[group.key] = None not in required
            for found in (t for t in required if t is not None):
                fields.update(found)
            for table in group.optional:
                fields.update(compat.resolve_fields(elf, table) or {})

        for table in (
            compat.THREAD_PRIORITY_FIELDS,
            compat.THREAD_STATE_FIELDS,
            compat.THREAD_OPTIONS_FIELDS,
            compat.THREAD_ENTRY_FIELDS,
            compat.THREAD_QNODE_FIELDS,
            compat.USAGE_STAMP_FIELDS,
        ):
            fields.update(compat.resolve_fields(elf, table) or {})

        return KernelLayout(**fields)

    def _resolve_addresses(self) -> None:
        elf = self._elf_inspector
        self._kernel_base_address = elf.get_symbol_info("_kernel", "address")[0]
        if self.has_thread_list:
            self._threads_address = self._kernel_base_address + self._layout.threads_head
        # z_idle_threads holds one idle thread per CPU.
        idle = elf.get_symbol_info("z_idle_threads", "address")[0]
        thread_size = elf.get_struct_size("k_thread")
        cpus = max(1, elf.get_symbol_info("z_idle_threads", "size")[0] // thread_size)
        self.idle_thread_addresses = tuple(idle + i * thread_size for i in range(cpus))

        self._usage_stamps: tuple[int, ...] = ()
        if self.has_usage:
            self._cpu_usage_address = self._kernel_base_address + self._layout.cpu_usage
            if self._layout.cpu_usage0 is not None:
                cpu_size = elf.get_struct_size("_cpu")
                self._usage_stamps = tuple(
                    self._kernel_base_address
                    + self._layout.kernel_cpus
                    + i * cpu_size
                    + self._layout.cpu_usage0
                    for i in range(cpus)
                )

    def _discover_groups(self) -> None:
        """Find each available group's rows; a group with none is not read."""
        session = self._session()
        for group in self.groups.values():
            found = group.discover(session) if self.available[group.key] else {}
            self.rows[group.key] = found
            self.available[group.key] = bool(found)

    def _session(self) -> Session:
        """What the groups read with, as of now."""
        return Session(
            scraper=self._m_scraper,
            elf=self._elf_inspector,
            layout=self._layout,
            waitq_flavor=self.waitq_flavor,
            thread_names=self._thread_names(),
            parts=frozenset(self.parts),
        )

    def capture_all(self) -> None:
        """Read every detail a view may ask for, as a recording holds them all."""
        for group in self.groups.values():
            group.capture_all()

    def _restrict_features_to_recording(self) -> None:
        """
        Limit a replayed session to the features its recording captured.

        Replay matches a strict read sequence. A recording with no feature list
        predates the primitives and is treated as threads and heaps only.
        """
        if self._m_scraper.is_live:
            return

        features = getattr(self._m_scraper, "features", None)
        if features is None:
            return

        for group in self.groups.values():
            self.available[group.key] = self.available[group.key] and group.key in features
        self.parts &= set(features)
        self.merged_thread_meta = "merged_thread_meta" in features
        # A recording from before the scan was skipped scanned every stack.
        self.has_stack_fill = self.has_stack_fill or "unfilled_stacks" not in features
        self.usage_snapshot = "usage_snapshot" in features

    def active_features(self) -> tuple[str, ...]:
        """The features this session polls, as recorded in a recording header."""
        features = ["threads"]
        if self.merged_thread_meta:
            features.append("merged_thread_meta")
        # A group read, then each of its parts that is on.
        for group in self.groups.values():
            if self.available[group.key]:
                features += [group.key, *(part for part in group.parts if part in self.parts)]
        for enabled, name in (
            (not self.has_stack_fill, "unfilled_stacks"),
            (self.has_usage and self.usage_snapshot, "usage_snapshot"),
        ):
            if enabled:
                features.append(name)

        return tuple(features)

    def absent_features(self) -> tuple[str, ...]:
        """One notice per capability the build's Kconfig leaves out."""
        return tuple(
            notice
            for enabled, notice in (
                (self.has_names, "Warning: no thread names (CONFIG_THREAD_NAME=n)"),
                (self.has_usage, "Warning: no cpu stats (CONFIG_THREAD_RUNTIME_STATS=n)"),
                (
                    # Only a build that declares heaps misses their stats.
                    self._layout.heap_free_bytes is not None
                    or not self._elf_inspector.find_struct_instances("k_heap"),
                    "Warning: no heap stats (CONFIG_SYS_HEAP_RUNTIME_STATS=n)",
                ),
                (
                    self.has_thread_list,
                    "Warning: static threads only (CONFIG_THREAD_MONITOR=n)",
                ),
                (self.has_stack_info, "Warning: no stack sizes (CONFIG_THREAD_STACK_INFO=n)"),
                (self.has_stack_fill, "Warning: no stack usage (CONFIG_INIT_STACKS=n)"),
            )
            if not enabled
        )

    def decl_site(self, address: int) -> tuple[str, int] | None:
        """``(path, line)`` where the object at ``address`` is declared, if known."""
        return self._elf_inspector.decl_site(address)

    def function_site(self, address: int) -> tuple[str, int] | None:
        """``(path, line)`` where the function at ``address`` is defined, if known."""
        return self._elf_inspector.function_site(address)

    def has_kernel_objects(self) -> bool:
        """True while any object group is live, so the objects view has rows to draw."""
        return any(self.available.values())

    def _reads_group(self, group: str) -> bool:
        """Both the session's selection and the view's read ``group``."""
        return self.selection.reads_group(group) and self.view.reads_group(group)

    def _reads(self, group: str, name: str) -> bool:
        """Both the session's selection and the view's read the row ``name``."""
        return self.selection.reads(group, name) and self.view.reads(group, name)

    def _poll_kernel_objects(self) -> dict:
        """
        Read the kernel objects. A failed read raises, and the frame is dropped.

        Wait queues are walked only on the ``simple`` layout; otherwise
        ``waiters`` is ``None``.
        """
        frame: dict = {}
        session = self._session()
        for group in self.groups.values():
            if not self.available[group.key] or not self._reads_group(group.key):
                continue

            # Matched against the labels the rows will carry.
            selected = {
                label: [address]
                for label, address in label_instances(self.rows[group.key])
                if self._reads(group.key, label)
            }
            frame[group.key] = list(group.walk(session, selected).values())

        return frame

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
        if self.has_thread_list:
            threads = walk_thread_list(
                self._m_scraper,
                self._elf_inspector,
                self._threads_address,
                self._layout,
                self._endianess,
                self.has_names,
                self._MAX_THREADS,
            )
        else:
            # Without the kernel's list, the threads are the ones allocated statically.
            threads = walk_static_threads(
                self._m_scraper,
                self._elf_inspector,
                self._elf_inspector.find_nested_instances("k_thread"),
                self._layout,
                self._endianess,
                self.has_names,
                self._MAX_THREADS,
            )
        self._all_threads_info.clear()
        self._all_threads_info.update(threads)

    def reset_thread_pool(self):
        self.thread_pool = list(self.all_threads.values())
        # A rebuilt list can hand the same address to a different thread.
        self._thread_entry_cache.clear()

    def reset_runtime_state(self) -> None:
        """Clear watermark/cycle caches; baselines are re-seeded by the next poll."""
        self._m_scraper.watermark_cache.clear()
        self.last_thread_cycles = {}

    def start_polling_thread(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        inspection_period: float,
    ):
        """Start the polling daemon thread. No-op if one is already running."""
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
        """Polling thread entry point. Tolerates ``_MAX_TOLERATED_ERRORS`` transient faults."""
        try:
            if not self._m_scraper.is_connected:
                self._m_scraper.connect()
        except Exception as e:
            data_queue.put({"fatal_error": f"Failed to connect to target: {e}"})
            return

        consecutive_errors = 0
        while not stop_event.is_set():
            cycle_start = time.perf_counter()
            in_batch = False
            try:
                self._m_scraper.begin_batch()
                in_batch = True

                deltas = self._read_usage_deltas()
                polled_raw = self._poll_threads(deltas)
                final_threads = self._finalize_threads(polled_raw, deltas)

                frame: dict = {"threads": final_threads}
                frame.update(self._poll_kernel_objects())
                data_queue.put(frame, block=False)

                consecutive_errors = 0
            except queue.Full:
                pass
            except ReplayComplete as e:
                data_queue.put({"replay_complete": str(e)})
                break
            except _BrokenFrame as e:
                logger.warning("Broken frame captured (%s); resyncing baselines.", e)
                self.reset_runtime_state()
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
                # One message per fault, not one per attempt: the message log
                # counts the repeats, which is the retry number.
                data_queue.put({"error": f"Transient read fault: {e}"}, block=False)
            finally:
                if in_batch:
                    with contextlib.suppress(Exception):
                        self._m_scraper.end_batch()

            # Sleep only the remainder of the period, so the poll cadence tracks
            # the requested period instead of period-plus-read-time. When the
            # reads already exceed the period, poll back-to-back (no sleep).
            elapsed = time.perf_counter() - cycle_start
            time.sleep(max(0.0, inspection_period - elapsed))

    def _usage_delta(self, address: int, cycles: int) -> int:
        """A thread's cycles since the last frame, 0 on its first. A regression breaks the frame."""
        prev = self.last_thread_cycles.get(address)
        self.last_thread_cycles[address] = cycles
        if prev is None:
            return 0
        if cycles < prev:
            raise _BrokenFrame(f"usage counter of the thread at 0x{address:X} regressed")
        return cycles - prev

    def _read_usage_deltas(self) -> dict[int, int]:
        """
        Every known thread's cycles since the last frame, ``{address: cycles}``.

        Empty when the build counts no usage or the view polls no thread, and on
        a recording from before the snapshot, whose counters sit beside each
        thread's stack.
        """
        if not self.has_usage:
            return {}

        if not self.usage_snapshot:
            # The kernel-wide counter that recording read first. Its value is unused.
            self._m_scraper.read32(self._cpu_usage_address)
            return {}

        # A muted thread keeps its counter, so the others' shares stay whole.
        if not self.thread_pool or not self._reads_group("threads"):
            return {}

        try:
            counters = read_usage_snapshot(
                self._m_scraper,
                [t.address for t in self.all_threads.values()],
                self._layout.thread_usage,
                self._usage_stamps,
            )
        except Exception as e:
            raise RuntimeError(f"Error polling thread usage: {e}") from e

        return {address: self._usage_delta(address, cycles) for address, cycles in counters.items()}

    def _poll_threads(self, deltas: dict[int, int]) -> list[dict]:
        """Read each polled thread's watermark and metadata, beside its share of ``deltas``."""
        if self.thread_pool is None:
            raise RuntimeError("Thread pool is uninitialized.")

        polled: list[dict] = []
        for thread in self.thread_pool:
            if not self._reads("threads", thread.name):
                continue
            usage_delta = deltas.get(thread.address, 0)
            if self.has_usage and not self.usage_snapshot:
                try:
                    cycles = self._m_scraper.read64(thread.address + self._layout.thread_usage)[0]
                except Exception as e:
                    raise RuntimeError(f"Error polling thread usage {thread.name}: {e}") from e
                usage_delta = self._usage_delta(thread.address, cycles)
            is_active = usage_delta > 0

            watermark = stack_usage_pct = None
            if self.has_stack_fill and thread.stack_size is not None:
                try:
                    watermark = self._m_scraper.calculate_dynamic_watermark(
                        thread.stack_start,
                        thread.stack_size,
                        thread_id=thread.address,
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"Error polling stack watermark for {thread.name}: {e}"
                    ) from e

                stack_usage_pct = (
                    (watermark / thread.stack_size * 100) if thread.stack_size > 0 else 0.0
                )

            meta = self._read_thread_meta(thread)

            polled.append(
                {
                    "info": thread,
                    "usage_delta": usage_delta,
                    "is_active": is_active,
                    "watermark": watermark,
                    "stack_usage_pct": stack_usage_pct,
                    "meta": meta,
                }
            )

        return polled

    def _read_thread_meta(self, thread: ThreadInfo) -> dict:
        """Read priority/state/options/entry for a thread; missing offsets yield ``None``."""
        meta: dict = {
            "priority": None,
            "state": None,
            "user_options": None,
            "entry_point": None,
            "entry_symbol": None,
        }
        try:
            meta.update(
                self._read_meta_span(thread)
                if self.merged_thread_meta
                else self._read_meta_fields(thread)
            )
            entry = self._read_thread_entry(thread)
        except ReplayMismatch:
            # A recording taken before the metadata was read has no such reads.
            # A drift leaves the cursor untouched, so the frame goes on.
            return meta
        except Exception as e:
            raise RuntimeError(f"Error polling metadata for {thread.name}: {e}") from e
        if entry is not None:
            meta["entry_point"] = entry
            meta["entry_symbol"] = self._resolve_entry_symbol(entry)
        return meta

    # The metadata bytes, by their offset in the layout.
    _META_BYTES = (
        ("priority", "thread_priority"),
        ("state", "thread_state"),
        ("user_options", "thread_user_options"),
    )

    def _read_meta_span(self, thread: ThreadInfo) -> dict:
        """The metadata bytes in one read: they sit within a few bytes of each other."""
        offsets = {
            name: offset
            for name, attribute in self._META_BYTES
            if (offset := getattr(self._layout, attribute)) is not None
        }
        if not offsets:
            return {}

        base = min(offsets.values())
        raw = self._m_scraper.read8(thread.address + base, max(offsets.values()) - base + 1)

        values = {name: raw[offset - base] for name, offset in offsets.items()}
        if "priority" in values:
            values["priority"] = self._signed_priority(values["priority"])
        return values

    def _read_meta_fields(self, thread: ThreadInfo) -> dict:
        """The metadata bytes one read apiece, as a recording made that way expects."""
        values = {}
        for name, attribute in self._META_BYTES:
            offset = getattr(self._layout, attribute)
            if offset is not None:
                values[name] = self._m_scraper.read8(thread.address + offset)[0]
        if "priority" in values:
            values["priority"] = self._signed_priority(values["priority"])
        return values

    @staticmethod
    def _signed_priority(raw: int) -> int:
        """A Zephyr priority is a signed 8-bit value: cooperative ones are negative."""
        return raw - 256 if raw >= 0x80 else raw

    def _read_thread_entry(self, thread: ThreadInfo) -> int | None:
        """The entry pointer, read once per thread: it is fixed while the thread lives."""
        if self._layout.thread_entry is None:
            return None
        if self.merged_thread_meta and thread.address in self._thread_entry_cache:
            return self._thread_entry_cache[thread.address]

        entry = self._m_scraper.read32(thread.address + self._layout.thread_entry)[0]
        self._thread_entry_cache[thread.address] = entry
        return entry

    def _resolve_entry_symbol(self, addr: int) -> str | None:
        """Cache-backed reverse symbol lookup for a function pointer."""
        if addr == 0:
            return None
        cache = getattr(self, "_entry_symbol_cache", None)
        if cache is None:
            self._entry_symbol_cache = cache = {}
        if addr in cache:
            return cache[addr]
        # Thumb code pointers have bit 0 set; mask before lookup.
        try:
            sym = self._elf_inspector.get_symbol_name_at(addr & ~1)
        except Exception:
            sym = None
        cache[addr] = sym
        return sym

    def _finalize_threads(self, polled: list[dict], deltas: dict[int, int]) -> list[ThreadInfo]:
        """
        Build ``ThreadInfo`` objects from raw polling state, computing CPU% and load%.

        Both are shares of the threads' own cycles: CPU% of every thread's, load%
        of every busy thread's. Read as one snapshot or not, the shares add up
        to 100% and none passes it.
        """
        # A recording from before the snapshot has counters for the polled threads only.
        deltas = deltas or {d["info"].address: d["usage_delta"] for d in polled}
        total = sum(deltas.values())
        busy = total - sum(deltas.get(address, 0) for address in self.idle_thread_addresses)

        final: list[ThreadInfo] = []
        for data in polled:
            absolute_cpu = (data["usage_delta"] / total * 100.0) if total > 0 else 0.0

            if data["info"].address in self.idle_thread_addresses:
                load_pct = 0.0
            elif busy > 0:
                load_pct = data["usage_delta"] / busy * 100.0
            else:
                load_pct = 0.0

            runtime = ThreadRuntime(
                cpu=load_pct if self.has_usage else None,
                cpu_normalized=absolute_cpu if self.has_usage else None,
                active=data["is_active"],
                stack_watermark=data["watermark"],
                stack_watermark_percent=data["stack_usage_pct"],
            )
            meta = data.get("meta") or {}
            final.append(
                ThreadInfo(
                    address=data["info"].address,
                    stack_start=data["info"].stack_start,
                    stack_size=data["info"].stack_size,
                    name=data["info"].name,
                    runtime=runtime,
                    priority=meta.get("priority"),
                    state=meta.get("state"),
                    user_options=meta.get("user_options"),
                    entry_point=meta.get("entry_point"),
                    entry_symbol=meta.get("entry_symbol"),
                )
            )
        return final

    def _thread_names(self) -> dict[int, str]:
        """``{address: name}`` for the threads a wait queue can name."""
        return {t.address: t.name for t in self._all_threads_info.values()}
