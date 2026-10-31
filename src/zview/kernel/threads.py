# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_thread`` group, with its thread list and static thread walkers."""

from collections import deque
from itertools import islice
from typing import Literal

from zview.backend.base import AbstractScraper, ThreadInfo, ThreadRuntime
from zview.backend.elf_inspector import ElfInspector
from zview.backend.replay import ReplayMismatch
from zview.kernel.compat import (
    CONFIG_INIT_STACKS,
    CONFIG_STACK_SENTINEL,
    FieldTable,
    has_kconfig,
    resolve_size,
)
from zview.kernel.group import BrokenFrame, Group
from zview.kernel.layout import KernelLayout
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.kernel.usage import read_usage_snapshot

# thread_state bits of an object that is not a live thread: the placeholder
# the kernel switches away from at boot, and a thread that has exited.
_THREAD_DUMMY = 1 << 0
_THREAD_DEAD = 1 << 3

# K_THREAD_DEFINE names its object after the thread's id.
_DEFINED_THREAD_PREFIX = "_k_thread_obj_"

# The most frames the CPU and load shares can be summed over.
USAGE_WINDOW_MAX = 64

# CONFIG_THREAD_MONITOR: the kernel's list of every thread.
THREAD_FIELDS: FieldTable = {
    "threads_head": ((("z_kernel", "threads"),),),
    "thread_next": ((("k_thread", "next_thread"),),),
}

# CONFIG_THREAD_STACK_INFO.
THREAD_STACK_FIELDS: FieldTable = {
    "stack_start": ((("k_thread", "stack_info"), ("_thread_stack_info", "start")),),
    "stack_size": ((("k_thread", "stack_info"), ("_thread_stack_info", "size")),),
}

# CONFIG_THREAD_NAME.
THREAD_NAME_FIELDS: FieldTable = {
    "thread_name": ((("k_thread", "name"),),),
}

# CONFIG_THREAD_RUNTIME_STATS. Each thread's cycle counter, from v3.3.0.
USAGE_FIELDS: FieldTable = {
    "thread_usage": (
        (("k_thread", "base"), ("_thread_base", "usage"), ("k_cycle_stats", "total")),
    ),
}

# CONFIG_SCHED_THREAD_USAGE_ALL. The kernel-wide counter, from v3.5.0, which a
# recording from before the usage snapshot read first.
KERNEL_USAGE_FIELDS: FieldTable = {
    "cpu_usage": ((("z_kernel", "usage"),),),
}

# CONFIG_SCHED_THREAD_USAGE. Each CPU's switch stamp, rewritten whenever the
# kernel adds to a usage counter.
USAGE_STAMP_FIELDS: FieldTable = {
    "kernel_cpus": ((("z_kernel", "cpus"),),),
    "cpu_usage0": ((("_cpu", "usage0"),),),
}

# Metadata mirroring `kernel threads list`, one table per member.
THREAD_PRIORITY_FIELDS: FieldTable = {
    "thread_priority": ((("k_thread", "base"), ("_thread_base", "prio")),),
}

THREAD_STATE_FIELDS: FieldTable = {
    "thread_state": ((("k_thread", "base"), ("_thread_base", "thread_state")),),
}

THREAD_OPTIONS_FIELDS: FieldTable = {
    "thread_user_options": ((("k_thread", "base"), ("_thread_base", "user_options")),),
}

THREAD_ENTRY_FIELDS: FieldTable = {
    "thread_entry": (
        (("k_thread", "entry"), ("__thread_entry", "pEntry")),
        # Single-underscore struct name carried by some vendor trees.
        (("k_thread", "entry"), ("_thread_entry", "pEntry")),
    ),
}

# Wait queues hold ``_thread_base.qnode_dlist`` nodes; a thread address is a
# node address minus this offset.
THREAD_QNODE_FIELDS: FieldTable = {
    "thread_qnode": ((("k_thread", "base"), ("_thread_base", "qnode_dlist")),),
}


def _word_at(words, offset: int | None) -> int | None:
    """The struct word at byte ``offset``, or None when the build has no such member."""
    return None if offset is None else words[offset // 4]


def _byte_at(words, offset: int, endianess: Literal["little", "big"]) -> int:
    """The struct byte at ``offset``."""
    return words[offset // 4].to_bytes(4, endianess)[offset % 4]


def _name_at(words, offset: int, endianess: Literal["little", "big"]) -> str:
    """The NUL-terminated string starting at byte ``offset`` of the struct."""
    raw = b"".join(w.to_bytes(4, endianess) for w in words[offset // 4 :])
    return raw.split(b"\0", 1)[0].decode(errors="ignore")


def walk_thread_list(
    scraper: AbstractScraper,
    elf: ElfInspector,
    threads_head_address: int,
    layout: KernelLayout,
    endianess: Literal["little", "big"],
    has_names: bool,
    max_threads: int = 64,
    *,
    is_distinct: bool = True,
) -> dict[str, ThreadInfo]:
    """
    Walk the kernel thread linked list and return ``{name: ThreadInfo}``.
    Wrapped in a single ``begin_batch``/``end_batch``. Raises ``RuntimeError``,
    after the batch ends, when a read fails.

    With ``is_distinct``, a thread with no name, or with one already taken, is
    named after its address. Without it, the later thread replaces the
    earlier, as a recording made before the ``distinct_names`` part reads.
    """
    try:
        if not scraper.is_connected:
            scraper.connect()

        scraper.begin_batch()
        thread_ptr = scraper.read32(threads_head_address)[0] if threads_head_address else 0
    except Exception as e:
        scraper.end_batch()
        raise RuntimeError("Unable to read kernel thread list.") from e

    stack_struct_size = elf.get_struct_size("k_thread")
    words_to_read = stack_struct_size // 4
    next_ptr_word_idx = layout.thread_next // 4

    threads: dict[str, ThreadInfo] = {}
    for _ in range(max_threads):
        if thread_ptr == 0:
            break

        try:
            thread_struct_words = scraper.read32(thread_ptr, words_to_read)
        except Exception as e:
            scraper.end_batch()
            raise RuntimeError(f"Error reading thread struct at 0x{thread_ptr:X}") from e

        if has_names:
            thread_name = _name_at(thread_struct_words, layout.thread_name, endianess)
        else:
            thread_name = f"thread @ 0x{thread_ptr:X}"
        if is_distinct:
            thread_name = thread_name or f"thread @ 0x{thread_ptr:X}"
            if thread_name in threads:
                thread_name = f"{thread_name} @ 0x{thread_ptr:X}"

        threads[thread_name] = ThreadInfo(
            thread_ptr,
            _word_at(thread_struct_words, layout.stack_start),
            _word_at(thread_struct_words, layout.stack_size),
            thread_name,
            None,
        )

        thread_ptr = thread_struct_words[next_ptr_word_idx]

    scraper.end_batch()
    return threads


def walk_static_threads(
    scraper: AbstractScraper,
    elf: ElfInspector,
    objects: dict[str, list[int]],
    layout: KernelLayout,
    endianess: Literal["little", "big"],
    has_names: bool,
    max_threads: int = 64,
) -> dict[str, ThreadInfo]:
    """
    Read the statically allocated ``k_thread`` objects, ``{path: addresses}``, and
    return the live ones as ``{name: ThreadInfo}``, for a build without the
    kernel's thread list. Wrapped in a single ``begin_batch``/``end_batch``.
    Raises ``RuntimeError``, after the batch ends, when a read fails.

    An object still all zero was never created. A thread the build does not name
    is named after its object.
    """
    if not scraper.is_connected:
        scraper.connect()
    scraper.begin_batch()

    words_to_read = elf.get_struct_size("k_thread") // 4
    threads: dict[str, ThreadInfo] = {}
    for path, address in ((p, a) for p, addresses in objects.items() for a in addresses):
        if len(threads) == max_threads:
            break

        try:
            words = scraper.read32(address, words_to_read)
        except Exception as e:
            scraper.end_batch()
            raise RuntimeError(f"Error reading thread struct at 0x{address:X}") from e

        state = (
            0 if layout.thread_state is None else _byte_at(words, layout.thread_state, endianess)
        )
        if not any(words) or state & (_THREAD_DUMMY | _THREAD_DEAD):
            continue

        name = _name_at(words, layout.thread_name, endianess) if has_names else ""
        name = name or path.removeprefix(_DEFINED_THREAD_PREFIX)
        if name in threads:
            name = f"{name} @ 0x{address:X}"

        threads[name] = ThreadInfo(
            address,
            _word_at(words, layout.stack_start),
            _word_at(words, layout.stack_size),
            name,
            None,
        )

    scraper.end_batch()
    return threads


class Threads(Group):
    key = "threads"
    struct = "k_thread"
    optional = (
        THREAD_FIELDS,
        THREAD_STACK_FIELDS,
        THREAD_NAME_FIELDS,
        USAGE_FIELDS,
        KERNEL_USAGE_FIELDS,
        USAGE_STAMP_FIELDS,
        THREAD_PRIORITY_FIELDS,
        THREAD_STATE_FIELDS,
        THREAD_OPTIONS_FIELDS,
        THREAD_ENTRY_FIELDS,
        THREAD_QNODE_FIELDS,
    )
    # The metadata bytes in one read and the entry pointer once per thread,
    # the stack scan skipped on a build that leaves its stacks unfilled,
    # every thread's usage counter read first, as of one instant, the scan
    # started above the stack sentinel, threads that share a name kept apart,
    # the threads' counters read on a build without the kernel-wide one, and
    # those counters read in one pass however the stamps moved. A recording
    # taken before each holds those reads the way they were made then.
    parts = (
        "merged_thread_meta",
        "unfilled_stacks",
        "usage_snapshot",
        "stack_sentinel",
        "distinct_names",
        "thread_usage_alone",
        "single_usage_pass",
    )
    max_threads = 64

    # The metadata bytes, by their offset in the layout.
    _META_BYTES = (
        ("priority", "thread_priority"),
        ("state", "thread_state"),
        ("user_options", "thread_user_options"),
    )

    def __init__(self) -> None:
        # The discovered threads by address, one idle thread per CPU, and each
        # thread's usage counter as of the last frame.
        self.infos: dict[int, ThreadInfo] = {}
        self.idle_addresses: tuple[int, ...] = ()
        self.last_cycles: dict[int, int] = {}
        # The frames the CPU and load shares are summed over, and the usage
        # deltas of the last ones.
        self.usage_window = 1
        self._window: deque[dict[int, int]] = deque(maxlen=USAGE_WINDOW_MAX)
        self._entries: dict[int, int] = {}
        self._symbols: dict[int, str | None] = {}
        self._cpu_usage_address = 0
        self._usage_stamps: tuple[int, ...] = ()
        # The metadata members wider than a byte, and the byte order they are in.
        self._meta_sizes: dict[str, int] = {}
        self._byteorder: Literal["little", "big"] = "little"

    @staticmethod
    def _counts_usage(session: Session) -> bool:
        """
        The threads' cycle counters are read: the build has them and, unless
        the session reads them alone, the kernel-wide counter beside them.
        """
        layout = session.layout
        return layout.thread_usage is not None and (
            layout.cpu_usage is not None or session.has_part("thread_usage_alone")
        )

    def discover(self, session: Session) -> dict[str, list[int]]:
        """
        ``{name: [address]}`` of the threads alive now: the kernel's list, or
        without it the statically allocated ``k_thread`` objects.
        """
        elf, layout = session.elf, session.layout
        kernel = elf.get_symbol_info("_kernel", "address")[0]
        # z_idle_threads holds one idle thread per CPU.
        idle = elf.get_symbol_info("z_idle_threads", "address")[0]
        thread_size = elf.get_struct_size("k_thread")
        cpus = max(1, elf.get_symbol_info("z_idle_threads", "size")[0] // thread_size)
        self.idle_addresses = tuple(idle + i * thread_size for i in range(cpus))

        if self._counts_usage(session):
            if layout.cpu_usage is not None:
                self._cpu_usage_address = kernel + layout.cpu_usage
            if layout.cpu_usage0 is not None:
                cpu_size = elf.get_struct_size("_cpu")
                self._usage_stamps = tuple(
                    kernel + layout.kernel_cpus + i * cpu_size + layout.cpu_usage0
                    for i in range(cpus)
                )

        endianess: Literal["little", "big"] = "little" if elf._is_little_endian else "big"
        # user_options is a uint8_t through v4.3.0 and a uint16_t from v4.4.0.
        options_size = resolve_size(elf, THREAD_OPTIONS_FIELDS["thread_user_options"])
        self._meta_sizes = {"user_options": options_size or 1}
        self._byteorder = endianess

        has_names = layout.thread_name is not None
        if layout.threads_head is not None:
            threads = walk_thread_list(
                session.scraper,
                elf,
                kernel + layout.threads_head,
                layout,
                endianess,
                has_names,
                self.max_threads,
                is_distinct=session.has_part("distinct_names"),
            )
        else:
            # Without the kernel's list, the threads are the ones allocated statically.
            threads = walk_static_threads(
                session.scraper,
                elf,
                elf.find_nested_instances("k_thread"),
                layout,
                endianess,
                has_names,
                self.max_threads,
            )

        self.infos = {thread.address: thread for thread in threads.values()}
        # A rebuilt list can hand the same address to a different thread.
        self._entries.clear()
        return {thread.name: [thread.address] for thread in threads.values()}

    def live_parts(self, session: Session) -> frozenset[str]:
        """
        Every part, but the scan skip on a build that fills its stacks, the
        usage snapshot on one that counts no usage, and the sentinel on one
        that places none.
        """
        off = {
            "unfilled_stacks": has_kconfig(session.elf, CONFIG_INIT_STACKS),
            "usage_snapshot": session.layout.thread_usage is None,
            "stack_sentinel": not has_kconfig(session.elf, CONFIG_STACK_SENTINEL),
        }
        return frozenset(part for part in self.parts if not off.get(part, False))

    def notices(self, session: Session) -> tuple[str, ...]:
        layout = session.layout
        return tuple(
            notice
            for enabled, notice in (
                (layout.thread_name is not None, "Warning: no thread names (CONFIG_THREAD_NAME=n)"),
                (
                    self._counts_usage(session),
                    "Warning: no cpu stats (CONFIG_THREAD_RUNTIME_STATS=n)",
                ),
                (
                    layout.threads_head is not None,
                    "Warning: static threads only (CONFIG_THREAD_MONITOR=n)",
                ),
                (
                    layout.stack_size is not None,
                    "Warning: no stack sizes (CONFIG_THREAD_STACK_INFO=n)",
                ),
                (
                    has_kconfig(session.elf, CONFIG_INIT_STACKS),
                    "Warning: no stack usage (CONFIG_INIT_STACKS=n)",
                ),
            )
            if not enabled
        )

    def reset(self) -> None:
        # Rebound, not cleared, since a poll may be reading them on its own thread.
        self.last_cycles = {}
        self._window = deque(maxlen=USAGE_WINDOW_MAX)

    def walk(self, session: Session, addresses: dict[str, list[int]]) -> dict[str, ThreadInfo]:
        """
        Read the threads at ``addresses`` and return ``{label: ThreadInfo}``.

        Every discovered thread's usage counter is read, the ones left out too,
        so each share is of all the threads' cycles. CPU% is a share of every
        thread's cycles and load% of every busy thread's: read as one snapshot
        or not, the shares add up to 100% and none passes it.
        """
        deltas, coherent = self._usage_deltas(session)
        summed = self._windowed(deltas)
        polled = [
            self._poll(session, self.infos[address], deltas, summed)
            for _, address in label_instances(addresses)
        ]
        finalized = self._finalize(session, polled, summed, coherent)
        return {thread.name: thread for thread in finalized}

    def _windowed(self, deltas: dict[int, int]) -> dict[int, int]:
        """
        Each thread's cycles over the last ``usage_window`` frames, ``deltas``
        being this frame's. Zephyr counts a thread's cycles as it is switched
        out, so a frame shorter than its run shows it idle, then the whole run.
        """
        if not deltas:
            return deltas

        self._window.append(deltas)
        summed: dict[int, int] = {}
        for frame in islice(reversed(self._window), self.usage_window):
            for address, cycles in frame.items():
                summed[address] = summed.get(address, 0) + cycles
        return summed

    def _usage_delta(self, address: int, cycles: int) -> int:
        """A thread's cycles since the last frame, 0 on its first. A regression breaks the frame."""
        prev = self.last_cycles.get(address)
        self.last_cycles[address] = cycles
        if prev is None:
            return 0
        if cycles < prev:
            raise BrokenFrame(f"usage counter of the thread at 0x{address:X} regressed")
        return cycles - prev

    def _usage_deltas(self, session: Session) -> tuple[dict[int, int], bool | None]:
        """
        Every discovered thread's cycles since the last frame, ``{address: cycles}``,
        and whether the counters hold as of one instant, None when that is unknown.

        Empty and unknown when the build counts no usage, and on a recording from
        before the snapshot, whose counters sit beside each thread's stack.
        """
        scraper, layout = session.scraper, session.layout
        if not self._counts_usage(session):
            return {}, None

        if not session.has_part("usage_snapshot"):
            # The kernel-wide counter that recording read first. Its value is unused.
            scraper.read32(self._cpu_usage_address)
            return {}, None

        if not self.infos:
            return {}, None

        try:
            counters, coherent = read_usage_snapshot(
                scraper,
                list(self.infos),
                layout.thread_usage,
                self._usage_stamps,
                # A recording from before the single pass read a moved pass again.
                attempts=1 if session.has_part("single_usage_pass") else 2,
            )
        except Exception as e:
            raise RuntimeError(f"Error polling thread usage: {e}") from e

        deltas = {
            address: self._usage_delta(address, cycles) for address, cycles in counters.items()
        }
        return deltas, coherent

    def _poll(
        self,
        session: Session,
        thread: ThreadInfo,
        deltas: dict[int, int],
        summed: dict[int, int],
    ) -> dict:
        """
        Read a thread's watermark and metadata, beside its share of ``summed``,
        active when it ran in this frame's ``deltas``.
        """
        scraper, layout = session.scraper, session.layout
        usage_delta = deltas.get(thread.address, 0)
        if self._counts_usage(session) and not session.has_part("usage_snapshot"):
            try:
                cycles = scraper.read64(thread.address + layout.thread_usage)[0]
            except Exception as e:
                raise RuntimeError(f"Error polling thread usage {thread.name}: {e}") from e
            usage_delta = self._usage_delta(thread.address, cycles)

        watermark = stack_usage_pct = None
        if not session.has_part("unfilled_stacks") and thread.stack_size is not None:
            try:
                watermark = scraper.calculate_dynamic_watermark(
                    thread.stack_start,
                    thread.stack_size,
                    thread_id=thread.address,
                    reserved_words=int(session.has_part("stack_sentinel")),
                )
            except Exception as e:
                raise RuntimeError(f"Error polling stack watermark for {thread.name}: {e}") from e

            stack_usage_pct = (
                (watermark / thread.stack_size * 100) if thread.stack_size > 0 else 0.0
            )

        return {
            "info": thread,
            # A recording from before the snapshot sums no window.
            "usage_delta": summed.get(thread.address, usage_delta),
            "is_active": usage_delta > 0,
            "watermark": watermark,
            "stack_usage_pct": stack_usage_pct,
            "meta": self._read_meta(session, thread),
        }

    def _read_meta(self, session: Session, thread: ThreadInfo) -> dict:
        """Read priority/state/options/entry for a thread; missing offsets yield ``None``."""
        meta: dict = {
            "priority": None,
            "state": None,
            "user_options": None,
            "entry_point": None,
            "entry_symbol": None,
            "entry_site": None,
        }
        try:
            meta.update(
                self._read_meta_span(session, thread)
                if session.has_part("merged_thread_meta")
                else self._read_meta_fields(session, thread)
            )
            entry = self._read_entry(session, thread)
        except ReplayMismatch:
            # A recording taken before the metadata was read has no such reads.
            # A drift leaves the cursor untouched, so the frame goes on.
            return meta
        except Exception as e:
            raise RuntimeError(f"Error polling metadata for {thread.name}: {e}") from e
        if entry is not None:
            meta["entry_point"] = entry
            meta["entry_symbol"] = self._entry_symbol(session.elf, entry)
            meta["entry_site"] = session.elf.function_site(entry)
        return meta

    def _read_meta_span(self, session: Session, thread: ThreadInfo) -> dict:
        """The metadata members in one read: they sit within a few bytes of each other."""
        offsets = {
            name: offset
            for name, attribute in self._META_BYTES
            if (offset := getattr(session.layout, attribute)) is not None
        }
        if not offsets:
            return {}

        sizes = {name: self._meta_sizes.get(name, 1) for name in offsets}
        base = min(offsets.values())
        end = max(offsets[name] + sizes[name] for name in offsets)
        raw = bytes(session.scraper.read8(thread.address + base, end - base))

        values = {
            name: int.from_bytes(raw[offset - base : offset - base + sizes[name]], self._byteorder)
            for name, offset in offsets.items()
        }
        if "priority" in values:
            values["priority"] = _signed_priority(values["priority"])
        return values

    def _read_meta_fields(self, session: Session, thread: ThreadInfo) -> dict:
        """The metadata bytes one read apiece, as a recording made that way expects."""
        values = {}
        for name, attribute in self._META_BYTES:
            offset = getattr(session.layout, attribute)
            if offset is not None:
                values[name] = session.scraper.read8(thread.address + offset)[0]
        if "priority" in values:
            values["priority"] = _signed_priority(values["priority"])
        return values

    def _read_entry(self, session: Session, thread: ThreadInfo) -> int | None:
        """The entry pointer, read once per thread: it is fixed while the thread lives."""
        if session.layout.thread_entry is None:
            return None
        if session.has_part("merged_thread_meta") and thread.address in self._entries:
            return self._entries[thread.address]

        entry = session.scraper.read32(thread.address + session.layout.thread_entry)[0]
        self._entries[thread.address] = entry
        return entry

    def _entry_symbol(self, elf: ElfInspector, addr: int) -> str | None:
        """Cache-backed reverse symbol lookup for a function pointer."""
        if addr == 0:
            return None
        if addr in self._symbols:
            return self._symbols[addr]
        # Thumb code pointers have bit 0 set; mask before lookup.
        try:
            sym = elf.get_symbol_name_at(addr & ~1)
        except Exception:
            sym = None
        self._symbols[addr] = sym
        return sym

    def _finalize(
        self,
        session: Session,
        polled: list[dict],
        deltas: dict[int, int],
        is_coherent: bool | None = None,
    ) -> list[ThreadInfo]:
        """
        Build ``ThreadInfo`` objects from raw polling state, computing CPU% and
        load%, marked ``is_coherent`` when the counters hold as of one instant.
        """
        has_usage = self._counts_usage(session)
        # A recording from before the snapshot has counters for the polled threads only.
        deltas = deltas or {d["info"].address: d["usage_delta"] for d in polled}
        total = sum(deltas.values())
        busy = total - sum(deltas.get(address, 0) for address in self.idle_addresses)

        final: list[ThreadInfo] = []
        for data in polled:
            absolute_cpu = (data["usage_delta"] / total * 100.0) if total > 0 else 0.0

            if data["info"].address in self.idle_addresses:
                load_pct = 0.0
            elif busy > 0:
                load_pct = data["usage_delta"] / busy * 100.0
            else:
                load_pct = 0.0

            runtime = ThreadRuntime(
                cpu=load_pct if has_usage else None,
                cpu_normalized=absolute_cpu if has_usage else None,
                is_active=data["is_active"],
                stack_watermark=data["watermark"],
                stack_watermark_percent=data["stack_usage_pct"],
                is_cpu_coherent=is_coherent if has_usage else None,
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
                    entry_site=meta.get("entry_site"),
                    is_idle=data["info"].address in self.idle_addresses,
                )
            )
        return final


def _signed_priority(raw: int) -> int:
    """A Zephyr priority is a signed 8-bit value: cooperative ones are negative."""
    return raw - 256 if raw >= 0x80 else raw
