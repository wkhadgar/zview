# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
AbstractScraper defines the common memory-read interface used by every live
probe backend (JLink, pyOCD, GDB RSP) and every synthetic one (recording,
replay). Shared kernel-level dataclasses and the probe error hierarchy live
here as well.
"""

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import ClassVar, Literal

# One read of a ``read_many`` call: the read method's name, then its ``at`` and ``amount``.
Read = tuple[Literal["read_bytes", "read8", "read32", "read64"], int, int]


class ProbeError(Exception):
    """Base for probe backend errors."""


class ProbeConnectFailure(ProbeError):
    """Probe failed to establish a session with the target."""


class ProbeReadFailure(ProbeError):
    """Probe returned no usable data for a memory read."""


class ProbeReadTimeout(ProbeReadFailure):
    """Probe read did not receive a response before the deadline."""


class ProbeReadError(ProbeReadFailure):
    """Probe returned an error reply to a memory read."""


class ProbeReadMalformed(ProbeReadFailure):
    """Probe returned an undecodable response to a memory read."""


class ReplayError(Exception):
    """Base for replay errors."""


@dataclass(frozen=True)
class ThreadRuntime:
    """Per-frame runtime state for a Zephyr thread."""

    # None on a build without runtime stats.
    cpu: float | None
    cpu_normalized: float | None
    active: bool
    # None on a build that does not fill its stacks.
    stack_watermark: int | None
    stack_watermark_percent: float | None


@dataclass(frozen=True)
class ThreadInfo:
    """Static identity and stack geometry of a Zephyr thread plus its latest runtime."""

    address: int
    # None on a build without CONFIG_THREAD_STACK_INFO.
    stack_start: int | None
    stack_size: int | None
    name: str
    runtime: ThreadRuntime | None
    priority: int | None = None
    state: int | None = None
    user_options: int | None = None
    entry_point: int | None = None
    entry_symbol: str | None = None


@dataclass(frozen=True)
class HeapInfo:
    """Snapshot of a Zephyr ``k_heap`` plus an optional chunk fragmentation map."""

    name: str
    address: int
    free_bytes: int
    allocated_bytes: int
    max_allocated_bytes: int
    usage_percent: float
    chunks: list[dict] | None
    # ``None``: the wait queue is not walkable, or the heap is not a ``k_heap``.
    waiters: tuple[str, ...] | None = None

    @property
    def total_bytes(self) -> int:
        return self.allocated_bytes + self.free_bytes

    @property
    def is_exhausted(self) -> bool:
        return self.total_bytes > 0 and self.free_bytes == 0


@dataclass(frozen=True)
class SemaphoreInfo:
    """Snapshot of a Zephyr ``k_sem``."""

    # K_SEM_MAX_LIMIT, the limit of a semaphore that counts without one.
    _MAX_LIMIT: ClassVar[int] = 0xFFFF_FFFF

    name: str
    address: int
    count: int
    limit: int
    # ``None``: the wait queue layout is not walkable. ``()``: nobody waiting.
    waiters: tuple[str, ...] | None = None

    @property
    def is_initialized(self) -> bool:
        """``k_sem_init`` refuses a zero limit, so a zero one has not run yet."""
        return self.limit != 0

    @property
    def is_unbounded(self) -> bool:
        return self.limit == self._MAX_LIMIT

    @property
    def fill_percent(self) -> float:
        """The count as a share of the limit, 0 where there is no limit to share."""
        if not self.is_initialized or self.is_unbounded:
            return 0.0
        return self.count / self.limit * 100.0


@dataclass(frozen=True)
class MsgqInfo:
    """Snapshot of a Zephyr ``k_msgq``."""

    name: str
    address: int
    used_msgs: int
    max_msgs: int
    msg_size: int
    # A ``k_msgq`` has one wait queue for both directions: senders queue on a
    # full one, receivers on an empty one.
    waiters: tuple[str, ...] | None = None

    @property
    def fill_percent(self) -> float:
        return (self.used_msgs / self.max_msgs * 100.0) if self.max_msgs else 0.0

    @property
    def is_full(self) -> bool:
        return self.max_msgs > 0 and self.used_msgs >= self.max_msgs


@dataclass(frozen=True)
class MemSlabInfo:
    """Snapshot of a Zephyr ``k_mem_slab``."""

    name: str
    address: int
    num_blocks: int
    block_size: int
    num_used: int
    # ``None`` unless the build sets CONFIG_MEM_SLAB_TRACE_MAX_UTILIZATION. A
    # high-water mark, so it holds its peak between reads.
    max_used: int | None = None
    waiters: tuple[str, ...] | None = None

    @property
    def fill_percent(self) -> float:
        return (self.num_used / self.num_blocks * 100.0) if self.num_blocks else 0.0

    @property
    def peak_percent(self) -> float | None:
        if self.max_used is None or not self.num_blocks:
            return None

        return self.max_used / self.num_blocks * 100.0

    @property
    def is_exhausted(self) -> bool:
        return self.num_blocks > 0 and self.num_used >= self.num_blocks

    @property
    def total_bytes(self) -> int:
        return self.num_blocks * self.block_size


@dataclass(frozen=True)
class WorkqInfo:
    """Snapshot of a Zephyr ``k_work_q``."""

    _STARTED: ClassVar[int] = 1 << 0
    _BUSY: ClassVar[int] = 1 << 1
    _DRAIN: ClassVar[int] = 1 << 2
    _PLUGGED: ClassVar[int] = 1 << 3

    name: str
    address: int
    flags: int
    # The item being run has left the list.
    pending: tuple[str, ...] = ()
    pending_truncated: bool = False
    thread_name: str | None = None
    waiters: tuple[str, ...] | None = None

    @property
    def depth(self) -> int:
        return len(self.pending)

    @property
    def is_started(self) -> bool:
        return bool(self.flags & self._STARTED)

    @property
    def is_busy(self) -> bool:
        """Running a handler, which an empty pending list does not rule out."""
        return bool(self.flags & self._BUSY)

    @property
    def is_draining(self) -> bool:
        return bool(self.flags & self._DRAIN)

    @property
    def is_plugged(self) -> bool:
        return bool(self.flags & self._PLUGGED)

    @property
    def states(self) -> tuple[str, ...]:
        """The state flags set, as words, in bit order."""
        return tuple(
            word
            for is_set, word in (
                (self.is_started, "started"),
                (self.is_busy, "busy"),
                (self.is_draining, "draining"),
                (self.is_plugged, "plugged"),
            )
            if is_set
        )


class MutexState(enum.IntEnum):
    """Lock state of a ``k_mutex``, ordered by contention."""

    FREE = 0
    LOCKED = 1
    CONTENDED = 2


@dataclass(frozen=True)
class MutexInfo:
    """Snapshot of a Zephyr ``k_mutex``."""

    name: str
    address: int
    lock_count: int
    owner_address: int
    # ``None`` when the mutex is free or the owner is not in the thread table.
    owner_name: str | None = None
    waiters: tuple[str, ...] | None = None

    @property
    def is_locked(self) -> bool:
        return self.owner_address != 0

    @property
    def state(self) -> MutexState:
        """Held with threads queued on it is contention; held alone is not."""
        if not self.is_locked:
            return MutexState.FREE

        return MutexState.CONTENDED if self.waiters else MutexState.LOCKED


class AbstractScraper(ABC):
    """Common interface for memory-read backends (JLink, pyOCD, GDB RSP)."""

    # True for live probe backends; False for synthetic backends (replay) that
    # cannot accept runtime mutations - no reconnect, no change to the polling
    # shape (thread pool, heap fragmentation toggle) mid-stream. Subclasses that
    # cannot absorb such mutations must set this to False.
    is_live: bool = True

    def __init__(self, target_mcu: str | None):
        self._target_mcu: str | None = target_mcu
        self._is_connected: bool = False
        self.watermark_cache = {}
        self.endianess: Literal["<", ">"] = "<"

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self.disconnect()

    @property
    def is_connected(self):
        return self._is_connected

    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def disconnect(self): ...

    # begin_batch/end_batch are optional hooks: GDB overrides them to halt/resume
    # the target; JLink and pyOCD inherit the no-op default because their probe
    # libraries do not require bracketing. Hence empty bodies on purpose.
    def begin_batch(self):  # noqa: B027
        pass

    def end_batch(self):  # noqa: B027
        pass

    @abstractmethod
    def read_bytes(self, at: int, amount: int) -> bytes: ...

    @abstractmethod
    def read8(self, at: int, amount: int = 1) -> Sequence[int]: ...

    @abstractmethod
    def read32(self, at: int, amount: int = 1) -> Sequence[int]: ...

    @abstractmethod
    def read64(self, at: int, amount: int = 1) -> Sequence[int]: ...

    def read_many(self, reads: Sequence[Read]) -> Iterator[bytes | Sequence[int]]:
        """
        The result of each of ``reads``, in order.

        One read at a time here. A backend that can queue reads sends them
        together. Either way a read that fails raises in its place, after the
        results before it.
        """
        for method, at, amount in reads:
            yield getattr(self, method)(at, amount)

    def calculate_dynamic_watermark(
        self,
        stack_start: int,
        stack_size: int,
        unused_pattern: int = 0xAA_AA_AA_AA,
        *,
        thread_id,
        reserved_words: int = 0,
    ) -> int:
        """
        Reads a stack memory and scans for the unused_pattern fill pattern
        to determine the current stack watermark (highest point of stack usage).

        Args:
            :param stack_start: The starting address of the thread's stack.
            :param stack_size: The total size of the thread's stack in bytes.
            :param unused_pattern: Unused stack fill word.
            :param id: Unique identification for the given thread.
            :param reserved_words: Words at the low end that hold no fill, such
                as Zephyr's stack sentinel. They are not scanned, and count as used.

        Returns:
            The calculated stack watermark in bytes, indicating the maximum
            amount of stack space that has been used.
        """
        if stack_size == 0:
            return 0

        reserved = 4 * reserved_words
        size = stack_size - reserved
        cache_watermark = self.watermark_cache.get(thread_id, 0)
        watermark = size - cache_watermark

        stack_words = self.read32(stack_start + reserved, (size // 4) - (cache_watermark // 4))

        for word in stack_words:
            if word == unused_pattern:
                watermark -= 4
            else:
                break

        self.watermark_cache[thread_id] = watermark + cache_watermark

        return self.watermark_cache[thread_id] + reserved
