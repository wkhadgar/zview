# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
AbstractScraper defines the common memory-read interface used by every live
probe backend (JLink, pyOCD, GDB RSP) and every synthetic one (recording,
replay). Shared kernel-level dataclasses and the probe error hierarchy live
here as well.
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


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


@dataclass(frozen=True)
class ThreadRuntime:
    """Per-frame runtime state for a Zephyr thread."""

    cpu: float
    cpu_normalized: float
    active: bool
    stack_watermark: int
    stack_watermark_percent: float


@dataclass(frozen=True)
class ThreadInfo:
    """Static identity and stack geometry of a Zephyr thread plus its latest runtime."""

    address: int
    stack_start: int
    stack_size: int
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

    def calculate_dynamic_watermark(
        self,
        stack_start: int,
        stack_size: int,
        unused_pattern: int = 0xAA_AA_AA_AA,
        *,
        thread_id,
    ) -> int:
        """
        Reads a stack memory and scans for the unused_pattern fill pattern
        to determine the current stack watermark (highest point of stack usage).

        Args:
            :param stack_start: The starting address of the thread's stack.
            :param stack_size: The total size of the thread's stack in bytes.
            :param unused_pattern: Unused stack fill word.
            :param id: Unique identification for the given thread.

        Returns:
            The calculated stack watermark in bytes, indicating the maximum
            amount of stack space that has been used.
        """
        if stack_size == 0:
            return 0

        cache_watermark = self.watermark_cache.get(thread_id, 0)
        watermark = stack_size - cache_watermark

        stack_words = self.read32(stack_start, (stack_size // 4) - (cache_watermark // 4))

        for word in stack_words:
            if word == unused_pattern:
                watermark -= 4
            else:
                break

        self.watermark_cache[thread_id] = watermark + cache_watermark

        return self.watermark_cache[thread_id]
