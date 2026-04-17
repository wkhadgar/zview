# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
AbstractScraper defines the common memory-read interface used by every live
probe backend (JLink, pyOCD, GDB RSP) and every synthetic one (recording,
replay, coredump). Shared kernel-level dataclasses live here as well.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


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


class AbstractScraper:
    """
    Base class for every memory-read backend. Subclasses implement the probe-
    or source-specific transport; shared state (watermark cache, endianess) and
    a default stack-watermark scanner live here.
    """

    def __init__(self, target_mcu: str | None):
        self._target_mcu: str | None = target_mcu
        self._is_connected: bool = False
        self.watermark_cache = {}
        self.endianess: Literal["<", ">"] = "<"  # default to little endian

    def __enter__(self):
        self.connect()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type
        del exc_val
        del exc_tb

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

    def begin_batch(self):
        pass

    def end_batch(self):
        pass

    def read_bytes(self, at: int, amount: int) -> bytes:
        print(f"Read {amount} raw bytes from {hex(at)}")
        return b""

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
        thread_id,
    ) -> int:
        """
        Scan a thread's stack for the unused-stack fill pattern and return
        the current high-water mark in bytes. Uses an internal per-thread
        cache so repeated calls only re-read the portion that could have
        grown since the last sample.
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
