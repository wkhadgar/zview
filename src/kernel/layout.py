# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""DWARF-resolved offsets needed to walk Zephyr kernel objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelLayout:
    """Byte offsets into Zephyr kernel structs, resolved from DWARF."""

    # Mandatory: thread walking
    threads_head: int  # z_kernel.threads
    thread_next: int  # k_thread.next_thread
    stack_start: int  # _thread_stack_info.start (resolved through k_thread.stack_info)
    stack_size: int  # _thread_stack_info.size (resolved through k_thread.stack_info)

    # Optional: thread names (CONFIG_THREAD_NAME)
    thread_name: int | None = None

    # Optional: per-thread CPU usage (CONFIG_THREAD_RUNTIME_STATS)
    cpu_usage: int | None = None  # z_kernel.usage
    thread_usage: int | None = None  # k_thread -> k_cycle_stats.total

    # Optional: thread metadata mirroring `kernel threads list` shell output.
    thread_priority: int | None = None  # k_thread.base.prio (signed 8-bit)
    thread_state: int | None = None  # k_thread.base.thread_state (8-bit bitfield)
    thread_user_options: int | None = None  # k_thread.base.user_options
    thread_entry: int | None = None  # k_thread.entry.pEntry (function pointer)

    # Optional: heap stats (CONFIG_SYS_HEAP_RUNTIME_STATS)
    heap_free_bytes: int | None = None
    heap_allocated_bytes: int | None = None
    heap_max_allocated_bytes: int | None = None
    heap_end_chunk: int | None = None
