# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""DWARF-resolved offsets needed to walk Zephyr kernel objects."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelLayout:
    """
    Byte offsets into Zephyr kernel structs, resolved from the ELF's DWARF.

    All fields are simple integer offsets. The owning ``ZScraper`` knows the
    base addresses (``_kernel_base_address``, per-thread struct address, per
    heap struct pointer) and adds them to these offsets at read time.
    """

    # Mandatory: thread walking
    threads_head: int  # z_kernel.threads
    thread_next: int  # k_thread.next_thread
    stack_start: int  # _thread_stack_info.start (resolved through k_thread.stack_info)
    stack_size: int  # _thread_stack_info.size (resolved through k_thread.stack_info)

    # Optional: thread names (CONFIG_THREAD_NAME)
    thread_name: int | None = None

    # Optional: per-thread CPU usage (CONFIG_THREAD_RUNTIME_STATS)
    cpu_usage: int | None = None  # z_kernel.usage
    thread_usage: int | None = None  # k_thread → k_cycle_stats.total

    # Optional: heap stats (CONFIG_SYS_HEAP_RUNTIME_STATS)
    heap_free_bytes: int | None = None
    heap_allocated_bytes: int | None = None
    heap_max_allocated_bytes: int | None = None
    heap_end_chunk: int | None = None
