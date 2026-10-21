# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr ``k_heap`` group, with its sys_heap chunk walker and wait queue reader."""

import struct

from zview.backend.base import AbstractScraper, HeapInfo
from zview.backend.replay import ReplayMismatch
from zview.kernel.compat import FieldTable
from zview.kernel.group import Group
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.kernel.wait_queues import resolve_waiter_names, walk_wait_queue

# CONFIG_SYS_HEAP_RUNTIME_STATS. max_allocated_bytes exists from v3.1.0; on
# v3.0 this group does not resolve.
HEAP_FIELDS: FieldTable = {
    "heap_free_bytes": ((("z_heap", "free_bytes"),),),
    "heap_allocated_bytes": ((("z_heap", "allocated_bytes"),),),
    "heap_max_allocated_bytes": ((("z_heap", "max_allocated_bytes"),),),
    "heap_end_chunk": ((("z_heap", "end_chunk"),),),
}

# A k_heap wraps its sys_heap with a wait queue. Resolved on its own: a heap
# found as a sys_heap has no queue.
HEAP_OPTIONAL_FIELDS: FieldTable = {
    "heap_wait_q": ((("k_heap", "wait_q"),),),
}


def walk_heap_fragmentation(
    scraper: AbstractScraper,
    z_heap_addr: int,
    end_chunk_offset: int,
) -> list[dict]:
    """
    Enumerate a sys_heap's chunks. Returns ``[{"used": bool, "size": int}]``
    in byte units, or ``[]`` when the heap is null/empty. Raises ``ValueError``
    on a corrupted header or sizes >32 MB, ``RuntimeError`` on a zero-size chunk.
    """
    if z_heap_addr == 0:
        return []

    end_chunk = scraper.read32(z_heap_addr + end_chunk_offset)[0]
    if end_chunk == 0:
        return []

    total_bytes = end_chunk * 8
    if total_bytes > (1024 * 1024 * 32):
        raise ValueError(f"Heap size exceeds sanity limit: {total_bytes} bytes.")

    raw_buffer = scraper.read_bytes(z_heap_addr, total_bytes)
    mv = memoryview(raw_buffer)

    # Evaluate the chunk0 signature structurally
    val16_raw = struct.unpack_from(f"{scraper.endianess}H", mv, 2)[0]
    val32_raw = struct.unpack_from(f"{scraper.endianess}I", mv, 4)[0]

    is_used_16 = bool(val16_raw & 1)
    is_used_32 = bool(val32_raw & 1)

    val16 = val16_raw >> 1
    val32 = val32_raw >> 1

    if (0 < val16 < end_chunk) and is_used_16:
        c = val16  # Start chunk ID
        fmt = f"{scraper.endianess}H"
        offset_in_chunk = 2
    elif (0 < val32 < end_chunk) and is_used_32:
        c = val32
        fmt = f"{scraper.endianess}I"
        offset_in_chunk = 4
    else:
        raise ValueError(
            f"Corrupted chunk0 header. "
            f"16-bit field: size {val16}, used {is_used_16} | "
            f"32-bit field: size {val32}, used {is_used_32}"
        )

    chunks = []

    while c < end_chunk:
        offset = (c * 8) + offset_in_chunk
        val = struct.unpack_from(fmt, mv, offset)[0]

        is_used = bool(val & 1)
        c_size = val >> 1

        if c_size == 0:
            raise RuntimeError(f"Infinite loop prevented: Chunk at ID {c} has size 0.")

        chunks.append({"used": is_used, "size": c_size * 8})
        c += c_size

    return chunks


def walk_heap_waiters(
    scraper: AbstractScraper,
    heap_address: int,
    wait_q_offset: int,
    qnode_offset: int,
    thread_names: dict[int, str] | None = None,
) -> tuple[str, ...]:
    """
    Return the names of the threads queued on a ``k_heap``, in queue order.

    ``heap_address`` is the ``k_heap`` itself, not the ``sys_heap`` it wraps;
    ``wait_q_offset`` is the queue's offset within it. A thread queues there by
    asking for memory with a timeout the heap cannot meet.
    """
    return resolve_waiter_names(
        walk_wait_queue(scraper, heap_address + wait_q_offset, qnode_offset),
        thread_names,
    )


class Heaps(Group):
    key = "heaps"
    struct = "k_heap"
    required = (HEAP_FIELDS,)
    optional = (HEAP_OPTIONAL_FIELDS,)
    # Each heap's wait queue, walked since after heaps were first read.
    parts = ("heap_waiters",)

    def __init__(self) -> None:
        # Every heap's chunk map is read each frame while ``all_chunks`` is
        # set, and the ``detailed`` one's is read for its detail view.
        self.all_chunks = False
        self.detailed: int | None = None

    def capture_all(self) -> None:
        """
        Read every heap's chunk map. A recording covers all of them or none,
        and a replay finds out on its first read and backs off if there is none.
        """
        self.all_chunks = True

    def notices(self, session: Session) -> tuple[str, ...]:
        # Only a build that declares heaps misses their stats.
        if session.layout.heap_free_bytes is None and self.discover(session):
            return ("Warning: no heap stats (CONFIG_SYS_HEAP_RUNTIME_STATS=n)",)
        return ()

    def walk(self, session: Session, addresses: dict[str, list[int]]) -> dict[str, HeapInfo]:
        """
        Read every statically declared ``k_heap`` and return ``{label: HeapInfo}``.

        Each heap costs four reads for its statistics, its wait queue with the
        ``heap_waiters`` part on, and its chunk map when ``all_chunks`` is set or
        it is the ``detailed`` heap. A failed read raises, naming the heap.
        """
        scraper, layout = session.scraper, session.layout
        walk_waiters = (
            session.has_part("heap_waiters")
            and session.walks_waiters
            and layout.heap_wait_q is not None
            and layout.thread_qnode is not None
        )

        heaps: dict[str, HeapInfo] = {}
        for label, address in label_instances(addresses):
            try:
                heap_struct = scraper.read32(address)[0]
                free = scraper.read32(heap_struct + layout.heap_free_bytes)[0]
                allocated = scraper.read32(heap_struct + layout.heap_allocated_bytes)[0]
                max_allocated = scraper.read32(heap_struct + layout.heap_max_allocated_bytes)[0]
            except Exception as e:
                raise RuntimeError(f"Error reading heap info for {label}: {e}") from e

            waiters = None
            if walk_waiters:
                try:
                    waiters = walk_heap_waiters(
                        scraper,
                        address,
                        layout.heap_wait_q,
                        layout.thread_qnode,
                        session.thread_names,
                    )
                except Exception as e:
                    raise RuntimeError(f"Error reading waiters for {label}: {e}") from e

            chunks = None
            if self.all_chunks or self.detailed == heap_struct:
                try:
                    chunks = walk_heap_fragmentation(scraper, heap_struct, layout.heap_end_chunk)
                except ReplayMismatch:
                    # A recording taken without the chunk map has no such read
                    # to serve. A drift leaves the cursor untouched, so the
                    # session goes on without asking again.
                    self.all_chunks = False
                    self.detailed = None
                except Exception as e:
                    raise RuntimeError(f"Error reading sparsity for {label}: {e}") from e

            total = allocated + free
            heaps[label] = HeapInfo(
                label,
                heap_struct,
                free,
                allocated,
                max_allocated,
                (allocated / total * 100) if total > 0 else 0.0,
                chunks,
                waiters,
            )

        return heaps
