# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Zephyr sys_heap chunk walker."""

import struct

from backend.base import AbstractScraper


def walk_heap_fragmentation(
    scraper: AbstractScraper,
    z_heap_addr: int,
    end_chunk_offset: int,
) -> list[dict]:
    """
    Enumerate a Zephyr sys_heap's chunks via a single bulk memory read.
    Each chunk is returned as ``{"used": bool, "size": int}`` in byte units.
    Returns an empty list when ``z_heap_addr`` is null or the heap's chunk
    count is zero. Raises ``ValueError`` when the heap header cannot be
    decoded or exceeds 32 MB, and ``RuntimeError`` on a zero-size chunk
    (infinite-loop guard).
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
