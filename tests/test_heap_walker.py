# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import struct

import pytest

from backend.z_scraper import ZScraper


class FakeScraper:
    """Minimal AbstractScraper stand-in for hermetic heap-walker tests."""

    def __init__(self, end_chunk: int, heap_bytes: bytes, endianess: str = "<"):
        self.end_chunk = end_chunk
        self.heap_bytes = heap_bytes
        self.endianess = endianess

    def read32(self, at: int, amount: int = 1):
        return (self.end_chunk,)

    def read_bytes(self, at: int, amount: int) -> bytes:
        return self.heap_bytes[:amount]


def _make_walker(scraper: FakeScraper) -> ZScraper:
    """Bypass ZScraper.__init__ (which requires an ELF) and seed only what the walker reads."""
    walker = object.__new__(ZScraper)
    walker._m_scraper = scraper
    walker._offsets = {"heap_info": {"end_chunk": 0}}
    return walker


def _chunk0_header_16(first_chunk_id: int, endian: str = "<") -> bytes:
    buf = bytearray(8)
    struct.pack_into(f"{endian}H", buf, 2, (first_chunk_id << 1) | 1)
    return bytes(buf)


def _chunk_body_16(size_units: int, used: bool, endian: str = "<") -> bytes:
    buf = bytearray(8)
    struct.pack_into(f"{endian}H", buf, 2, (size_units << 1) | (1 if used else 0))
    return bytes(buf)


def _chunk0_header_32(first_chunk_id: int, endian: str = "<") -> bytes:
    buf = bytearray(8)
    struct.pack_into(f"{endian}I", buf, 4, (first_chunk_id << 1) | 1)
    return bytes(buf)


def _chunk_body_32(size_units: int, used: bool, endian: str = "<") -> bytes:
    buf = bytearray(8)
    struct.pack_into(f"{endian}I", buf, 4, (size_units << 1) | (1 if used else 0))
    return bytes(buf)


def test_null_address_returns_empty():
    walker = _make_walker(FakeScraper(end_chunk=0, heap_bytes=b""))
    assert walker.get_heap_fragmentation(0) == []


def test_end_chunk_zero_returns_empty():
    walker = _make_walker(FakeScraper(end_chunk=0, heap_bytes=b""))
    assert walker.get_heap_fragmentation(0x1000) == []


def test_16bit_walk():
    image = (
        _chunk0_header_16(first_chunk_id=1)
        + _chunk_body_16(size_units=2, used=True)  # c=1
        + b"\x00" * 8  # c=2 (inside chunk 1)
        + _chunk_body_16(size_units=2, used=False)  # c=3
        + b"\x00" * 8  # c=4 (inside chunk 3)
    )
    walker = _make_walker(FakeScraper(end_chunk=5, heap_bytes=image))

    assert walker.get_heap_fragmentation(0x1000) == [
        {"used": True, "size": 16},
        {"used": False, "size": 16},
    ]


def test_32bit_walk():
    image = (
        _chunk0_header_32(first_chunk_id=1)
        + _chunk_body_32(size_units=2, used=True)
        + b"\x00" * 8
        + _chunk_body_32(size_units=2, used=False)
        + b"\x00" * 8
    )
    walker = _make_walker(FakeScraper(end_chunk=5, heap_bytes=image))

    assert walker.get_heap_fragmentation(0x1000) == [
        {"used": True, "size": 16},
        {"used": False, "size": 16},
    ]


def test_oversize_heap_raises():
    # end_chunk * 8 = 40 MB, over the 32 MB sanity ceiling.
    walker = _make_walker(FakeScraper(end_chunk=5_000_000, heap_bytes=b""))
    with pytest.raises(ValueError, match="sanity"):
        walker.get_heap_fragmentation(0x1000)


def test_corrupted_chunk0_raises():
    # All zeros: both 16-bit and 32-bit interpretations fail bounds+used check.
    walker = _make_walker(FakeScraper(end_chunk=3, heap_bytes=bytes(24)))
    with pytest.raises(ValueError, match="Corrupted chunk0"):
        walker.get_heap_fragmentation(0x1000)


def test_zero_size_mid_walk_raises():
    image = (
        _chunk0_header_16(first_chunk_id=1)
        + _chunk_body_16(size_units=0, used=True)  # triggers infinite-loop guard
        + b"\x00" * 8
    )
    walker = _make_walker(FakeScraper(end_chunk=3, heap_bytes=image))
    with pytest.raises(RuntimeError, match="Infinite loop"):
        walker.get_heap_fragmentation(0x1000)


def test_big_endian_16bit_walk():
    image = (
        _chunk0_header_16(first_chunk_id=1, endian=">")
        + _chunk_body_16(size_units=2, used=True, endian=">")
        + b"\x00" * 8
    )
    walker = _make_walker(FakeScraper(end_chunk=3, heap_bytes=image, endianess=">"))

    assert walker.get_heap_fragmentation(0x1000) == [
        {"used": True, "size": 16},
    ]
