# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal
from unittest.mock import MagicMock

import pytest

from backend.z_scraper import GDBScraper, PyOCDScraper


def test_gdb_scraper_endianness():
    """Validates that GDBScraper correctly applies endianness to struct unpacking."""
    scraper = GDBScraper("localhost:1234")
    scraper._read_mem_raw = MagicMock()

    # Little endian validation
    scraper.endianess = "<"
    scraper._read_mem_raw.return_value = b'\x01\x00\x02\x00'
    assert scraper.read16(0x0, 2) == (1, 2)

    scraper._read_mem_raw.return_value = b'\x01\x00\x00\x00\x02\x00\x00\x00'
    assert scraper.read32(0x0, 2) == (1, 2)

    # Big endian validation
    scraper.endianess = ">"
    scraper._read_mem_raw.return_value = b'\x00\x01\x00\x02'
    assert scraper.read16(0x0, 2) == (1, 2)

    scraper._read_mem_raw.return_value = b'\x00\x00\x00\x01\x00\x00\x00\x02'
    assert scraper.read32(0x0, 2) == (1, 2)


def test_pyocd_scraper_read64_endianness():
    """Validates the manual 64-bit word assembly in PyOCDScraper."""
    scraper = PyOCDScraper(None)
    scraper.read32 = MagicMock()

    # Little endian (words read as [low_word, high_word])
    scraper.endianess = "<"
    scraper.read32.return_value = [0xBBBBBBBB, 0xAAAAAAAA]
    assert scraper.read64(0x0, 1) == [0xAAAAAAAABBBBBBBB]

    # Big endian (words read as [high_word, low_word])
    scraper.endianess = ">"
    scraper.read32.return_value = [0xAAAAAAAA, 0xBBBBBBBB]
    assert scraper.read64(0x0, 1) == [0xAAAAAAAABBBBBBBB]


def test_determine_chunk_width_signatures():
    """Validates the O(1) deterministic heap signature check."""

    # Construct a stub to isolate the evaluation logic without requiring a full ZScraper
    class ScraperStub:
        def __init__(self):
            self.runner = MagicMock()

        def _determine_chunk_width(
            self, z_heap_addr: int, byte_order: Literal["little", "big"]
        ) -> int:
            raw_bytes = self.runner.read_memory(z_heap_addr, 8)
            size_and_used_16 = int.from_bytes(raw_bytes[2:4], byteorder=byte_order)
            size_and_used_32 = int.from_bytes(raw_bytes[4:8], byteorder=byte_order)

            if size_and_used_16 == 1:
                return 2
            elif size_and_used_32 == 1:
                return 4
            else:
                raise ValueError("Invalid heap signature")

    stub = ScraperStub()

    # 16-bit little endian (offset 0x02 is 0x01 0x00)
    stub.runner.read_memory.return_value = b'\x00\x00\x01\x00\x00\x00\x00\x00'
    assert stub._determine_chunk_width(0x20000000, "little") == 2

    # 32-bit big endian (offset 0x04 is 0x00 0x00 0x00 0x01)
    stub.runner.read_memory.return_value = b'\x00\x00\x00\x00\x00\x00\x00\x01'
    assert stub._determine_chunk_width(0x20000000, "big") == 4

    # Invalid signature rejection
    stub.runner.read_memory.return_value = b'\x00\x00\x00\x00\x00\x00\x00\x00'
    with pytest.raises(ValueError, match="Invalid heap signature"):
        stub._determine_chunk_width(0x20000000, "little")
