# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.z_scraper import GDBScraper, PyOCDScraper, ThreadInfo, ZScraper


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
    """Validates the struct unpacking of 64-bit words in PyOCDScraper."""
    scraper = PyOCDScraper(None)
    scraper.target = MagicMock()

    scraper.endianess = "<"
    scraper.target.read_memory_block8.return_value = [
        0xBB,
        0xBB,
        0xBB,
        0xBB,
        0xAA,
        0xAA,
        0xAA,
        0xAA,
    ]
    assert scraper.read64(0x0, 1) == (0xAAAAAAAABBBBBBBB,)

    scraper.endianess = ">"
    scraper.target.read_memory_block8.return_value = [
        0xAA,
        0xAA,
        0xAA,
        0xAA,
        0xBB,
        0xBB,
        0xBB,
        0xBB,
    ]
    assert scraper.read64(0x0, 1) == (0xAAAAAAAABBBBBBBB,)


@pytest.fixture
def elf_path():
    base_dir = Path(__file__).parent
    return base_dir / "fixtures" / "zephyr.elf"


def test_poll_thread_worker_math_and_queue(elf_path):
    """Validates CPU normalization and watermark percentage calculation via queue emission."""

    mock_meta_scraper = MagicMock()
    mock_meta_scraper.is_connected = True

    scraper = ZScraper(mock_meta_scraper, elf_path=elf_path, max_threads=2)
    scraper.has_heaps = False
    scraper.has_usage = True
    scraper.idle_threads_address = 0x2000

    scraper._offsets = {"thread_info": {"usage": 0x10}}
    scraper._cpu_usage_address = 0x1000
    scraper.last_cpu_cycles = 100
    scraper.last_cpu_delta = 100

    scraper.thread_pool = [
        ThreadInfo(address=0x1000, stack_start=0x0, stack_size=1000, name="main", runtime=None),
        ThreadInfo(address=0x2000, stack_start=0x0, stack_size=1000, name="idle", runtime=None),
    ]

    scraper.last_thread_cycles = {0x1000: 50, 0x2000: 50}

    mock_meta_scraper.read32.return_value = [200]

    def mock_read64(address):
        if address == 0x1000 + 0x10:
            return [60]
        elif address == 0x2000 + 0x10:
            return [130]
        return [0]

    mock_meta_scraper.read64.side_effect = mock_read64
    mock_meta_scraper.calculate_dynamic_watermark.return_value = 250

    data_queue = queue.Queue()
    stop_event = threading.Event()
    stop_event.clear()  # Ensure the loop starts

    with patch("time.sleep", side_effect=lambda _: stop_event.set()):
        scraper._poll_thread_worker(data_queue, stop_event, 0)

    result = data_queue.get(timeout=1.0)

    assert "threads" in result
    threads = result["threads"]
    assert len(threads) == 2

    main_thread = next(t for t in threads if t.name == "main")
    idle_thread = next(t for t in threads if t.name == "idle")

    # Idle thread: 80 / 180 = 44.444% Absolute CPU
    # cpu_normalized represents the Absolute CPU %
    assert idle_thread.runtime.cpu_normalized == pytest.approx(44.4444, abs=1e-3)
    # The idle thread by definition has 0% relative Load
    assert idle_thread.runtime.cpu == 0.0

    # Main thread: 10 / 180 = 5.555% Absolute CPU
    assert main_thread.runtime.cpu_normalized == pytest.approx(5.5555, abs=1e-3)

    # Main thread Load: 10 cycles / 100 non-idle cycles = 10.0%
    # cpu represents the Relative Load %
    assert main_thread.runtime.cpu == pytest.approx(10.0, abs=1e-3)

    # Stack watermark remains 250 / 1000 = 25%
    assert main_thread.runtime.stack_watermark_percent == 25.0
