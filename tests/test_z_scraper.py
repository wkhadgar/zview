# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.base import ProbeReadError, ProbeReadMalformed, ProbeReadTimeout, ThreadInfo
from backend.gdb import GDBScraper
from backend.pyocd import PyOCDScraper
from orchestrator import ZScraper


def test_gdb_scraper_endianness():
    """``GDBScraper`` applies endianness to struct unpacking."""
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
    """``PyOCDScraper.read64`` unpacks with endianness."""
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
    """CPU normalization and watermark percentage propagate to the emitted frame."""

    mock_meta_scraper = MagicMock()
    mock_meta_scraper.is_connected = True

    scraper = ZScraper(mock_meta_scraper, elf_path=elf_path, max_threads=2)
    scraper.has_heaps = False
    scraper.has_usage = True
    scraper.idle_threads_address = 0x2000

    from dataclasses import replace as dc_replace

    scraper._layout = dc_replace(scraper._layout, thread_usage=0x10)
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


def _make_strike_scraper(elf_path):
    """``ZScraper`` shim with ``has_usage``/``has_heaps`` disabled."""
    mock = MagicMock()
    mock.is_connected = True
    scraper = ZScraper(mock, elf_path=elf_path, max_threads=2)
    scraper.has_heaps = False
    scraper.has_usage = False
    scraper.idle_threads_address = 0x9999
    scraper.thread_pool = [
        ThreadInfo(address=0x1000, stack_start=0x0, stack_size=1000, name="t1", runtime=None),
    ]
    return scraper, mock


def test_single_strike_emits_transient_error(elf_path):
    """One read failure emits a transient error; loop continues."""
    scraper, mock = _make_strike_scraper(elf_path)
    mock.calculate_dynamic_watermark.side_effect = RuntimeError("read fail")

    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    with patch("time.sleep", side_effect=lambda _: stop_event.set()):
        scraper._poll_thread_worker(q, stop_event, 0)

    msg = q.get(timeout=1)
    assert "error" in msg
    assert "Transient" in msg["error"]
    assert "read fail" in msg["error"]


def test_three_strikes_emit_fatal_and_break(elf_path):
    """Three consecutive failures emit ``fatal_error`` and terminate the loop."""
    scraper, mock = _make_strike_scraper(elf_path)
    mock.calculate_dynamic_watermark.side_effect = RuntimeError("persistent fail")

    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    # The loop breaks itself on the 3rd strike; time.sleep should not need to stop it.
    with patch("time.sleep"):
        scraper._poll_thread_worker(q, stop_event, 0)

    messages = []
    while not q.empty():
        messages.append(q.get_nowait())

    assert len(messages) == 3
    # The same fault worded the same way twice, so the message log collapses it.
    assert messages[0]["error"] == messages[1]["error"]
    assert "persistent fail" in messages[0]["error"]
    assert "fatal_error" in messages[2]
    assert "Target lost" in messages[2]["fatal_error"]


def test_strike_counter_resets_on_success(elf_path):
    """Two failures + one success resets the counter; no ``fatal_error``."""
    scraper, mock = _make_strike_scraper(elf_path)
    mock.calculate_dynamic_watermark.side_effect = [
        RuntimeError("fail-1"),
        RuntimeError("fail-2"),
        250,  # success
    ]

    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    frame_count = [0]

    def fake_sleep(_):
        frame_count[0] += 1
        if frame_count[0] >= 3:
            stop_event.set()

    with patch("time.sleep", side_effect=fake_sleep):
        scraper._poll_thread_worker(q, stop_event, 0)

    messages = []
    while not q.empty():
        messages.append(q.get_nowait())

    assert len(messages) == 3
    assert "fail-1" in messages[0]["error"]
    assert "fail-2" in messages[1]["error"]
    assert "threads" in messages[2]
    assert "fatal_error" not in messages[2]


def test_queue_full_is_not_counted_as_strike(elf_path):
    """``queue.Full`` is swallowed; not counted as a strike."""
    scraper, mock = _make_strike_scraper(elf_path)
    mock.calculate_dynamic_watermark.return_value = 250

    # maxsize=1 pre-filled with a sentinel; successful frame put() will raise queue.Full.
    q: queue.Queue = queue.Queue(maxsize=1)
    q.put({"sentinel": True})

    stop_event = threading.Event()

    # Run a single frame; time.sleep stops the loop. If queue.Full were a strike,
    # a transient error message would be appended here, but the queue is full so
    # nothing gets appended regardless. The real assertion is that no fatal_error
    # fires even across many iterations.
    iterations = [0]

    def fake_sleep(_):
        iterations[0] += 1
        if iterations[0] >= 5:
            stop_event.set()

    with patch("time.sleep", side_effect=fake_sleep):
        scraper._poll_thread_worker(q, stop_event, 0)

    # Queue still holds only the sentinel: no transient or fatal_error emitted.
    assert q.get_nowait() == {"sentinel": True}
    assert q.empty()


def test_reset_runtime_state_clears_caches_and_invalidates_baseline(elf_path):
    """Clears watermark/cycle caches and marks the CPU baseline invalid."""
    scraper = ZScraper.__new__(ZScraper)
    mock = MagicMock()
    mock.watermark_cache = {0x1000: 100, 0x2000: 200}
    scraper._m_scraper = mock
    scraper.has_usage = True
    scraper.last_thread_cycles = {0x1000: 50, 0x2000: 75}
    scraper.last_cpu_cycles = 1234
    scraper.last_cpu_delta = 1234

    scraper.reset_runtime_state()

    assert mock.watermark_cache == {}
    assert scraper.last_thread_cycles == {}
    assert scraper.last_cpu_cycles is None
    assert scraper.last_cpu_delta == 0
    mock.read64.assert_not_called()


def test_broken_frame_resyncs_and_skips_emission(elf_path):
    """Counter regression triggers resync; no frame is emitted, no error counted."""
    mock_meta_scraper = MagicMock()
    mock_meta_scraper.is_connected = True
    mock_meta_scraper.watermark_cache = {0x1000: 100}

    scraper = ZScraper(mock_meta_scraper, elf_path=elf_path, max_threads=2)
    scraper.has_heaps = False
    scraper.has_usage = True
    scraper.idle_threads_address = 0x2000
    scraper._cpu_usage_address = 0x1000
    scraper.thread_pool = [
        ThreadInfo(address=0x1000, stack_start=0x0, stack_size=1000, name="main", runtime=None),
    ]

    # Seed a stale baseline; current read goes backwards -> _BrokenFrame.
    scraper.last_cpu_cycles = 5_000
    scraper.last_thread_cycles = {0x1000: 1_000}
    mock_meta_scraper.read32.return_value = [100]
    mock_meta_scraper.read64.return_value = [0]

    q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    iters = [0]

    def fake_sleep(_):
        iters[0] += 1
        if iters[0] >= 1:
            stop_event.set()

    with patch("time.sleep", side_effect=fake_sleep):
        scraper._poll_thread_worker(q, stop_event, 0)

    # Nothing was emitted (no frame, no error, no fatal_error).
    assert q.empty()
    # State was wiped: next poll will re-baseline cleanly.
    assert scraper.last_cpu_cycles is None
    assert scraper.last_thread_cycles == {}
    assert mock_meta_scraper.watermark_cache == {}


def _make_gdb_with_mock_sock() -> GDBScraper:
    scraper = GDBScraper("localhost:1234")
    scraper.sock = MagicMock()
    return scraper


def test_read_mem_raw_timeout_raises_probe_read_timeout():
    scraper = _make_gdb_with_mock_sock()
    with (
        patch.object(GDBScraper, "_read_response", side_effect=TimeoutError("deadline")),
        pytest.raises(ProbeReadTimeout, match="Timeout reading"),
    ):
        scraper._read_mem_raw(0x1000, 16)


def test_read_mem_raw_error_reply_raises_probe_read_error():
    scraper = _make_gdb_with_mock_sock()
    with (
        patch.object(GDBScraper, "_read_response", return_value=b"E01"),
        pytest.raises(ProbeReadError, match="GDB error"),
    ):
        scraper._read_mem_raw(0x1000, 16)


def test_read_mem_raw_malformed_hex_raises_probe_read_malformed():
    scraper = _make_gdb_with_mock_sock()
    with (
        patch.object(GDBScraper, "_read_response", return_value=b"ZZZZ"),
        pytest.raises(ProbeReadMalformed, match="Malformed hex"),
    ):
        scraper._read_mem_raw(0x1000, 16)


def test_read_mem_raw_happy_path_returns_decoded_bytes():
    scraper = _make_gdb_with_mock_sock()
    # Stub returns 4 bytes of hex for the 4-byte requested read.
    with patch.object(GDBScraper, "_read_response", return_value=b"deadbeef"):
        result = scraper._read_mem_raw(0x1000, 4)
    assert result == b"\xde\xad\xbe\xef"


def test_cpu_share_cannot_exceed_the_total_it_divides(elf_path):
    """A thread reading more cycles than the global counter caps at 100%."""
    mock_meta_scraper = MagicMock()
    mock_meta_scraper.is_connected = True

    scraper = ZScraper(mock_meta_scraper, elf_path=elf_path, max_threads=2)
    scraper.has_heaps = False
    scraper.has_usage = True
    scraper.idle_threads_address = 0x2000

    from dataclasses import replace as dc_replace

    scraper._layout = dc_replace(scraper._layout, thread_usage=0x10)
    scraper._cpu_usage_address = 0x1000
    scraper.last_cpu_cycles = 100
    scraper.last_cpu_delta = 100

    scraper.thread_pool = [
        ThreadInfo(address=0x1000, stack_start=0x0, stack_size=1000, name="burst", runtime=None),
        ThreadInfo(address=0x2000, stack_start=0x0, stack_size=1000, name="idle", runtime=None),
    ]
    scraper.last_thread_cycles = {0x1000: 50, 0x2000: 50}

    # The global counter advances by 100, the thread's own by 4950.
    mock_meta_scraper.read32.return_value = [200]
    mock_meta_scraper.read64.side_effect = lambda address: {
        0x1000 + 0x10: [5000],
        0x2000 + 0x10: [55],
    }.get(address, [0])
    mock_meta_scraper.calculate_dynamic_watermark.return_value = 250

    data_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    with patch("time.sleep", side_effect=lambda _: stop_event.set()):
        scraper._poll_thread_worker(data_queue, stop_event, 0)

    burst = next(t for t in data_queue.get(timeout=1.0)["threads"] if t.name == "burst")

    assert burst.runtime.cpu_normalized == 100.0
    assert burst.runtime.cpu == 100.0


def _meta_scraper(elf_path) -> tuple[ZScraper, list[tuple[str, int, int]]]:
    """``ZScraper`` over a recording scraper stand-in that logs every read it serves."""
    from dataclasses import replace as dc_replace

    reads: list[tuple[str, int, int]] = []

    class _Logging(MagicMock):
        def read8(self, at, amount=1):
            reads.append(("read8", at, amount))
            return [0x7F] * amount

        def read32(self, at, amount=1):
            reads.append(("read32", at, amount))
            return [0xDEAD] * amount

    scraper = ZScraper(_Logging(is_connected=True), elf_path=elf_path, max_threads=2)
    scraper._layout = dc_replace(
        scraper._layout,
        thread_priority=14,
        thread_state=16,
        thread_user_options=12,
        thread_entry=112,
    )
    return scraper, reads


def test_thread_metadata_bytes_come_in_one_read(elf_path):
    """``user_options``, ``prio`` and ``state`` sit within one span of the struct."""
    scraper, reads = _meta_scraper(elf_path)
    thread = ThreadInfo(address=0x2000, stack_start=0, stack_size=0, name="t", runtime=None)

    meta = scraper._read_thread_meta(thread)

    assert [r for r in reads if r[0] == "read8"] == [("read8", 0x2000 + 12, 5)]
    assert (meta["user_options"], meta["priority"], meta["state"]) == (0x7F, 0x7F, 0x7F)


def test_the_entry_pointer_is_read_once_per_thread(elf_path):
    scraper, reads = _meta_scraper(elf_path)
    thread = ThreadInfo(address=0x2000, stack_start=0, stack_size=0, name="t", runtime=None)

    for _ in range(4):
        scraper._read_thread_meta(thread)

    assert [r for r in reads if r[0] == "read32"] == [("read32", 0x2000 + 112, 1)]


def test_a_rebuilt_thread_pool_reads_the_entry_again(elf_path):
    """The address can belong to another thread after the list is walked again."""
    scraper, reads = _meta_scraper(elf_path)
    thread = ThreadInfo(address=0x2000, stack_start=0, stack_size=0, name="t", runtime=None)

    scraper._read_thread_meta(thread)
    scraper.reset_thread_pool()
    scraper._read_thread_meta(thread)

    assert len([r for r in reads if r[0] == "read32"]) == 2


def test_a_legacy_recording_reads_each_metadata_field_apiece(elf_path):
    scraper, reads = _meta_scraper(elf_path)
    scraper.merged_thread_meta = False
    thread = ThreadInfo(address=0x2000, stack_start=0, stack_size=0, name="t", runtime=None)

    for _ in range(2):
        scraper._read_thread_meta(thread)

    assert [r for r in reads if r[0] == "read8"] == [
        ("read8", 0x2000 + 14, 1),
        ("read8", 0x2000 + 16, 1),
        ("read8", 0x2000 + 12, 1),
    ] * 2
    assert len([r for r in reads if r[0] == "read32"]) == 2
