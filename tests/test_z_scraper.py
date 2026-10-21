# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from zview.backend.base import (
    AbstractScraper,
    ProbeReadError,
    ProbeReadMalformed,
    ProbeReadTimeout,
    ThreadInfo,
)
from zview.backend.gdb import GDBScraper
from zview.backend.pyocd import PyOCDScraper
from zview.kernel.threads import Threads
from zview.orchestrator import ZScraper
from zview.selection import Selection


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


_USAGE = 0x10
_MAIN, _IDLE, _WORKER = 0x1000, 0x2000, 0x3000


def _thread(address: int, name: str) -> ThreadInfo:
    return ThreadInfo(address=address, stack_start=0x0, stack_size=1000, name=name, runtime=None)


def _seed(scraper: ZScraper, threads, idle=(_IDLE,)) -> Threads:
    """The scraper's threads group, as a discovery of ``threads`` leaves it."""
    group = scraper.groups["threads"]
    group.infos = {t.address: t for t in threads}
    group.idle_addresses = idle
    scraper.rows["threads"] = {t.name: [t.address] for t in threads}
    return group


def _usage_scraper(elf_path, threads, before: dict[int, int], now: dict[int, int]):
    """``ZScraper`` over a target whose usage counters moved from ``before`` to ``now``."""
    from dataclasses import replace as dc_replace

    mock = MagicMock()
    mock.is_connected = True
    mock.watermark_cache = {}
    scraper = ZScraper(mock, elf_path=elf_path)
    scraper.available["heaps"] = False
    scraper._layout = dc_replace(scraper._layout, thread_usage=_USAGE)
    _seed(scraper, threads).last_cycles = dict(before)
    mock.read64.side_effect = lambda address, amount=1: [now.get(address - _USAGE, 0)]
    mock.read_many.side_effect = lambda reads: AbstractScraper.read_many(mock, reads)
    mock.read8.side_effect = lambda address, amount=1: [0] * amount
    mock.calculate_dynamic_watermark.return_value = 250
    return scraper, mock


def _one_frame(scraper: ZScraper) -> dict:
    """Run the polling loop for exactly one frame and return what it emitted."""
    data_queue: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    with patch("time.sleep", side_effect=lambda _: stop_event.set()):
        scraper._poll_thread_worker(data_queue, stop_event, 0)
    return data_queue.get(timeout=1.0)


def test_shares_are_of_the_threads_own_cycles(elf_path):
    """CPU% divides by every thread's cycles, load% by the busy threads' only."""
    scraper, _ = _usage_scraper(
        elf_path,
        [_thread(_MAIN, "main"), _thread(_WORKER, "worker"), _thread(_IDLE, "idle")],
        before={_MAIN: 50, _WORKER: 0, _IDLE: 50},
        now={_MAIN: 60, _WORKER: 30, _IDLE: 110},
    )

    runtimes = {t.name: t.runtime for t in _one_frame(scraper)["threads"]}

    # 10, 30 and 60 cycles out of 100.
    assert {n: r.cpu_normalized for n, r in runtimes.items()} == pytest.approx(
        {"main": 10.0, "worker": 30.0, "idle": 60.0}
    )
    # 10 and 30 out of the 40 busy cycles.
    assert {n: r.cpu for n, r in runtimes.items()} == pytest.approx(
        {"main": 25.0, "worker": 75.0, "idle": 0.0}
    )
    assert runtimes["main"].stack_watermark_percent == 25.0


def test_every_idle_thread_counts_as_idle(elf_path):
    """An SMP build idles on each CPU, and neither idle thread is load."""
    second_idle = 0x2100
    scraper, _ = _usage_scraper(
        elf_path,
        [_thread(_MAIN, "main"), _thread(_IDLE, "idle 00"), _thread(second_idle, "idle 01")],
        before={_MAIN: 0, _IDLE: 0, second_idle: 0},
        now={_MAIN: 20, _IDLE: 90, second_idle: 90},
    )
    scraper.groups["threads"].idle_addresses = (_IDLE, second_idle)

    runtimes = {t.name: t.runtime for t in _one_frame(scraper)["threads"]}

    assert runtimes["main"].cpu == 100.0
    assert runtimes["idle 01"].cpu == 0.0
    assert runtimes["main"].cpu_normalized == pytest.approx(10.0)


def test_a_detail_view_shares_count_the_threads_it_does_not_scan(elf_path):
    """Only the view's stacks are scanned, but every thread's cycles are read."""
    main, worker, idle = _thread(_MAIN, "main"), _thread(_WORKER, "worker"), _thread(_IDLE, "idle")
    scraper, mock = _usage_scraper(
        elf_path,
        [main, worker, idle],
        before={_MAIN: 0, _WORKER: 0, _IDLE: 0},
        now={_MAIN: 10, _WORKER: 30, _IDLE: 60},
    )
    scraper.view = Selection(include=("main", "idle"))

    runtimes = {t.name: t.runtime for t in _one_frame(scraper)["threads"]}

    assert set(runtimes) == {"main", "idle"}
    assert runtimes["main"].cpu == pytest.approx(25.0)
    assert mock.calculate_dynamic_watermark.call_count == 2


def test_a_muted_thread_is_not_scanned_but_its_cycles_still_count(elf_path):
    """Dropping its counter would hand its cycles to the threads still listed."""
    from zview.selection import Selection

    main, worker, idle = _thread(_MAIN, "main"), _thread(_WORKER, "worker"), _thread(_IDLE, "idle")
    scraper, mock = _usage_scraper(
        elf_path,
        [main, worker, idle],
        before={_MAIN: 0, _WORKER: 0, _IDLE: 0},
        now={_MAIN: 10, _WORKER: 30, _IDLE: 60},
    )
    scraper.selection = Selection(muted=frozenset({("threads", "worker")}))

    runtimes = {t.name: t.runtime for t in _one_frame(scraper)["threads"]}

    # 10 of the 40 busy cycles, the muted worker's 30 included.
    assert set(runtimes) == {"main", "idle"}
    assert runtimes["main"].cpu == pytest.approx(25.0)
    assert mock.calculate_dynamic_watermark.call_count == 2


def test_a_recording_from_before_the_snapshot_reads_counters_beside_the_stacks(elf_path):
    """It read the kernel-wide counter first, and each thread's own after the one before's stack."""
    scraper, mock = _usage_scraper(
        elf_path,
        [_thread(_MAIN, "main"), _thread(_IDLE, "idle")],
        before={_MAIN: 0, _IDLE: 0},
        now={_MAIN: 10, _IDLE: 30},
    )
    scraper.parts.discard("usage_snapshot")
    scraper.groups["threads"]._cpu_usage_address = 0x9000

    runtimes = {t.name: t.runtime for t in _one_frame(scraper)["threads"]}

    reads = [
        c
        for c in mock.mock_calls
        if c[0] in ("read8", "read32", "read64", "calculate_dynamic_watermark")
    ]
    # The metadata and entry reads sit between a stack and the next counter.
    assert [c[0] for c in reads[:6]] == [
        "read32",
        "read64",
        "calculate_dynamic_watermark",
        "read8",
        "read32",
        "read64",
    ]
    assert reads[0].args == (0x9000,)
    assert runtimes["main"].cpu_normalized == pytest.approx(25.0)


def _make_strike_scraper(elf_path):
    """``ZScraper`` shim with its usage counters and heaps left unread."""
    from dataclasses import replace as dc_replace

    mock = MagicMock()
    mock.is_connected = True
    mock.read8.side_effect = lambda address, amount=1: [0] * amount
    scraper = ZScraper(mock, elf_path=elf_path)
    scraper.available["heaps"] = False
    scraper._layout = dc_replace(scraper._layout, thread_usage=None)
    _seed(
        scraper,
        [ThreadInfo(address=0x1000, stack_start=0x0, stack_size=1000, name="t1", runtime=None)],
        idle=(0x9999,),
    )
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
    scraper.groups = {"threads": Threads()}
    scraper.groups["threads"].last_cycles = {0x1000: 50, 0x2000: 75}

    scraper.reset_runtime_state()

    assert mock.watermark_cache == {}
    assert scraper.groups["threads"].last_cycles == {}
    mock.read64.assert_not_called()


def test_broken_frame_resyncs_and_skips_emission(elf_path):
    """Counter regression triggers resync; no frame is emitted, no error counted."""
    # The thread's counter reads below its baseline.
    scraper, mock_meta_scraper = _usage_scraper(
        elf_path, [_thread(_MAIN, "main")], before={_MAIN: 1_000}, now={_MAIN: 0}
    )
    mock_meta_scraper.watermark_cache = {_MAIN: 100}

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
    assert scraper.groups["threads"].last_cycles == {}
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


def test_gdb_failed_connect_leaves_no_socket_to_resume_on():
    """A refused connect surfaces as itself, not as a broken pipe from the batch cleanup."""
    scraper = GDBScraper("localhost:1234")
    refused = MagicMock()
    refused.connect.side_effect = ConnectionRefusedError()

    with (
        patch("zview.backend.gdb.socket.socket", return_value=refused),
        pytest.raises(ConnectionError, match="Is QEMU running"),
    ):
        scraper.connect()

    scraper.end_batch()

    assert scraper.sock is None
    refused.sendall.assert_not_called()


def test_reload_elf_rebuilds_features_and_keeps_tui_settings(elf_path):
    meta = MagicMock()
    meta.is_live = True
    scraper = ZScraper(meta, elf_path=elf_path)
    has_heaps = scraper.available["heaps"]
    scraper.available["heaps"] = not has_heaps
    scraper.view = Selection(groups=frozenset({"threads"}))
    scraper.inspection_period = 1.5

    scraper.reload_elf()

    assert scraper.available["heaps"] is has_heaps
    assert scraper.view == Selection(groups=frozenset({"threads"}))
    assert scraper.inspection_period == 1.5


def _serving(memory: dict[int, int], reply_cap: int | None = None):
    """
    ``_read_response`` stand-in answering each ``m`` packet from ``memory``.

    ``reply_cap`` answers with at most that many bytes, as a stub may.
    """
    packets: list[tuple[int, int]] = []

    def send(data: bytes) -> None:
        addr, length = (int(field, 16) for field in data[1:].decode().split(","))
        packets.append((addr, length))

    def respond() -> bytes:
        addr, length = packets[-1]
        served = length if reply_cap is None else min(length, reply_cap)
        return bytes(memory.get(addr + i, 0) for i in range(served)).hex().encode()

    return packets, send, respond


def test_a_read_past_the_packet_size_goes_out_in_pieces():
    """A heap's chunk map read reaches past what one packet can carry on QEMU."""
    scraper = _make_gdb_with_mock_sock()
    memory = {0x2000 + i: i & 0xFF for i in range(2112)}
    packets, send, respond = _serving(memory)

    with (
        patch.object(GDBScraper, "_send_packet", side_effect=send),
        patch.object(GDBScraper, "_read_response", side_effect=respond),
    ):
        data = scraper._read_mem_raw(0x2000, 2112)

    assert packets == [(0x2000, 1024), (0x2400, 1024), (0x2800, 64)]
    assert data == bytes(i & 0xFF for i in range(2112))


def test_a_short_reply_is_read_on_from_where_it_stopped():
    """A stub may answer with fewer bytes than asked; they are not zeros."""
    scraper = _make_gdb_with_mock_sock()
    memory = {0x1000 + i: 0xA0 + i for i in range(8)}
    packets, send, respond = _serving(memory, reply_cap=3)

    with (
        patch.object(GDBScraper, "_send_packet", side_effect=send),
        patch.object(GDBScraper, "_read_response", side_effect=respond),
    ):
        data = scraper._read_mem_raw(0x1000, 8)

    assert data == bytes(0xA0 + i for i in range(8))
    assert [addr for addr, _ in packets] == [0x1000, 0x1003, 0x1006]


def test_an_empty_reply_is_malformed():
    scraper = _make_gdb_with_mock_sock()
    with (
        patch.object(GDBScraper, "_read_response", return_value=b""),
        pytest.raises(ProbeReadMalformed, match="Empty reply"),
    ):
        scraper._read_mem_raw(0x1000, 16)


def test_bursts_read_after_the_frame_began_still_share_out_to_100(elf_path):
    """
    The six philosophers' frame. Divided by a kernel-wide counter that saw 400 of
    their 1000 busy cycles, these loads read 100%, 100% and 50%.
    """
    first, second, third = 0x4000, 0x4100, 0x4200
    scraper, _ = _usage_scraper(
        elf_path,
        [_thread(first, "P0"), _thread(second, "P1"), _thread(third, "P2"), _thread(_IDLE, "idle")],
        before={first: 0, second: 0, third: 0, _IDLE: 0},
        now={first: 400, second: 400, third: 200, _IDLE: 9000},
    )

    runtimes = {t.name: t.runtime for t in _one_frame(scraper)["threads"]}

    assert {n: r.cpu for n, r in runtimes.items()} == pytest.approx(
        {"P0": 40.0, "P1": 40.0, "P2": 20.0, "idle": 0.0}
    )
    assert sum(r.cpu_normalized for r in runtimes.values()) == pytest.approx(100.0)


def _one_thread(scraper: ZScraper, stack_start=0x1000, stack_size=512) -> ThreadInfo:
    """Poll and finalize one frame of a single thread."""
    thread = ThreadInfo(
        address=0x2000, stack_start=stack_start, stack_size=stack_size, name="t", runtime=None
    )
    return _seed(scraper, [thread]).walk(scraper._session(), {"t": [0x2000]})["t"]


def _without_usage(scraper: ZScraper) -> None:
    """The layout of a build that counts no usage."""
    from dataclasses import replace as dc_replace

    scraper._layout = dc_replace(scraper._layout, thread_usage=None)


def test_an_unfilled_build_skips_the_scan_and_reports_no_stack_usage(elf_path):
    """Without the fill pattern the scan would find every stack full."""
    scraper, _ = _meta_scraper(elf_path)
    _without_usage(scraper)
    scraper.parts.add("unfilled_stacks")

    runtime = _one_thread(scraper).runtime

    scraper._m_scraper.calculate_dynamic_watermark.assert_not_called()
    assert (runtime.stack_watermark, runtime.stack_watermark_percent) == (None, None)


def test_a_build_without_runtime_stats_reports_no_cpu_share(elf_path):
    """A 0% share would claim the thread idle, not unmeasured."""
    scraper, _ = _meta_scraper(elf_path)
    _without_usage(scraper)
    scraper._m_scraper.calculate_dynamic_watermark.return_value = 128

    runtime = _one_thread(scraper).runtime

    assert (runtime.cpu, runtime.cpu_normalized) == (None, None)
    assert runtime.stack_watermark_percent == 25.0


def test_a_thread_with_no_stack_geometry_is_not_scanned(elf_path):
    """A filled stack still needs a start and a size to be read."""
    scraper, _ = _meta_scraper(elf_path)
    _without_usage(scraper)

    runtime = _one_thread(scraper, stack_start=None, stack_size=None).runtime

    scraper._m_scraper.calculate_dynamic_watermark.assert_not_called()
    assert (runtime.stack_watermark, runtime.stack_watermark_percent) == (None, None)


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

    scraper = ZScraper(_Logging(is_connected=True), elf_path=elf_path)
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

    meta = scraper.groups["threads"]._read_meta(scraper._session(), thread)

    assert [r for r in reads if r[0] == "read8"] == [("read8", 0x2000 + 12, 5)]
    assert (meta["user_options"], meta["priority"], meta["state"]) == (0x7F, 0x7F, 0x7F)


def test_the_entry_pointer_is_read_once_per_thread(elf_path):
    scraper, reads = _meta_scraper(elf_path)
    thread = ThreadInfo(address=0x2000, stack_start=0, stack_size=0, name="t", runtime=None)

    for _ in range(4):
        scraper.groups["threads"]._read_meta(scraper._session(), thread)

    assert [r for r in reads if r[0] == "read32"] == [("read32", 0x2000 + 112, 1)]


def test_a_rediscovered_thread_list_reads_the_entry_again(elf_path):
    """The address can belong to another thread after the list is walked again."""
    scraper, reads = _meta_scraper(elf_path)
    thread = ThreadInfo(address=0x2000, stack_start=0, stack_size=0, name="t", runtime=None)
    group = scraper.groups["threads"]

    group._read_meta(scraper._session(), thread)
    with patch("zview.kernel.threads.walk_thread_list", return_value={"t": thread}):
        scraper.discover()
    group._read_meta(scraper._session(), thread)

    assert len([r for r in reads if r[0] == "read32"]) == 2


def test_a_legacy_recording_reads_each_metadata_field_apiece(elf_path):
    scraper, reads = _meta_scraper(elf_path)
    scraper.parts.discard("merged_thread_meta")
    thread = ThreadInfo(address=0x2000, stack_start=0, stack_size=0, name="t", runtime=None)

    for _ in range(2):
        scraper.groups["threads"]._read_meta(scraper._session(), thread)

    assert [r for r in reads if r[0] == "read8"] == [
        ("read8", 0x2000 + 14, 1),
        ("read8", 0x2000 + 16, 1),
        ("read8", 0x2000 + 12, 1),
    ] * 2
    assert len([r for r in reads if r[0] == "read32"]) == 2
