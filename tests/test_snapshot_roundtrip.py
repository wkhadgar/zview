# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
Covers the headless snapshot module: frame serialization, single-frame dump
against a ReplayScraper, and Replay→Recording roundtrip yielding a valid
recording that replays cleanly on its own.
"""

import json
from pathlib import Path

import pytest

from backend.base import HeapInfo, ThreadInfo, ThreadRuntime
from backend.replay import ReplayScraper
from snapshot import dump_single_frame, record_session, serialize_frame

_TESTS_DIR = Path(__file__).parent
_FIXTURE = _TESTS_DIR / "fixtures" / "recordings" / "sys_heap_v4.3.ndjson.gz"
_ELF = _TESTS_DIR / "fixtures" / "zephyr.elf"


def test_serialize_frame_produces_json_safe_dict():
    runtime = ThreadRuntime(
        cpu=1.0,
        cpu_normalized=0.5,
        active=True,
        stack_watermark=100,
        stack_watermark_percent=25.0,
    )
    thread = ThreadInfo(
        address=0x1000,
        stack_start=0x2000,
        stack_size=400,
        name="main",
        runtime=runtime,
    )
    heap = HeapInfo(
        name="my_heap",
        address=0x3000,
        free_bytes=1024,
        allocated_bytes=512,
        max_allocated_bytes=800,
        usage_percent=50.0,
        chunks=None,
    )

    out = serialize_frame({"threads": [thread], "heaps": [heap]})

    # Roundtrips through json without error.
    serialized = json.dumps(out)
    restored = json.loads(serialized)

    assert restored["threads"][0]["name"] == "main"
    assert restored["threads"][0]["address"] == 0x1000
    assert restored["threads"][0]["runtime"]["cpu"] == 1.0
    assert restored["heaps"][0]["free_bytes"] == 1024
    assert restored["heaps"][0]["chunks"] is None


def test_serialize_frame_omits_runtime_when_none():
    thread = ThreadInfo(
        address=0x1000,
        stack_start=0x2000,
        stack_size=400,
        name="probe",
        runtime=None,
    )
    out = serialize_frame({"threads": [thread]})
    assert "runtime" not in out["threads"][0]


def test_serialize_frame_handles_empty():
    assert serialize_frame({}) == {}
    assert serialize_frame({"threads": []}) == {"threads": []}


def test_dump_single_frame_via_replay():
    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    frame = dump_single_frame(backend, str(_ELF), period=0.001, timeout=3.0)

    assert "threads" in frame
    names = {t.name for t in frame["threads"]}
    assert names == {"stress_id", "idle", "main"}

    # JSON shape sanity
    payload = serialize_frame(frame)
    json.dumps(payload)  # must not raise


def test_record_session_roundtrip(tmp_path):
    """ReplayScraper fed into RecordingScraper should emit a replayable recording."""
    source = ReplayScraper(_FIXTURE, honor_timing=False)
    out_path = tmp_path / "roundtrip.ndjson.gz"

    captured = record_session(source, str(_ELF), out_path, duration=1.5, period=0.001)
    assert out_path.exists()
    assert captured >= 1

    # Replay the recorded session — it must yield at least one valid frame.
    replay2 = ReplayScraper(out_path, honor_timing=False)
    frame2 = dump_single_frame(replay2, str(_ELF), period=0.001, timeout=3.0)
    assert "threads" in frame2
    assert {t.name for t in frame2["threads"]} == {"stress_id", "idle", "main"}


def test_record_session_respects_frames_bound(tmp_path):
    source = ReplayScraper(_FIXTURE, honor_timing=False)
    out_path = tmp_path / "bounded.ndjson.gz"

    captured = record_session(source, str(_ELF), out_path, frames=2, period=0.001)
    assert captured == 2


def test_record_session_requires_a_bound(tmp_path):
    source = ReplayScraper(_FIXTURE, honor_timing=False)
    with pytest.raises(ValueError, match="duration or frames"):
        record_session(source, str(_ELF), tmp_path / "no_bound.ndjson.gz")


def test_record_session_duration_bound(tmp_path):
    source = ReplayScraper(_FIXTURE, honor_timing=False)
    out_path = tmp_path / "duration.ndjson.gz"

    # period dominates frame spacing; duration=0.1s / period=0.05s yields ~2 frames.
    captured = record_session(source, str(_ELF), out_path, duration=0.1, period=0.05)

    assert out_path.exists()
    assert 1 <= captured <= 5


def test_record_session_dual_bound_first_to_hit_wins(tmp_path):
    """With a short duration and a much larger frames target, duration terminates first."""
    source = ReplayScraper(_FIXTURE, honor_timing=False)
    out_path = tmp_path / "dual.ndjson.gz"

    captured = record_session(source, str(_ELF), out_path, duration=0.1, frames=100, period=0.05)

    assert out_path.exists()
    # Duration wins: captured is nowhere near the frames target.
    assert captured < 100
    assert captured <= 5


def test_configure_heap_detail_sets_extra_info_address():
    """Dereferences the heap symbol (via read32) and stores the struct pointer."""
    from unittest.mock import MagicMock

    from orchestrator import ZScraper as _ZScraper
    from snapshot import _configure_heap_detail

    scraper = _ZScraper.__new__(_ZScraper)
    scraper.has_heaps = True
    scraper._k_heap_addresses = {"my_heap": [0x1000]}
    scraper.extra_info_heap_address = None

    recorder = MagicMock()
    recorder.read32.return_value = [0x5000]

    _configure_heap_detail(scraper, recorder, "my_heap")

    assert scraper.extra_info_heap_address == 0x5000
    recorder.begin_batch.assert_called_once()
    recorder.end_batch.assert_called_once()
    recorder.read32.assert_called_once_with(0x1000)


def test_configure_heap_detail_unknown_heap_raises():
    from unittest.mock import MagicMock

    from orchestrator import ZScraper as _ZScraper
    from snapshot import _configure_heap_detail

    scraper = _ZScraper.__new__(_ZScraper)
    scraper.has_heaps = True
    scraper._k_heap_addresses = {"my_heap": [0x1000]}

    with pytest.raises(ValueError, match="not found"):
        _configure_heap_detail(scraper, MagicMock(), "nonexistent")


def test_configure_heap_detail_requires_heap_support():
    from unittest.mock import MagicMock

    from orchestrator import ZScraper as _ZScraper
    from snapshot import _configure_heap_detail

    scraper = _ZScraper.__new__(_ZScraper)
    scraper.has_heaps = False

    with pytest.raises(ValueError, match="no k_heap"):
        _configure_heap_detail(scraper, MagicMock(), "any")


def test_dump_single_frame_propagates_fatal_error(monkeypatch):
    """A fatal_error emitted by the polling thread surfaces as RuntimeError."""

    def fake_start(self, data_queue, stop_event, period):
        data_queue.put({"fatal_error": "synthetic backend loss"})

    monkeypatch.setattr("orchestrator.ZScraper.start_polling_thread", fake_start)

    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    with pytest.raises(RuntimeError, match="synthetic backend loss"):
        dump_single_frame(backend, str(_ELF), period=0.001, timeout=1.0)


def test_dump_single_frame_rejects_invalid_frame():
    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    with pytest.raises(ValueError, match="frame must be >= 1"):
        dump_single_frame(backend, str(_ELF), frame=0)


def test_dump_single_frame_skips_to_requested_frame():
    """frame=N returns the Nth valid data frame, distinct from the first."""
    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    f1 = dump_single_frame(backend, str(_ELF), period=0.001, frame=1, timeout=3.0)

    backend2 = ReplayScraper(_FIXTURE, honor_timing=False)
    f3 = dump_single_frame(backend2, str(_ELF), period=0.001, frame=3, timeout=3.0)

    assert {t.name for t in f1["threads"]} == {t.name for t in f3["threads"]}
    cpu_f1 = sorted(t.runtime.cpu_normalized for t in f1["threads"])
    cpu_f3 = sorted(t.runtime.cpu_normalized for t in f3["threads"])
    assert cpu_f1 != cpu_f3


def test_dump_single_frame_raises_when_recording_too_short():
    """frame=N where N exceeds recorded frames surfaces a clear RuntimeError."""
    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    with pytest.raises(RuntimeError, match="out of range"):
        dump_single_frame(backend, str(_ELF), period=0.001, frame=10_000, timeout=3.0)
