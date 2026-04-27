# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end replay integration against a recorded fixture and its golden JSON."""

import json
import queue
import threading
import time
from pathlib import Path

import pytest

from backend.base import HeapInfo, ThreadInfo
from backend.replay import ReplayScraper
from orchestrator import ZScraper

_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "recordings"
_ELF_PATH = Path(__file__).parent / "fixtures" / "zephyr.elf"
_REPLAY_BUDGET_SECS = 2.0


def _discover_fixtures() -> list[tuple[Path, Path]]:
    """Find (.ndjson.gz, .golden.json) pairs under tests/fixtures/recordings/."""
    if not _FIXTURES_DIR.exists():
        return []
    pairs = []
    for recording in sorted(_FIXTURES_DIR.glob("*.ndjson.gz")):
        golden = recording.with_suffix("").with_suffix(".golden.json")
        if golden.exists():
            pairs.append((recording, golden))
    return pairs


def _drive_replay(recording: Path) -> list[dict]:
    """Replay a fixture through a full ZScraper polling loop; return captured frames."""
    replay = ReplayScraper(recording, honor_timing=False)
    with replay:
        scraper = ZScraper(replay, str(_ELF_PATH))
        scraper.update_available_threads()
        scraper.reset_thread_pool()

        data_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        scraper.start_polling_thread(data_queue, stop_event, 0.001)

        time.sleep(_REPLAY_BUDGET_SECS)
        stop_event.set()
        scraper.finish_polling_thread()

        frames = []
        while not data_queue.empty():
            frames.append(data_queue.get_nowait())
    return frames


@pytest.fixture(scope="module", params=_discover_fixtures(), ids=lambda p: p[0].name)
def replay_run(request) -> tuple[list[dict], dict]:
    recording, golden_path = request.param
    frames = _drive_replay(recording)
    golden = json.loads(golden_path.read_text())
    return frames, golden


def _last_valid_frame(frames: list[dict]) -> dict:
    valid = [f for f in frames if "threads" in f]
    assert valid, "Replay produced no valid frames"
    return valid[-1]


def test_replay_produces_frames(replay_run):
    """Replay drains the recording into one or more valid data frames."""
    frames, _ = replay_run
    valid = [f for f in frames if "threads" in f]
    assert valid, "Replay produced no valid frames"


def test_thread_shape(replay_run):
    """Every emitted thread has a non-empty name and a positive stack size."""
    frames, _ = replay_run
    last = _last_valid_frame(frames)
    threads: list[ThreadInfo] = last["threads"]
    assert threads, "Frame has no threads"
    for t in threads:
        assert isinstance(t, ThreadInfo)
        assert t.name
        assert t.stack_size > 0
        if t.runtime is not None:
            assert 0 <= t.runtime.stack_watermark <= t.stack_size


def test_heap_shape(replay_run):
    """Emitted heaps report non-negative sizes; max ≥ allocated."""
    frames, _ = replay_run
    last = _last_valid_frame(frames)
    heaps: list[HeapInfo] = last.get("heaps", [])
    for h in heaps:
        assert isinstance(h, HeapInfo)
        assert h.free_bytes >= 0
        assert h.allocated_bytes >= 0
        assert h.max_allocated_bytes >= h.allocated_bytes


def test_snapshot_thread_set_matches_golden(replay_run):
    """Thread names + stack starts + stack sizes equal the golden."""
    frames, golden = replay_run
    last = _last_valid_frame(frames)
    captured = {(t.name, t.stack_start, t.stack_size) for t in last["threads"]}
    expected = {(t["name"], t["stack_start"], t["stack_size"]) for t in golden["threads"]}
    assert captured == expected


def test_snapshot_watermarks_match_golden(replay_run):
    """Per-thread watermark equals the golden."""
    frames, golden = replay_run
    last = _last_valid_frame(frames)
    captured = {t.name: t.runtime.stack_watermark for t in last["threads"] if t.runtime}
    expected = {
        t["name"]: t["stack_watermark"]
        for t in golden["threads"]
        if t["stack_watermark"] is not None
    }
    assert captured == expected


def test_snapshot_heaps_match_golden(replay_run):
    """Heap identity + byte counters equal the golden."""
    frames, golden = replay_run
    last = _last_valid_frame(frames)
    captured = [
        (h.name, h.address, h.free_bytes, h.allocated_bytes, h.max_allocated_bytes)
        for h in last.get("heaps", [])
    ]
    expected = [
        (h["name"], h["address"], h["free_bytes"], h["allocated_bytes"], h["max_allocated_bytes"])
        for h in golden.get("heaps", [])
    ]
    assert captured == expected
