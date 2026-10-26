# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the headless snapshot module."""

import json
import time
from pathlib import Path

import pytest

from zview.backend.base import HeapInfo, MemSlabInfo, MsgqInfo, ThreadInfo, ThreadRuntime, WorkqInfo
from zview.backend.replay import ReplayScraper
from zview.frames import Fatal
from zview.snapshot import dump_single_frame, record_session, serialize_frame

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


def test_serialize_frame_carries_every_object_group():
    frame = {
        "msgqs": [
            MsgqInfo(name="q", address=0x4000, used_msgs=2, max_msgs=8, msg_size=16, waiters=())
        ],
        "mem_slabs": [
            MemSlabInfo(
                name="slab", address=0x5000, num_blocks=8, block_size=64, num_used=3, waiters=()
            )
        ],
        "workqs": [
            WorkqInfo(
                name="bench_workq",
                address=0x6000,
                flags=0b11,
                pending=("trail_work (work_handler)",),
                thread_name="bench_workq",
                waiters=(),
            )
        ],
    }

    out = json.loads(json.dumps(serialize_frame(frame)))

    assert out["msgqs"][0]["used_msgs"] == 2
    assert out["mem_slabs"][0]["num_used"] == 3
    assert out["workqs"][0]["pending"] == ["trail_work (work_handler)"]
    assert out["workqs"][0]["flags"] == 0b11


def test_serialize_frame_keeps_runtime_key_as_none():
    """``runtime`` field stays present as null when no runtime data is set."""
    thread = ThreadInfo(
        address=0x1000,
        stack_start=0x2000,
        stack_size=400,
        name="probe",
        runtime=None,
    )
    out = serialize_frame({"threads": [thread]})
    assert out["threads"][0]["runtime"] is None


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


def test_a_replayed_dump_shows_only_the_selected_groups():
    """The replay reads all its recording holds, so the selection filters the frame after."""
    from zview.selection import Selection

    replay = ReplayScraper(_FIXTURE, honor_timing=False)
    frame = dump_single_frame(
        replay, str(_ELF), period=0.001, selection=Selection(groups=frozenset({"heaps"}))
    )

    assert set(frame) == {"heaps"}


def test_record_session_roundtrip(tmp_path):
    """ReplayScraper -> RecordingScraper produces a replayable recording."""
    source = ReplayScraper(_FIXTURE, honor_timing=False)
    out_path = tmp_path / "roundtrip.ndjson.gz"

    captured = record_session(source, str(_ELF), out_path, duration=1.5, period=0.001)
    assert out_path.exists()
    assert captured >= 1

    # Replay the recorded session: it must yield at least one valid frame.
    replay2 = ReplayScraper(out_path, honor_timing=False)
    frame2 = dump_single_frame(replay2, str(_ELF), period=0.001, timeout=3.0)
    assert "threads" in frame2
    assert {t.name for t in frame2["threads"]} == {"stress_id", "idle", "main"}


def test_record_session_respects_frames_bound(tmp_path):
    source = ReplayScraper(_FIXTURE, honor_timing=False)
    out_path = tmp_path / "bounded.ndjson.gz"

    captured = record_session(source, str(_ELF), out_path, frames=2, period=0.001)
    assert captured == 2


def test_record_ends_with_a_poll_whose_frame_was_dropped(tmp_path, monkeypatch):
    """A frame a full queue dropped would leave the record waiting for it."""
    from zview.core import Core
    from zview.frames import Frame

    publish = Core._publish
    dropped: list = []

    def drop_the_first_frame(self, item):
        if isinstance(item, Frame) and not dropped:
            dropped.append(item)
            return
        publish(self, item)

    monkeypatch.setattr(Core, "_publish", drop_the_first_frame)
    source = ReplayScraper(_FIXTURE, honor_timing=False)

    captured = record_session(source, str(_ELF), tmp_path / "r.ndjson.gz", frames=3, period=0.001)

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
    """Duration terminates first when frames is much larger."""
    source = ReplayScraper(_FIXTURE, honor_timing=False)
    out_path = tmp_path / "dual.ndjson.gz"

    captured = record_session(source, str(_ELF), out_path, duration=0.1, frames=100, period=0.05)

    assert out_path.exists()
    # Duration wins: captured is nowhere near the frames target.
    assert captured < 100
    assert captured <= 5


def test_dump_single_frame_propagates_fatal_error(monkeypatch):
    """``fatal_error`` from the polling thread surfaces as ``RuntimeError``."""

    def fake_poll(self, period=None, frames=None):
        self._publish(Fatal("synthetic backend loss"))

    monkeypatch.setattr("zview.core.Core.poll", fake_poll)

    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    with pytest.raises(RuntimeError, match="synthetic backend loss"):
        dump_single_frame(backend, str(_ELF), period=0.001, timeout=1.0)


def test_dump_single_frame_rejects_invalid_frame():
    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    with pytest.raises(ValueError, match="frame must be >= 1"):
        dump_single_frame(backend, str(_ELF), frame=0)


def test_dump_single_frame_skips_to_requested_frame():
    """``frame=N`` returns the Nth valid data frame."""
    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    f1 = dump_single_frame(backend, str(_ELF), period=0.001, frame=1, timeout=3.0)

    backend2 = ReplayScraper(_FIXTURE, honor_timing=False)
    f3 = dump_single_frame(backend2, str(_ELF), period=0.001, frame=3, timeout=3.0)

    assert {t.name for t in f1["threads"]} == {t.name for t in f3["threads"]}
    cpu_f1 = sorted(t.runtime.cpu_normalized for t in f1["threads"])
    cpu_f3 = sorted(t.runtime.cpu_normalized for t in f3["threads"])
    assert cpu_f1 != cpu_f3


class _SlowFrames(ReplayScraper):
    """A replay whose frames take over a second, as pyOCD's through a J-Link OB."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []
        self._batches = 0

    def begin_batch(self):
        self._batches += 1
        # The walk at discovery comes quick.
        if self._batches > 1:
            time.sleep(1.5)
        super().begin_batch()

    def end_batch(self):
        super().end_batch()
        self.calls.append("end_batch")

    def disconnect(self):
        self.calls.append("disconnect")
        super().disconnect()


def test_dump_closes_the_probe_after_the_frame_in_flight():
    backend = _SlowFrames(_FIXTURE, honor_timing=False)

    with pytest.raises(TimeoutError):
        dump_single_frame(backend, str(_ELF), period=0.001, frame=1, timeout=0.2)

    # The walk at discovery, then the frame in flight, then the probe.
    assert backend.calls == ["end_batch", "end_batch", "disconnect"]


def test_dump_single_frame_raises_when_recording_too_short():
    """``frame`` past recording exhaustion raises ``RuntimeError``."""
    backend = ReplayScraper(_FIXTURE, honor_timing=False)
    with pytest.raises(RuntimeError, match="out of range"):
        dump_single_frame(backend, str(_ELF), period=0.001, frame=10_000, timeout=3.0)
