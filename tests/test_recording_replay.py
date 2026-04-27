# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path

import pytest

from backend.base import AbstractScraper
from backend.recording import SCHEMA_VERSION, RecordingScraper
from backend.replay import (
    ReplayComplete,
    ReplayExhausted,
    ReplayMismatch,
    ReplayScraper,
    UnsupportedSchema,
)


class FakeBackend(AbstractScraper):
    """Deterministic in-memory scraper used as the record source."""

    def __init__(self, memory: dict[tuple[str, int, int], object], endianess: str = "<"):
        super().__init__(target_mcu=None)
        self._memory = memory
        self.endianess = endianess
        self.connected_calls = 0
        self.disconnected_calls = 0

    def connect(self) -> None:
        self.connected_calls += 1
        self._is_connected = True

    def disconnect(self) -> None:
        self.disconnected_calls += 1
        self._is_connected = False

    def read_bytes(self, at: int, amount: int) -> bytes:
        return self._memory[("read_bytes", at, amount)]

    def read8(self, at: int, amount: int = 1):
        return self._memory[("read8", at, amount)]

    def read32(self, at: int, amount: int = 1):
        return self._memory[("read32", at, amount)]

    def read64(self, at: int, amount: int = 1):
        return self._memory[("read64", at, amount)]


def _record_session(path: Path, backend: FakeBackend) -> None:
    """Runs a small fixed call sequence through the recorder."""
    with RecordingScraper(backend, path) as rec:
        rec.begin_batch()
        rec.read32(0x1000, 1)
        rec.read32(0x1004, 4)
        rec.read64(0x2000)
        rec.read_bytes(0x3000, 16)
        rec.end_batch()


@pytest.fixture
def sample_path(tmp_path: Path) -> Path:
    return tmp_path / "session.ndjson.gz"


@pytest.fixture
def sample_backend() -> FakeBackend:
    return FakeBackend(
        memory={
            ("read32", 0x1000, 1): (0xDEADBEEF,),
            ("read32", 0x1004, 4): (1, 2, 3, 4),
            ("read64", 0x2000, 1): (0x0123456789ABCDEF,),
            ("read_bytes", 0x3000, 16): bytes(range(16)),
        }
    )


def test_roundtrip_read_results_match(sample_path, sample_backend):
    _record_session(sample_path, sample_backend)
    assert sample_backend.connected_calls == 1
    assert sample_backend.disconnected_calls == 1

    replay = ReplayScraper(sample_path)
    replay.connect()
    replay.begin_batch()
    assert replay.read32(0x1000, 1) == (0xDEADBEEF,)
    assert replay.read32(0x1004, 4) == (1, 2, 3, 4)
    assert replay.read64(0x2000) == (0x0123456789ABCDEF,)
    assert replay.read_bytes(0x3000, 16) == bytes(range(16))
    replay.end_batch()
    replay.disconnect()


def test_mismatched_args_raise(sample_path, sample_backend):
    _record_session(sample_path, sample_backend)

    replay = ReplayScraper(sample_path)
    replay.connect()
    replay.begin_batch()
    with pytest.raises(ReplayMismatch, match="drift"):
        replay.read32(0x9999, 1)


def test_exhaustion_raises(sample_path, sample_backend):
    _record_session(sample_path, sample_backend)

    replay = ReplayScraper(sample_path)
    replay.connect()
    replay.begin_batch()
    replay.read32(0x1000, 1)
    replay.read32(0x1004, 4)
    replay.read64(0x2000)
    replay.read_bytes(0x3000, 16)
    replay.end_batch()
    replay.disconnect()

    with pytest.raises(ReplayExhausted):
        replay.begin_batch()


def test_header_endianess_round_trip(tmp_path):
    backend = FakeBackend(memory={}, endianess=">")
    path = tmp_path / "be.ndjson.gz"

    with RecordingScraper(backend, path) as rec:
        assert rec.endianess == ">"

    replay = ReplayScraper(path)
    replay.connect()
    assert replay.endianess == ">"


def test_unsupported_schema_raises(tmp_path):
    path = tmp_path / "bogus.ndjson.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fp:
        fp.write(json.dumps({"schema": "nope/0"}) + "\n")

    with pytest.raises(UnsupportedSchema):
        ReplayScraper(path).connect()


def test_begin_batch_at_trailing_disconnect_signals_completion(sample_path, sample_backend):
    """Starting a new frame after the recording's last op raises ReplayComplete, not drift."""
    _record_session(sample_path, sample_backend)

    replay = ReplayScraper(sample_path, honor_timing=False)
    replay.connect()
    replay.begin_batch()
    replay.read32(0x1000, 1)
    replay.read32(0x1004, 4)
    replay.read64(0x2000)
    replay.read_bytes(0x3000, 16)
    replay.end_batch()

    # Cursor is now at the trailing "disconnect" entry. Starting another frame
    # is a clean end-of-stream rather than a mismatch.
    with pytest.raises(ReplayComplete):
        replay.begin_batch()


def test_partial_replay_allows_early_disconnect(sample_path, sample_backend):
    """Stopping replay mid-recording must not raise; disconnect is a lifecycle hint."""
    _record_session(sample_path, sample_backend)

    replay = ReplayScraper(sample_path)
    replay.connect()
    replay.begin_batch()
    replay.read32(0x1000, 1)

    replay.disconnect()  # early stop; further entries remain unconsumed

    assert not replay.is_connected


def test_payload_is_gzipped(sample_path, sample_backend):
    _record_session(sample_path, sample_backend)

    assert sample_path.exists()
    with sample_path.open("rb") as fp:
        magic = fp.read(2)
    assert magic == b"\x1f\x8b"  # gzip magic

    with gzip.open(sample_path, "rt", encoding="utf-8") as fp:
        header = json.loads(fp.readline())
    assert header["schema"] == SCHEMA_VERSION


def _write_paced_recording(path: Path, deltas_sec: list[float]) -> None:
    """Write a minimal recording with the given inter-op wall-clock deltas."""
    import time as _time

    t0 = _time.time()
    header = {"schema": SCHEMA_VERSION, "endianess": "<", "created_at": t0}
    entries = [{"t": t0, "op": "connect", "args": {}}]
    cursor = t0
    for i, delta in enumerate(deltas_sec):
        cursor += delta
        entries.append(
            {
                "t": cursor,
                "op": "read32",
                "args": {"at": i, "amount": 1},
                "result": [i],
            }
        )
    with gzip.open(path, "wt", encoding="utf-8") as fp:
        fp.write(json.dumps(header) + "\n")
        for entry in entries:
            fp.write(json.dumps(entry) + "\n")


def test_replay_is_not_live(sample_path, sample_backend):
    """ReplayScraper advertises is_live=False so the TUI can block mutations."""
    _record_session(sample_path, sample_backend)
    replay = ReplayScraper(sample_path)
    assert replay.is_live is False


def test_live_backends_advertise_is_live_by_default():
    """AbstractScraper default is is_live=True; concrete live backends inherit it."""
    from backend.gdb import GDBScraper

    scraper = GDBScraper("localhost:1234")
    assert scraper.is_live is True


def test_honor_timing_sleeps_to_recorded_wall_clock(tmp_path):
    """With honor_timing=True, replay paces itself to the recorded inter-op gap."""
    import time as _time

    path = tmp_path / "paced.ndjson.gz"
    _write_paced_recording(path, deltas_sec=[0.0, 0.25])

    replay = ReplayScraper(path, honor_timing=True)
    replay.connect()
    replay.read32(0, 1)

    start = _time.monotonic()
    replay.read32(1, 1)
    elapsed = _time.monotonic() - start

    # Second read is 0.25 s past the first; allow slack for scheduler jitter.
    assert elapsed >= 0.2


def test_honor_timing_false_drains_instantly(tmp_path):
    """With honor_timing=False, the same recording replays without any sleep."""
    import time as _time

    path = tmp_path / "fast.ndjson.gz"
    _write_paced_recording(path, deltas_sec=[1.0, 1.0])

    replay = ReplayScraper(path, honor_timing=False)

    start = _time.monotonic()
    replay.connect()
    replay.read32(0, 1)
    replay.read32(1, 1)
    elapsed = _time.monotonic() - start

    # Recording spans 2 s; honor_timing=False drains it in a few ms.
    assert elapsed < 0.1
