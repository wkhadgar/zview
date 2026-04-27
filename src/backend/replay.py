# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""ReplayScraper: AbstractScraper backed by a RecordingScraper output file."""

import gzip
import json
import time
from collections.abc import Sequence
from pathlib import Path

from backend.base import AbstractScraper
from backend.recording import SCHEMA_VERSION


class ReplayError(Exception):
    """Base for replay errors."""


class ReplayMismatch(ReplayError):
    """Replay op/args did not match the next recorded entry."""


class ReplayExhausted(ReplayError):
    """Recording ran out of entries before replay completed."""


class ReplayComplete(ReplayError):
    """Caller attempted a new frame past the trailing disconnect — the recording ended cleanly."""


class UnsupportedSchema(ReplayError):
    """Recording header advertises a schema this reader does not handle."""


class ReplayScraper(AbstractScraper):
    """
    Replays a RecordingScraper file. Strict: every call's op and args must
    match the next recorded entry, else raises ReplayMismatch. Past the end
    raises ReplayExhausted. ``honor_timing=True`` paces calls to the recorded
    timestamps.
    """

    # A replay cursor can neither rewind nor absorb changes to the polling
    # shape (thread pool reductions, heap fragmentation toggles, etc.) without
    # drifting against the recording.
    is_live = False

    def __init__(self, source_path: Path | str, honor_timing: bool = True):
        super().__init__(target_mcu=None)
        self._source_path = Path(source_path)
        self._entries: list[dict] = []
        self._cursor: int = 0
        self._loaded: bool = False
        self._honor_timing = honor_timing
        self._anchor_wall: float | None = None
        self._anchor_recording: float | None = None

    def _load(self) -> None:
        """Parse header and buffer all entries. Idempotent. Raises UnsupportedSchema."""
        if self._loaded:
            return

        with gzip.open(self._source_path, "rt", encoding="utf-8") as fp:
            header_line = fp.readline()
            if not header_line:
                raise UnsupportedSchema(f"Empty recording: {self._source_path}")

            header = json.loads(header_line)
            if header.get("schema") != SCHEMA_VERSION:
                raise UnsupportedSchema(
                    f"Expected schema {SCHEMA_VERSION}, got {header.get('schema')!r}"
                )

            self.endianess = header.get("endianess", "<")

            for line in fp:
                line = line.strip()
                if not line:
                    continue
                self._entries.append(json.loads(line))

        self._loaded = True

    def _next(self, op: str, args: dict):
        """
        Advance cursor and return the entry's stored result.
        Raises ReplayMismatch on op/args drift, ReplayExhausted past end,
        ReplayComplete when ``op == 'begin_batch'`` lands on a trailing ``disconnect``.
        """
        if self._cursor >= len(self._entries):
            raise ReplayExhausted(f"Recording exhausted before op {op}({args}).")

        entry = self._entries[self._cursor]

        # A caller starting a new frame while the cursor sits on the trailing
        # ``disconnect`` entry means the recording has been fully consumed —
        # that's a clean end-of-stream, not a drift.
        if op == "begin_batch" and entry["op"] == "disconnect":
            raise ReplayComplete(f"Recording ended cleanly at index {self._cursor}.")

        if entry["op"] != op or entry["args"] != args:
            raise ReplayMismatch(
                f"Replay drift at index {self._cursor}: "
                f"expected {entry['op']}({entry['args']}), got {op}({args})."
            )

        if self._honor_timing:
            self._pace_to(entry.get("t"))

        self._cursor += 1
        return entry.get("result")

    def _pace_to(self, recording_t: float | None) -> None:
        """Sleep until the wall-clock matches ``recording_t`` (anchored on first call)."""
        if recording_t is None:
            return

        now = time.monotonic()
        if self._anchor_wall is None:
            self._anchor_wall = now
            self._anchor_recording = recording_t
            return

        target = self._anchor_wall + (recording_t - self._anchor_recording)
        delay = target - now
        if delay > 0:
            time.sleep(delay)

    def connect(self) -> None:
        self._load()
        self._next("connect", {})
        self._is_connected = True

    def disconnect(self) -> None:
        """Consume a trailing ``disconnect`` entry if present; mark disconnected."""
        if (
            self._cursor < len(self._entries)
            and self._entries[self._cursor]["op"] == "disconnect"
            and self._entries[self._cursor]["args"] == {}
        ):
            self._cursor += 1
        self._is_connected = False

    def begin_batch(self) -> None:
        self._next("begin_batch", {})

    def end_batch(self) -> None:
        self._next("end_batch", {})

    def read_bytes(self, at: int, amount: int) -> bytes:
        return bytes.fromhex(self._next("read_bytes", {"at": at, "amount": amount}))

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        return tuple(self._next("read8", {"at": at, "amount": amount}))

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        return tuple(self._next("read32", {"at": at, "amount": amount}))

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        return tuple(self._next("read64", {"at": at, "amount": amount}))
