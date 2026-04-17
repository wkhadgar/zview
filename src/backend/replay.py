# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
ReplayScraper reads an NDJSON-over-gzip recording produced by RecordingScraper
and satisfies the AbstractScraper interface by returning each recorded result
in sequence. Strict drift detection raises ReplayMismatch when the caller's
op or args diverge from the next recorded entry.
"""

import gzip
import json
from collections.abc import Sequence
from pathlib import Path

from backend.recording import SCHEMA_VERSION
from backend.z_scraper import AbstractScraper


class ReplayError(Exception):
    """Base for replay errors."""


class ReplayMismatch(ReplayError):
    """Replay op/args did not match the next recorded entry."""


class ReplayExhausted(ReplayError):
    """Recording ran out of entries before replay completed."""


class UnsupportedSchema(ReplayError):
    """Recording header advertises a schema this reader does not handle."""


class ReplayScraper(AbstractScraper):
    """
    Replays a recording produced by RecordingScraper. Strict: every call must
    match the next entry's op and args, else ReplayMismatch is raised. Running
    past the end raises ReplayExhausted.
    """

    def __init__(self, source_path: Path | str):
        super().__init__(target_mcu=None)
        self._source_path = Path(source_path)
        self._entries: list[dict] = []
        self._cursor: int = 0
        self._loaded: bool = False

    def _load(self) -> None:
        """
        Parse the header and eagerly buffer every entry. Idempotent: subsequent
        calls are no-ops. Raises UnsupportedSchema if the header is missing or
        advertises a schema this reader does not understand.
        """
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
        Advance the cursor by one entry and return its stored result. The
        caller's ``op`` and ``args`` must match the entry exactly; any drift
        raises ReplayMismatch, and advancing past the last entry raises
        ReplayExhausted.
        """
        if self._cursor >= len(self._entries):
            raise ReplayExhausted(f"Recording exhausted before op {op}({args}).")

        entry = self._entries[self._cursor]
        if entry["op"] != op or entry["args"] != args:
            raise ReplayMismatch(
                f"Replay drift at index {self._cursor}: "
                f"expected {entry['op']}({entry['args']}), got {op}({args})."
            )
        self._cursor += 1
        return entry.get("result")

    def connect(self) -> None:
        self._load()
        self._next("connect", {})
        self._is_connected = True

    def disconnect(self) -> None:
        self._next("disconnect", {})
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
