# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
RecordingScraper captures every call issued to a live AbstractScraper and
streams it to a gzip-compressed NDJSON file. The resulting recording can be
played back offline through ReplayScraper without a probe or target attached.
"""

import gzip
import json
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import IO

from backend.base import AbstractScraper

SCHEMA_VERSION = "zview-recording/1"


class RecordingScraper(AbstractScraper):
    """
    Wraps an AbstractScraper and serializes every call to a gzip-compressed
    NDJSON stream. The first line is a header (schema version, endianess,
    creation timestamp); subsequent lines are call entries in issue order.
    Bulk ``read_bytes`` payloads are hex-encoded; gzip reclaims the bloat.
    """

    def __init__(self, wrapped: AbstractScraper, out_path: Path | str):
        super().__init__(target_mcu=None)
        self._wrapped = wrapped
        self._out_path = Path(out_path)
        self._fp: IO | None = None
        self._stack = ExitStack()
        self.endianess = wrapped.endianess

    def _open(self) -> None:
        """
        Open the gzip stream and write the header line. Idempotent: repeat
        calls on an already-open recorder are no-ops.
        """
        if self._fp is not None:
            return
        # File lifetime spans connect()/disconnect(), tracked via ExitStack.
        # SIM115's syntactic check does not recognize ExitStack.enter_context.
        self._fp = self._stack.enter_context(
            gzip.open(self._out_path, "wt", encoding="utf-8")  # noqa: SIM115
        )
        header = {
            "schema": SCHEMA_VERSION,
            "endianess": self.endianess,
            "created_at": time.time(),
        }
        self._fp.write(json.dumps(header) + "\n")

    def _emit(self, op: str, args: dict, result=None) -> None:
        """
        Append one call entry to the stream. Entries are JSON objects with
        fields ``t`` (wall-clock timestamp), ``op`` (method name), ``args``
        (keyword arguments dict) and, for read ops, ``result`` (decoded return
        value; ``bytes`` payloads are stored as hex strings).
        """
        if self._fp is None:
            return
        entry: dict = {"t": time.time(), "op": op, "args": args}
        if result is not None:
            entry["result"] = result
        self._fp.write(json.dumps(entry) + "\n")

    def connect(self) -> None:
        self._open()
        self._wrapped.connect()
        self.endianess = self._wrapped.endianess
        self._is_connected = self._wrapped.is_connected
        self._emit("connect", {})

    def disconnect(self) -> None:
        self._wrapped.disconnect()
        self._is_connected = self._wrapped.is_connected
        self._emit("disconnect", {})
        self._stack.close()
        self._fp = None

    def begin_batch(self) -> None:
        self._wrapped.begin_batch()
        self._emit("begin_batch", {})

    def end_batch(self) -> None:
        self._wrapped.end_batch()
        self._emit("end_batch", {})

    def read_bytes(self, at: int, amount: int) -> bytes:
        result = self._wrapped.read_bytes(at, amount)
        self._emit("read_bytes", {"at": at, "amount": amount}, bytes(result).hex())
        return result

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        result = self._wrapped.read8(at, amount)
        self._emit("read8", {"at": at, "amount": amount}, list(result))
        return result

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        result = self._wrapped.read32(at, amount)
        self._emit("read32", {"at": at, "amount": amount}, list(result))
        return result

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        result = self._wrapped.read64(at, amount)
        self._emit("read64", {"at": at, "amount": amount}, list(result))
        return result
