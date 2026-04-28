# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""RecordingScraper: AbstractScraper wrapper that streams every call to a gzipped NDJSON file."""

import gzip
import json
import time
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import IO

from backend.base import AbstractScraper

SCHEMA_VERSION = "zview-recording/2"


class RecordingScraper(AbstractScraper):
    """
    AbstractScraper wrapper that emits each call as one NDJSON line.
    First line is a header ({schema, endianess, created_at}); ``read_bytes``
    payloads are hex-encoded.
    """

    def __init__(self, wrapped: AbstractScraper, out_path: Path | str):
        super().__init__(target_mcu=None)
        self._wrapped = wrapped
        self._out_path = Path(out_path)
        self._fp: IO | None = None
        self._stack = ExitStack()
        self.endianess = wrapped.endianess

    def _open(self) -> None:
        """Open the gzip stream and write the header. Idempotent."""
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
        """Append one call entry: ``{t, op, args[, result]}``."""
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

    def _record_read(self, op: str, fn, at: int, amount: int) -> Sequence[int]:
        result = fn(at, amount)
        self._emit(op, {"at": at, "amount": amount}, list(result))
        return result

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        return self._record_read("read8", self._wrapped.read8, at, amount)

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        return self._record_read("read32", self._wrapped.read32, at, amount)

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        return self._record_read("read64", self._wrapped.read64, at, amount)
