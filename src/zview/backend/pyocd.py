# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""PyOCDScraper: AbstractScraper backend wrapping a pyOCD session."""

import struct
from collections.abc import Callable, Iterator, Sequence

from pyocd.core.helpers import ConnectHelper
from pyocd.core.session import Session
from pyocd.core.target import Target

from zview.backend.base import AbstractScraper, ProbeConnectFailure, ProbeReadFailure, Read

# A read32 or read64 of up to this many words is queued word by word. A longer
# one is a block read, which moves each word in fewer bytes than a queued
# read's address and data.
_QUEUED_WORDS = 16

# The SWD clock, in Hz, when none is given. pyOCD's own default is 1 MHz.
DEFAULT_FREQUENCY = 4_000_000


class PyOCDScraper(AbstractScraper):
    def __init__(self, target_mcu: str | None, frequency: int | None = None):
        super().__init__(target_mcu)
        self.session: Session | None = None
        self.target: Target | None = None
        self.frequency = frequency or DEFAULT_FREQUENCY

    def connect(self):
        if self._is_connected:
            return

        self.session = ConnectHelper.session_with_chosen_probe(
            target_override=self._target_mcu, connect_mode="attach", frequency=self.frequency
        )
        if self.session is None:
            raise ProbeConnectFailure("Unable to create a PyOCD session.")

        try:
            self.session.open()
            self.target = self.session.target
        except Exception as e:
            raise ProbeConnectFailure(
                f"Unable to connect PyOCD to MCU [{self._target_mcu}]: {e}"
            ) from e

        self._is_connected = True

    def disconnect(self):
        if self.session is None or not self._is_connected:
            return

        self.session.close()
        self._is_connected = False

    def read_bytes(self, at: int, amount: int) -> bytes:
        if self.target is None:
            raise ProbeReadFailure("No target available.")

        return bytes(self.target.read_memory_block8(at, amount))

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        if self.target is None:
            raise ProbeReadFailure("No target available.")

        return self.target.read_memory_block8(at, amount)

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        if self.target is None:
            raise ProbeReadFailure("No target available.")

        return self.target.read_memory_block32(at, amount)

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        if self.target is None:
            raise ProbeReadFailure("No target available.")

        raw_bytes = bytes(self.target.read_memory_block8(at, amount * 8))
        return struct.unpack(f"{self.endianess}{amount}Q", raw_bytes)

    def read_many(self, reads: Sequence[Read]) -> Iterator[bytes | Sequence[int]]:
        """
        Queue the small reads' words with ``now=False`` and collect them after.

        pyOCD packs queued transfers into as few probe packets as fit, and sends
        them before any read made at once, so the probe sees the reads in order.
        The others are made at once in their place. A fault aborts every queued
        transfer and is reported at the first, so a batch that raises is made
        again one read at a time.
        """
        if self.target is None:
            raise ProbeReadFailure("No target available.")

        try:
            collectors = [self._queue(method, at, amount) for method, at, amount in reads]
            results = [collect() for collect in collectors]
        except Exception:
            yield from super().read_many(reads)
            return

        yield from results

    def _queue(self, method: str, at: int, amount: int) -> Callable[[], bytes | Sequence[int]]:
        """A callable returning the read's result: queued words, or a read made now."""
        words_per_item = {"read32": 1, "read64": 2}.get(method)
        if words_per_item is None or amount * words_per_item > _QUEUED_WORDS:
            result = getattr(self, method)(at, amount)
            return lambda: result

        words = [
            self.target.read_memory(at + 4 * i, 32, now=False)
            for i in range(amount * words_per_item)
        ]
        if method == "read32":
            return lambda: tuple(word() for word in words)

        def as_longs() -> tuple[int, ...]:
            raw = b"".join(
                word().to_bytes(4, "little" if self.endianess == "<" else "big") for word in words
            )
            return struct.unpack(f"{self.endianess}{amount}Q", raw)

        return as_longs
