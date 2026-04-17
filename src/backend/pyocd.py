# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""PyOCDScraper: AbstractScraper backend wrapping a pyOCD session."""

import struct
from collections.abc import Sequence

from pyocd.core.helpers import ConnectHelper
from pyocd.core.session import Session
from pyocd.core.target import Target

from backend.base import AbstractScraper


class PyOCDScraper(AbstractScraper):
    def __init__(self, target_mcu: str | None):
        super().__init__(target_mcu)
        self.session: Session | None = None
        self.target: Target | None = None

    def connect(self):
        if self._is_connected:
            return

        self.session = ConnectHelper.session_with_chosen_probe(
            target_override=self._target_mcu, connect_mode="attach"
        )
        if self.session is None:
            raise Exception("Unable to create a PyOCD session.")

        try:
            self.session.open()
            self.target = self.session.target
        except Exception as e:
            raise Exception(f"\nUnable to connect with MCU [{self._target_mcu}].\n") from e

        self._is_connected = True

    def disconnect(self):
        if self.session is None or not self._is_connected:
            return

        self.session.close()
        self._is_connected = False

    def read_bytes(self, at: int, amount: int) -> bytes:
        if self.target is None:
            raise Exception("No target available.")

        return bytes(self.target.read_memory_block8(at, amount))

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        if self.target is None:
            raise Exception("No target available.")

        return self.target.read_memory_block8(at, amount)

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        if self.target is None:
            raise Exception("No target available.")

        return self.target.read_memory_block32(at, amount)

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        if self.target is None:
            raise Exception("No target available.")

        raw_bytes = bytes(self.target.read_memory_block8(at, amount * 8))
        return struct.unpack(f'{self.endianess}{amount}Q', raw_bytes)
