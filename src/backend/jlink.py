# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""JLinkScraper: AbstractScraper backend driving a Segger J-Link probe via pylink."""

from collections.abc import Sequence

from pylink import JLink, JLinkException, JLinkInterfaces

from backend.base import AbstractScraper, ProbeConnectFailure


class JLinkScraper(AbstractScraper):
    def __init__(self, target_mcu: str | None):
        super().__init__(target_mcu)
        self.probe = JLink()

    def connect(self):
        if self._is_connected:
            return

        try:
            self.probe.open()
        except JLinkException as e:
            raise ProbeConnectFailure(f"Unable to open JLink probe: {e}") from e

        self.probe.set_tif(JLinkInterfaces.SWD)

        try:
            self.probe.connect(self._target_mcu)
        except JLinkException as e:
            self.probe.close()
            raise ProbeConnectFailure(
                f"Unable to connect JLink to [{self._target_mcu}]: {e}"
            ) from e
        self._is_connected = True

    def disconnect(self):
        if not self._is_connected:
            return

        self.probe.close()
        self._is_connected = False

    def read_bytes(self, at: int, amount: int) -> bytes:
        return bytes(self.probe.memory_read8(at, amount))

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read8(at, amount)

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read32(at, amount)

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read64(at, amount)
