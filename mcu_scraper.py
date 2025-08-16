# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Sequence

from pylink import JLink, JLinkException, JLinkInterfaces
from pyocd.core.helpers import ConnectHelper
from pyocd.core.session import Session
from pyocd.core.target import Target


class AbstractScraper:
    def __init__(self, target_mcu: str | None):
        self.target_mcu: str | None = target_mcu

    def __enter__(self):
        self.connect()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        print("Connect was called")

    def disconnect(self):
        print("Disconnect was called")

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        print(f"Read {amount} bytes from {hex(at)}")
        return []

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        print(f"Read {amount} words from {hex(at)}")
        return []

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        print(f"Read {amount} double words from {hex(at)}")
        return []


class PyOCDScraper(AbstractScraper):
    def __init__(self, target_mcu: str | None):
        super().__init__(target_mcu)
        self.session: Session | None = None
        self.target: Target | None = None

    def connect(self):
        try:
            if self.target_mcu is not None:
                self.session = ConnectHelper.session_with_chosen_probe(target_override=self.target_mcu,
                                                                       connect_mode="attach")
            else:
                self.session = ConnectHelper.session_with_chosen_probe(connect_mode="attach")
            self.session.open()
            self.target = self.session.target
        except:
            raise Exception(f"\nUnable to connect with MCU [{self.target_mcu}].")

    def disconnect(self):
        self.session.close()

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.target.read_memory_block8(at, amount)

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.target.read_memory_block32(at, amount)

    def read64(self, at, amount: int = 1) -> Sequence[int]:
        words = self.read32(at, amount * 2)
        dwords = []
        for i in range(0, len(words), 2):
            dwords.append(((words[i] << 32) | words[i + 1]))

        return dwords


class JLinkScraper(AbstractScraper):
    def __init__(self, target_mcu: str | None):
        super().__init__(target_mcu)
        self.probe = JLink()

    def connect(self):
        try:
            self.probe.open()
        except JLinkException:
            raise Exception("\nNão foi possível iniciar o JLink, ele está conectado?")

        self.probe.set_tif(JLinkInterfaces.SWD)
        print(f"Pesquisando MCU via {self.probe.product_name}")

        try:
            self.probe.connect(self.target_mcu)
        except JLinkException:
            self.probe.close()
            raise Exception(f"\nNão foi possível conectar com a MCU [{self.target_mcu}], verifique suas conexões.")

    def disconnect(self):
        self.probe.close()

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read8(at, amount)

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read32(at, amount)

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        return self.probe.memory_read64(at, amount)
