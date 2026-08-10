# Copyright (c) 2026 Flavio Ceolin (flavio.ceolin@gmail.com)
#
# SPDX-License-Identifier: Apache-2.0

"""NrfutilScraper: AbstractScraper backend driving a probe via the ``nrfutil`` CLI."""

import json
import shutil
import struct
import subprocess
from collections.abc import Sequence

from backend.base import (
    AbstractScraper,
    ProbeConnectFailure,
    ProbeReadError,
    ProbeReadMalformed,
    ProbeReadTimeout,
)


class NrfutilScraper(AbstractScraper):
    """
    Memory-read backend that shells out to ``nrfutil device read``.

    Unlike the pylink/pyOCD backends there is no persistent session object:
    ``nrfutil`` is a stateless CLI, so every read spawns a fresh subprocess.
    ``target_mcu`` is interpreted as the probe serial number; when omitted the
    ``jlink`` device trait is used, which requires exactly one J-Link attached.
    Reads use ``--direct`` (a pure probe memory read) so arbitrary RAM/flash
    addresses are accepted without configuring programming memory controllers.
    """

    def __init__(self, target_mcu: str | None, timeout: float = 20.0):
        super().__init__(target_mcu)
        self._timeout = timeout
        self._nrfutil = shutil.which("nrfutil")

    def _selector(self) -> list[str]:
        """Probe-selection flags: serial number when known, else the jlink trait."""
        if self._target_mcu:
            return ["--serial-number", self._target_mcu]
        return ["--traits", "jlink"]

    def connect(self):
        if self._is_connected:
            return

        if self._nrfutil is None:
            raise ProbeConnectFailure("`nrfutil` executable not found on PATH.")

        # nrfutil holds no session; validate reachability with a small read.
        try:
            self._read_mem_raw(0x0, 4)
        except (ProbeReadError, ProbeReadTimeout, ProbeReadMalformed) as e:
            raise ProbeConnectFailure(
                f"Unable to reach target via nrfutil [{self._target_mcu or 'jlink'}]: {e}"
            ) from e

        self._is_connected = True

    def disconnect(self):
        # Stateless CLI: nothing to tear down beyond the connected flag.
        self._is_connected = False

    def _read_mem_raw(self, at: int, amount: int) -> bytes:
        """
        Run ``nrfutil device read`` for ``amount`` bytes at ``at`` and return them.

        Raises ``ProbeReadTimeout`` if the subprocess exceeds the deadline,
        ``ProbeReadError`` if nrfutil reports a failed read, and
        ``ProbeReadMalformed`` if the JSON output cannot be decoded.
        """
        if amount <= 0:
            return b""

        cmd = [
            self._nrfutil or "nrfutil",
            "--json",
            "device",
            "read",
            "--address",
            hex(at),
            "--bytes",
            str(amount),
            "--direct",
            *self._selector(),
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            raise ProbeReadTimeout(f"nrfutil read timed out reading {amount}B at {hex(at)}.") from e

        values = self._extract_values(proc.stdout)
        if values is None:
            detail = self._extract_error(proc.stdout) or proc.stderr.strip() or "no data returned"
            raise ProbeReadError(f"nrfutil failed to read {amount}B at {hex(at)}: {detail}")

        try:
            raw = bytes(values)
        except ValueError as e:
            raise ProbeReadMalformed(f"Malformed data for address {hex(at)}") from e

        if len(raw) < amount:
            raise ProbeReadMalformed(f"nrfutil failed to read {amount}B at {hex(at)}")

        return raw[:amount]

    @staticmethod
    def _iter_json_lines(stdout: str):
        """Yield each decodable JSON object from nrfutil's JSON-lines stdout."""
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

    @classmethod
    def _extract_values(cls, stdout: str) -> list[int] | None:
        """
        Pull the byte list from a successful ``memory-read`` task_end record.

        nrfutil emits one JSON object per line; the payload of interest is::

            {"type":"task_end","data":{...,"result":"success",
             "data":{"memoryData":[{"startAddress":"0x0","values":[...]}]}}}
        """
        for obj in cls._iter_json_lines(stdout):
            if obj.get("type") != "task_end":
                continue
            data = obj.get("data", {})
            if data.get("result") != "success":
                continue
            memory = data.get("data", {}).get("memoryData")
            if memory:
                return [v for block in memory for v in block.get("values", [])]
        return None

    @classmethod
    def _extract_error(cls, stdout: str) -> str | None:
        """Return the message of the first failed task_end record, if any."""
        for obj in cls._iter_json_lines(stdout):
            if obj.get("type") != "task_end":
                continue
            data = obj.get("data", {})
            if data.get("result") == "fail":
                return data.get("message")
        return None

    def read_bytes(self, at: int, amount: int) -> bytes:
        return self._read_mem_raw(at, amount)

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        return list(self._read_mem_raw(at, amount))

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        return struct.unpack(f"{self.endianess}{amount}I", self._read_mem_raw(at, amount * 4))

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        return struct.unpack(f"{self.endianess}{amount}Q", self._read_mem_raw(at, amount * 8))
