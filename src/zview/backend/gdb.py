# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""GDBScraper: AbstractScraper backend implementing the GDB remote serial protocol."""

import contextlib
import logging
import socket
import struct
import sys
from collections.abc import Sequence

from zview.backend.base import (
    AbstractScraper,
    ProbeConnectFailure,
    ProbeReadError,
    ProbeReadMalformed,
    ProbeReadTimeout,
)

logger = logging.getLogger("zview.scraper")


class GDBScraper(AbstractScraper):
    def __init__(self, host: str = 'localhost:1234', timeout: float = 3.0):
        super().__init__("GDB Server")
        try:
            host, port = host.strip().split(":")
        except ValueError:
            raise ValueError(
                "GDB target must be in the format 'host:port'. e.g.: 'localhost:1234'"
            ) from None

        self.host: str = host
        self.port: int = int(port)
        self.timeout: float = timeout
        self.sock: socket.SocketType | None = None
        # A 'c' is pending, the only state a stub takes the interrupt byte in.
        self._running = False
        # The stub expects a '+' for each reply.
        self._acks = True

    def begin_batch(self):
        self._halt()

    def end_batch(self):
        self._resume()

    def connect(self):
        """
        Connect to a GDB RSP stub and perform the initial handshake.

        Switches to no-ack mode for faster transfers. The stub holds the target
        stopped on attach; halt/resume is managed per batch.
        """
        if self._is_connected:
            return

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # An ack and the next packet go out apiece, not held for a TCP ACK.
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
        except OSError as e:
            self.sock.close()
            self.sock = None
            raise ProbeConnectFailure(
                f"Could not connect to {self.host}:{self.port}. Is QEMU running with -s?"
            ) from e

        self._drain()

        self._acks = True
        self.sock.sendall(b'+')
        self._send_packet(b'QStartNoAckMode')
        try:
            # A stub without no-ack mode, such as QEMU's, answers empty and has
            # each of its replies acked.
            self._acks = self._read_response() != b'OK'
        except OSError as e:
            self.sock.close()
            self.sock = None
            raise ProbeConnectFailure(
                f"GDB server at {self.host}:{self.port} did not answer."
            ) from e

        self._running = False
        self._is_connected = True
        print(f"Connected to GDB server at {self.host}:{self.port}", file=sys.stderr)
        logger.info("Connected to GDB server at %s:%d", self.host, self.port)

    def disconnect(self):
        if self.sock is None:
            return

        with contextlib.suppress(Exception):
            self._halt()
            self._send_packet(b'D')
            self.sock.settimeout(1.0)
            self._read_response()

        self.sock.close()
        self.sock = None
        self._is_connected = False

    def _drain(self):
        """Discard buffered bytes before the RSP handshake (e.g. QEMU connection banners)."""
        if self.sock is None:
            return
        self.sock.settimeout(0.2)
        with contextlib.suppress(socket.timeout, BlockingIOError):
            while self.sock.recv(4096):
                pass

        self.sock.settimeout(self.timeout)

    def _halt(self):
        """
        Interrupt the running target via the out-of-band SIGINT byte (0x03).

        Sent raw outside RSP packet framing per the GDB remote serial protocol
        spec, and only while a 'c' is pending: a stopped target's stub ignores
        it. Consumes the resulting stop-reply (T05/S05); tolerates stubs that
        omit it.
        """
        if self.sock is None or not self._running:
            return

        self.sock.sendall(b'\x03')
        self._running = False
        with contextlib.suppress(socket.timeout):
            self._read_response()

    def _resume(self):
        """Resume target execution via the RSP 'c' command."""
        if self.sock is None:
            return
        self._send_packet(b'c')
        self._running = True

    def _send_packet(self, data: bytes):
        """Transmit one RSP packet: ``$<data>#<checksum>``."""
        if self.sock is None:
            raise ConnectionError("No GDB server connected.")
        checksum = sum(data) % 256
        self.sock.sendall(b'$' + data + f'#{checksum:02x}'.encode())

    def _read_response(self) -> bytes:
        """
        Receive one RSP response packet and return its payload, acked when the
        stub expects it.

        Strips leading ``+`` ack bytes and the ``$...#cc`` framing. Raises
        ``socket.timeout`` on receive deadline; callers that treat a missing
        reply as non-fatal should catch it explicitly.
        """
        if self.sock is None:
            raise ConnectionError("No GDB server connected.")

        buffer = b''
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("GDB server closed the connection.")
            buffer += chunk

            while buffer.startswith(b'+'):
                buffer = buffer[1:]

            if b'$' not in buffer:
                continue

            start = buffer.index(b'$')
            if b'#' not in buffer[start:]:
                continue
            end = buffer.index(b'#', start)
            if len(buffer) < end + 3:
                continue

            if self._acks:
                self.sock.sendall(b'+')
            return buffer[start + 1 : end]

    # Largest read per packet: its hex reply fits the 4 KB buffer of QEMU's
    # stub, and the larger ones of OpenOCD and J-Link.
    _MAX_READ_BYTES = 1024

    def _read_mem_raw(self, addr: int, length: int) -> bytes:
        """
        Read ``length`` bytes at ``addr`` over as many ``m`` packets as it takes.

        A stub may answer with fewer bytes than asked, so each packet starts
        where the last reply ended. Raises ``ProbeReadTimeout``,
        ``ProbeReadError``, or ``ProbeReadMalformed`` when a reply is missing,
        rejected, undecodable or empty.
        """
        if self.sock is None:
            raise ConnectionError("No GDB server connected.")

        data = bytearray()
        while len(data) < length:
            at = addr + len(data)
            chunk = self._read_mem_packet(at, min(length - len(data), self._MAX_READ_BYTES))
            if not chunk:
                raise ProbeReadMalformed(f"Empty reply reading {hex(at)}.")
            data += chunk

        return bytes(data)

    def _read_mem_packet(self, addr: int, length: int) -> bytes:
        """Execute one ``m<addr>,<length>`` and return the bytes it decoded, at most ``length``."""
        self._send_packet(f'm{addr:x},{length:x}'.encode())

        try:
            resp = self._read_response()
        except TimeoutError as e:
            raise ProbeReadTimeout(f"Timeout reading {length}B at {hex(addr)}.") from e

        if resp.startswith(b'E'):
            raise ProbeReadError(f"GDB error at {hex(addr)}: {resp.decode()}")

        try:
            raw = bytes.fromhex(resp.decode('ascii'))
        except ValueError as e:
            raise ProbeReadMalformed(f"Malformed hex response at {hex(addr)}: {resp!r}") from e

        return raw[:length]

    def read_bytes(self, at: int, amount: int = 1) -> bytes:
        return self._read_mem_raw(at, amount)

    def read8(self, at: int, amount: int = 1) -> Sequence[int]:
        return list(self._read_mem_raw(at, amount))

    def read16(self, at: int, amount: int = 1) -> Sequence[int]:
        return struct.unpack(f'{self.endianess}{amount}H', self._read_mem_raw(at, amount * 2))

    def read32(self, at: int, amount: int = 1) -> Sequence[int]:
        return struct.unpack(f'{self.endianess}{amount}I', self._read_mem_raw(at, amount * 4))

    def read64(self, at: int, amount: int = 1) -> Sequence[int]:
        return struct.unpack(f'{self.endianess}{amount}Q', self._read_mem_raw(at, amount * 8))
