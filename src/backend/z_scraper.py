# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0


import contextlib
import logging
import queue
import socket
import struct
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Literal

from pylink import JLink, JLinkException, JLinkInterfaces
from pyocd.core.helpers import ConnectHelper
from pyocd.core.session import Session
from pyocd.core.target import Target
from yaml import safe_load

from backend.elf_inspector import ElfInspector

logger = logging.getLogger("zview.scraper")


class ProbeError(Exception):
    """Base for probe backend errors."""


class ProbeConnectFailure(ProbeError):
    """Probe failed to establish a session with the target."""


class ProbeReadFailure(ProbeError):
    """Probe returned no usable data for a memory read."""


class ProbeReadTimeout(ProbeReadFailure):
    """Probe read did not receive a response before the deadline."""


class ProbeReadError(ProbeReadFailure):
    """Probe returned an error reply to a memory read."""


class ProbeReadMalformed(ProbeReadFailure):
    """Probe returned an undecodable response to a memory read."""


class RunnerConfig:
    def __init__(self, runners_yaml_path):
        self.path = Path(runners_yaml_path)
        if not self.path.exists():
            raise FileNotFoundError(f"runners.yaml not found at {runners_yaml_path}")

        with open(self.path) as f:
            self.data = safe_load(f)

    def get_config(self, preferred_runner=None):
        """
        Returns a tuple of (runner_name, mcu_target)
        """
        runner = preferred_runner or self.data.get("flash-runner") or "jlink"

        args = self.data.get("args", {}).get(runner, [])

        mcu = None
        if runner == "jlink":
            mcu = self._find_arg(args, "--device")
        elif runner == "pyocd":
            mcu = self._find_arg(args, "--target")

        return runner, mcu

    def _find_arg(self, args_list, prefix):
        """Helper to find an argument starting with a specific prefix"""
        for arg in args_list:
            if arg.startswith(prefix):
                if "=" in arg:
                    return arg.split("=")[1]
                idx = args_list.index(arg)
                if idx + 1 < len(args_list):
                    return args_list[idx + 1]
        return None


@dataclass(frozen=True)
class ThreadRuntime:
    """
    Data class to hold runtime information about the thread.
    """

    cpu: float
    cpu_normalized: float
    active: bool
    stack_watermark: int
    stack_watermark_percent: float


@dataclass(frozen=True)
class ThreadInfo:
    """
    Data class to hold information about a single Zephyr RTOS thread.
    """

    address: int
    stack_start: int
    stack_size: int
    name: str
    runtime: ThreadRuntime | None


@dataclass(frozen=True)
class HeapInfo:
    name: str
    address: int
    free_bytes: int
    allocated_bytes: int
    max_allocated_bytes: int
    usage_percent: float
    chunks: list[dict] | None


class AbstractScraper(ABC):
    """Common interface for memory-read backends (JLink, pyOCD, GDB RSP)."""

    def __init__(self, target_mcu: str | None):
        self._target_mcu: str | None = target_mcu
        self._is_connected: bool = False
        self.watermark_cache = {}
        self.endianess: Literal["<", ">"] = "<"

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        self.disconnect()

    @property
    def is_connected(self):
        return self._is_connected

    @abstractmethod
    def connect(self): ...

    @abstractmethod
    def disconnect(self): ...

    # begin_batch/end_batch are optional hooks: GDB overrides them to halt/resume
    # the target; JLink and pyOCD inherit the no-op default because their probe
    # libraries do not require bracketing. Hence empty bodies on purpose.
    def begin_batch(self):  # noqa: B027
        pass

    def end_batch(self):  # noqa: B027
        pass

    @abstractmethod
    def read_bytes(self, at: int, amount: int) -> bytes: ...

    @abstractmethod
    def read8(self, at: int, amount: int = 1) -> Sequence[int]: ...

    @abstractmethod
    def read32(self, at: int, amount: int = 1) -> Sequence[int]: ...

    @abstractmethod
    def read64(self, at: int, amount: int = 1) -> Sequence[int]: ...

    def calculate_dynamic_watermark(
        self,
        stack_start: int,
        stack_size: int,
        unused_pattern: int = 0xAA_AA_AA_AA,
        *,
        thread_id,
    ) -> int:
        """
        Reads a stack memory and scans for the unused_pattern fill pattern
        to determine the current stack watermark (highest point of stack usage).

        Args:
            :param stack_start: The starting address of the thread's stack.
            :param stack_size: The total size of the thread's stack in bytes.
            :param unused_pattern: Unused stack fill word.
            :param id: Unique identification for the given thread.

        Returns:
            The calculated stack watermark in bytes, indicating the maximum
            amount of stack space that has been used.
        """
        if stack_size == 0:
            return 0

        cache_watermark = self.watermark_cache.get(thread_id, 0)
        watermark = stack_size - cache_watermark

        stack_words = self.read32(stack_start, (stack_size // 4) - (cache_watermark // 4))

        for word in stack_words:
            if word == unused_pattern:
                watermark -= 4
            else:
                break

        self.watermark_cache[thread_id] = watermark + cache_watermark

        return self.watermark_cache[thread_id]


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

    def begin_batch(self):
        self._halt()

    def end_batch(self):
        self._resume()

    def connect(self):
        """
        Connect to a GDB RSP stub and perform the initial handshake.

        Switches to no-ack mode for faster transfers. Target is left running;
        halt/resume is managed per read in _read_mem_raw.
        """
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
        except (ConnectionRefusedError, ConnectionResetError) as e:
            raise ConnectionError(
                f"Could not connect to {self.host}:{self.port}. Is QEMU running with -s?"
            ) from e

        self._drain()

        self.sock.sendall(b'+')
        self._send_packet(b'QStartNoAckMode')

        self._is_connected = True
        print(f"Connected to GDB server at {self.host}:{self.port}")
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
        Interrupt the target via the out-of-band SIGINT byte (0x03).

        Sent raw outside RSP packet framing per the GDB remote serial protocol
        spec. Consumes the resulting stop-reply (T05/S05); tolerates stubs that
        omit it.
        """
        if self.sock is None:
            return

        self.sock.sendall(b'\x03')
        with contextlib.suppress(socket.timeout):
            self._read_response()

    def _resume(self):
        """Resume target execution via the RSP 'c' command."""
        if self.sock is None:
            return
        self._send_packet(b'c')

    def _send_packet(self, data: bytes):
        """Transmit one RSP packet: ``$<data>#<checksum>``."""
        if self.sock is None:
            raise ConnectionError("No GDB server connected.")
        checksum = sum(data) % 256
        self.sock.sendall(b'$' + data + f'#{checksum:02x}'.encode())

    def _read_response(self) -> bytes:
        """
        Receive one RSP response packet and return its payload.

        Strips leading ``+`` ack bytes and the ``$…#cc`` framing. Raises
        ``socket.timeout`` on receive deadline — callers that treat a missing
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

            return buffer[start + 1 : end]

    def _read_mem_raw(self, addr: int, length: int) -> bytes:
        """
        Execute ``m<addr>,<length>`` and return the decoded byte string.
        Raises ``ProbeReadTimeout``, ``ProbeReadError``, or ``ProbeReadMalformed``
        when the reply is missing, rejected, or undecodable.
        """
        if self.sock is None:
            raise ConnectionError("No GDB server connected.")

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

        if len(raw) < length:
            raw += b'\x00' * (length - len(raw))

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
        return struct.unpack(f'{self.endianess}{amount}Q', raw_bytes)


class ZScraper:
    def __init__(
        self,
        meta_scraper: AbstractScraper,
        elf_path,
        max_threads: int = 64,
    ):
        self._all_threads_info: dict[str, ThreadInfo] = {}
        self._elf_inspector: ElfInspector = ElfInspector(elf_path)
        self._endianess: Literal["little", "big"] = (
            "little" if self._elf_inspector._little_endian else "big"
        )
        self._m_scraper: AbstractScraper = meta_scraper
        self._m_scraper.endianess = "<" if self._endianess == "little" else ">"
        self._polling_thread: threading.Thread | None = None
        self._thread_pool: list[ThreadInfo] | None = None
        self._stop_event: Event | None = None
        self.inspection_period = 0.2

        self.has_heaps: bool = True
        self.has_usage: bool = True
        self.has_names: bool = True

        self._MAX_THREADS: int = max_threads

        self._offsets = {
            "kernel": {
                "cpu": self._elf_inspector.get_struct_member_offset("z_kernel", "cpus"),
                "threads": self._elf_inspector.get_struct_member_offset("z_kernel", "threads"),
            },
            "k_thread": {
                "base": self._elf_inspector.get_struct_member_offset("k_thread", "base"),
                "stack_info": self._elf_inspector.get_struct_member_offset(
                    "k_thread", "stack_info"
                ),
                "next_thread": self._elf_inspector.get_struct_member_offset(
                    "k_thread", "next_thread"
                ),
            },
        }

        self._offsets["thread_info"] = {
            "stack_start": self._offsets["k_thread"]["stack_info"]
            + self._elf_inspector.get_struct_member_offset("_thread_stack_info", "start"),
            "stack_size": self._offsets["k_thread"]["stack_info"]
            + self._elf_inspector.get_struct_member_offset("_thread_stack_info", "size"),
            "stack_delta": self._offsets["k_thread"]["stack_info"]
            + self._elf_inspector.get_struct_member_offset("_thread_stack_info", "delta"),
        }

        # Known to be only one (at least, expected)
        self._kernel_base_address = self._elf_inspector.get_symbol_info("_kernel", info="address")[
            0
        ]
        self._threads_address = self._kernel_base_address + self._offsets["kernel"]["threads"]

        self.idle_threads_address = self._elf_inspector.get_symbol_info(
            "z_idle_threads", "address"
        )[0]

        try:
            self._offsets["k_thread"]["name"] = self._elf_inspector.get_struct_member_offset(
                "k_thread", "name"
            )
        except LookupError:
            self.has_names = False

        try:
            self._offsets["kernel"]["usage"] = self._elf_inspector.get_struct_member_offset(
                "z_kernel", "usage"
            )
            self._offsets["thread_info"].update(
                {
                    "usage_base": self._offsets["k_thread"]["base"]
                    + self._elf_inspector.get_struct_member_offset("_thread_base", "usage")
                }
            )
            self._cpu_usage_address = self._kernel_base_address + self._offsets["kernel"]["usage"]
            self._m_scraper.begin_batch()
            self.init_cpu_cycles = self._m_scraper.read64(self._cpu_usage_address)[0]
            self._m_scraper.end_batch()
            self.last_cpu_delta = self.init_cpu_cycles
            self.last_cpu_cycles = self.init_cpu_cycles
            self.last_thread_cycles = {}
        except LookupError:
            self.has_usage = False

        try:
            self._offsets["thread_info"]["usage"] = self._offsets["thread_info"][
                "usage_base"
            ] + self._elf_inspector.get_struct_member_offset("k_cycle_stats", "total")
            self._offsets["thread_info"]["longest_window"] = self._offsets["thread_info"][
                "usage_base"
            ] + self._elf_inspector.get_struct_member_offset("k_cycle_stats", "longest")

            self._offsets["thread_info"]["amount_windows"] = self._offsets["thread_info"][
                "usage_base"
            ] + self._elf_inspector.get_struct_member_offset("k_cycle_stats", "num_windows")
        except LookupError:
            self.has_extra_info = False

        try:
            self._offsets["heap_info"] = {
                "z_heap_base": self._elf_inspector.get_struct_member_offset("k_heap", "heap")
                + self._elf_inspector.get_struct_member_offset("sys_heap", "heap"),
                "free_bytes": self._elf_inspector.get_struct_member_offset("z_heap", "free_bytes"),
                "allocated_bytes": self._elf_inspector.get_struct_member_offset(
                    "z_heap", "allocated_bytes"
                ),
                "max_allocated_bytes": self._elf_inspector.get_struct_member_offset(
                    "z_heap", "max_allocated_bytes"
                ),
                "end_chunk": self._elf_inspector.get_struct_member_offset("z_heap", "end_chunk"),
            }

        except LookupError:
            self.has_heaps = False
            return

        all_heaps = self._elf_inspector.find_struct_variable_names("k_heap")

        if all_heaps is None or len(all_heaps) == 0:
            self.has_heaps = False
            return

        self._k_heap_addresses = {}
        self.extra_info_heap_address: int | None = None
        for heap in all_heaps:
            self._k_heap_addresses[heap] = self._elf_inspector.get_symbol_info(heap, "address")

    def __enter__(self):
        self._m_scraper.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._m_scraper.__exit__(exc_type, exc_val, exc_tb)

    @property
    def all_threads(self):
        return self._all_threads_info

    @property
    def thread_pool(self):
        return self._thread_pool

    @thread_pool.setter
    def thread_pool(self, new: list[ThreadInfo]):
        self._thread_pool = new

    def read_variable(self, var_name: str) -> list[Sequence[int]]:
        if not self._m_scraper.is_connected:
            self._m_scraper.connect()

        var_addresses = self._elf_inspector.get_symbol_info(var_name, "address")
        var_sizes = self._elf_inspector.get_symbol_info(var_name, "size")

        return [
            self._m_scraper.read8(var_address, var_size)
            for var_address, var_size in zip(var_addresses, var_sizes, strict=True)
        ]

    def update_available_threads(self):
        try:
            if not self._m_scraper.is_connected:
                self._m_scraper.connect()

            self._m_scraper.begin_batch()
            thread_ptr = (
                self._m_scraper.read32(self._threads_address)[0] if self._threads_address else 0
            )
        except Exception as e:
            self._m_scraper.end_batch()
            raise RuntimeError("Unable to read kernel thread list.") from e

        stack_struct_size = self._elf_inspector.get_struct_size("k_thread")
        words_to_read = stack_struct_size // 4
        next_ptr_word_idx = self._offsets["k_thread"]["next_thread"] // 4
        stack_start_word_idx = self._offsets["thread_info"]["stack_start"] // 4
        name_word_idx = self._offsets["k_thread"]["name"] // 4 if self.has_names else 0
        stack_size_word_idx = self._offsets["thread_info"]["stack_size"] // 4

        self.all_threads.clear()
        for _ in range(self._MAX_THREADS):
            if thread_ptr == 0:
                break

            try:
                thread_struct_words = self._m_scraper.read32(thread_ptr, words_to_read)
            except Exception as e:
                raise Exception(f"Error reading thread struct at 0x{thread_ptr:X}") from e

            if self.has_names:
                words = thread_struct_words[name_word_idx:]
                full_bytes = b''.join(w.to_bytes(4, self._endianess) for w in words)
                thread_name = full_bytes.split(b'\0', 1)[0].decode(errors="ignore")
            else:
                thread_name = f"thread @ 0x{thread_ptr:X}"

            self.all_threads[thread_name] = ThreadInfo(
                thread_ptr,
                thread_struct_words[stack_start_word_idx],
                thread_struct_words[stack_size_word_idx],
                thread_name,
                None,
            )

            thread_ptr = thread_struct_words[next_ptr_word_idx]

        self._m_scraper.end_batch()

    def get_heap_fragmentation(self, z_heap_addr: int) -> list[dict]:
        """
        Extracts the physical sequence of heap chunks via a single bulk memory read.
        """
        if z_heap_addr == 0:
            return []

        end_chunk = self._m_scraper.read32(z_heap_addr + self._offsets["heap_info"]["end_chunk"])[0]
        if end_chunk == 0:
            return []

        total_bytes = end_chunk * 8
        if total_bytes > (1024 * 1024 * 32):
            raise ValueError(f"Heap size exceeds sanity limit: {total_bytes} bytes.")

        raw_buffer = self._m_scraper.read_bytes(z_heap_addr, total_bytes)
        mv = memoryview(raw_buffer)

        # Evaluate the chunk0 signature structurally
        val16_raw = struct.unpack_from(f"{self._m_scraper.endianess}H", mv, 2)[0]
        val32_raw = struct.unpack_from(f"{self._m_scraper.endianess}I", mv, 4)[0]

        is_used_16 = bool(val16_raw & 1)
        is_used_32 = bool(val32_raw & 1)

        val16 = val16_raw >> 1
        val32 = val32_raw >> 1

        if (0 < val16 < end_chunk) and is_used_16:
            c = val16  # Start chunk ID
            fmt = f"{self._m_scraper.endianess}H"
            offset_in_chunk = 2
        elif (0 < val32 < end_chunk) and is_used_32:
            c = val32
            fmt = f"{self._m_scraper.endianess}I"
            offset_in_chunk = 4
        else:
            raise ValueError(
                f"Corrupted chunk0 header. "
                f"16-bit field: size {val16}, used {is_used_16} | "
                f"32-bit field: size {val32}, used {is_used_32}"
            )

        chunks = []

        while c < end_chunk:
            offset = (c * 8) + offset_in_chunk
            val = struct.unpack_from(fmt, mv, offset)[0]

            is_used = bool(val & 1)
            c_size = val >> 1

            if c_size == 0:
                raise RuntimeError(f"Infinite loop prevented: Chunk at ID {c} has size 0.")

            chunks.append({"used": is_used, "size": c_size * 8})
            c += c_size

        return chunks

    def reset_thread_pool(self):
        self.thread_pool = list(self.all_threads.values())

    def reset_runtime_state(self):
        """
        Drop all per-thread caches and re-baseline CPU cycle counters.
        Call this whenever the target has been reset or reconnected so that
        stale watermark assumptions and cycle deltas do not leak across sessions.
        """
        self._m_scraper.watermark_cache.clear()

        if not self.has_usage:
            return

        self.last_thread_cycles.clear()
        self._m_scraper.begin_batch()
        try:
            self.init_cpu_cycles = self._m_scraper.read64(self._cpu_usage_address)[0]
        finally:
            self._m_scraper.end_batch()
        self.last_cpu_cycles = self.init_cpu_cycles
        self.last_cpu_delta = self.init_cpu_cycles

    def start_polling_thread(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        inspection_period: float,
    ):
        """
        Starts a separate daemon thread to continuously poll data from the MCU.
        """
        if self._polling_thread is not None:
            data_queue.put({"error": "Already started..."})
            return

        self.inspection_period = inspection_period
        self._polling_thread = threading.Thread(
            target=self._poll_thread_worker,
            args=(data_queue, stop_event, inspection_period),
            daemon=True,
        )
        self._stop_event = stop_event
        self._polling_thread.daemon = True
        self._polling_thread.start()

    def finish_polling_thread(self):
        if self._polling_thread is None or self._stop_event is None:
            return

        self._stop_event.set()
        if self._polling_thread.is_alive():
            self._polling_thread.join(timeout=1.0)
            if self._polling_thread.is_alive():
                logger.warning("Polling thread did not terminate gracefully.")

        self._polling_thread = None

    def _poll_thread_worker(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        inspection_period: float,
    ):
        """
        Worker function executed by the polling thread.

        Connects to the MCU via J-Link, analyzes Zephyr kernel symbols from the ELF
        file, and continuously reads and processes runtime data, pushing it to
        `data_queue`.

        Args:
            data_queue: A queue for sending processed data to the main UI thread.
            stop_event: An event to signal the polling thread to terminate.
        """
        consecutive_errors = 0
        MAX_TOLERATED_ERRORS = 3

        try:
            if not self._m_scraper.is_connected:
                self._m_scraper.connect()
        except Exception as e:
            data_queue.put({"fatal_error": f"Failed to connect to target: {e}"})
            return

        while not stop_event.is_set():
            in_batch = False
            heap_info = []
            try:
                self._m_scraper.begin_batch()
                in_batch = True

                if self.has_usage:
                    try:
                        current_cpu_cycles = self._m_scraper.read32(self._cpu_usage_address)[0]
                    except Exception as e:
                        raise RuntimeError(f"Error reading global CPU cycles: {e}") from e

                    cpu_cycles_delta = current_cpu_cycles - self.last_cpu_cycles
                    self.last_cpu_cycles = current_cpu_cycles
                    if cpu_cycles_delta > 0:
                        self.last_cpu_delta = cpu_cycles_delta
                    elif (
                        cpu_cycles_delta <= 0
                    ):  # When current_cpu_cycles is 0 due to context resets
                        cpu_cycles_delta = self.last_cpu_delta
                else:
                    cpu_cycles_delta = -1

                if self.thread_pool is None:
                    raise RuntimeError("Thread pool is uninitialized.")

                polled_raw_data = []
                for thread_info in self.thread_pool:
                    if self.has_usage:
                        try:
                            thread_usage = self._m_scraper.read64(
                                thread_info.address + self._offsets["thread_info"]["usage"]
                            )[0]
                        except Exception as e:
                            raise RuntimeError(
                                f"Error polling thread usage {thread_info.name}: {e}"
                            ) from e

                        thread_usage_delta = thread_usage - self.last_thread_cycles.get(
                            thread_info.address, 0
                        )
                        self.last_thread_cycles[thread_info.address] = thread_usage

                        if thread_info.address == self.idle_threads_address:
                            cpu_cycles_delta += thread_usage_delta

                        is_active = thread_usage_delta > 0
                    else:
                        thread_usage_delta = 0
                        is_active = False

                    try:
                        watermark = self._m_scraper.calculate_dynamic_watermark(
                            thread_info.stack_start,
                            thread_info.stack_size,
                            thread_id=thread_info.address,
                        )
                    except Exception as e:
                        raise RuntimeError(
                            f"Error polling stack watermark for {thread_info.name}: {e}"
                        ) from e

                    stack_usage_pct = (
                        (watermark / thread_info.stack_size * 100)
                        if thread_info.stack_size > 0
                        else 0.0
                    )

                    polled_raw_data.append(
                        {
                            "info": thread_info,
                            "usage_delta": thread_usage_delta,
                            "is_active": is_active,
                            "watermark": watermark,
                            "stack_usage_pct": stack_usage_pct,
                        }
                    )

                idle_thread_data = next(
                    (d for d in polled_raw_data if d["info"].address == self.idle_threads_address),
                    None,
                )

                if idle_thread_data:
                    active_cycles_total = cpu_cycles_delta - idle_thread_data["usage_delta"]
                else:
                    active_cycles_total = cpu_cycles_delta

                final_threads = []
                for data in polled_raw_data:
                    # Absolute CPU % (t / total_system_time)
                    absolute_cpu = (
                        (data["usage_delta"] / cpu_cycles_delta * 100)
                        if cpu_cycles_delta > 0
                        else 0.0
                    )

                    # Relative Load % (t / total_active_time)
                    if data["info"].address == self.idle_threads_address:
                        load_pct = 0.0
                    elif active_cycles_total > 0:
                        load_pct = min((data["usage_delta"] / active_cycles_total) * 100.0, 100.0)
                    else:
                        load_pct = 0.0

                    final_runtime = ThreadRuntime(
                        cpu=load_pct,
                        cpu_normalized=absolute_cpu,
                        active=data["is_active"],
                        stack_watermark=data["watermark"],
                        stack_watermark_percent=data["stack_usage_pct"],
                    )
                    final_threads.append(
                        ThreadInfo(
                            address=data["info"].address,
                            stack_start=data["info"].stack_start,
                            stack_size=data["info"].stack_size,
                            name=data["info"].name,
                            runtime=final_runtime,
                        )
                    )

                if self.has_heaps:
                    for heap_name, heap_addresses in self._k_heap_addresses.items():
                        for heap_address in heap_addresses:
                            try:
                                heap_address_val = self._m_scraper.read32(heap_address)[0]
                                free_bytes = self._m_scraper.read32(
                                    heap_address_val + self._offsets["heap_info"]["free_bytes"]
                                )[0]
                                allocated_bytes = self._m_scraper.read32(
                                    heap_address_val + self._offsets["heap_info"]["allocated_bytes"]
                                )[0]
                                max_allocated_bytes = self._m_scraper.read32(
                                    heap_address_val
                                    + self._offsets["heap_info"]["max_allocated_bytes"]
                                )[0]
                            except Exception as e:
                                data_queue.put(
                                    {"error": f"Error reading heap info for {heap_name}: {e}"},
                                    block=False,
                                )
                                continue

                            chunks = None
                            if self.extra_info_heap_address == heap_address_val:
                                try:
                                    chunks = self.get_heap_fragmentation(heap_address_val)
                                except Exception as e:
                                    data_queue.put(
                                        {"error": f"Error reading sparsity for {heap_name}: {e}"},
                                        block=False,
                                    )

                            total_heap_size = allocated_bytes + free_bytes
                            heap_usage_pct = (
                                (allocated_bytes / total_heap_size * 100)
                                if total_heap_size > 0
                                else 0.0
                            )

                            heap_info.append(
                                HeapInfo(
                                    heap_name,
                                    heap_address_val,
                                    free_bytes,
                                    allocated_bytes,
                                    max_allocated_bytes,
                                    heap_usage_pct,
                                    chunks,
                                )
                            )

                if self.has_heaps:
                    data_queue.put({"threads": final_threads, "heaps": heap_info}, block=False)
                else:
                    data_queue.put({"threads": final_threads}, block=False)

                # Frame successful. Reset error counter.
                consecutive_errors = 0

            except queue.Full:
                pass
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= MAX_TOLERATED_ERRORS:
                    data_queue.put(
                        {"fatal_error": f"Target lost after {MAX_TOLERATED_ERRORS} retries: {e}"}
                    )
                    break  # Terminate loop cleanly
                else:
                    data_queue.put(
                        {
                            "error": f"Transient read fault "
                            f"({consecutive_errors}/{MAX_TOLERATED_ERRORS}): {e}"
                        },
                        block=False,
                    )

            finally:
                if in_batch:
                    with contextlib.suppress(Exception):
                        self._m_scraper.end_batch()

            time.sleep(inspection_period)
