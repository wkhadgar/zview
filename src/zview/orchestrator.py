# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""ZScraper: backend + ELF + kernel groups + polling thread."""

import contextlib
import logging
import queue
import threading
import time
from threading import Event
from typing import Literal

from zview.backend.base import AbstractScraper
from zview.backend.elf_inspector import ElfInspector
from zview.backend.replay import ReplayComplete
from zview.frames import Fatal, Frame, Notice, ReplayEnd
from zview.kernel import compat
from zview.kernel.group import BrokenFrame, Group
from zview.kernel.groups import GROUPS
from zview.kernel.layout import KernelLayout
from zview.kernel.object_names import label_instances
from zview.kernel.session import Session
from zview.selection import Selection

logger = logging.getLogger("zview.scraper")


class ZScraper:
    # What the session reads, and what the view on screen needs: a frame reads
    # what both allow. The TUI sets ``view`` on a live target only, dump and
    # record leave it reading everything.
    selection: Selection = Selection()
    view: Selection = Selection()

    def __init__(self, meta_scraper: AbstractScraper, elf_path):
        self._m_scraper: AbstractScraper = meta_scraper
        self._polling_thread: threading.Thread | None = None
        self._stop_event: Event | None = None
        self.inspection_period = 0.2

        self._load_elf(elf_path)

    def reload_elf(self) -> None:
        """Re-parse the ELF at the same path and rebuild everything derived from it."""
        self._load_elf(self.elf_path)

    def _load_elf(self, elf_path) -> None:
        """Parse ``elf_path`` and resolve the layout and the groups it provides."""
        self.elf_path = elf_path
        self._elf_inspector: ElfInspector = ElfInspector(elf_path)
        self._endianess: Literal["little", "big"] = (
            "little" if self._elf_inspector._little_endian else "big"
        )
        self._m_scraper.endianess = "<" if self._endianess == "little" else ">"

        # One of each group per ELF, in the order a frame reads them, and per
        # group key whether it is read and the rows its discover found.
        self.groups: dict[str, Group] = {cls.key: cls() for cls in GROUPS}
        self.available: dict[str, bool] = {}
        self.rows: dict[str, dict] = {key: {} for key in self.groups}
        self.parts: set[str] = set()

        # Built Zephyr version (when the ELF is in its build tree) and the wait
        # queue layout.
        self.zephyr_version: str | None = compat.detect_zephyr_version(elf_path)
        self.waitq_flavor: compat.WaitQFlavor = compat.waitq_flavor(self._elf_inspector)

        self._layout = self._resolve_layout()
        # Each group's live parts start on, and a replay keeps those its recording names.
        self.parts = {
            part for group in self.groups.values() for part in group.live_parts(self._session())
        }
        self._restrict_features_to_recording()

        if not self._m_scraper.is_live:
            self.capture_all()

        self.reset_runtime_state()

    def _resolve_layout(self) -> KernelLayout:
        """Resolve each group's tables; a group whose required tables miss is not read."""
        elf = self._elf_inspector

        fields: dict = {}
        for group in self.groups.values():
            required = [compat.resolve_fields(elf, table) for table in group.required]
            self.available[group.key] = None not in required
            for found in (t for t in required if t is not None):
                fields.update(found)
            for table in group.optional:
                fields.update(compat.resolve_fields(elf, table) or {})

        return KernelLayout(**fields)

    def discover(self) -> None:
        """
        Find each available group's rows on the target; a group with none is
        not read. Runs after connect, and again on a refresh.
        """
        session = self._session()
        for group in self.groups.values():
            if self.available[group.key]:
                self.rows[group.key] = group.discover(session)
                self.available[group.key] = bool(self.rows[group.key])

    def _session(self) -> Session:
        """What the groups read with, as of now."""
        return Session(
            scraper=self._m_scraper,
            elf=self._elf_inspector,
            layout=self._layout,
            waitq_flavor=self.waitq_flavor,
            rows=self.rows,
            parts=frozenset(self.parts),
        )

    def capture_all(self) -> None:
        """Read every detail a view may ask for, as a recording holds them all."""
        for group in self.groups.values():
            group.capture_all()

    def _restrict_features_to_recording(self) -> None:
        """
        Limit a replayed session to the features its recording captured.

        Replay matches a strict read sequence. A recording with no feature list
        predates the primitives and is treated as threads and heaps only.
        """
        if self._m_scraper.is_live:
            return

        features = getattr(self._m_scraper, "features", None)
        if features is None:
            return

        for group in self.groups.values():
            self.available[group.key] = self.available[group.key] and group.key in features
        self.parts &= set(features)

    def active_features(self) -> tuple[str, ...]:
        """The features this session polls, as recorded in a recording header."""
        features: list[str] = []
        # A group read, then each of its parts that is on.
        for group in self.groups.values():
            if self.available[group.key]:
                features += [group.key, *(part for part in group.parts if part in self.parts)]
        return tuple(features)

    def absent_features(self) -> tuple[str, ...]:
        """One notice per capability the build's Kconfig leaves out."""
        session = self._session()
        return tuple(notice for group in self.groups.values() for notice in group.notices(session))

    def decl_site(self, address: int) -> tuple[str, int] | None:
        """``(path, line)`` where the object at ``address`` is declared, if known."""
        return self._elf_inspector.decl_site(address)

    def function_site(self, address: int) -> tuple[str, int] | None:
        """``(path, line)`` where the function at ``address`` is defined, if known."""
        return self._elf_inspector.function_site(address)

    def check_selection(self, selection: Selection) -> None:
        """Raises ``ValueError`` when ``selection`` names a group no frame holds."""
        if unknown := set(selection.groups or ()) - set(self.groups):
            named, known = ", ".join(sorted(unknown)), ", ".join(self.groups)
            raise ValueError(f"Unknown group {named}; choose from {known}.")

    def _reads_group(self, group: str) -> bool:
        """Both the session's selection and the view's read ``group``."""
        return self.selection.reads_group(group) and self.view.reads_group(group)

    def _reads(self, group: str, name: str) -> bool:
        """Both the session's selection and the view's read the row ``name``."""
        return self.selection.reads(group, name) and self.view.reads(group, name)

    def _poll_groups(self) -> dict:
        """
        Read every group the selection and the view read, in order. A failed
        read raises, and the frame is dropped.

        Wait queues are walked only on the ``simple`` layout; otherwise
        ``waiters`` is ``None``.
        """
        frame: dict = {}
        session = self._session()
        for group in self.groups.values():
            if not self.available[group.key] or not self._reads_group(group.key):
                continue

            # Matched against the labels the rows will carry.
            selected = {
                label: [address]
                for label, address in label_instances(self.rows[group.key])
                if self._reads(group.key, label)
            }
            frame[group.key] = list(group.walk(session, selected).values())

        return frame

    def __enter__(self):
        self._m_scraper.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._m_scraper.__exit__(exc_type, exc_val, exc_tb)

    def read_variable(self, var_name: str) -> list:
        if not self._m_scraper.is_connected:
            self._m_scraper.connect()

        var_addresses = self._elf_inspector.get_symbol_info(var_name, "address")
        var_sizes = self._elf_inspector.get_symbol_info(var_name, "size")

        return [
            self._m_scraper.read8(var_address, var_size)
            for var_address, var_size in zip(var_addresses, var_sizes, strict=True)
        ]

    def reset_runtime_state(self) -> None:
        """Clear the watermark cache and each group's baselines; the next poll seeds them."""
        self._m_scraper.watermark_cache.clear()
        for group in self.groups.values():
            group.reset()

    def start_polling_thread(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        inspection_period: float,
    ):
        """Start the polling daemon thread. No-op if one is already running."""
        if self._polling_thread is not None:
            data_queue.put(Notice("Already started..."))
            return

        self.inspection_period = inspection_period
        self._polling_thread = threading.Thread(
            target=self._poll_thread_worker,
            args=(data_queue, stop_event, inspection_period),
            daemon=True,
        )
        self._stop_event = stop_event
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

    _MAX_TOLERATED_ERRORS = 3

    def _poll_thread_worker(
        self,
        data_queue: queue.Queue,
        stop_event: threading.Event,
        inspection_period: float,
    ):
        """Polling thread entry point. Tolerates ``_MAX_TOLERATED_ERRORS`` transient faults."""
        try:
            if not self._m_scraper.is_connected:
                self._m_scraper.connect()
        except Exception as e:
            data_queue.put(Fatal(f"Failed to connect to target: {e}"))
            return

        consecutive_errors = 0
        while not stop_event.is_set():
            cycle_start = time.perf_counter()
            in_batch = False
            try:
                self._m_scraper.begin_batch()
                in_batch = True

                data_queue.put(Frame(self._poll_groups()), block=False)

                consecutive_errors = 0
            except queue.Full:
                pass
            except ReplayComplete as e:
                data_queue.put(ReplayEnd(str(e)))
                break
            except BrokenFrame as e:
                logger.warning("Broken frame captured (%s); resyncing baselines.", e)
                self.reset_runtime_state()
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors >= self._MAX_TOLERATED_ERRORS:
                    data_queue.put(
                        Fatal(f"Target lost after {self._MAX_TOLERATED_ERRORS} retries: {e}")
                    )
                    break
                # One message per fault, not one per attempt: the message log
                # counts the repeats, which is the retry number.
                data_queue.put(Notice(f"Transient read fault: {e}"), block=False)
            finally:
                if in_batch:
                    with contextlib.suppress(Exception):
                        self._m_scraper.end_batch()

            # Sleep only the remainder of the period, so the poll cadence tracks
            # the requested period instead of period-plus-read-time. When the
            # reads already exceed the period, poll back-to-back (no sleep).
            elapsed = time.perf_counter() - cycle_start
            time.sleep(max(0.0, inspection_period - elapsed))
