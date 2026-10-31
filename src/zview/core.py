# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""One target or recording, polled for the frontend attached to it."""

import contextlib
import enum
import logging
import queue
import threading
import time
from dataclasses import dataclass

from zview.backend.replay import ReplayComplete
from zview.frames import Fatal, Frame, FrameHistory, Notice, ReplayEnd
from zview.kernel.group import BrokenFrame
from zview.kernel.threads import USAGE_WINDOW_MAX
from zview.orchestrator import ZScraper
from zview.selection import Selection

logger = logging.getLogger("zview.core")

# Faults in a row a poll goes on after.
MAX_TOLERATED_ERRORS = 3
# Events kept for a frontend that only reads the core's state.
EVENTS_KEPT = 256


class Status(enum.Enum):
    """What the core's poll is doing."""

    # Not polling: before the first poll, after a stop, or between one-frame polls.
    IDLE = "idle"
    POLLING = "polling"
    # The target never connected, or it was lost.
    LOST = "lost"
    # The recording being replayed ended.
    ENDED = "ended"


@dataclass(frozen=True)
class Build:
    """What the firmware fixes: the Kconfig it leaves out, and its wait queue layout."""

    # One notice per capability the Kconfig leaves out.
    missing: tuple[str, ...]
    is_waitq_walkable: bool

    @classmethod
    def of(cls, scraper: ZScraper) -> "Build":
        return cls(tuple(scraper.absent_features()), scraper.waitq_flavor == "simple")


class Core:
    """The poll of one target or recording, its state, and the events it hands the frontend."""

    def __init__(self, scraper: ZScraper):
        self.scraper = scraper
        self.build = Build.of(scraper)
        self.status = Status.IDLE
        # The text of the poll's latest fault, transient or fatal.
        self.last_fault: str | None = None
        # Each event of the poll, for the frontend to take.
        self.events: queue.Queue = queue.Queue(maxsize=EVENTS_KEPT)
        # What the frontend shows: the session's selection, then what it changes.
        self.selection: Selection = scraper.selection
        self.frames = FrameHistory()
        # The period of the poll ``start`` runs, kept for the next one, and its thread.
        self.period: float | None = None
        self._poller: threading.Thread | None = None
        self._stop = threading.Event()
        # Held while a start or a stop changes the poller.
        self._lifecycle = threading.Lock()
        # Held over each frame's reads, so a refresh never reads beside the poll.
        self._reading = threading.Lock()
        self._faults = 0
        self._frames_read = 0

    def select(self, selection: Selection) -> None:
        """Show what ``selection`` reads, and read only that from a live target."""
        self.selection = selection
        if self.scraper.is_live:
            self.scraper.selection = selection

    @property
    def is_live(self) -> bool:
        """A target is read, not a recording."""
        return self.scraper.is_live

    @property
    def groups(self) -> tuple[str, ...]:
        """The groups the latest discovery found rows of."""
        return tuple(key for key, available in self.scraper.available.items() if available)

    def check(self, selection: Selection) -> tuple[str, ...]:
        """
        One warning per pattern of ``selection`` the latest discovery found no
        row for. Raises ``ValueError`` when it names a group no frame holds.
        """
        self.scraper.check_selection(selection)
        return self.scraper.unmatched(selection)

    @property
    def usage_window(self) -> int:
        """
        The polls the CPU and load shares are summed over, 1 on a recording
        from before the usage snapshot, whose shares are of each poll alone.
        """
        summed = "usage_snapshot" in self.scraper.parts
        return self.scraper.groups["threads"].usage_window if summed else 1

    @usage_window.setter
    def usage_window(self, polls: int) -> None:
        self.scraper.groups["threads"].usage_window = max(1, min(USAGE_WINDOW_MAX, polls))

    def poll(self, period: float | None = None, frames: int | None = None) -> None:
        """
        Read one frame and put it on ``events``. With ``period``, read one
        every ``period`` seconds until ``stop``, ``frames`` frames when given,
        the end of a recording, or ``MAX_TOLERATED_ERRORS`` faults in a row.
        """
        self._faults = 0
        self._frames_read = 0
        try:
            self.scraper.connect()
        except Exception as e:
            self._publish(Fatal(f"Failed to connect to target: {e}"))
            return

        if period is None:
            self._read()
            return

        self.status = Status.POLLING
        while not self._stop.is_set() and (frames is None or self._frames_read < frames):
            started = time.perf_counter()
            if not self._read():
                break
            # Sleep only the remainder of the period, so the poll cadence tracks
            # the requested period instead of period-plus-read-time. When the
            # reads already exceed the period, poll back-to-back (no sleep).
            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, period - elapsed))
        if self.status is Status.POLLING:
            self.status = Status.IDLE

    @property
    def is_polling(self) -> bool:
        """A poll ``start`` began is still running."""
        return self._poller is not None and self._poller.is_alive()

    def start(self, period: float, frames: int | None = None) -> None:
        """
        Poll with ``period`` on a thread of its own, unless a poll is running.
        A stop from before this call is for no poll, and one after it ends this one.
        """
        with self._lifecycle:
            if self.is_polling:
                return
            self._stop.clear()
            self.period = period
            self._poller = threading.Thread(target=self.poll, args=(period, frames), daemon=True)
            self._poller.start()

    def stop(self) -> None:
        """
        End a poll running with a period after its current frame, and wait
        for it: a slow probe takes over a second to read one, and stays open
        until then. The poll's own thread stops it without waiting.
        """
        with self._lifecycle:
            self._stop.set()
            poller = self._poller
        if poller is None or poller is threading.current_thread():
            return
        poller.join()
        with self._lifecycle:
            if self._poller is poller:
                self._poller = None

    def discover(self) -> None:
        """Find the target's rows. A failure is a notice on ``events``."""
        try:
            self.scraper.discover()
        except RuntimeError as e:
            self._publish(Notice(f"Unable to update available threads [{e}]"))

    def detail(self, group: str, address: int | None) -> None:
        """
        Read the object at ``address`` in full, as a detail view shows it, or
        no object of ``group`` in full. A heap's is its chunk map.
        """
        with self._reading:
            self.scraper.groups[group].detailed = address

    def refresh(self) -> None:
        """Find the target's rows again, and clear the baselines the next frame seeds."""
        with self._reading:
            self.scraper.discover()
            self.scraper.reset_runtime_state()

    def reattach(self) -> None:
        """
        With the poll stopped, reattach the target and find its rows again.
        The history starts over, since a rebuilt ELF can reassign the
        addresses rows are kept by. The fault of the lost target is cleared
        once it is reattached.
        """
        self.frames = FrameHistory()
        self.scraper.reattach()
        self.build = Build.of(self.scraper)
        self.refresh()
        self.status, self.last_fault = Status.IDLE, None

    def _read(self) -> bool:
        """Read one frame and hand it out. False once the poll must end."""
        try:
            with self._reading:
                frame = Frame(self.scraper.read_frame())
            self.frames.add(frame)
            self._publish(frame)
            self._frames_read += 1
            self._faults = 0
        except ReplayComplete as e:
            self._publish(ReplayEnd(str(e)))
            return False
        except BrokenFrame as e:
            logger.warning("Broken frame captured (%s); resyncing baselines.", e)
            with self._reading:
                self.scraper.reset_runtime_state()
        except Exception as e:
            self._faults += 1
            if self._faults >= MAX_TOLERATED_ERRORS:
                self._publish(Fatal(f"Target lost after {MAX_TOLERATED_ERRORS} retries: {e}"))
                return False
            # One message per fault, not one per attempt: the message log
            # counts the repeats, which is the retry number.
            self._publish(Notice(f"Transient read fault: {e}"))
        return True

    def _publish(self, item: Frame | Notice | Fatal | ReplayEnd) -> None:
        """
        Keep what ``item`` says of the poll, then put it on ``events``. Full,
        they drop a frame or a notice, and an end of the poll takes the place
        of the oldest event.
        """
        match item:
            case Fatal(text):
                self.status, self.last_fault = Status.LOST, text
            case Notice(text):
                self.last_fault = text
            case ReplayEnd():
                self.status = Status.ENDED
        while True:
            try:
                self.events.put_nowait(item)
                return
            except queue.Full:
                if not isinstance(item, Fatal | ReplayEnd):
                    return
                with contextlib.suppress(queue.Empty):
                    self.events.get_nowait()
