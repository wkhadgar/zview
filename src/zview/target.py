# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Where a session reads: an ELF, and the probe or recording it is read through."""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass

from zview.backend.base import AbstractScraper
from zview.backend.gdb import GDBScraper
from zview.backend.image import verify_target_image
from zview.backend.jlink import JLinkScraper
from zview.backend.nrfutil import NrfutilScraper
from zview.backend.pyocd import PyOCDScraper
from zview.backend.replay import ReplayScraper
from zview.core import Core
from zview.orchestrator import ZScraper
from zview.selection import Selection

_SCRAPERS: dict[str, type[AbstractScraper]] = {
    "gdb": GDBScraper,
    "jlink": JLinkScraper,
    "nrfutil": NrfutilScraper,
    "pyocd": PyOCDScraper,
}
AVAILABLE_RUNNERS = tuple(_SCRAPERS)
# Runners whose probe clock ``frequency`` sets.
CLOCKED_RUNNERS = ("pyocd", "jlink")


@dataclass(frozen=True)
class Target:
    """
    An ELF, read through a probe or a recording. ``runner`` names the probe's
    scraper, pyocd when unset, and ``target`` its MCU or, for gdb, the server's
    ``host:port``. ``frequency`` is the SWD clock of pyocd and jlink, in Hz.
    A ``recording`` is replayed in place of a probe, at its own cadence when
    ``pacing``.
    """

    elf: str
    runner: str | None = None
    target: str | None = None
    frequency: int | None = None
    recording: str | None = None
    pacing: bool = True

    def backend(self) -> AbstractScraper:
        """The scraper that reads this target, not yet connected."""
        if self.recording is not None:
            return ReplayScraper(self.recording, honor_timing=self.pacing)
        runner = self.runner if self.runner in _SCRAPERS else "pyocd"
        cls = _SCRAPERS[runner]
        if runner in CLOCKED_RUNNERS:
            return cls(self.target, self.frequency)
        return cls(self.target)

    @contextlib.contextmanager
    def open(self, selection: Selection | None = None) -> Iterator[Core]:
        """
        Connect, check that a live target runs the ELF, find its rows, and
        yield the core reading them, stopping its poll and closing the
        connection after. The core
        shows what ``selection`` reads, which is all a live target reads, and
        a replay without one shows its recording's own. A discovery that
        fails is an event on the core. Raises ``ValueError`` when
        ``selection`` names a group the ELF has no table for.
        """
        with self.backend() as backend:
            if backend.is_live:
                verify_target_image(backend, self.elf)
            scraper = ZScraper(backend, self.elf)
            core = Core(scraper)
            if selection is not None:
                scraper.check_selection(selection)
                core.select(selection)
            core.discover()
            try:
                yield core
            finally:
                core.stop()
