# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from zview.backend.replay import ReplayScraper
from zview.frames import Notice
from zview.orchestrator import ZScraper
from zview.target import Target

_TESTS_DIR = Path(__file__).parent
_FIXTURE = _TESTS_DIR / "fixtures" / "recordings" / "sys_heap_v4.3.ndjson.gz"
_ELF = _TESTS_DIR / "fixtures" / "zephyr.elf"


def test_a_discovery_that_fails_is_an_event_on_the_core(monkeypatch):
    """The frontend opens and says why rows are missing, instead of the launch failing."""

    def discover(self):
        raise RuntimeError("thread list unreadable")

    monkeypatch.setattr(ZScraper, "discover", discover)

    with Target(str(_ELF), recording=str(_FIXTURE), pacing=False).open() as core:
        event = core.events.get_nowait()

    assert event == Notice("Unable to update available threads [thread list unreadable]")


def test_leaving_the_session_stops_its_poll_before_the_disconnect(monkeypatch):
    """A poll left running would read from a closed connection."""
    disconnects = []
    monkeypatch.setattr(ReplayScraper, "disconnect", lambda self: disconnects.append(1))

    with Target(str(_ELF), recording=str(_FIXTURE), pacing=False).open() as core:
        core.start(0.01)
        poller = core._poller
        stop = core.stop
        monkeypatch.setattr(core, "stop", lambda: (disconnects.append(0), stop()))

    assert disconnects[:2] == [0, 1]
    assert not poller.is_alive()
