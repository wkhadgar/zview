# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import threading
from unittest.mock import MagicMock, patch

from zview.core import Core


def _core() -> Core:
    scraper = MagicMock()
    scraper.read_frame.return_value = {"threads": []}
    return Core(scraper)


def test_a_stop_after_a_poll_ended_leaves_the_next_poll_running():
    """R stops a poll a fatal fault already ended, then polls again."""
    core = _core()
    core.stop()
    sleeps = []

    def sleep(_):
        sleeps.append(1)
        if len(sleeps) == 2:
            core.stop()

    with patch("zview.core.time.sleep", side_effect=sleep):
        core.start(0)
        core._poller.join()

    assert core.events.qsize() == 2


def test_a_stop_right_after_the_start_ends_the_poll_and_waits_for_it():
    """It can land before the poll thread runs, and the poll would never end."""
    core = _core()

    with patch("zview.core.time.sleep"):
        core.start(0)
        poller = core._poller
        stopper = threading.Thread(target=core.stop)
        stopper.start()
        stopper.join(timeout=2.0)

    assert not stopper.is_alive()
    assert not poller.is_alive()


def test_a_stop_while_the_poll_thread_starts_waits_for_it():
    """A stop from another thread before the thread runs would join one never started."""
    core = _core()
    starting, release = threading.Event(), threading.Event()

    class GatedThread(threading.Thread):
        def start(self):
            starting.set()
            release.wait()
            super().start()

    errors = []

    def stop():
        starting.wait()
        try:
            core.stop()
        except RuntimeError as e:
            errors.append(e)

    starter = threading.Thread(target=core.start, args=(0,))
    stopper = threading.Thread(target=stop)
    with patch("zview.core.time.sleep"), patch("zview.core.threading.Thread", GatedThread):
        starter.start()
        stopper.start()
        starting.wait()
        stopper.join(timeout=0.2)
        assert stopper.is_alive()
        release.set()
        starter.join(timeout=2.0)
        stopper.join(timeout=2.0)

    assert errors == []
    assert not stopper.is_alive()
    assert not core.is_polling


def test_a_second_start_leaves_the_running_poll_alone():
    """A second poller would double the reads, and stop would wait for one of them only."""
    core = _core()

    with patch("zview.core.time.sleep"):
        core.start(0)
        first = core._poller
        core.start(0)
        assert core._poller is first
        core.stop()


def test_a_poll_bounded_in_frames_ends_after_them():
    """A recording bounded in frames gets no frame the poll would run ahead to."""
    core = _core()

    with patch("zview.core.time.sleep"):
        core.poll(0, frames=3)

    assert core.events.qsize() == 3
    assert core.scraper.read_frame.call_count == 3
