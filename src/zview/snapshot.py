# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Headless recording, replay, and single-frame helpers used by the CLI commands."""

import queue
import sys
import time

from zview.backend.base import AbstractScraper
from zview.backend.image import verify_target_image
from zview.backend.recording import RecordingScraper
from zview.core import Core
from zview.frames import Fatal, Frame, Notice, ReplayEnd
from zview.kinds import KINDS
from zview.orchestrator import ZScraper
from zview.selection import Selection
from zview.target import Target


def serialize_frame(frame: dict) -> dict:
    """A polling frame's rows, each as its kind writes it to JSON."""
    return {
        kind.key: [kind.to_json(info) for info in frame[kind.key]]
        for kind in KINDS
        if kind.key in frame
    }


def dump_single_frame(
    target: Target,
    period: float = 0.1,
    frame: int = 1,
    timeout: float | None = None,
    selection: Selection | None = None,
) -> dict:
    """
    Return the Nth valid polling frame (1-indexed; default 1), with only the
    rows ``selection`` reads. A live target reads only those, and a recording,
    which holds the reads it was made with, is filtered after.
    Raises ``ValueError`` for ``frame < 1``, ``RuntimeError`` on fatal scraper
    error or recording exhaustion, ``TimeoutError`` past ``timeout`` seconds.
    ``timeout`` defaults to ``max(5.0, frame * max(period, 0.1) * 5.0)``.
    """
    if frame < 1:
        raise ValueError("frame must be >= 1")
    selection = selection or Selection()
    if timeout is None:
        timeout = max(5.0, frame * max(period, 0.1) * 5.0)

    with target.open(selection) as core:
        for notice in core.scraper.unmatched(selection):
            print(notice, file=sys.stderr)
        core.start(period)
        deadline = time.monotonic() + timeout
        seen = 0
        try:
            while time.monotonic() < deadline:
                try:
                    frame_data = core.events.get(timeout=0.1)
                except queue.Empty:
                    continue
                match frame_data:
                    case Fatal(text):
                        raise RuntimeError(text)
                    case Notice(text):
                        print(text, file=sys.stderr)
                    case ReplayEnd():
                        raise RuntimeError(
                            f"Recording has only {seen} frames; --frame {frame} out of range."
                        )
                    case Frame(groups):
                        seen += 1
                        if seen >= frame:
                            return selection.keep(groups)
            raise TimeoutError(f"No frame {frame} within {timeout} seconds")
        finally:
            core.stop()


def record_session(
    backend: AbstractScraper,
    elf_path,
    out_path,
    duration: float | None = None,
    frames: int | None = None,
    period: float = 0.1,
    selection: Selection | None = None,
    source: dict | None = None,
) -> int:
    """
    Record a polling session to ``out_path``. Requires one of ``duration``
    (seconds) or ``frames`` (count). Captures fragmentation maps for every
    heap so replay can navigate to any heap detail without drift. A live
    target reads only the rows ``selection`` reads, and the header says so;
    a replay keeps its recording's. The header holds ``source``, the runner,
    target and clock read through, beside ``period``. Returns frame count.
    """
    if duration is None and frames is None:
        raise ValueError("record_session requires either duration or frames bound")

    # Checked on the bare backend, so the read stays out of the recording.
    if backend.is_live:
        backend.connect()
        verify_target_image(backend, elf_path)

    recorder = RecordingScraper(backend, out_path)
    # Built before connect: the header is written there, and the constructor
    # only reads the ELF.
    scraper = ZScraper(recorder, elf_path)
    core = Core(scraper)
    selection = selection or Selection()
    scraper.check_selection(selection)
    core.select(selection)
    recorder.declare_features(scraper.active_features())
    recorder.declare_selection(scraper.selection)
    recorder.declare_source({**(source or {}), "period": period})

    with recorder:
        scraper.discover()
        for notice in scraper.unmatched(scraper.selection):
            print(notice, file=sys.stderr)
        scraper.capture_all()

        # The poll stops at the bound itself: it runs a frame ahead of this loop.
        core.start(period, frames)

        captured = 0
        start = time.monotonic()
        try:
            while True:
                try:
                    frame = core.events.get(timeout=0.1)
                except queue.Empty:
                    # A poll bounded in frames that ended has nothing more to hand out.
                    if not core.is_polling and core.events.empty():
                        break
                else:
                    match frame:
                        case Fatal(text):
                            raise RuntimeError(text)
                        case ReplayEnd():
                            break
                        case Frame():
                            captured += 1
                            if frames is not None and captured >= frames:
                                break
                if duration is not None and (time.monotonic() - start) >= duration:
                    break
        finally:
            core.stop()
        return captured
