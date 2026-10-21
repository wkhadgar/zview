# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Headless recording, replay, and single-frame helpers used by the CLI commands."""

import dataclasses
import queue
import threading
import time

from zview.backend.base import AbstractScraper
from zview.backend.image import verify_target_image
from zview.backend.recording import RecordingScraper
from zview.frames import Fatal, Frame, ReplayEnd
from zview.kernel.groups import GROUPS
from zview.orchestrator import ZScraper


def serialize_frame(frame: dict) -> dict:
    """Convert a polling frame's dataclasses into a JSON-serializable dict."""
    out: dict = {}
    for key in (group.key for group in GROUPS):
        if key in frame:
            out[key] = [dataclasses.asdict(entry) for entry in frame[key]]
    return out


def dump_single_frame(
    backend: AbstractScraper,
    elf_path,
    period: float = 0.1,
    frame: int = 1,
    timeout: float | None = None,
) -> dict:
    """
    Return the Nth valid polling frame (1-indexed; default 1).
    Raises ``ValueError`` for ``frame < 1``, ``RuntimeError`` on fatal scraper
    error or recording exhaustion, ``TimeoutError`` past ``timeout`` seconds.
    ``timeout`` defaults to ``max(5.0, frame * max(period, 0.1) * 5.0)``.
    """
    if frame < 1:
        raise ValueError("frame must be >= 1")
    if timeout is None:
        timeout = max(5.0, frame * max(period, 0.1) * 5.0)

    with backend:
        if backend.is_live:
            verify_target_image(backend, elf_path)
        scraper = ZScraper(backend, elf_path)
        scraper.discover()
        data_queue: queue.Queue = queue.Queue()
        stop = threading.Event()
        scraper.start_polling_thread(data_queue, stop, period)
        deadline = time.monotonic() + timeout
        seen = 0
        try:
            while time.monotonic() < deadline:
                try:
                    frame_data = data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                match frame_data:
                    case Fatal(text):
                        raise RuntimeError(text)
                    case ReplayEnd():
                        raise RuntimeError(
                            f"Recording has only {seen} frames; --frame {frame} out of range."
                        )
                    case Frame(groups):
                        seen += 1
                        if seen >= frame:
                            return groups
            raise TimeoutError(f"No frame {frame} within {timeout} seconds")
        finally:
            stop.set()
            scraper.finish_polling_thread()


def record_session(
    backend: AbstractScraper,
    elf_path,
    out_path,
    duration: float | None = None,
    frames: int | None = None,
    period: float = 0.1,
) -> int:
    """
    Record a polling session to ``out_path``. Requires one of ``duration``
    (seconds) or ``frames`` (count). Captures fragmentation maps for every
    heap so replay can navigate to any heap detail without drift. Returns
    frame count.
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
    recorder.declare_features(scraper.active_features())

    with recorder:
        scraper.discover()
        scraper.capture_all()

        data_queue: queue.Queue = queue.Queue()
        stop = threading.Event()
        scraper.start_polling_thread(data_queue, stop, period)

        captured = 0
        start = time.monotonic()
        try:
            while not stop.is_set():
                try:
                    frame = data_queue.get(timeout=0.1)
                except queue.Empty:
                    pass
                else:
                    if isinstance(frame, Frame):
                        captured += 1
                        if frames is not None and captured >= frames:
                            break
                if duration is not None and (time.monotonic() - start) >= duration:
                    break
        finally:
            stop.set()
            scraper.finish_polling_thread()
        return captured
