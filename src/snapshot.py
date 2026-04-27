# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Headless recording, replay, and single-frame helpers used by the CLI commands."""

import dataclasses
import queue
import threading
import time

from backend.base import AbstractScraper
from backend.recording import RecordingScraper
from orchestrator import ZScraper


def serialize_frame(frame: dict) -> dict:
    """Convert a polling frame's dataclasses into a JSON-serializable dict."""
    out: dict = {}
    if "threads" in frame:
        out["threads"] = [dataclasses.asdict(t) for t in frame["threads"]]
    if "heaps" in frame:
        out["heaps"] = [dataclasses.asdict(h) for h in frame["heaps"]]
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
        scraper = ZScraper(backend, elf_path)
        scraper.update_available_threads()
        scraper.reset_thread_pool()
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
                if "fatal_error" in frame_data:
                    raise RuntimeError(frame_data["fatal_error"])
                if "replay_complete" in frame_data:
                    raise RuntimeError(
                        f"Recording has only {seen} frames; --frame {frame} out of range."
                    )
                if "threads" in frame_data:
                    seen += 1
                    if seen >= frame:
                        return frame_data
            raise TimeoutError(f"No frame {frame} within {timeout} seconds")
        finally:
            stop.set()
            scraper.finish_polling_thread()


def _configure_heap_detail(scraper: ZScraper, recorder: AbstractScraper, heap_name: str) -> None:
    """
    Set ``scraper.extra_info_heap_address`` from the named ``k_heap`` global.
    Raises ``ValueError`` when the ELF has no heaps or ``heap_name`` is unknown.
    """
    if not scraper.has_heaps or not getattr(scraper, "_k_heap_addresses", None):
        raise ValueError("This ELF exposes no k_heap symbols; --heap-detail unavailable.")

    if heap_name not in scraper._k_heap_addresses:
        available = ", ".join(sorted(scraper._k_heap_addresses.keys()))
        raise ValueError(f"Heap {heap_name!r} not found. Available heaps: {available}.")

    symbol_addrs = scraper._k_heap_addresses[heap_name]
    recorder.begin_batch()
    try:
        heap_struct_ptr = recorder.read32(symbol_addrs[0])[0]
    finally:
        recorder.end_batch()
    scraper.extra_info_heap_address = heap_struct_ptr


def record_session(
    backend: AbstractScraper,
    elf_path,
    out_path,
    duration: float | None = None,
    frames: int | None = None,
    period: float = 0.1,
    heap_detail: str | None = None,
) -> int:
    """
    Record a polling session to ``out_path``. Requires one of ``duration``
    (seconds) or ``frames`` (count). ``heap_detail`` names a ``k_heap``
    global to capture per-frame fragmentation for. Returns frame count.
    """
    if duration is None and frames is None:
        raise ValueError("record_session requires either duration or frames bound")

    recorder = RecordingScraper(backend, out_path)
    with recorder:
        scraper = ZScraper(recorder, elf_path)
        scraper.update_available_threads()
        scraper.reset_thread_pool()

        if heap_detail is not None:
            _configure_heap_detail(scraper, recorder, heap_detail)

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
                    if "threads" in frame:
                        captured += 1
                        if frames is not None and captured >= frames:
                            break
                if duration is not None and (time.monotonic() - start) >= duration:
                    break
        finally:
            stop.set()
            scraper.finish_polling_thread()
        return captured
