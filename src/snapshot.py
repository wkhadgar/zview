# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
Headless recording and single-frame JSON snapshot helpers backing the
``--snapshot``, ``--replay``, and ``--once --json`` CLI modes.
"""

import queue
import threading
import time

from backend.base import AbstractScraper, HeapInfo, ThreadInfo, ThreadRuntime
from backend.recording import RecordingScraper
from orchestrator import ZScraper


def _runtime_to_dict(runtime: ThreadRuntime) -> dict:
    return {
        "cpu": runtime.cpu,
        "cpu_normalized": runtime.cpu_normalized,
        "active": runtime.active,
        "stack_watermark": runtime.stack_watermark,
        "stack_watermark_percent": runtime.stack_watermark_percent,
    }


def _thread_to_dict(thread: ThreadInfo) -> dict:
    out: dict = {
        "name": thread.name,
        "address": thread.address,
        "stack_start": thread.stack_start,
        "stack_size": thread.stack_size,
    }
    if thread.runtime is not None:
        out["runtime"] = _runtime_to_dict(thread.runtime)
    return out


def _heap_to_dict(heap: HeapInfo) -> dict:
    return {
        "name": heap.name,
        "address": heap.address,
        "free_bytes": heap.free_bytes,
        "allocated_bytes": heap.allocated_bytes,
        "max_allocated_bytes": heap.max_allocated_bytes,
        "usage_percent": heap.usage_percent,
        "chunks": heap.chunks,
    }


def serialize_frame(frame: dict) -> dict:
    """
    Convert a polling-thread frame dict (carrying ThreadInfo and HeapInfo
    dataclasses) into a JSON-serializable plain-dict shape.
    """
    out: dict = {}
    if "threads" in frame:
        out["threads"] = [_thread_to_dict(t) for t in frame["threads"]]
    if "heaps" in frame:
        out["heaps"] = [_heap_to_dict(h) for h in frame["heaps"]]
    return out


def dump_single_frame(
    backend: AbstractScraper,
    elf_path,
    period: float = 0.1,
    timeout: float = 5.0,
) -> dict:
    """
    Capture the first valid polling frame from ``backend``. Raises
    ``RuntimeError`` on a fatal scraper error emitted by the polling thread,
    and ``TimeoutError`` if no valid frame arrives within ``timeout`` seconds.
    """
    with backend:
        scraper = ZScraper(backend, elf_path)
        scraper.update_available_threads()
        scraper.reset_thread_pool()
        data_queue: queue.Queue = queue.Queue()
        stop = threading.Event()
        scraper.start_polling_thread(data_queue, stop, period)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                try:
                    frame = data_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if "fatal_error" in frame:
                    raise RuntimeError(frame["fatal_error"])
                if "threads" in frame:
                    return frame
            raise TimeoutError(f"No valid frame within {timeout} seconds")
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
    Record a polling session to ``out_path``. Bounded by either ``duration``
    (wall-clock seconds) or ``frames`` (count of valid data frames); at
    least one must be provided. Returns the number of data frames captured.
    """
    if duration is None and frames is None:
        raise ValueError("record_session requires either duration or frames bound")

    recorder = RecordingScraper(backend, out_path)
    with recorder:
        scraper = ZScraper(recorder, elf_path)
        scraper.update_available_threads()
        scraper.reset_thread_pool()
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
