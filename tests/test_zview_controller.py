# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for ``ZView`` controller logic that doesn't require real curses."""

import queue
from unittest.mock import MagicMock

import pytest

from backend.base import HeapInfo, ThreadInfo
from frontend.tui.views.base import ZViewState
from frontend.zview_tui import ZView


@pytest.fixture
def app() -> ZView:
    """``ZView`` instance without going through curses init."""
    a = ZView.__new__(ZView)
    a.threads_data = []
    a.heaps_data = []
    a.status_message = ""
    a.update_count = 0
    a.idle_thread = None
    a.data_queue = queue.Queue()
    a.scraper = MagicMock()
    a.scraper.idle_threads_address = 0xDEAD
    a.state = ZViewState.THREAD_LIST_VIEW
    return a


def test_purge_queue_clears_pending_items(app):
    for i in range(5):
        app.data_queue.put({"i": i})
    assert app.data_queue.qsize() == 5
    app.purge_queue()
    assert app.data_queue.empty()


def test_process_data_fatal_transitions_to_error_state(app):
    app.process_data({"fatal_error": "synthetic loss"})
    assert app.state is ZViewState.FATAL_ERROR
    assert "TARGET LOST" in app.status_message
    assert "synthetic loss" in app.status_message


def test_process_data_replay_complete_sets_status_only(app):
    app.process_data({"replay_complete": "drained"})
    assert "Recording ended" in app.status_message
    # State is unchanged on a clean replay end.
    assert app.state is ZViewState.THREAD_LIST_VIEW


def test_process_data_error_sets_error_status_without_state_change(app):
    app.process_data({"error": "transient"})
    assert app.status_message.startswith("Error")
    assert "transient" in app.status_message
    assert app.state is ZViewState.THREAD_LIST_VIEW


def test_process_data_running_status_increments_with_threads_payload(app):
    """A normal frame: status reads ``Running...``, threads payload is captured."""
    threads = [
        ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None),
    ]
    app.process_data({"threads": threads})
    assert app.status_message.startswith("Running")
    assert app.threads_data == threads
    assert app.update_count == 1


def test_process_data_idle_thread_pulled_into_dedicated_field(app):
    """Idle thread is recognized by address and stripped out of ``threads_data``."""
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None)
    idle = ThreadInfo(address=0xDEAD, stack_start=0, stack_size=128, name="idle", runtime=None)
    app.process_data({"threads": [main, idle]})
    assert app.idle_thread == idle
    assert app.threads_data == [main]


def test_process_data_heaps_payload_captured(app):
    heap = HeapInfo(
        name="my_heap",
        address=0x4000,
        free_bytes=100,
        allocated_bytes=50,
        max_allocated_bytes=80,
        usage_percent=33.0,
        chunks=None,
    )
    app.process_data({"heaps": [heap]})
    assert app.heaps_data == [heap]


def test_process_data_empty_payload_does_not_clear_existing(app):
    """A no-op frame must not erase last-known data."""
    main = ThreadInfo(address=0x1000, stack_start=0, stack_size=512, name="main", runtime=None)
    app.threads_data = [main]
    app.process_data({})
    assert app.threads_data == [main]
