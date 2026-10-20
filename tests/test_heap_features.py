# SPDX-License-Identifier: Apache-2.0

import queue
import threading
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zview.backend.elf_inspector import ElfInspector
from zview.kernel import compat
from zview.kernel.groups import GROUPS
from zview.orchestrator import ZScraper


@pytest.mark.parametrize('fields_available', (True, False))
@pytest.mark.parametrize('objects', ({}, {'test_heap': [0x1000]}))
def test_heap_warnings_and_polling(monkeypatch, tmp_path, fields_available, objects):
    monkeypatch.setattr(ElfInspector, '_resolve_os_cache_dir', staticmethod(lambda: tmp_path))
    resolve_fields = compat.resolve_fields
    find_instances = ElfInspector.find_struct_instances

    def resolve(elf, fields):
        if fields is compat.HEAP_FIELDS and not fields_available:
            return None
        return resolve_fields(elf, fields)

    def discover(elf, name):
        if name == 'k_heap':
            return objects
        return find_instances(elf, name)

    monkeypatch.setattr(compat, 'resolve_fields', resolve)
    monkeypatch.setattr(ElfInspector, 'find_struct_instances', discover)
    elf = Path(__file__).parent / 'fixtures' / 'zephyr.elf'
    scraper = ZScraper(MagicMock(is_live=True, is_connected=True, watermark_cache={}), elf)

    can_poll = bool(objects) and fields_available
    assert scraper.available['heaps'] == can_poll
    assert scraper.rows['heaps'] == (objects if can_poll else {})
    notices = tuple(n for n in scraper.absent_features() if 'heap' in n)
    if objects and not fields_available:
        assert notices == ('Warning: no heap stats (CONFIG_SYS_HEAP_RUNTIME_STATS=n)',)
    else:
        assert notices == ()

    assert ('heaps' in scraper.active_features()) == can_poll
    assert ('heap_waiters' in scraper.active_features()) == can_poll

    monkeypatch.setattr(scraper, '_read_usage_deltas', lambda: {})
    monkeypatch.setattr(scraper, '_poll_threads', lambda deltas: [])
    monkeypatch.setattr(scraper, '_finalize_threads', lambda threads, deltas: [])
    walked = []
    monkeypatch.setattr(
        'zview.orchestrator.GROUPS',
        tuple(
            replace(group, walk=lambda session, rows, key=group.key: walked.append(key) or {})
            for group in GROUPS
        ),
    )
    stop = threading.Event()
    monkeypatch.setattr('zview.orchestrator.time.sleep', lambda _: stop.set())
    data_queue = queue.Queue()

    scraper._poll_thread_worker(data_queue, stop, 0)

    assert walked.count('heaps') == int(can_poll)
    assert ('heaps' in data_queue.get_nowait()) == can_poll
    assert data_queue.empty()
