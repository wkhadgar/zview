# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zview.backend.elf_inspector import ElfInspector
from zview.core import Core
from zview.kernel import compat
from zview.kernel.heaps import HEAP_FIELDS
from zview.orchestrator import ZScraper


@pytest.mark.parametrize('fields_available', (True, False))
@pytest.mark.parametrize('objects', ({}, {'test_heap': [0x1000]}))
def test_heap_warnings_and_polling(monkeypatch, tmp_path, fields_available, objects):
    monkeypatch.setattr(ElfInspector, '_resolve_os_cache_dir', staticmethod(lambda: tmp_path))
    resolve_fields = compat.resolve_fields
    find_instances = ElfInspector.find_struct_instances

    def resolve(elf, fields):
        if fields is HEAP_FIELDS and not fields_available:
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
    # The mock target holds no thread list to walk.
    monkeypatch.setattr(scraper.groups['threads'], 'discover', lambda session: {})
    scraper.discover()

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

    walked = []
    for group in scraper.groups.values():
        group.walk = lambda session, rows, key=group.key: walked.append(key) or {}
    core = Core(scraper)
    data_queue = core.events

    core.poll()

    assert walked.count('heaps') == int(can_poll)
    assert ('heaps' in data_queue.get_nowait().groups) == can_poll
    assert data_queue.empty()
