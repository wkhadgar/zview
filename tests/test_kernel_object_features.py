# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the kernel-object feature list a recording carries."""

import queue
from unittest.mock import MagicMock, patch

from zview.kernel.groups import GROUPS
from zview.kernel.layout import KernelLayout
from zview.orchestrator import ZScraper


def _scraper(**enabled: bool) -> ZScraper:
    """``ZScraper`` with only the feature flags set, without touching an ELF."""
    s = ZScraper.__new__(ZScraper)
    s.available = {group.key: enabled.get(group.key, False) for group in GROUPS}
    s.rows = {group.key: {} for group in GROUPS}
    s.group_state = {}
    s.parts = {"heap_waiters"} if enabled.get("heap_waiters", s.available["heaps"]) else set()
    s.has_stack_fill = enabled.get("stack_fill", True)
    s.has_thread_list = enabled.get("thread_list", True)
    s.has_usage = enabled.get("usage", False)
    s.usage_snapshot = True
    s.merged_thread_meta = True
    s._layout = KernelLayout(heap_free_bytes=4 if enabled.get("heap_stats", True) else None)
    s._elf_inspector = MagicMock(find_struct_instances=lambda struct: {"test_heap": [0x1000]})
    return s


def _polling_scraper(**rows: dict[str, list[int]]) -> ZScraper:
    """``_scraper`` with ``rows`` discovered per group, ready to poll them."""
    s = _scraper(**{key: True for key in rows})
    s.rows.update(rows)
    s._m_scraper = MagicMock()
    s._elf_inspector = MagicMock()
    s._layout = MagicMock()
    s.waitq_flavor = "simple"
    s._thread_names = lambda: {}
    return s


def _with_walks(**walks):
    """``GROUPS`` with these groups' ``walk`` replaced."""
    from dataclasses import replace

    return tuple(replace(group, walk=walks.get(group.key, group.walk)) for group in GROUPS)


def test_active_features_lists_only_what_is_polled():
    s = _scraper(heaps=True, semaphores=True, msgqs=True, mem_slabs=True)

    assert s.active_features() == (
        "threads",
        "merged_thread_meta",
        "heaps",
        "heap_waiters",
        "semaphores",
        "msgqs",
        "mem_slabs",
    )


def test_a_recording_without_the_heap_wait_queue_reads_does_not_replay_them():
    """The walk sits inside the heaps group, so an older recording has no reads for it."""
    s = _scraper(heaps=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "heaps"))

    s._restrict_features_to_recording()

    assert s.available["heaps"]
    assert "heap_waiters" not in s.parts


def test_threads_are_always_a_feature():
    assert _scraper().active_features() == ("threads", "merged_thread_meta")


def test_a_live_session_is_not_restricted():
    s = _scraper(semaphores=True, mutexes=True, msgqs=True)
    s._m_scraper = MagicMock(is_live=True)

    s._restrict_features_to_recording()

    assert [s.available[key] for key in ("semaphores", "mutexes", "msgqs")] == [True] * 3


def test_a_recording_without_a_group_turns_it_off():
    """Replay matches a strict read sequence, so an absent group must not be read."""
    s = _scraper(semaphores=True, mutexes=True, msgqs=True, mem_slabs=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "semaphores"))

    s._restrict_features_to_recording()

    assert s.available["semaphores"]
    assert not s.available["mutexes"]
    assert not s.available["msgqs"]
    assert not s.available["mem_slabs"]


def test_a_group_absent_from_the_elf_stays_off_even_if_recorded():
    s = _scraper(semaphores=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "semaphores", "msgqs"))

    s._restrict_features_to_recording()

    assert not s.available["msgqs"]


def test_a_scraper_reporting_no_features_is_left_alone():
    s = _scraper(msgqs=True)
    s._m_scraper = MagicMock(is_live=False, spec=["is_live"])

    s._restrict_features_to_recording()

    assert s.available["msgqs"]


def test_nothing_is_polled_when_no_group_is_enabled():
    s = _polling_scraper()

    assert not s.has_kernel_objects()
    assert s._poll_kernel_objects(queue.Queue()) == {}


def test_a_build_with_only_heaps_still_has_objects_to_show():
    """The objects view lists heaps, so the gate cannot rest on the other groups."""
    assert _scraper(heaps=True).has_kernel_objects()


def test_a_view_without_a_group_skips_its_reads():
    from zview.selection import Selection

    s = _polling_scraper(msgqs={"msgq": [0x100]})
    s.view = Selection(groups=frozenset({"threads"}))

    assert s._poll_kernel_objects(queue.Queue()) == {}
    assert not s._m_scraper.method_calls


def test_absent_features_name_the_kconfig_they_need():
    s = _scraper(stack_fill=False, thread_list=False, heap_stats=False)
    s.has_names = False
    s.has_usage = True
    s.has_stack_info = False

    assert s.absent_features() == (
        "Warning: no thread names (CONFIG_THREAD_NAME=n)",
        "Warning: no heap stats (CONFIG_SYS_HEAP_RUNTIME_STATS=n)",
        "Warning: static threads only (CONFIG_THREAD_MONITOR=n)",
        "Warning: no stack sizes (CONFIG_THREAD_STACK_INFO=n)",
        "Warning: no stack usage (CONFIG_INIT_STACKS=n)",
    )


def test_a_full_build_has_no_notices():
    s = _scraper(heaps=True)
    s.has_names = True
    s.has_usage = True
    s.has_stack_info = True

    assert s.absent_features() == ()


def test_a_recording_made_before_the_merged_metadata_reads_keeps_the_old_shape():
    """Its stream holds one read per metadata field, so the poll must ask that way."""
    s = _scraper()
    s.merged_thread_meta = True
    s._m_scraper = MagicMock(is_live=False, features=("threads", "heaps"))

    s._restrict_features_to_recording()

    assert not s.merged_thread_meta


def test_a_recording_that_declares_the_merged_reads_replays_them():
    s = _scraper()
    s._m_scraper = MagicMock(is_live=False, features=("threads", "merged_thread_meta"))

    s._restrict_features_to_recording()

    assert s.merged_thread_meta


def test_work_queues_are_a_recorded_feature():
    """Their reads sit in every frame, so a recording has to say it holds them."""
    assert "workqs" in _scraper(workqs=True).active_features()
    assert "workqs" not in _scraper().active_features()


def test_a_recording_without_work_queues_does_not_replay_them():
    s = _scraper(workqs=True, semaphores=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "semaphores"))

    s._restrict_features_to_recording()

    assert not s.available["workqs"]
    assert s.available["semaphores"]


def test_a_skipped_stack_scan_is_a_recorded_feature():
    """The scan reads each stack every frame, so a recording has to say it left them out."""
    assert "unfilled_stacks" in _scraper(stack_fill=False).active_features()
    assert "unfilled_stacks" not in _scraper().active_features()


def test_a_recording_that_scanned_an_unfilled_build_replays_the_scan():
    """Before the scan was skipped, every stack was read, filled or not."""
    s = _scraper(stack_fill=False)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "merged_thread_meta"))

    s._restrict_features_to_recording()

    assert s.has_stack_fill


def test_a_recording_that_skipped_the_scan_skips_it_again():
    s = _scraper(stack_fill=False)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "unfilled_stacks"))

    s._restrict_features_to_recording()

    assert not s.has_stack_fill


def test_the_usage_snapshot_is_a_recorded_feature():
    """It moves every counter read to the start of the frame."""
    assert "usage_snapshot" in _scraper(usage=True).active_features()
    assert "usage_snapshot" not in _scraper().active_features()


def test_a_recording_from_before_the_snapshot_reads_the_counters_its_way():
    s = _scraper(usage=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "merged_thread_meta"))

    s._restrict_features_to_recording()

    assert not s.usage_snapshot


def test_a_build_with_only_work_queues_has_objects_to_show():
    assert _scraper(workqs=True).has_kernel_objects()


def test_a_group_walker_gets_only_the_selected_labels():
    """Matched against the labels rows carry, ``name@0x...`` for a name with several."""
    from zview.selection import Selection

    s = _polling_scraper(semaphores={"sem_a": [0x100], "sem_b": [0x200, 0x300]})
    s.selection = Selection(exclude=("sem_a",), muted=frozenset({("semaphores", "sem_b@0x300")}))
    given: list[dict] = []

    with patch(
        "zview.orchestrator.GROUPS",
        _with_walks(semaphores=lambda session, rows: given.append(rows) or {}),
    ):
        s._poll_kernel_objects(queue.Queue())

    assert given == [{"sem_b@0x200": [0x200]}]


def test_the_poll_reads_the_work_queues_last():
    """Appended after the other groups, so a recording's earlier reads keep their order."""
    s = _polling_scraper(semaphores={"sem": [0x100]}, workqs={"k_sys_work_q": [0x200]})
    order: list[str] = []

    def walker(key):
        return lambda session, rows: order.append(key) or {key: key}

    with patch(
        "zview.orchestrator.GROUPS",
        _with_walks(semaphores=walker("semaphores"), workqs=walker("workqs")),
    ):
        frame = s._poll_kernel_objects(queue.Queue())

    assert order == ["semaphores", "workqs"]
    assert frame == {"semaphores": ["semaphores"], "workqs": ["workqs"]}
