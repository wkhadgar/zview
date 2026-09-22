# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the kernel-object feature list a recording carries."""

import queue
from unittest.mock import MagicMock

from orchestrator import ZScraper


def _scraper(**enabled: bool) -> ZScraper:
    """``ZScraper`` with only the feature flags set, without touching an ELF."""
    s = ZScraper.__new__(ZScraper)
    s.has_heaps = enabled.get("heaps", False)
    s.has_semaphores = enabled.get("semaphores", False)
    s.has_mutexes = enabled.get("mutexes", False)
    s.has_msgqs = enabled.get("msgqs", False)
    s.has_mem_slabs = enabled.get("mem_slabs", False)
    s.has_heap_waiters = enabled.get("heap_waiters", s.has_heaps)
    s.poll_kernel_objects = True
    return s


def test_active_features_lists_only_what_is_polled():
    s = _scraper(heaps=True, semaphores=True, msgqs=True, mem_slabs=True)

    assert s.active_features() == (
        "threads",
        "merged_thread_meta",
        "heaps",
        "semaphores",
        "msgqs",
        "mem_slabs",
        "heap_waiters",
    )


def test_a_recording_without_the_heap_wait_queue_reads_does_not_replay_them():
    """The walk sits inside the heaps group, so an older recording has no reads for it."""
    s = _scraper(heaps=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "heaps"))

    s._restrict_features_to_recording()

    assert s.has_heaps
    assert not s.has_heap_waiters


def test_threads_are_always_a_feature():
    assert _scraper().active_features() == ("threads", "merged_thread_meta")


def test_a_live_session_is_not_restricted():
    s = _scraper(semaphores=True, mutexes=True, msgqs=True)
    s._m_scraper = MagicMock(is_live=True)

    s._restrict_features_to_recording()

    assert (s.has_semaphores, s.has_mutexes, s.has_msgqs) == (True, True, True)


def test_a_recording_without_a_group_turns_it_off():
    """Replay matches a strict read sequence, so an absent group must not be read."""
    s = _scraper(semaphores=True, mutexes=True, msgqs=True, mem_slabs=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "semaphores"))

    s._restrict_features_to_recording()

    assert s.has_semaphores
    assert not s.has_mutexes
    assert not s.has_msgqs
    assert not s.has_mem_slabs


def test_a_group_absent_from_the_elf_stays_off_even_if_recorded():
    s = _scraper(semaphores=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "semaphores", "msgqs"))

    s._restrict_features_to_recording()

    assert not s.has_msgqs


def test_a_scraper_reporting_no_features_is_left_alone():
    s = _scraper(msgqs=True)
    s._m_scraper = MagicMock(is_live=False, spec=["is_live"])

    s._restrict_features_to_recording()

    assert s.has_msgqs


def test_nothing_is_polled_when_no_group_is_enabled():
    s = _scraper()

    assert not s.has_kernel_objects()
    assert s._poll_kernel_objects(queue.Queue()) == {}


def test_a_build_with_only_heaps_still_has_objects_to_show():
    """The objects view lists heaps, so the gate cannot rest on the other groups."""
    assert _scraper(heaps=True).has_kernel_objects()


def test_the_view_gate_skips_the_reads():
    s = _scraper(msgqs=True)
    s.poll_kernel_objects = False

    assert s._poll_kernel_objects(queue.Queue()) == {}


def test_absent_features_name_the_kconfig_they_need():
    s = _scraper()
    s.has_names = False
    s.has_usage = True
    s.has_heaps = False

    assert s.absent_features() == (
        "Warning: no thread names (CONFIG_THREAD_NAME=n)",
        "Warning: no heap stats (CONFIG_SYS_HEAP_RUNTIME_STATS=n)",
    )


def test_a_full_build_has_no_notices():
    s = _scraper(heaps=True)
    s.has_names = True
    s.has_usage = True

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
