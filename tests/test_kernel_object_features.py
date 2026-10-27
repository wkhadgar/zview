# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the kernel-object feature list a recording carries."""

from unittest.mock import MagicMock

from zview.kernel.compat import CONFIG_INIT_STACKS
from zview.kernel.groups import GROUPS
from zview.kernel.layout import KernelLayout
from zview.orchestrator import ZScraper

# The layout fields each capability resolves.
_CAPABILITY_FIELDS = {
    "thread_list": ("threads_head", "thread_next"),
    "names": ("thread_name",),
    "stack_info": ("stack_start", "stack_size"),
    "usage": ("cpu_usage", "thread_usage"),
    "heap_stats": (
        "heap_free_bytes",
        "heap_allocated_bytes",
        "heap_max_allocated_bytes",
        "heap_end_chunk",
    ),
}


class _Kconfig:
    """An ELF whose symbol table holds only these ``CONFIG_*`` options, and these heaps."""

    def __init__(self, *options: str, heaps: dict[str, list[int]]):
        self._options = set(options)
        self._heaps = heaps

    def get_symbol_info(self, name: str, _attribute: str) -> list[int]:
        if name not in self._options:
            raise LookupError(name)
        return [1]

    def find_struct_instances(self, struct: str) -> dict[str, list[int]]:
        return self._heaps if struct == "k_heap" else {}


def _scraper(**enabled: bool) -> ZScraper:
    """``ZScraper`` with only the named groups, capabilities and Kconfig, without an ELF."""
    on = {
        "threads": True,
        "thread_list": True,
        "names": True,
        "stack_info": True,
        "stack_fill": True,
        "heap_stats": True,
        "declared_heaps": True,
        "usage": False,
    } | enabled
    s = ZScraper.__new__(ZScraper)
    s.groups = {cls.key: cls() for cls in GROUPS}
    s.available = {key: on.get(key, False) for key in s.groups}
    s.rows = {key: {} for key in s.groups}
    s._m_scraper = MagicMock(is_live=True)
    s._elf_inspector = _Kconfig(
        *([CONFIG_INIT_STACKS] if on["stack_fill"] else []),
        heaps={"test_heap": [0x1000]} if on["declared_heaps"] else {},
    )
    s._layout = KernelLayout(
        **{
            field: 0
            for capability, fields in _CAPABILITY_FIELDS.items()
            if on[capability]
            for field in fields
        }
    )
    s.waitq_flavor = "simple"
    s.parts = set()
    s.parts = {part for group in s.groups.values() for part in group.live_parts(s._session())}
    return s


def _polling_scraper(**rows: dict[str, list[int]]) -> ZScraper:
    """``_scraper`` with ``rows`` discovered per group, ready to poll them."""
    s = _scraper(threads="threads" in rows, **{key: True for key in rows})
    s.rows.update(rows)
    s._m_scraper = MagicMock()
    return s


def _objects_shown(s: ZScraper) -> bool:
    """Whether the TUI's objects view has rows to draw over ``s``."""
    from zview.frontend.zview_tui import ZView

    view = ZView.__new__(ZView)
    view.scraper = s
    view.selection = s.selection
    return view.has_kernel_objects()


def test_active_features_lists_only_what_is_polled():
    s = _scraper(heaps=True, semaphores=True, msgqs=True, mem_slabs=True)

    assert s.active_features() == (
        "threads",
        "merged_thread_meta",
        "distinct_names",
        "thread_usage_alone",
        "single_usage_pass",
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
    assert _scraper().active_features() == (
        "threads",
        "merged_thread_meta",
        "distinct_names",
        "thread_usage_alone",
        "single_usage_pass",
    )


def test_a_recording_from_before_reading_the_counters_alone_counts_no_usage():
    """Without the kernel-wide counter, that zview read no usage, so its replay does not."""
    from zview.kernel.session import Session
    from zview.kernel.threads import Threads

    def session(parts: frozenset[str], cpu_usage: int | None = None) -> Session:
        layout = KernelLayout(thread_usage=48, cpu_usage=cpu_usage)
        return Session(MagicMock(), MagicMock(), layout, "simple", {}, parts)

    assert Threads._counts_usage(session(frozenset({"thread_usage_alone"})))
    assert not Threads._counts_usage(session(frozenset()))
    assert Threads._counts_usage(session(frozenset(), cpu_usage=16))


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


def test_a_replay_reads_the_rows_its_recording_read():
    """A recording made with a selection holds only its reads, so its replay asks for those."""
    from zview.selection import Selection

    recorded = Selection(groups=frozenset({"threads", "mutexes"}))
    s = _scraper(mutexes=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "mutexes"), selection=recorded)

    s._restrict_features_to_recording()

    assert s.selection == recorded


def test_a_scraper_reporting_no_features_is_left_alone():
    s = _scraper(msgqs=True)
    s._m_scraper = MagicMock(is_live=False, spec=["is_live"])

    s._restrict_features_to_recording()

    assert s.available["msgqs"]


def test_nothing_is_polled_when_no_group_is_enabled():
    s = _polling_scraper()

    assert s._poll_groups() == {}


def test_threads_alone_leave_the_objects_view_empty():
    assert not _objects_shown(_scraper())


def test_a_selection_of_threads_only_leaves_the_objects_view_empty():
    from zview.selection import Selection

    s = _scraper(heaps=True, mutexes=True)
    s.selection = Selection(groups=frozenset({"threads"}))

    assert not _objects_shown(s)


def test_a_build_with_only_heaps_still_has_objects_to_show():
    """The objects view lists heaps, so the gate cannot rest on the other groups."""
    assert _objects_shown(_scraper(heaps=True))


def test_a_view_without_a_group_skips_its_reads():
    from zview.selection import Selection

    s = _polling_scraper(msgqs={"msgq": [0x100]})
    s.view = Selection(groups=frozenset({"semaphores"}))

    assert s._poll_groups() == {}
    assert not s._m_scraper.method_calls


def test_absent_features_name_the_kconfig_they_need():
    s = _scraper(
        stack_fill=False, thread_list=False, names=False, stack_info=False, heap_stats=False
    )

    assert s.absent_features() == (
        "Warning: no thread names (CONFIG_THREAD_NAME=n)",
        "Warning: no cpu stats (CONFIG_THREAD_RUNTIME_STATS=n)",
        "Warning: static threads only (CONFIG_THREAD_MONITOR=n)",
        "Warning: no stack sizes (CONFIG_THREAD_STACK_INFO=n)",
        "Warning: no stack usage (CONFIG_INIT_STACKS=n)",
        "Warning: no heap stats (CONFIG_SYS_HEAP_RUNTIME_STATS=n)",
    )


def test_a_full_build_has_no_notices():
    assert _scraper(heaps=True, usage=True).absent_features() == ()


def test_a_build_without_heaps_misses_no_heap_stats():
    assert _scraper(usage=True, heap_stats=False, declared_heaps=False).absent_features() == ()


def test_a_recording_made_before_the_merged_metadata_reads_keeps_the_old_shape():
    """Its stream holds one read per metadata field, so the poll must ask that way."""
    s = _scraper()
    s._m_scraper = MagicMock(is_live=False, features=("threads", "heaps"))

    s._restrict_features_to_recording()

    assert "merged_thread_meta" not in s.parts


def test_a_recording_that_declares_the_merged_reads_replays_them():
    s = _scraper()
    s._m_scraper = MagicMock(is_live=False, features=("threads", "merged_thread_meta"))

    s._restrict_features_to_recording()

    assert "merged_thread_meta" in s.parts


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

    assert "unfilled_stacks" not in s.parts


def test_a_recording_that_skipped_the_scan_skips_it_again():
    s = _scraper(stack_fill=False)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "unfilled_stacks"))

    s._restrict_features_to_recording()

    assert "unfilled_stacks" in s.parts


def test_the_usage_snapshot_is_a_recorded_feature():
    """It moves every counter read to the start of the frame."""
    assert "usage_snapshot" in _scraper(usage=True).active_features()
    assert "usage_snapshot" not in _scraper().active_features()


def test_a_recording_from_before_the_snapshot_reads_the_counters_its_way():
    s = _scraper(usage=True)
    s._m_scraper = MagicMock(is_live=False, features=("threads", "merged_thread_meta"))

    s._restrict_features_to_recording()

    assert "usage_snapshot" not in s.parts


def test_a_build_with_only_work_queues_has_objects_to_show():
    assert _objects_shown(_scraper(workqs=True))


def test_a_group_walker_gets_only_the_selected_labels():
    """Matched against the labels rows carry, ``name@0x...`` for a name with several."""
    from zview.selection import Selection

    s = _polling_scraper(semaphores={"sem_a": [0x100], "sem_b": [0x200, 0x300]})
    s.selection = Selection(
        exclude=((None, "sem_a"),), muted=frozenset({("semaphores", "sem_b@0x300")})
    )
    given: list[dict] = []
    s.groups["semaphores"].walk = lambda session, rows: given.append(rows) or {}

    s._poll_groups()

    assert given == [{"sem_b@0x200": [0x200]}]


def test_the_poll_reads_the_work_queues_last():
    """Appended after the other groups, so a recording's earlier reads keep their order."""
    s = _polling_scraper(semaphores={"sem": [0x100]}, workqs={"k_sys_work_q": [0x200]})
    order: list[str] = []

    def walker(key):
        return lambda session, rows: order.append(key) or {key: key}

    s.groups["semaphores"].walk = walker("semaphores")
    s.groups["workqs"].walk = walker("workqs")

    frame = s._poll_groups()

    assert order == ["semaphores", "workqs"]
    assert frame == {"semaphores": ["semaphores"], "workqs": ["workqs"]}
