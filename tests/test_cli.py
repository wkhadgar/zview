# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""CLI wiring coverage for zview.cli.main."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from zview import cli
from zview.backend.image import ImageMismatch

_TESTS_DIR = Path(__file__).parent
_FIXTURE = _TESTS_DIR / "fixtures" / "recordings" / "sys_heap_v4.3.ndjson.gz"
_ELF = _TESTS_DIR / "fixtures" / "zephyr.elf"


def _invoke(monkeypatch, capsys, argv: list[str]):
    """Run zview.cli.main with ``argv``; return (rc, stdout, stderr)."""
    monkeypatch.setattr(sys, "argv", ["zview", *argv])
    try:
        rc = cli.main()
    except SystemExit as exc:
        rc = exc.code
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_normalize_argv_splices_live_for_bare_invocation():
    assert cli._normalize_argv(["-e", "x", "-r", "jlink", "-t", "y"]) == [
        "live",
        "-e",
        "x",
        "-r",
        "jlink",
        "-t",
        "y",
    ]


def test_normalize_argv_passes_known_commands_through():
    for cmd in cli.KNOWN_COMMANDS:
        assert cli._normalize_argv([cmd, "-e", "x"]) == [cmd, "-e", "x"]


def test_normalize_argv_passes_help_through():
    assert cli._normalize_argv(["--help"]) == ["--help"]
    assert cli._normalize_argv(["-h"]) == ["-h"]


def test_only_names_the_groups_the_session_does_not_have(monkeypatch, capsys):
    """The session's own groups decide, so a group added to the poll needs no list here."""
    rc, _, err = _invoke(
        monkeypatch,
        capsys,
        ["dump", "-e", str(_ELF), "-i", str(_FIXTURE), "--only", "threads,timer"],
    )

    assert rc != 0
    assert "Unknown group timer; choose from threads, heaps, semaphores" in err
    assert cli._groups("threads, heaps") == {"threads", "heaps"}


def test_dump_replay_json_stdout_is_valid_json(monkeypatch, capsys):
    rc, out, _ = _invoke(
        monkeypatch,
        capsys,
        ["dump", "-e", str(_ELF), "-i", str(_FIXTURE), "--json"],
    )
    assert rc == 0
    data = json.loads(out)
    assert "threads" in data and len(data["threads"]) >= 1


def test_dump_replay_json_keeps_stdout_clean(monkeypatch, capsys):
    rc, out, err = _invoke(
        monkeypatch,
        capsys,
        ["dump", "-e", str(_ELF), "-i", str(_FIXTURE), "--json"],
    )
    assert rc == 0
    stripped = out.strip()
    assert stripped.startswith("{") and stripped.endswith("}")
    assert "Loaded cached ELF" not in out
    assert "Loaded cached ELF" in err or "Loading ELF" in err


def test_dump_replay_human_readable(monkeypatch, capsys):
    rc, out, _ = _invoke(
        monkeypatch,
        capsys,
        ["dump", "-e", str(_ELF), "-i", str(_FIXTURE)],
    )
    assert rc == 0
    for thread in ("stress_id", "idle", "main"):
        assert thread in out
    assert "stack=" in out and "watermark=" in out


def test_dump_prints_a_semaphore_without_a_limit_or_an_init(monkeypatch, capsys):
    from zview.backend.base import SemaphoreInfo

    frame = {
        "semaphores": [
            SemaphoreInfo(
                name="unbounded", address=0x1000, count=14, limit=0xFFFF_FFFF, waiters=()
            ),
            SemaphoreInfo(name="late", address=0x1010, count=0, limit=0, waiters=()),
            SemaphoreInfo(name="bounded", address=0x1020, count=2, limit=4, waiters=()),
        ]
    }
    monkeypatch.setattr(cli, "dump_single_frame", lambda *a, **kw: frame)

    rc, out, _ = _invoke(monkeypatch, capsys, ["dump", "-e", str(_ELF), "-i", str(_FIXTURE)])

    assert rc == 0
    assert "sem  unbounded            14/-  waiters=-" in out
    assert "sem  late                 not initialized  waiters=-" in out
    assert "sem  bounded              2/4  waiters=-" in out


def test_dump_prints_an_unmeasured_watermark_as_a_dash(monkeypatch, capsys):
    from zview.backend.base import ThreadInfo, ThreadRuntime

    unmeasured = ThreadRuntime(
        cpu=None,
        cpu_normalized=None,
        active=False,
        stack_watermark=None,
        stack_watermark_percent=None,
    )
    frame = {
        "threads": [
            ThreadInfo(0x1000, 0x2000, 512, "worker", unmeasured),
            ThreadInfo(0x1100, None, None, "unsized", unmeasured),
        ]
    }
    monkeypatch.setattr(cli, "dump_single_frame", lambda *a, **kw: frame)

    rc, out, _ = _invoke(monkeypatch, capsys, ["dump", "-e", str(_ELF), "-i", str(_FIXTURE)])

    assert rc == 0
    assert "worker                         stack=   512  watermark=-" in out
    assert "unsized                        stack=     -  watermark=-" in out


def test_dump_prints_every_group_it_polled(monkeypatch, capsys):
    """The text dump covers what the JSON one carries, groups added later included."""
    from zview.backend.base import HeapInfo, MemSlabInfo, MsgqInfo, WorkqInfo

    frame = {
        "heaps": [
            HeapInfo(
                name="bench_heap",
                address=0x5000,
                free_bytes=768,
                allocated_bytes=768,
                max_allocated_bytes=1536,
                usage_percent=50.0,
                chunks=None,
                waiters=("heap_waiter_id",),
            )
        ],
        "msgqs": [
            MsgqInfo(
                name="bench_q", address=0x4000, used_msgs=8, max_msgs=8, msg_size=4, waiters=()
            )
        ],
        "mem_slabs": [
            MemSlabInfo(
                name="bench_slab",
                address=0x6000,
                num_blocks=8,
                block_size=64,
                num_used=6,
                max_used=7,
                waiters=("slab_id",),
            )
        ],
        "workqs": [
            WorkqInfo(
                name="bench_workq",
                address=0x7000,
                flags=0b11,
                pending=("trail_work (work_handler)",),
                thread_name="bench_workq",
                waiters=(),
            ),
            WorkqInfo(name="idle_q", address=0x7100, flags=0, waiters=None),
        ],
    }
    monkeypatch.setattr(cli, "dump_single_frame", lambda *a, **kw: frame)

    rc, out, _ = _invoke(monkeypatch, capsys, ["dump", "-e", str(_ELF), "-i", str(_FIXTURE)])

    assert rc == 0
    assert "msgq bench_q              8/8 x 4B waiters=-" in out
    assert "slab bench_slab           6/8 x 64B peak=7 waiters=slab_id" in out
    assert "waiters=heap_waiter_id" in out
    assert (
        "wkq  bench_workq          1 pending flags=started,busy thread=bench_workq "
        "items=trail_work (work_handler) waiters=-"
    ) in out
    assert "wkq  idle_q               0 pending flags=- thread=- items=- waiters=unknown" in out


def test_dump_frame_arg_skips_to_requested_frame(monkeypatch, capsys):
    """--frame 3 emits a different frame than --frame 1."""
    rc1, out1, _ = _invoke(
        monkeypatch,
        capsys,
        ["dump", "-e", str(_ELF), "-i", str(_FIXTURE), "--json", "--frame", "1"],
    )
    rc3, out3, _ = _invoke(
        monkeypatch,
        capsys,
        ["dump", "-e", str(_ELF), "-i", str(_FIXTURE), "--json", "--frame", "3"],
    )
    assert rc1 == 0 and rc3 == 0
    f1 = json.loads(out1)
    f3 = json.loads(out3)
    # Same threads, different runtime state across frames.
    assert {t["name"] for t in f1["threads"]} == {t["name"] for t in f3["threads"]}
    cpu1 = sorted(t["runtime"]["cpu_normalized"] for t in f1["threads"])
    cpu3 = sorted(t["runtime"]["cpu_normalized"] for t in f3["threads"])
    assert cpu1 != cpu3


def test_dump_rejects_zero_or_negative_frame(monkeypatch, capsys):
    rc, _, err = _invoke(
        monkeypatch,
        capsys,
        ["dump", "-e", str(_ELF), "-i", str(_FIXTURE), "--frame", "0"],
    )
    assert rc == 2
    assert "--frame" in err


def test_dump_requires_input_or_runner_target(monkeypatch, capsys):
    rc, _, err = _invoke(monkeypatch, capsys, ["dump", "-e", str(_ELF)])
    assert rc == 2
    assert "--input" in err


def test_dump_input_and_runner_are_mutually_exclusive(monkeypatch, capsys):
    rc, _, err = _invoke(
        monkeypatch,
        capsys,
        ["dump", "-e", str(_ELF), "-i", str(_FIXTURE), "-r", "gdb", "-t", "localhost:1"],
    )
    assert rc == 2
    assert "mutually exclusive" in err


def test_live_requires_runner(monkeypatch, capsys):
    rc, _, err = _invoke(monkeypatch, capsys, ["-e", str(_ELF)])
    assert rc == 2
    assert "--runner" in err


def test_live_requires_runner_target(monkeypatch, capsys):
    rc, _, err = _invoke(monkeypatch, capsys, ["-e", str(_ELF), "-r", "gdb"])
    assert rc == 2
    assert "--runner-target" in err


def test_record_requires_duration_or_frames(monkeypatch, capsys, tmp_path):
    rc, _, err = _invoke(
        monkeypatch,
        capsys,
        [
            "record",
            "-e",
            str(_ELF),
            "-r",
            "gdb",
            "-t",
            "localhost:1",
            "-o",
            str(tmp_path / "x.ndjson.gz"),
        ],
    )
    assert rc == 2
    assert "--duration" in err or "--frames" in err


def test_record_dispatches_with_frames_bound(monkeypatch, capsys, tmp_path):
    recorded: list[dict] = []

    def fake_record(backend, elf_path, out_path, **kwargs):
        recorded.append({"out_path": str(out_path), **kwargs})
        return 7

    monkeypatch.setattr(cli, "record_session", fake_record)

    out_file = tmp_path / "snap.ndjson.gz"
    rc, out, _ = _invoke(
        monkeypatch,
        capsys,
        [
            "record",
            "-e",
            str(_ELF),
            "-r",
            "gdb",
            "-t",
            "localhost:1",
            "-o",
            str(out_file),
            "--frames",
            "5",
        ],
    )
    assert rc == 0
    assert len(recorded) == 1
    assert recorded[0]["frames"] == 5
    assert recorded[0]["duration"] is None
    assert f"Recorded 7 frames to {out_file}" in out


def test_replay_no_pacing_passes_to_scraper(monkeypatch, capsys):
    """--no-pacing -> ReplayScraper(honor_timing=False)."""
    captured: dict = {}
    real_cls = cli.ReplayScraper

    class SpyReplay(real_cls):
        def __init__(self, *args, **kwargs):
            captured["honor_timing"] = kwargs.get("honor_timing", True)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cli, "ReplayScraper", SpyReplay)
    monkeypatch.setattr(cli, "tui_run", lambda *a, **kw: None)
    monkeypatch.setattr(cli.curses, "wrapper", lambda fn, *a, **kw: fn(None, *a, **kw))

    rc, _, _ = _invoke(
        monkeypatch,
        capsys,
        ["replay", "-e", str(_ELF), "-i", str(_FIXTURE), "--no-pacing"],
    )
    assert rc == 0
    assert captured["honor_timing"] is False


_RUNNERS_YAML = (
    "flash-runner: jlink\n"
    "config:\n"
    "  elf_file: zephyr.elf\n"
    "args:\n"
    "  jlink:\n"
    "    - --device=nRF5340_xxAA\n"
)


def _write_build(image: Path, runners_yaml: str | None = _RUNNERS_YAML) -> Path:
    """Populate ``image/zephyr`` with an empty ELF and a runners.yaml; return the ELF."""
    zephyr_dir = image / "zephyr"
    zephyr_dir.mkdir(parents=True)
    elf = zephyr_dir / "zephyr.elf"
    elf.write_bytes(b"")
    if runners_yaml is not None:
        (zephyr_dir / "runners.yaml").write_text(runners_yaml)
    return elf


def _capture_dispatch(monkeypatch) -> list[argparse.Namespace]:
    """Replace ``dispatch``; return the list each dispatched namespace lands in."""
    dispatched = []

    def fake_dispatch(args: argparse.Namespace) -> int:
        dispatched.append(args)
        return 0

    monkeypatch.setattr(cli, "dispatch", fake_dispatch)
    return dispatched


@pytest.mark.parametrize(
    ("cmd", "input_file", "expected"),
    [
        ("live", None, True),
        ("record", None, True),
        ("dump", None, True),
        ("dump", "x.gz", False),
        ("replay", "x.gz", False),
    ],
)
def test_wants_runner_by_command(cmd, input_file, expected):
    args = argparse.Namespace(cmd=cmd, input=input_file)

    assert cli._wants_runner(args) is expected


def test_needs_no_build_dir_for_a_replay_with_an_elf():
    args = argparse.Namespace(cmd="replay", input="x.gz", elf_file="x.elf")

    assert cli.needs_build_dir(args) is False


def test_build_dir_fills_the_live_elf_runner_and_target(monkeypatch, capsys, tmp_path):
    elf = _write_build(tmp_path)
    dispatched = _capture_dispatch(monkeypatch)

    rc, _, _ = _invoke(monkeypatch, capsys, ["-d", str(tmp_path)])

    assert rc == 0
    assert dispatched[0].elf_file == str(elf)
    assert dispatched[0].runner == "jlink"
    assert dispatched[0].runner_target == "nRF5340_xxAA"


def test_neither_elf_nor_build_dir_is_a_usage_error(monkeypatch, capsys):
    rc, _, err = _invoke(monkeypatch, capsys, ["replay", "-i", str(_FIXTURE)])

    assert rc == 2
    assert "--build-dir" in err


def test_build_dir_without_runners_yaml_asks_for_the_runner(monkeypatch, capsys, tmp_path):
    _write_build(tmp_path, runners_yaml=None)

    rc, _, err = _invoke(monkeypatch, capsys, ["-d", str(tmp_path)])

    assert rc == 2
    assert "Please specify runner" in err


def test_foreign_flash_runner_falls_back_to_pyocd(monkeypatch, capsys, tmp_path):
    _write_build(
        tmp_path,
        runners_yaml=(
            "flash-runner: stm32cubeprogrammer\nargs:\n  pyocd:\n    - --target=stm32h753zitx\n"
        ),
    )
    dispatched = _capture_dispatch(monkeypatch)

    rc, _, err = _invoke(monkeypatch, capsys, ["-d", str(tmp_path)])

    assert rc == 0
    assert dispatched[0].runner == "pyocd"
    assert dispatched[0].runner_target == "stm32h753zitx"
    assert "zview: warning: 'stm32cubeprogrammer'" in err


@pytest.mark.parametrize("runner", ["nrfutil", "pyocd"])
def test_build_dir_may_leave_the_target_unset(monkeypatch, capsys, tmp_path, runner):
    """nrfutil picks the sole J-Link and pyocd auto-detects the MCU without a target."""
    _write_build(tmp_path, runners_yaml=f"flash-runner: {runner}\n")
    dispatched = _capture_dispatch(monkeypatch)

    rc, _, _ = _invoke(monkeypatch, capsys, ["-d", str(tmp_path)])

    assert rc == 0
    assert dispatched[0].runner == runner
    assert dispatched[0].runner_target is None


@pytest.mark.parametrize("runner", ["gdb", "jlink"])
def test_runner_needing_a_target_rejects_an_unset_one(monkeypatch, capsys, tmp_path, runner):
    _write_build(tmp_path, runners_yaml="flash-runner: jlink\n")

    rc, _, err = _invoke(monkeypatch, capsys, ["-d", str(tmp_path), "-r", runner])

    assert rc == 2
    assert f"runner '{runner}' requires --runner-target" in err


def test_explicit_runner_and_target_need_no_runners_yaml(monkeypatch, capsys, tmp_path):
    """A QEMU build without runners.yaml takes only its ELF from the build directory."""
    elf = _write_build(tmp_path, runners_yaml=None)
    dispatched = _capture_dispatch(monkeypatch)

    rc, _, _ = _invoke(
        monkeypatch, capsys, ["-d", str(tmp_path), "-r", "gdb", "-t", "localhost:1234"]
    )

    assert rc == 0
    assert dispatched[0].elf_file == str(elf)
    assert dispatched[0].runner_target == "localhost:1234"


@pytest.fixture
def spied_log(monkeypatch, tmp_path):
    """Point ``configure_logging`` at ``tmp_path``; return (log path, logger spy)."""
    log_path = tmp_path / "zview.log"
    spy = MagicMock()
    monkeypatch.setattr(cli, "configure_logging", lambda: log_path)
    monkeypatch.setattr(cli, "logger", spy)
    return log_path, spy


def test_a_failing_command_reports_one_line(monkeypatch, capsys, spied_log):
    log_path, spy = spied_log
    error = ImageMismatch("target image differs from the ELF")

    def failing_dispatch(args: argparse.Namespace) -> int:
        raise error

    monkeypatch.setattr(cli, "dispatch", failing_dispatch)

    rc, _, err = _invoke(monkeypatch, capsys, ["replay", "-e", str(_ELF), "-i", str(_FIXTURE)])

    assert rc == 1
    assert err == f"zview: {error} (traceback in {log_path})\n"
    assert spy.error.call_args.kwargs["exc_info"] is error


def test_a_failure_without_a_message_reports_its_type(spied_log):
    log_path, _ = spied_log

    assert cli.failure_report(ValueError()) == f"ValueError (traceback in {log_path})"
