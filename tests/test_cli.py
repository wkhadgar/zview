# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""CLI wiring coverage for zview_cli.main."""

import json
import sys
from pathlib import Path

import zview_cli

_TESTS_DIR = Path(__file__).parent
_FIXTURE = _TESTS_DIR / "fixtures" / "recordings" / "sys_heap_v4.3.ndjson.gz"
_ELF = _TESTS_DIR / "fixtures" / "zephyr.elf"


def _invoke(monkeypatch, capsys, argv: list[str]):
    """Run zview_cli.main with ``argv``; return (rc, stdout, stderr)."""
    monkeypatch.setattr(sys, "argv", ["zview", *argv])
    try:
        rc = zview_cli.main()
    except SystemExit as exc:
        rc = exc.code
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


def test_normalize_argv_splices_live_for_bare_invocation():
    assert zview_cli._normalize_argv(["-e", "x", "-r", "jlink", "-t", "y"]) == [
        "live",
        "-e",
        "x",
        "-r",
        "jlink",
        "-t",
        "y",
    ]


def test_normalize_argv_passes_known_commands_through():
    for cmd in zview_cli.KNOWN_COMMANDS:
        assert zview_cli._normalize_argv([cmd, "-e", "x"]) == [cmd, "-e", "x"]


def test_normalize_argv_passes_help_through():
    assert zview_cli._normalize_argv(["--help"]) == ["--help"]
    assert zview_cli._normalize_argv(["-h"]) == ["-h"]


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

    monkeypatch.setattr(zview_cli, "record_session", fake_record)

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
    real_cls = zview_cli.ReplayScraper

    class SpyReplay(real_cls):
        def __init__(self, *args, **kwargs):
            captured["honor_timing"] = kwargs.get("honor_timing", True)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(zview_cli, "ReplayScraper", SpyReplay)
    monkeypatch.setattr(zview_cli, "tui_run", lambda *a, **kw: None)
    monkeypatch.setattr(zview_cli.curses, "wrapper", lambda fn, *a, **kw: fn(None, *a, **kw))

    rc, _, _ = _invoke(
        monkeypatch,
        capsys,
        ["replay", "-e", str(_ELF), "-i", str(_FIXTURE), "--no-pacing"],
    )
    assert rc == 0
    assert captured["honor_timing"] is False
