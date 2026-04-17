# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
CLI wiring coverage for zview_cli.main: argparse dispatch, flag validation,
stdout/stderr separation for headless modes, and --once rendering.
"""

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


def test_once_json_stdout_is_valid_json(monkeypatch, capsys):
    rc, out, _ = _invoke(
        monkeypatch,
        capsys,
        ["-e", str(_ELF), "--replay", str(_FIXTURE), "--once", "--json"],
    )
    assert rc == 0
    data = json.loads(out)
    assert "threads" in data and len(data["threads"]) >= 1


def test_once_json_keeps_stdout_clean_of_status_lines(monkeypatch, capsys):
    rc, out, err = _invoke(
        monkeypatch,
        capsys,
        ["-e", str(_ELF), "--replay", str(_FIXTURE), "--once", "--json"],
    )
    assert rc == 0
    # Stdout must contain ONLY the JSON document; any status line on stdout
    # would both fail json.loads and leak diagnostic strings.
    stripped = out.strip()
    assert stripped.startswith("{")
    assert stripped.endswith("}")
    assert "Loaded cached ELF" not in out
    assert "Connected to GDB" not in out
    # Diagnostic that DID fire must be on stderr.
    assert "Loaded cached ELF" in err or "Loading ELF" in err


def test_once_human_readable_lists_threads(monkeypatch, capsys):
    rc, out, _ = _invoke(
        monkeypatch,
        capsys,
        ["-e", str(_ELF), "--replay", str(_FIXTURE), "--once"],
    )
    assert rc == 0
    for expected_thread in ("stress_id", "idle", "main"):
        assert expected_thread in out
    assert "stack=" in out
    assert "watermark=" in out


def test_requires_runner_without_replay(monkeypatch, capsys):
    rc, _, err = _invoke(monkeypatch, capsys, ["-e", str(_ELF)])
    assert rc == 2
    assert "--runner" in err


def test_requires_runner_target_without_replay(monkeypatch, capsys):
    rc, _, err = _invoke(monkeypatch, capsys, ["-e", str(_ELF), "-r", "gdb"])
    assert rc == 2
    assert "--runner-target" in err


def test_snapshot_and_replay_are_mutually_exclusive(monkeypatch, capsys, tmp_path):
    rc, _, err = _invoke(
        monkeypatch,
        capsys,
        [
            "-e",
            str(_ELF),
            "--replay",
            str(_FIXTURE),
            "--snapshot",
            str(tmp_path / "x.ndjson.gz"),
            "--frames",
            "1",
        ],
    )
    assert rc == 2
    assert "mutually exclusive" in err


def test_snapshot_and_once_are_mutually_exclusive(monkeypatch, capsys, tmp_path):
    rc, _, err = _invoke(
        monkeypatch,
        capsys,
        [
            "-e",
            str(_ELF),
            "-r",
            "gdb",
            "-t",
            "localhost:1",
            "--snapshot",
            str(tmp_path / "x.ndjson.gz"),
            "--once",
            "--frames",
            "1",
        ],
    )
    assert rc == 2
    assert "mutually exclusive" in err


def test_snapshot_requires_duration_or_frames(monkeypatch, capsys, tmp_path):
    rc, _, err = _invoke(
        monkeypatch,
        capsys,
        [
            "-e",
            str(_ELF),
            "-r",
            "gdb",
            "-t",
            "localhost:1",
            "--snapshot",
            str(tmp_path / "x.ndjson.gz"),
        ],
    )
    assert rc == 2
    assert "--duration" in err or "--frames" in err


def test_json_requires_once(monkeypatch, capsys):
    rc, _, err = _invoke(
        monkeypatch,
        capsys,
        ["-e", str(_ELF), "--replay", str(_FIXTURE), "--json"],
    )
    assert rc == 2
    assert "--once" in err


def test_snapshot_dispatches_with_bound(monkeypatch, capsys, tmp_path):
    """main() forwards --frames to record_session and reports the captured count."""
    recorded_calls: list[dict] = []

    def fake_record(backend, elf_path, out_path, duration=None, frames=None, period=0.1):
        recorded_calls.append(
            {
                "out_path": str(out_path),
                "duration": duration,
                "frames": frames,
                "period": period,
            }
        )
        return 7

    monkeypatch.setattr(zview_cli, "record_session", fake_record)

    out_file = tmp_path / "snap.ndjson.gz"
    rc, out, _ = _invoke(
        monkeypatch,
        capsys,
        [
            "-e",
            str(_ELF),
            "-r",
            "gdb",
            "-t",
            "localhost:1",
            "--snapshot",
            str(out_file),
            "--frames",
            "5",
        ],
    )

    assert rc == 0
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["frames"] == 5
    assert recorded_calls[0]["duration"] is None
    assert f"Recorded 7 frames to {out_file}" in out
