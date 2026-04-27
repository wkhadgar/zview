# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
Coverage for the west zview command: predicate helpers plus end-to-end
verification of the ELF and runner injection paths. Skipped when west is
not importable.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("west")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "west_commands"))

from zview_tui_cmd import (  # noqa: E402
    ZViewCommand,
    _command_wants_runner,
    _has_flag,
    _help_requested,
)


def test_has_flag_short():
    assert _has_flag(["-e", "x"], "-e", "--elf-file") is True


def test_has_flag_long():
    assert _has_flag(["--elf-file", "x"], "-e", "--elf-file") is True


def test_has_flag_long_equals():
    assert _has_flag(["--elf-file=x"], "-e", "--elf-file") is True


def test_has_flag_absent():
    assert _has_flag(["-r", "jlink"], "-e", "--elf-file") is False


def test_command_wants_runner_live():
    assert _command_wants_runner("live", []) is True


def test_command_wants_runner_record():
    assert _command_wants_runner("record", ["-o", "x.gz", "--frames", "1"]) is True


def test_command_wants_runner_replay_never():
    assert _command_wants_runner("replay", ["-i", "x.gz"]) is False


def test_command_wants_runner_dump_with_input_no():
    assert _command_wants_runner("dump", ["-i", "x.gz", "--json"]) is False


def test_command_wants_runner_dump_without_input_yes():
    assert _command_wants_runner("dump", ["--json"]) is True


def test_help_requested_short_long():
    assert _help_requested(["-h"]) is True
    assert _help_requested(["--help"]) is True
    assert _help_requested(["record", "--help"]) is True
    assert _help_requested(["live", "-e", "x"]) is False


# --- ELF injection ---


def _make_cmd() -> ZViewCommand:
    """ZViewCommand with the minimal config that die/err/wrn need outside a west session."""
    cmd = ZViewCommand()
    cmd.config = MagicMock()
    cmd.config.getboolean.return_value = False
    return cmd


@pytest.fixture
def fake_build_dir(tmp_path, monkeypatch):
    """tmp_path with a populated build/zephyr/zephyr.elf and runners.yaml."""
    build = tmp_path / "build" / "zephyr"
    build.mkdir(parents=True)
    elf = build / "zephyr.elf"
    elf.write_bytes(b"")
    runners_yaml = build / "runners.yaml"
    runners_yaml.write_text("flash-runner: jlink\nargs:\n  jlink:\n    - --device=nRF5340_xxAA\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_inject_elf_when_missing_uses_build_path(fake_build_dir):
    cmd = _make_cmd()
    expected = str(fake_build_dir / "build" / "zephyr" / "zephyr.elf")

    out = cmd._inject_elf_if_missing(["live"])
    assert out == ["live", "-e", expected]


def test_inject_elf_at_front_when_no_command(fake_build_dir):
    """No command present → ELF flag goes at the front (becomes implicit live)."""
    cmd = _make_cmd()
    expected = str(fake_build_dir / "build" / "zephyr" / "zephyr.elf")

    out = cmd._inject_elf_if_missing([])
    assert out == ["-e", expected]


def test_inject_elf_keeps_command_at_front_with_other_flags(fake_build_dir):
    """The verb stays at argv[0]; -e is inserted just after it."""
    cmd = _make_cmd()
    expected = str(fake_build_dir / "build" / "zephyr" / "zephyr.elf")

    out = cmd._inject_elf_if_missing(["dump", "--json"])
    assert out == ["dump", "-e", expected, "--json"]


def test_inject_elf_skips_when_e_present(fake_build_dir):
    cmd = _make_cmd()
    out = cmd._inject_elf_if_missing(["live", "-e", "/custom/path.elf"])
    assert out == ["live", "-e", "/custom/path.elf"]


def test_inject_elf_skips_when_long_form_present(fake_build_dir):
    cmd = _make_cmd()
    out = cmd._inject_elf_if_missing(["--elf-file=/x.elf", "live"])
    assert out == ["--elf-file=/x.elf", "live"]


def test_inject_elf_skips_when_help_requested(tmp_path, monkeypatch):
    """No build dir, but --help must still pass through without dying."""
    monkeypatch.chdir(tmp_path)
    cmd = _make_cmd()
    out = cmd._inject_elf_if_missing(["--help"])
    assert out == ["--help"]


def test_inject_elf_dies_when_build_missing(tmp_path, monkeypatch):
    """No build dir and no --help → log.die must be invoked."""
    monkeypatch.chdir(tmp_path)
    cmd = _make_cmd()
    with pytest.raises(SystemExit):
        cmd._inject_elf_if_missing(["live"])


# --- Runner injection ---


def test_inject_runner_for_live(fake_build_dir):
    cmd = _make_cmd()
    out = cmd._inject_runner_if_missing(["live"])
    assert out == ["live", "-r", "jlink", "-t", "nRF5340_xxAA"]


def test_inject_runner_for_record(fake_build_dir):
    cmd = _make_cmd()
    out = cmd._inject_runner_if_missing(["record", "-o", "x.gz", "--frames", "1"])
    # Verb stays at argv[0]; -r/-t inserted right after.
    assert out[0] == "record"
    assert out[1:5] == ["-r", "jlink", "-t", "nRF5340_xxAA"]
    assert out[5:] == ["-o", "x.gz", "--frames", "1"]


def test_inject_runner_keeps_dump_command_at_front(fake_build_dir):
    """Regression: dump --json must not have -r/-t injected before the verb."""
    cmd = _make_cmd()
    out = cmd._inject_runner_if_missing(["dump", "--json"])
    assert out[0] == "dump"
    assert "-r" in out and "-t" in out
    assert out.index("-r") > 0  # not at front


def test_inject_runner_skips_replay(fake_build_dir):
    cmd = _make_cmd()
    argv = ["replay", "-i", "x.gz"]
    out = cmd._inject_runner_if_missing(argv)
    assert out == argv


def test_inject_runner_skips_dump_with_input(fake_build_dir):
    cmd = _make_cmd()
    argv = ["dump", "-i", "x.gz", "--json"]
    out = cmd._inject_runner_if_missing(argv)
    assert out == argv


def test_inject_runner_for_dump_without_input(fake_build_dir):
    cmd = _make_cmd()
    out = cmd._inject_runner_if_missing(["dump", "--json"])
    assert "-r" in out and "-t" in out


def test_inject_runner_skips_when_both_present(fake_build_dir):
    cmd = _make_cmd()
    argv = ["live", "-r", "pyocd", "-t", "stm32h753zitx"]
    out = cmd._inject_runner_if_missing(argv)
    assert out == argv  # no duplication


def test_inject_runner_fills_only_missing_target(fake_build_dir):
    """Runner explicit, target missing → only -t injected."""
    cmd = _make_cmd()
    out = cmd._inject_runner_if_missing(["live", "-r", "jlink"])
    # -t injected, -r untouched. The user's explicit "-r jlink" is preserved.
    assert "-t" in out
    assert out.count("-r") == 1


def test_inject_runner_skips_when_help_requested(fake_build_dir):
    cmd = _make_cmd()
    out = cmd._inject_runner_if_missing(["live", "--help"])
    assert out == ["live", "--help"]
    # Specifically: no runners.yaml read attempted.
