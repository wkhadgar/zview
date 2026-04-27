# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Coverage for the west zview command. Skipped when west is not importable."""

import argparse
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

pytest.importorskip("west")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "west_commands"))

from zview_tui_cmd import ZViewCommand, _command_wants_runner  # noqa: E402


def _make_cmd() -> ZViewCommand:
    """``ZViewCommand`` with a stub ``config`` so ``die``/``err``/``wrn`` work."""
    cmd = ZViewCommand()
    cmd.config = MagicMock()
    cmd.config.getboolean.return_value = False
    return cmd


def _ns(**kwargs) -> argparse.Namespace:
    """Build a namespace with default fields; ``kwargs`` override."""
    defaults = {
        "cmd": "live",
        "elf_file": None,
        "runner": None,
        "runner_target": None,
        "input": None,
        "period": 0.10,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_command_wants_runner_live():
    assert _command_wants_runner(_ns(cmd="live")) is True


def test_command_wants_runner_record():
    assert _command_wants_runner(_ns(cmd="record")) is True


def test_command_wants_runner_replay_never():
    assert _command_wants_runner(_ns(cmd="replay")) is False


def test_command_wants_runner_dump_with_input_no():
    assert _command_wants_runner(_ns(cmd="dump", input="x.gz")) is False


def test_command_wants_runner_dump_without_input_yes():
    assert _command_wants_runner(_ns(cmd="dump", input=None)) is True


@pytest.fixture
def fake_build_dir(tmp_path, monkeypatch):
    """``tmp_path`` with ``build/zephyr/zephyr.elf`` + ``runners.yaml`` populated."""
    build = tmp_path / "build" / "zephyr"
    build.mkdir(parents=True)
    elf = build / "zephyr.elf"
    elf.write_bytes(b"")
    runners_yaml = build / "runners.yaml"
    runners_yaml.write_text("flash-runner: jlink\nargs:\n  jlink:\n    - --device=nRF5340_xxAA\n")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_fill_elf_when_missing_uses_build_path(fake_build_dir):
    cmd = _make_cmd()
    args = _ns()
    cmd._fill_elf(args)
    assert args.elf_file == str(fake_build_dir / "build" / "zephyr" / "zephyr.elf")


def test_fill_elf_skips_when_explicit(fake_build_dir):
    cmd = _make_cmd()
    args = _ns(elf_file="/custom/path.elf")
    cmd._fill_elf(args)
    assert args.elf_file == "/custom/path.elf"


def test_fill_elf_dies_when_build_missing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cmd = _make_cmd()
    args = _ns()
    with pytest.raises(SystemExit):
        cmd._fill_elf(args)


def test_fill_runner_for_live(fake_build_dir):
    cmd = _make_cmd()
    args = _ns(cmd="live")
    cmd._fill_runner(args)
    assert args.runner == "jlink"
    assert args.runner_target == "nRF5340_xxAA"


def test_fill_runner_for_record(fake_build_dir):
    cmd = _make_cmd()
    args = _ns(cmd="record")
    cmd._fill_runner(args)
    assert args.runner == "jlink"
    assert args.runner_target == "nRF5340_xxAA"


def test_fill_runner_skips_when_both_explicit(fake_build_dir):
    cmd = _make_cmd()
    args = _ns(cmd="live", runner="pyocd", runner_target="stm32h753zitx")
    cmd._fill_runner(args)
    assert args.runner == "pyocd"
    assert args.runner_target == "stm32h753zitx"


def test_fill_runner_fills_only_missing_target(fake_build_dir):
    cmd = _make_cmd()
    args = _ns(cmd="live", runner="jlink", runner_target=None)
    cmd._fill_runner(args)
    assert args.runner == "jlink"  # untouched
    assert args.runner_target == "nRF5340_xxAA"


def test_parser_bare_invocation_defaults_to_live():
    cmd = _make_cmd()
    subparsers = argparse.ArgumentParser().add_subparsers()
    parser = cmd.do_add_parser(subparsers)
    args = parser.parse_args([])
    assert args.cmd == "live"
    assert args.elf_file is None
    assert args.runner is None
    assert args.runner_target is None


def test_parser_dump_command_parses_input_and_json():
    cmd = _make_cmd()
    subparsers = argparse.ArgumentParser().add_subparsers()
    parser = cmd.do_add_parser(subparsers)
    args = parser.parse_args(["dump", "-i", "x.gz", "--json"])
    assert args.cmd == "dump"
    assert args.input == "x.gz"
    assert args.json is True
    # Auto-filled fields stay None at parse time:
    assert args.elf_file is None
    assert args.runner is None
