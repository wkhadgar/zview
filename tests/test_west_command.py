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

from zview_tui_cmd import ZViewCommand  # noqa: E402

from zview import cli  # noqa: E402

_RUNNERS_YAML = (
    "flash-runner: jlink\n"
    "config:\n"
    "  elf_file: zephyr.elf\n"
    "args:\n"
    "  jlink:\n"
    "    - --device=nRF5340_xxAA\n"
)


def _make_cmd() -> ZViewCommand:
    """``ZViewCommand`` with a stub ``config`` so ``die``/``err``/``wrn`` work."""
    cmd = ZViewCommand()
    cmd.config = MagicMock()
    cmd.config.getboolean.return_value = False
    cmd.parser = cmd.do_add_parser(argparse.ArgumentParser().add_subparsers())
    return cmd


def _run(monkeypatch, argv: list[str]) -> tuple[int, argparse.Namespace | None]:
    """Run ``west zview <argv>``; return (exit code, namespace handed to dispatch)."""
    dispatched = []

    def fake_dispatch(args: argparse.Namespace) -> int:
        dispatched.append(args)
        return 0

    monkeypatch.setattr(cli, "dispatch", fake_dispatch)
    cmd = _make_cmd()
    args = cmd.parser.parse_args(argv)

    with pytest.raises(SystemExit) as exc:
        cmd.do_run(args, [])

    return exc.value.code, (dispatched[0] if dispatched else None)


def _write_build(image: Path, runners_yaml: str | None = _RUNNERS_YAML) -> Path:
    """Populate ``image/zephyr`` with an empty ELF and a runners.yaml; return the ELF."""
    zephyr_dir = image / "zephyr"
    zephyr_dir.mkdir(parents=True)
    elf = zephyr_dir / "zephyr.elf"
    elf.write_bytes(b"")
    if runners_yaml is not None:
        (zephyr_dir / "runners.yaml").write_text(runners_yaml)
    return elf


@pytest.fixture
def fake_zephyr(tmp_path, monkeypatch):
    """``ZEPHYR_BASE`` whose ``run_common.get_build_dir`` answers ``tmp_path / "build"``."""
    build = tmp_path / "build"
    west_commands = tmp_path / "zephyr" / "scripts" / "west_commands"
    west_commands.mkdir(parents=True)
    (west_commands / "run_common.py").write_text(
        f"def get_build_dir(args):\n    return {str(build)!r}\n"
    )
    monkeypatch.setenv("ZEPHYR_BASE", str(tmp_path / "zephyr"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "run_common", raising=False)
    return build


@pytest.fixture
def no_zephyr(monkeypatch):
    monkeypatch.delenv("ZEPHYR_BASE", raising=False)


def test_bare_invocation_takes_everything_from_the_default_build_dir(fake_zephyr, monkeypatch):
    elf = _write_build(fake_zephyr)

    rc, args = _run(monkeypatch, [])

    assert rc == 0
    assert args.build_dir == str(fake_zephyr)
    assert args.elf_file == str(elf)
    assert args.runner == "jlink"
    assert args.runner_target == "nRF5340_xxAA"


def test_explicit_elf_still_takes_the_runner_from_the_build_dir(fake_zephyr, monkeypatch):
    _write_build(fake_zephyr)

    rc, args = _run(monkeypatch, ["live", "-e", "/custom/path.elf"])

    assert rc == 0
    assert args.elf_file == "/custom/path.elf"
    assert args.runner == "jlink"
    assert args.runner_target == "nRF5340_xxAA"


def test_explicit_build_dir_needs_no_zephyr_base(tmp_path, monkeypatch, no_zephyr):
    elf = _write_build(tmp_path / "out")

    rc, args = _run(monkeypatch, ["live", "-d", str(tmp_path / "out")])

    assert rc == 0
    assert args.elf_file == str(elf)


def test_fully_explicit_run_skips_the_build_dir(monkeypatch, no_zephyr):
    rc, args = _run(
        monkeypatch, ["live", "-e", "/custom/path.elf", "-r", "pyocd", "-t", "stm32h753zitx"]
    )

    assert rc == 0
    assert args.build_dir is None
    assert args.runner == "pyocd"
    assert args.runner_target == "stm32h753zitx"


def test_replay_takes_only_the_elf(tmp_path, monkeypatch, no_zephyr):
    elf = _write_build(tmp_path, runners_yaml=None)

    rc, args = _run(monkeypatch, ["replay", "-d", str(tmp_path), "-i", "x.ndjson.gz"])

    assert rc == 0
    assert args.elf_file == str(elf)
    assert args.runner is None


def test_domain_selects_the_sysbuild_image(tmp_path, monkeypatch, no_zephyr):
    (tmp_path / "domains.yaml").write_text(
        f"default: app\n"
        f"build_dir: {tmp_path}\n"
        f"domains:\n"
        f"  - name: app\n"
        f"    build_dir: {tmp_path / 'app'}\n"
        f"  - name: remote\n"
        f"    build_dir: {tmp_path / 'remote'}\n"
    )
    _write_build(tmp_path / "app")
    remote_elf = _write_build(tmp_path / "remote")

    rc, args = _run(monkeypatch, ["live", "-d", str(tmp_path), "--domain", "remote"])

    assert rc == 0
    assert args.elf_file == str(remote_elf)


def test_missing_zephyr_base_is_fatal(monkeypatch, capsys, no_zephyr):
    rc, args = _run(monkeypatch, [])

    assert rc != 0
    assert args is None
    assert "ZEPHYR_BASE" in capsys.readouterr().err


def test_missing_firmware_is_fatal(tmp_path, monkeypatch, capsys, no_zephyr):
    rc, args = _run(monkeypatch, ["live", "-d", str(tmp_path)])

    assert rc != 0
    assert args is None
    assert "Could not find firmware" in capsys.readouterr().err


def test_a_failing_command_dies_with_one_line(monkeypatch, capsys, tmp_path, no_zephyr):
    log_path = tmp_path / "zview.log"

    def failing_dispatch(args: argparse.Namespace) -> int:
        raise ConnectionError("Could not connect to localhost:1.")

    monkeypatch.setattr(cli, "configure_logging", lambda: log_path)
    monkeypatch.setattr(cli, "logger", MagicMock())
    monkeypatch.setattr(cli, "dispatch", failing_dispatch)
    cmd = _make_cmd()
    args = cmd.parser.parse_args(["live", "-e", "/x.elf", "-r", "gdb", "-t", "localhost:1"])

    with pytest.raises(SystemExit) as exc:
        cmd.do_run(args, [])

    err = capsys.readouterr().err
    assert exc.value.code == 1
    assert f"Could not connect to localhost:1. (traceback in {log_path})" in err
    assert "Traceback" not in err


def test_parser_bare_invocation_defaults_to_live():
    cmd = _make_cmd()
    args = cmd.parser.parse_args([])
    assert args.cmd == "live"
    assert args.elf_file is None
    assert args.build_dir is None
    assert args.domain is None
    assert args.runner is None
    assert args.runner_target is None


def test_parser_dump_command_parses_input_and_json():
    cmd = _make_cmd()
    args = cmd.parser.parse_args(["dump", "-i", "x.gz", "--json"])
    assert args.cmd == "dump"
    assert args.input == "x.gz"
    assert args.json is True
    # Auto-filled fields stay None at parse time:
    assert args.elf_file is None
    assert args.runner is None
