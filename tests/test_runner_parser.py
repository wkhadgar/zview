# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from zview.runner_config import (
    NON_SYSBUILD_DOMAIN,
    BuildDirError,
    RunnerConfig,
    find_elf,
    image_dir,
)


def _write_image(image: Path, elf_name: str = "zephyr.elf", runners_yaml: dict | None = None):
    """Populate ``image/zephyr`` with an empty ELF and, optionally, a runners.yaml."""
    zephyr_dir = image / "zephyr"
    zephyr_dir.mkdir(parents=True)
    (zephyr_dir / elf_name).write_bytes(b"")
    if runners_yaml is not None:
        (zephyr_dir / "runners.yaml").write_text(yaml.dump(runners_yaml))


def _write_sysbuild(build: Path, recorded_build_dir: Path) -> None:
    """Sysbuild layout with images ``hello_world`` (default) and ``mcuboot``."""
    build.mkdir(parents=True, exist_ok=True)
    domains = {
        "default": "hello_world",
        "build_dir": str(recorded_build_dir),
        "domains": [
            {"name": "hello_world", "build_dir": str(recorded_build_dir / "hello_world")},
            {"name": "mcuboot", "build_dir": str(recorded_build_dir / "mcuboot")},
        ],
        "flash_order": ["mcuboot", "hello_world"],
    }
    (build / "domains.yaml").write_text(yaml.dump(domains))
    _write_image(build / "hello_world")
    _write_image(build / "mcuboot")


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        RunnerConfig("non_existent_file.yaml")


def test_get_config_jlink_defaults(tmp_path: Path):
    f = tmp_path / "runners.yaml"
    data = {
        "flash-runner": "jlink",
        "args": {"jlink": ["--device=NRF52840", "--speed", "4000"]},
    }
    f.write_text(yaml.dump(data))

    parser = RunnerConfig(f)
    runner, mcu = parser.get_config()

    assert runner == "jlink"
    assert mcu == "NRF52840"


def test_get_config_pyocd_space_args(tmp_path: Path):
    # Test handling of space separated args (e.g. --target STM32)
    f = tmp_path / "runners.yaml"
    data = {"args": {"pyocd": ["--target", "STM32F4", "--pack", "somepack"]}}
    f.write_text(yaml.dump(data))

    # Force preferred runner
    parser = RunnerConfig(f)
    runner, mcu = parser.get_config(preferred_runner="pyocd")

    assert runner == "pyocd"
    assert mcu == "STM32F4"


def test_arg_not_found(tmp_path: Path):
    f = tmp_path / "runners.yaml"
    data = {"flash-runner": "jlink", "args": {"jlink": []}}
    f.write_text(yaml.dump(data))

    parser = RunnerConfig(f)
    runner, mcu = parser.get_config()

    assert runner == "jlink"
    assert mcu is None


def test_image_dir_without_domains_yaml_is_the_build_dir(tmp_path: Path):
    assert image_dir(tmp_path) == tmp_path
    assert image_dir(tmp_path, NON_SYSBUILD_DOMAIN) == tmp_path


def test_image_dir_follows_the_sysbuild_default_domain(tmp_path: Path):
    _write_sysbuild(tmp_path, tmp_path)

    assert image_dir(tmp_path) == tmp_path / "hello_world"


def test_image_dir_picks_the_requested_domain(tmp_path: Path):
    _write_sysbuild(tmp_path, tmp_path)

    assert image_dir(tmp_path, "mcuboot") == tmp_path / "mcuboot"


def test_image_dir_lists_valid_domains_on_unknown_name(tmp_path: Path):
    _write_sysbuild(tmp_path, tmp_path)

    with pytest.raises(BuildDirError, match="valid domains are: hello_world, mcuboot"):
        image_dir(tmp_path, "netcore")


def test_image_dir_rebases_a_moved_build(tmp_path: Path):
    moved = tmp_path / "moved"
    _write_sysbuild(moved, tmp_path / "original")

    assert image_dir(moved) == moved / "hello_world"


def test_image_dir_rejects_a_malformed_domains_yaml(tmp_path: Path):
    (tmp_path / "domains.yaml").write_text(yaml.dump({"build_dir": str(tmp_path), "domains": []}))

    with pytest.raises(BuildDirError, match="Malformed"):
        image_dir(tmp_path)


def test_find_elf_uses_the_runners_yaml_name(tmp_path: Path):
    _write_image(tmp_path, "app.elf", {"config": {"elf_file": "app.elf"}})

    assert find_elf(tmp_path) == tmp_path / "zephyr" / "app.elf"


def test_find_elf_defaults_to_zephyr_elf_without_runners_yaml(tmp_path: Path):
    _write_image(tmp_path)

    assert find_elf(tmp_path) == tmp_path / "zephyr" / "zephyr.elf"


def test_find_elf_defaults_to_zephyr_elf_when_runners_yaml_omits_it(tmp_path: Path):
    _write_image(tmp_path, runners_yaml={"flash-runner": "jlink"})

    assert find_elf(tmp_path) == tmp_path / "zephyr" / "zephyr.elf"
