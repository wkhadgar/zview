# Copyright (c) 2025 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from backend.z_scraper import RunnerConfig


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
