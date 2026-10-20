# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Parsers for Zephyr's ``domains.yaml`` and ``runners.yaml`` west-build artifacts."""

import logging
import os
from pathlib import Path

from pyocd.utility.cmdline import convert_frequency
from yaml import safe_load

logger = logging.getLogger("zview.build")

DOMAINS_YAML_NAME = "domains.yaml"
RUNNERS_YAML_NAME = "runners.yaml"
IMAGE_ZEPHYR_DIR = "zephyr"
NON_SYSBUILD_DOMAIN = "app"
DEFAULT_ELF_NAME = "zephyr.elf"


class BuildDirError(Exception):
    """A build directory lacks an artifact ZView needs."""


class RunnerConfig:
    def __init__(self, runners_yaml_path):
        self.path = Path(runners_yaml_path)
        if not self.path.exists():
            raise FileNotFoundError(f"runners.yaml not found at {runners_yaml_path}")

        with open(self.path) as f:
            self.data = safe_load(f)

    @property
    def elf_file(self) -> Path | None:
        """The ELF recorded under ``config``, relative to this file, or ``None``."""
        elf_name = self.data.get("config", {}).get("elf_file")
        if elf_name is None:
            return None

        return self.path.parent / elf_name

    def get_config(self, preferred_runner=None):
        """
        Returns a tuple of (runner_name, mcu_target)
        """
        runner = preferred_runner or self.data.get("flash-runner") or "jlink"

        args = self.data.get("args", {}).get(runner, [])

        mcu = None
        if runner == "jlink":
            mcu = self._find_arg(args, "--device")
        elif runner == "pyocd":
            mcu = self._find_arg(args, "--target")

        return runner, mcu

    def frequency(self, runner: str) -> int | None:
        """The SWD clock the board passes ``runner``, in Hz, or ``None`` for the probe's own."""
        args = self.data.get("args", {}).get(runner, [])
        if runner == "jlink":
            # In kHz, or "auto" and "adaptive", which leave it to the J-Link.
            speed = self._find_arg(args, "--speed")
            return int(speed) * 1000 if speed and speed.isdigit() else None

        value = self._find_arg(args, "--frequency")
        return None if value is None else convert_frequency(value)

    def _find_arg(self, args_list, prefix):
        """Helper to find an argument starting with a specific prefix"""
        for arg in args_list:
            if arg.startswith(prefix):
                if "=" in arg:
                    return arg.split("=")[1]
                idx = args_list.index(arg)
                if idx + 1 < len(args_list):
                    return args_list[idx + 1]
        return None


def image_dir(build_dir, domain: str | None = None) -> Path:
    """
    Return the build directory of ``domain`` under ``build_dir``.
    ``domain`` defaults to the one ``domains.yaml`` marks as default; a build without
    ``domains.yaml`` holds the single domain ``app``.
    """
    build_dir = Path(os.path.abspath(build_dir))
    domains_yaml = build_dir / DOMAINS_YAML_NAME

    if domains_yaml.is_file():
        with open(domains_yaml) as f:
            data = safe_load(f)

        try:
            default = data["default"]
            recorded_build_dir = data["build_dir"]
            # Paths recorded before the build directory was moved are rebased onto it.
            domains = {
                d["name"]: Path(d["build_dir"].replace(recorded_build_dir, str(build_dir), 1))
                for d in data["domains"]
            }
        except KeyError as e:
            raise BuildDirError(f"Malformed '{domains_yaml}': missing {e}.") from e
    else:
        default = NON_SYSBUILD_DOMAIN
        domains = {NON_SYSBUILD_DOMAIN: build_dir}

    name = domain or default
    if name not in domains:
        raise BuildDirError(
            f"Domain '{name}' not found in '{build_dir}', valid domains are: {', '.join(domains)}."
        )

    logger.debug("Domain '%s' of '%s' is built in '%s'.", name, build_dir, domains[name])

    return domains[name]


def runners_yaml_path(image: Path) -> Path:
    """Path of the ``runners.yaml`` that ``west build`` writes for ``image``."""
    return image / IMAGE_ZEPHYR_DIR / RUNNERS_YAML_NAME


def find_elf(image: Path) -> Path:
    """Return the ELF that ``runners.yaml`` records for ``image``, else ``zephyr/zephyr.elf``."""
    runners_yaml = runners_yaml_path(image)

    elf = None
    if runners_yaml.is_file():
        elf = RunnerConfig(runners_yaml).elf_file
    if elf is None:
        elf = image / IMAGE_ZEPHYR_DIR / DEFAULT_ELF_NAME

    if not elf.is_file():
        raise BuildDirError(
            f"Could not find firmware at '{elf}'. Please specify '-e' or run west build."
        )

    logger.debug("Firmware of '%s' is '%s'.", image, elf)

    return elf
