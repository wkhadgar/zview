# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""Parser for Zephyr's ``build/zephyr/runners.yaml`` west-build artifact."""

from pathlib import Path

from yaml import safe_load


class RunnerConfig:
    def __init__(self, runners_yaml_path):
        self.path = Path(runners_yaml_path)
        if not self.path.exists():
            raise FileNotFoundError(f"runners.yaml not found at {runners_yaml_path}")

        with open(self.path) as f:
            self.data = safe_load(f)

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
