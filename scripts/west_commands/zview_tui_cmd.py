import argparse
import sys
from pathlib import Path

from west.commands import WestCommand

_zview_base = Path(__file__).parent.parent.parent
sys.path.append(str(_zview_base / "src"))

import zview_cli  # noqa: E402
from logging_setup import configure as configure_logging  # noqa: E402
from runner_config import RunnerConfig  # noqa: E402

_COMMANDS_NEEDING_RUNNER = ("live", "record")


def _command_wants_runner(args: argparse.Namespace) -> bool:
    if args.cmd in _COMMANDS_NEEDING_RUNNER:
        return True
    if args.cmd == "dump":
        return getattr(args, "input", None) is None
    return False


def _build_subcommand_epilogs() -> dict[str, str]:
    elf_line = "  -e/--elf-file       from build/zephyr/zephyr.elf"
    runner_line = "  -r/--runner         from build/zephyr/runners.yaml"
    target_line = "  -t/--runner-target  from build/zephyr/runners.yaml"

    live_record = "Auto-detected when omitted:\n" + "\n".join((elf_line, runner_line, target_line))
    replay = "Auto-detected when omitted:\n" + elf_line
    dump = (
        "Auto-detected when omitted:\n"
        + elf_line
        + "\n"
        + runner_line
        + " (when -i is not given)\n"
        + target_line
        + " (when -i is not given)"
    )
    return {"live": live_record, "record": live_record, "replay": replay, "dump": dump}


class ZViewCommand(WestCommand):
    def __init__(self):
        super().__init__(
            'zview',
            'Zephyr RTOS system-wide runtime visualizer via SWD probe',
            'Take a broader look on your Zephyr application with a non-heavy, small footprint, '
            'Kconfig-only thread stats analyser. '
            'Run `west zview --help` for the full command list.',
        )

    def do_add_parser(self, parser_adder):
        parser = parser_adder.add_parser(
            self.name,
            help=self.help,
            description=self.description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")
        zview_cli.register_commands(
            sub,
            auto_filled=frozenset({"elf", "target"}),
            subcommand_epilogs=_build_subcommand_epilogs(),
        )
        parser.set_defaults(
            cmd="live",
            elf_file=None,
            runner=None,
            runner_target=None,
            period=0.10,
        )
        return parser

    def do_run(self, args, unknown):
        del unknown
        configure_logging()

        self._fill_elf(args)
        if _command_wants_runner(args):
            self._fill_runner(args)

        zview_cli.validate_args(self.parser, args)
        sys.exit(zview_cli.dispatch(args))

    def _fill_elf(self, args: argparse.Namespace) -> None:
        if args.elf_file is not None:
            return
        elf_path = Path("build").resolve() / "zephyr" / "zephyr.elf"
        if not elf_path.exists():
            self.die(
                f"Could not find firmware at '{elf_path}'. Please specify '-e' or run west build."
            )
        args.elf_file = str(elf_path)

    def _fill_runner(self, args: argparse.Namespace) -> None:
        if args.runner is not None and args.runner_target is not None:
            return

        runners_yaml_path = Path("build").resolve() / "zephyr" / "runners.yaml"
        try:
            cfg = RunnerConfig(runners_yaml_path)
        except Exception as e:
            self.err(e)
            self.die(
                "Failed to parse runners.yaml. Please specify runner ('-r') and target ('-t')."
            )

        runner, target_mcu = cfg.get_config(preferred_runner=args.runner)

        if runner not in zview_cli.AVAILABLE_RUNNERS:
            self.wrn(
                f"'{runner}' is not a valid runner for ZView. Try explicitly passing one of "
                f"the following: {', '.join(zview_cli.AVAILABLE_RUNNERS)}"
            )
            self.wrn("Trying to continue with PyOCD as runner...")
            runner, target_mcu = cfg.get_config(preferred_runner="pyocd")

        if args.runner is None:
            args.runner = runner
        if args.runner_target is None and target_mcu is not None:
            args.runner_target = str(target_mcu)
