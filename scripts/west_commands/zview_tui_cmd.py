import argparse
import os
import sys
from pathlib import Path

from west.commands import WestCommand

_zview_base = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_zview_base / "src"))

from zview import cli  # noqa: E402
from zview.logging_setup import configure as configure_logging  # noqa: E402
from zview.runner_config import BuildDirError  # noqa: E402


def _build_subcommand_epilogs() -> dict[str, str]:
    build_dir_line = "  -d/--build-dir      build.dir-fmt, the current directory, or ./build"
    elf_line = "  -e/--elf-file       from the build directory's runners.yaml"
    runner_line = "  -r/--runner         from the build directory's runners.yaml"
    target_line = "  -t/--runner-target  from the build directory's runners.yaml"

    live_record = "Auto-detected when omitted:\n" + "\n".join(
        (build_dir_line, elf_line, runner_line, target_line)
    )
    replay = "Auto-detected when omitted:\n" + "\n".join((build_dir_line, elf_line))
    dump = (
        "Auto-detected when omitted:\n"
        + build_dir_line
        + "\n"
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
        cli.register_commands(sub, subcommand_epilogs=_build_subcommand_epilogs())
        parser.set_defaults(
            cmd="live",
            elf_file=None,
            build_dir=None,
            domain=None,
            runner=None,
            runner_target=None,
            frequency=None,
            period=0.10,
        )
        return parser

    def do_run(self, args, unknown):
        del unknown
        configure_logging()

        if cli.needs_build_dir(args) and args.build_dir is None:
            args.build_dir = self._find_build_dir(args)

        try:
            cli.fill_from_build_dir(args, warn=self.wrn)
        except BuildDirError as e:
            self.die(str(e))

        cli.validate_args(self.parser, args)

        try:
            rc = cli.dispatch(args)
        except Exception as error:
            self.die(cli.failure_report(error))

        sys.exit(rc)

    def _find_build_dir(self, args: argparse.Namespace) -> str:
        zephyr_base = os.environ.get("ZEPHYR_BASE")
        if zephyr_base is None:
            self.die("ZEPHYR_BASE is not set. Please specify '-d', or '-e' with '-r' and '-t'.")

        sys.path.append(str(Path(zephyr_base) / "scripts" / "west_commands"))
        from run_common import get_build_dir

        return get_build_dir(args)
