import sys
from pathlib import Path

from west import log
from west.commands import WestCommand

_zview_base = Path(__file__).parent.parent.parent
sys.path.append(str(_zview_base / "src"))

import zview_cli  # noqa: E402
from logging_setup import configure as configure_logging  # noqa: E402
from runner_config import RunnerConfig  # noqa: E402

_COMMANDS_NEEDING_RUNNER = ("live", "record")


def _has_flag(argv: list[str], short: str, long_: str) -> bool:
    """True if ``short``, ``long_``, or ``long_=...`` appears in ``argv``."""
    return any(a in (short, long_) or a.startswith(long_ + "=") for a in argv)


def _help_requested(argv: list[str]) -> bool:
    return any(a in ("-h", "--help") for a in argv)


def _insert_after_command(argv: list[str], extras: list[str]) -> list[str]:
    """
    Insert ``extras`` immediately after the first known command in ``argv``,
    or at the front when no command is present (which zview_cli will normalize
    into the implicit ``live`` command). Keeps the user-visible verb at the
    head of argv so downstream parsing isn't confused by injected flags.
    """
    for i, a in enumerate(argv):
        if a in zview_cli.KNOWN_COMMANDS:
            return [*argv[: i + 1], *extras, *argv[i + 1 :]]
    return [*extras, *argv]


def _command_wants_runner(cmd: str, argv: list[str]) -> bool:
    if cmd in _COMMANDS_NEEDING_RUNNER:
        return True
    if cmd == "dump":
        return not _has_flag(argv, "-i", "--input")
    return False


class ZViewCommand(WestCommand):
    def __init__(self):
        super().__init__(
            'zview',
            'Zephyr RTOS system-wide runtime visualizer via SWD probe',
            'Take a broader look on your Zephyr application with a non-heavy, small footprint, '
            'Kconfig-only thread stats analyser. '
            'Run `west zview --help` for the full command list.',
            # All arguments are forwarded to zview_cli; west must not reject them itself.
            accepts_unknown_args=True,
        )

    def do_add_parser(self, parser_adder):
        # Forward every argument to zview_cli unmodified; west auto-detection
        # of ELF and runner happens in do_run before delegation.
        parser = parser_adder.add_parser(
            self.name,
            help=self.help,
            description=self.description,
            add_help=False,
        )
        return parser

    def do_run(self, args, unknown):
        del args
        configure_logging()

        argv = list(unknown)
        argv = self._inject_elf_if_missing(argv)
        argv = self._inject_runner_if_missing(argv)

        elf_line = "  -e/--elf-file       from build/zephyr/zephyr.elf"
        runner_line = "  -r/--runner         from build/zephyr/runners.yaml"
        target_line = "  -t/--runner-target  from build/zephyr/runners.yaml"

        live_record_note = "Auto-detected when omitted:\n" + "\n".join(
            (elf_line, runner_line, target_line)
        )
        replay_note = "Auto-detected when omitted:\n" + elf_line
        dump_note = (
            "Auto-detected when omitted:\n"
            + elf_line
            + "\n"
            + runner_line
            + " (when -i is not given)\n"
            + target_line
            + " (when -i is not given)"
        )

        sys.exit(
            zview_cli.main(
                argv,
                prog="west zview",
                subcommand_epilogs={
                    "live": live_record_note,
                    "record": live_record_note,
                    "replay": replay_note,
                    "dump": dump_note,
                },
            )
        )

    def _inject_elf_if_missing(self, argv: list[str]) -> list[str]:
        """Prepend ``-e build/zephyr/zephyr.elf`` if no ELF flag is present."""
        if _has_flag(argv, "-e", "--elf-file") or _help_requested(argv):
            return argv

        elf_path = Path("build").resolve() / "zephyr" / "zephyr.elf"
        if not elf_path.exists():
            log.die(
                f"Could not find firmware at '{elf_path}'. Please specify '-e' or run west build."
            )
        return _insert_after_command(argv, ["-e", str(elf_path)])

    def _inject_runner_if_missing(self, argv: list[str]) -> list[str]:
        """Fill ``-r``/``-t`` from build/zephyr/runners.yaml when the command needs them."""
        if _help_requested(argv):
            return argv

        normalized = zview_cli._normalize_argv(argv)
        if not normalized:
            return argv
        cmd = normalized[0]
        if cmd not in zview_cli.KNOWN_COMMANDS:
            return argv
        if not _command_wants_runner(cmd, argv):
            return argv

        has_runner = _has_flag(argv, "-r", "--runner")
        has_target = _has_flag(argv, "-t", "--runner-target")
        if has_runner and has_target:
            return argv

        runners_yaml_path = Path("build").resolve() / "zephyr" / "runners.yaml"
        try:
            runner_config = RunnerConfig(runners_yaml_path)
        except Exception as e:
            log.err(e)
            log.die("Failed to parse runners.yaml. Please specify runner ('-r') and target ('-t').")

        preferred = None
        for i, a in enumerate(argv):
            if a in ("-r", "--runner"):
                preferred = argv[i + 1] if i + 1 < len(argv) else None
                break
            if a.startswith("--runner="):
                preferred = a.split("=", 1)[1]
                break

        runner, target_mcu = runner_config.get_config(preferred_runner=preferred)

        if runner not in zview_cli.AVAILABLE_RUNNERS:
            log.wrn(
                f"'{runner}' is not a valid runner for ZView. Try explicitly passing one of "
                f"the following: {', '.join(zview_cli.AVAILABLE_RUNNERS)}"
            )
            log.wrn("Trying to continue with PyOCD as runner...")
            runner, target_mcu = runner_config.get_config(preferred_runner="pyocd")

        injected: list[str] = []
        if not has_runner:
            injected += ["-r", runner]
        if not has_target and target_mcu:
            injected += ["-t", str(target_mcu)]
        return _insert_after_command(argv, injected)
