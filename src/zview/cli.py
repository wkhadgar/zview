import argparse
import curses
import json
import logging
import sys
from collections.abc import Callable

from pyocd.utility.cmdline import convert_frequency
from yaml import YAMLError

from zview.backend.gdb import GDBScraper
from zview.backend.image import verify_target_image
from zview.backend.jlink import JLinkScraper
from zview.backend.nrfutil import NrfutilScraper
from zview.backend.pyocd import PyOCDScraper
from zview.backend.replay import ReplayScraper
from zview.frontend.zview_tui import tui_run
from zview.kinds import KINDS
from zview.logging_setup import configure as configure_logging
from zview.orchestrator import ZScraper
from zview.runner_config import BuildDirError, RunnerConfig, find_elf, image_dir, runners_yaml_path
from zview.selection import Selection, parse_pattern
from zview.snapshot import dump_single_frame, record_session, serialize_frame

logger = logging.getLogger("zview.cli")

AVAILABLE_RUNNERS = ("gdb", "jlink", "nrfutil", "pyocd")
KNOWN_COMMANDS = ("live", "record", "replay", "dump")
PROBE_COMMANDS = ("live", "record")
RUNNERS_NEEDING_TARGET = ("gdb", "jlink")
# Runners whose probe clock --frequency sets.
CLOCKED_RUNNERS = ("pyocd", "jlink")


def _normalize_argv(argv: list[str]) -> list[str]:
    """Prepend ``live`` when ``argv`` does not start with a known command."""
    if not argv:
        return argv
    first = argv[0]
    if first in KNOWN_COMMANDS or first in ("-h", "--help"):
        return argv
    return ["live", *argv]


def _add_elf(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-e",
        "--elf-file",
        default=None,
        help="Path to the application's .elf firmware file.",
    )
    parser.add_argument(
        "-d",
        "--build-dir",
        metavar="DIR",
        default=None,
        help="Application build directory to read unset arguments from.",
    )
    parser.add_argument(
        "--domain",
        default=None,
        help="Sysbuild domain to view (default: the build's default domain).",
    )


def _add_live_target(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-r",
        "--runner",
        choices=AVAILABLE_RUNNERS,
        default=None,
        help="Debug runner to attach to the live target.",
    )
    parser.add_argument(
        "-t",
        "--runner-target",
        default=None,
        help="MCU descriptor for the chosen runner.",
    )
    parser.add_argument(
        "--frequency",
        type=convert_frequency,
        default=None,
        metavar="HZ",
        help=(
            "SWD clock for the pyocd and jlink runners, such as 4000000 or 10M "
            "(default: 4M on pyocd, the J-Link's own on jlink)."
        ),
    )


def _add_period(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--period",
        type=float,
        default=0.10,
        help="Minimum polling period in seconds (default: 0.10).",
    )


def _groups(value: str) -> frozenset[str]:
    """``--only``'s comma-separated groups. The session checks them against its own."""
    return frozenset(group.strip() for group in value.split(",") if group.strip())


def _add_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--only",
        type=_groups,
        default=None,
        metavar="GROUPS",
        help="Read only these groups, comma-separated, such as threads,mutexes.",
    )
    parser.add_argument(
        "--include",
        action="append",
        type=parse_pattern,
        default=[],
        metavar="PATTERN",
        help="Read only rows whose name matches this glob, or GROUP:GLOB within one group. "
        "Repeatable.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        type=parse_pattern,
        default=[],
        metavar="PATTERN",
        help="Skip rows whose name matches this glob, or GROUP:GLOB within one group. Repeatable.",
    )


def _selection(args: argparse.Namespace) -> Selection:
    return Selection(groups=args.only, include=tuple(args.include), exclude=tuple(args.exclude))


def register_commands(
    sub: argparse._SubParsersAction,
    *,
    subcommand_epilogs: dict[str, str] | None = None,
) -> None:
    """
    Add the live/record/replay/dump subparsers to ``sub``.
    ``subcommand_epilogs`` maps command name -> epilog text.
    """
    epilogs = subcommand_epilogs or {}

    def _sub(name: str, summary: str) -> argparse.ArgumentParser:
        return sub.add_parser(
            name,
            help=summary,
            epilog=epilogs.get(name),
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

    p_live = _sub("live", "Attach to a probe and render the TUI (default).")
    _add_elf(p_live)
    _add_live_target(p_live)
    _add_period(p_live)
    _add_selection(p_live)

    p_record = _sub("record", "Capture a live session to a .ndjson.gz recording file.")
    _add_elf(p_record)
    _add_live_target(p_record)
    p_record.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="FILE",
        help="Recording target path (.ndjson.gz).",
    )
    bound = p_record.add_mutually_exclusive_group(required=True)
    bound.add_argument("--duration", type=float, help="Recording bound, in seconds.")
    bound.add_argument("--frames", type=int, help="Recording bound, in data frames.")
    _add_period(p_record)
    _add_selection(p_record)

    p_replay = _sub("replay", "Render the TUI from a recording file.")
    _add_elf(p_replay)
    p_replay.add_argument(
        "-i",
        "--input",
        required=True,
        metavar="FILE",
        help="Recording source path (.ndjson.gz).",
    )
    p_replay.add_argument(
        "--no-pacing",
        action="store_true",
        help="Drain the recording as fast as possible instead of honoring its wall-clock cadence.",
    )

    p_dump = _sub("dump", "Capture a single polling frame and exit (no TUI).")
    _add_elf(p_dump)
    _add_live_target(p_dump)
    p_dump.add_argument(
        "-i",
        "--input",
        metavar="FILE",
        help="Replay source path (.ndjson.gz). Mutually exclusive with --runner/--runner-target.",
    )
    p_dump.add_argument(
        "--frame",
        type=int,
        default=1,
        help="Which polling frame to dump (1-indexed). Default: 1.",
    )
    p_dump.add_argument(
        "--json",
        action="store_true",
        help="Emit the frame as JSON on stdout.",
    )
    _add_period(p_dump)
    _add_selection(p_dump)


def _wants_runner(args: argparse.Namespace) -> bool:
    """True when ``args`` attaches to a probe."""
    if args.cmd == "dump":
        return args.input is None

    return args.cmd in PROBE_COMMANDS


def needs_build_dir(args: argparse.Namespace) -> bool:
    """True when ``args`` leaves the ELF, or the runner it attaches with, unset."""
    if args.elf_file is None:
        return True

    if not _wants_runner(args):
        return False

    return args.runner is None or args.runner_target is None


def fill_from_build_dir(args: argparse.Namespace, warn: Callable[[str], None]) -> None:
    """
    Fill the ELF, runner and target left unset in ``args`` from ``args.build_dir``.
    Raises ``BuildDirError`` when the build directory lacks an unset artifact.
    """
    if not needs_build_dir(args):
        return

    image = image_dir(args.build_dir, args.domain)

    if args.elf_file is None:
        args.elf_file = str(find_elf(image))

    if not _wants_runner(args):
        return

    if args.runner is not None and args.runner_target is not None:
        return

    yaml_path = runners_yaml_path(image)
    try:
        cfg = RunnerConfig(yaml_path)
    except (OSError, YAMLError) as e:
        raise BuildDirError(
            f"Failed to parse '{yaml_path}': {e}. Please specify runner ('-r') and target ('-t')."
        ) from e

    runner, target_mcu = cfg.get_config(preferred_runner=args.runner)
    if runner not in AVAILABLE_RUNNERS:
        warn(
            f"'{runner}' is not a valid runner for ZView. Try explicitly passing one of "
            f"the following: {', '.join(AVAILABLE_RUNNERS)}"
        )
        warn("Trying to continue with PyOCD as runner...")
        runner, target_mcu = cfg.get_config(preferred_runner="pyocd")

    if args.runner is None:
        args.runner = runner
    if args.runner_target is None and target_mcu is not None:
        args.runner_target = str(target_mcu)
    if args.frequency is None and args.runner in CLOCKED_RUNNERS:
        args.frequency = cfg.frequency(args.runner)

    logger.debug(
        "Runner '%s', target '%s', frequency %s from '%s'.",
        args.runner,
        args.runner_target,
        args.frequency,
        yaml_path,
    )


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Post-parse cross-flag validation. Calls ``parser.error()`` on failure."""
    if args.elf_file is None:
        parser.error(f"{args.cmd}: requires --elf-file or --build-dir.")

    probe_unset = args.cmd in PROBE_COMMANDS and (args.runner is None or args.runner_target is None)
    if probe_unset and args.build_dir is None:
        parser.error(f"{args.cmd}: requires --runner and --runner-target, or --build-dir.")

    target_needed = args.cmd in PROBE_COMMANDS and args.runner in RUNNERS_NEEDING_TARGET
    if target_needed and args.runner_target is None:
        parser.error(f"{args.cmd}: runner '{args.runner}' requires --runner-target.")

    if getattr(args, "frequency", None) is not None and args.runner not in (None, *CLOCKED_RUNNERS):
        parser.error(f"{args.cmd}: --frequency sets the pyocd and jlink runners' clock only.")

    if args.cmd == "dump":
        if args.input and (args.runner or args.runner_target):
            parser.error("dump: --input is mutually exclusive with --runner/--runner-target.")
        if not args.input and not (args.runner and args.runner_target):
            parser.error("dump: requires either --input FILE or both --runner and --runner-target.")
        if args.frame < 1:
            parser.error("dump: --frame must be >= 1.")


def _build_parser(
    prog: str = "zview",
    subcommand_epilogs: dict[str, str] | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="ZView - Zephyr RTOS runtime visualizer over SWD.",
        epilog=f"Run `{prog} <command> --help` for command-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND", required=True)
    register_commands(sub, subcommand_epilogs=subcommand_epilogs)
    return parser


def _parse_args(
    argv: list[str],
    prog: str = "zview",
    subcommand_epilogs: dict[str, str] | None = None,
) -> argparse.Namespace:
    parser = _build_parser(prog=prog, subcommand_epilogs=subcommand_epilogs)
    args = parser.parse_args(argv)

    if args.build_dir is not None:
        try:
            fill_from_build_dir(args, warn=_warn)
        except BuildDirError as e:
            parser.error(str(e))

    validate_args(parser, args)

    return args


def _warn(message: str) -> None:
    print(f"zview: warning: {message}", file=sys.stderr)


def _build_live_backend(args):
    scraper_map = {
        "gdb": GDBScraper,
        "jlink": JLinkScraper,
        "nrfutil": NrfutilScraper,
        "pyocd": PyOCDScraper,
    }
    cls = scraper_map.get(args.runner, PyOCDScraper)
    if cls in (PyOCDScraper, JLinkScraper):
        return cls(args.runner_target, args.frequency)
    return cls(args.runner_target)


def _do_live(args) -> int:
    with _build_live_backend(args) as backend:
        verify_target_image(backend, args.elf_file)
        scraper = ZScraper(backend, args.elf_file)
        selection = _selection(args)
        scraper.check_selection(selection)
        scraper.selection = selection
        for notice in scraper.absent_features():
            print(notice, file=sys.stderr)
        curses.wrapper(tui_run, scraper, args.period)
    return 0


def _do_record(args) -> int:
    backend = _build_live_backend(args)
    captured = record_session(
        backend,
        args.elf_file,
        args.output,
        duration=args.duration,
        frames=args.frames,
        period=args.period,
        selection=_selection(args),
    )
    print(f"Recorded {captured} frames to {args.output}")
    return 0


def _do_replay(args) -> int:
    backend = ReplayScraper(args.input, honor_timing=not args.no_pacing)
    with backend:
        scraper = ZScraper(backend, args.elf_file)
        # Pacing comes from the recording; the polling-thread sleep is just a floor.
        curses.wrapper(tui_run, scraper, 0.05)
    return 0


def _do_dump(args) -> int:
    if args.input:
        backend = ReplayScraper(args.input, honor_timing=False)
    else:
        backend = _build_live_backend(args)

    frame = dump_single_frame(
        backend,
        args.elf_file,
        period=args.period,
        frame=args.frame,
        selection=_selection(args),
    )
    if args.json:
        print(json.dumps(serialize_frame(frame)))
    else:
        for kind in KINDS:
            for info in frame.get(kind.key, []):
                print(kind.dump(info))
    return 0


_DISPATCH = {
    "live": _do_live,
    "record": _do_record,
    "replay": _do_replay,
    "dump": _do_dump,
}


def dispatch(args: argparse.Namespace) -> int:
    """Run the command named by ``args.cmd``."""
    return _DISPATCH[args.cmd](args)


def failure_report(error: Exception) -> str:
    """Log ``error`` with its traceback; return the one line reported to the user."""
    log_path = configure_logging()
    message = str(error) or type(error).__name__

    logger.error("Command failed: %s", message, exc_info=error)

    return f"{message} (traceback in {log_path})"


def main(
    argv: list[str] | None = None,
    prog: str = "zview",
    subcommand_epilogs: dict[str, str] | None = None,
) -> int:
    configure_logging()
    raw = sys.argv[1:] if argv is None else argv
    args = _parse_args(_normalize_argv(raw), prog=prog, subcommand_epilogs=subcommand_epilogs)

    try:
        return dispatch(args)
    except Exception as error:
        print(f"zview: {failure_report(error)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
