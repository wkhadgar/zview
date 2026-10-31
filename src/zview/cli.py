import argparse
import curses
import json
import logging
import sys
from collections.abc import Callable

from pyocd.utility.cmdline import convert_frequency

from zview.frontend.zview_tui import tui_run
from zview.kernel.threads import USAGE_WINDOW_MAX
from zview.kinds import KINDS
from zview.logging_setup import configure as configure_logging
from zview.runner_config import BuildDirError, find_elf, image_dir
from zview.selection import Selection
from zview.snapshot import dump_single_frame, record_session, serialize_frame
from zview.target import AVAILABLE_RUNNERS, CLOCKED_RUNNERS, Target

logger = logging.getLogger("zview.cli")

KNOWN_COMMANDS = ("live", "record", "replay", "dump")
PROBE_COMMANDS = ("live", "record")
RUNNERS_NEEDING_TARGET = ("gdb", "jlink")


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


def _add_usage_window(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--usage-window",
        type=int,
        default=1,
        metavar="POLLS",
        help=f"Sum the CPU and load shares over the last POLLS polls, up to {USAGE_WINDOW_MAX} "
        "(default: 1, each poll alone).",
    )


def _add_selection(parser: argparse.ArgumentParser) -> None:
    every_row = Selection().pattern_line()
    parser.add_argument(
        "--match",
        default=every_row,
        metavar="LINE",
        help="Read only the rows this line matches, as / edits it, such as '*, heaps:!*' for "
        f"all but the heaps (default: {every_row}).",
    )


def _selection(args: argparse.Namespace) -> Selection:
    return Selection().with_pattern_line(args.match)


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
    _add_usage_window(p_live)
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
    _add_usage_window(p_replay)
    p_replay.add_argument(
        "--match",
        metavar="LINE",
        help="Show only the rows this line matches, as / edits it (default: the recording's line).",
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
        default=2,
        help=(
            "Which polling frame to dump (1-indexed). Default: 2, the first with CPU shares, "
            "as frame 1 has no earlier frame to count cycles from."
        ),
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

    if not _wants_runner(args):
        if args.elf_file is None:
            args.elf_file = str(find_elf(image_dir(args.build_dir, args.domain)))
        return

    probe = Target.from_build_dir(
        args.build_dir,
        args.domain,
        elf=args.elf_file,
        runner=args.runner,
        target=args.runner_target,
        frequency=args.frequency,
        warn=warn,
    )
    args.elf_file, args.runner = probe.elf, probe.runner
    args.runner_target, args.frequency = probe.target, probe.frequency


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

    # A replay has no runner, and no clock either.
    if getattr(args, "frequency", None) is not None and args.runner not in CLOCKED_RUNNERS:
        parser.error(f"{args.cmd}: --frequency sets the pyocd and jlink runners' clock only.")

    if getattr(args, "period", 0) < 0:
        parser.error(f"{args.cmd}: --period must be >= 0.")

    if getattr(args, "frames", None) is not None and args.frames < 1:
        parser.error(f"{args.cmd}: --frames must be >= 1.")

    if not 1 <= getattr(args, "usage_window", 1) <= USAGE_WINDOW_MAX:
        parser.error(f"{args.cmd}: --usage-window must be from 1 to {USAGE_WINDOW_MAX}.")

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


def _probe(args) -> Target:
    """The probe ``args`` name, with the ELF it is checked against."""
    return Target(
        args.elf_file, runner=args.runner, target=args.runner_target, frequency=args.frequency
    )


def _do_live(args) -> int:
    with _probe(args).open(_selection(args)) as core:
        core.usage_window = args.usage_window
        for notice in core.build.missing:
            print(notice, file=sys.stderr)
        curses.wrapper(tui_run, core, args.period)
    return 0


def _do_record(args) -> int:
    backend = _probe(args).backend()
    captured = record_session(
        backend,
        args.elf_file,
        args.output,
        duration=args.duration,
        frames=args.frames,
        period=args.period,
        selection=_selection(args),
        source={
            "runner": args.runner,
            "runner_target": args.runner_target,
            "frequency": args.frequency,
        },
    )
    print(f"Recorded {captured} frames to {args.output}")
    return 0


def _do_replay(args) -> int:
    replay = Target(args.elf_file, recording=args.input, is_paced=not args.no_pacing)
    # Without --match, the replay shows the recording's own line.
    shown = None if args.match is None else _selection(args)
    with replay.open(shown) as core:
        core.usage_window = args.usage_window
        # Pacing comes from the recording; the polling-thread sleep is just a floor.
        curses.wrapper(tui_run, core, 0.05)
    return 0


def _do_dump(args) -> int:
    if args.input:
        target = Target(args.elf_file, recording=args.input, is_paced=False)
    else:
        target = _probe(args)

    frame = dump_single_frame(
        target,
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
