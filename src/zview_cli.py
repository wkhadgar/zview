import argparse
import curses
import json
import sys

from backend.gdb import GDBScraper
from backend.jlink import JLinkScraper
from backend.pyocd import PyOCDScraper
from backend.replay import ReplayScraper
from frontend.zview_tui import tui_run
from logging_setup import configure as configure_logging
from orchestrator import ZScraper
from snapshot import dump_single_frame, record_session, serialize_frame

AVAILABLE_RUNNERS = ("gdb", "jlink", "pyocd")
KNOWN_COMMANDS = ("live", "record", "replay", "dump")


def _normalize_argv(argv: list[str]) -> list[str]:
    """Splice the implicit ``live`` command when zview is invoked without one."""
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
        required=True,
        help="Path to the application's .elf firmware file.",
    )


def _add_live_target(parser: argparse.ArgumentParser, *, required: bool = True) -> None:
    parser.add_argument(
        "-r",
        "--runner",
        choices=AVAILABLE_RUNNERS,
        required=required,
        default=None,
        help="Debug runner to attach to the live target.",
    )
    parser.add_argument(
        "-t",
        "--runner-target",
        required=required,
        default=None,
        help="MCU descriptor for the chosen runner.",
    )


def _add_period(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--period",
        type=float,
        default=0.10,
        help="Minimum polling period in seconds (default: 0.10).",
    )


def _build_parser(
    prog: str = "zview",
    subcommand_epilogs: dict[str, str] | None = None,
) -> argparse.ArgumentParser:
    epilogs = subcommand_epilogs or {}
    parser = argparse.ArgumentParser(
        prog=prog,
        description="ZView - Zephyr RTOS runtime visualizer over SWD.",
        epilog=f"Run `{prog} <command> --help` for command-specific options.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND", required=True)

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
    p_record.add_argument(
        "--heap",
        metavar="NAME",
        help="Capture per-frame fragmentation reads for the named k_heap variable.",
    )
    _add_period(p_record)

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
    _add_live_target(p_dump, required=False)
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

    return parser


def _parse_args(
    argv: list[str],
    prog: str = "zview",
    subcommand_epilogs: dict[str, str] | None = None,
) -> argparse.Namespace:
    parser = _build_parser(prog=prog, subcommand_epilogs=subcommand_epilogs)
    args = parser.parse_args(argv)

    if args.cmd == "dump":
        if args.input and (args.runner or args.runner_target):
            parser.error("dump: --input is mutually exclusive with --runner/--runner-target.")
        if not args.input and not (args.runner and args.runner_target):
            parser.error("dump: requires either --input FILE or both --runner and --runner-target.")
        if args.frame < 1:
            parser.error("dump: --frame must be >= 1.")

    return args


def _build_live_backend(args):
    scraper_map = {
        "gdb": GDBScraper,
        "jlink": JLinkScraper,
        "pyocd": PyOCDScraper,
    }
    cls = scraper_map.get(args.runner, PyOCDScraper)
    return cls(args.runner_target)


def _do_live(args) -> int:
    with _build_live_backend(args) as backend:
        scraper = ZScraper(backend, args.elf_file)
        if not scraper.has_names:
            print("NO thread names available (CONFIG_THREAD_NAME=n)", file=sys.stderr)
        if not scraper.has_usage:
            print("NO cpu stats available (CONFIG_THREAD_RUNTIME_STATS=n)", file=sys.stderr)
        if not scraper.has_heaps:
            print("NO heap stats available (CONFIG_SYS_HEAP_RUNTIME_STATS=n)", file=sys.stderr)
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
        heap_detail=args.heap,
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
    )
    if args.json:
        print(json.dumps(serialize_frame(frame)))
    else:
        for t in frame.get("threads", []):
            wm = t.runtime.stack_watermark if t.runtime else "-"
            print(f"thread {t.name:30s} stack={t.stack_size:>6}  watermark={wm}")
        for h in frame.get("heaps", []):
            print(
                f"heap {h.name:20s} free={h.free_bytes} "
                f"alloc={h.allocated_bytes} max={h.max_allocated_bytes}"
            )
    return 0


_DISPATCH = {
    "live": _do_live,
    "record": _do_record,
    "replay": _do_replay,
    "dump": _do_dump,
}


def main(
    argv: list[str] | None = None,
    prog: str = "zview",
    subcommand_epilogs: dict[str, str] | None = None,
) -> int:
    configure_logging()
    raw = sys.argv[1:] if argv is None else argv
    args = _parse_args(_normalize_argv(raw), prog=prog, subcommand_epilogs=subcommand_epilogs)
    return _DISPATCH[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
