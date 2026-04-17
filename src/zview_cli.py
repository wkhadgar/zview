import argparse
import curses
import json
import sys

from backend.base import AbstractScraper
from backend.gdb import GDBScraper
from backend.jlink import JLinkScraper
from backend.pyocd import PyOCDScraper
from backend.replay import ReplayScraper
from frontend.zview_tui import tui_run
from logging_setup import configure as configure_logging
from orchestrator import ZScraper
from snapshot import dump_single_frame, record_session, serialize_frame

AVAILABLE_RUNNERS = ("gdb", "jlink", "pyocd")


def _build_backend(args) -> AbstractScraper:
    if args.replay:
        return ReplayScraper(args.replay)

    scraper_map = {
        "gdb": GDBScraper,
        "jlink": JLinkScraper,
        "pyocd": PyOCDScraper,
    }
    scraper_cls = scraper_map.get(args.runner, PyOCDScraper)
    return scraper_cls(args.runner_target)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ZView - A real-time thread viewer for Zephyr RTOS."
    )
    parser.add_argument(
        "-e",
        "--elf-file",
        required=True,
        help="Path to the application's .elf firmware file.",
    )
    parser.add_argument(
        "-r",
        "--runner",
        choices=AVAILABLE_RUNNERS,
        default=None,
        help="Runner to start analysis with (required unless --replay is given).",
    )
    parser.add_argument(
        "-t",
        "--runner-target",
        default=None,
        help="Descriptor name for the target MCU on the chosen runner.",
    )
    parser.add_argument(
        "--period",
        default=0.10,
        type=float,
        help="Minimum period to update system information.",
    )
    parser.add_argument(
        "--replay",
        metavar="PATH",
        help="Replay a .ndjson.gz recording instead of attaching to a live target.",
    )
    parser.add_argument(
        "--snapshot",
        metavar="PATH",
        help="Record a live polling session to a .ndjson.gz file and exit.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        help="Snapshot upper bound in seconds (with --snapshot).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        help="Snapshot upper bound in data frames (with --snapshot).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Emit a single polling frame and exit (no TUI).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="With --once, emit the frame as JSON to stdout.",
    )

    args = parser.parse_args()

    if not args.replay:
        if not args.runner:
            parser.error("-r/--runner is required unless --replay is given.")
        if not args.runner_target:
            parser.error("-t/--runner-target is required unless --replay is given.")

    if args.snapshot and args.replay:
        parser.error("--snapshot and --replay are mutually exclusive.")
    if args.snapshot and args.once:
        parser.error("--snapshot and --once are mutually exclusive.")
    if args.snapshot and args.duration is None and args.frames is None:
        parser.error("--snapshot requires --duration or --frames.")
    if args.json and not args.once:
        parser.error("--json only applies to --once.")

    return args


def main() -> int:
    configure_logging()
    args = _parse_args()

    backend = _build_backend(args)

    if args.snapshot:
        captured = record_session(
            backend,
            args.elf_file,
            args.snapshot,
            duration=args.duration,
            frames=args.frames,
            period=args.period,
        )
        print(f"Recorded {captured} frames to {args.snapshot}")
        return 0

    if args.once:
        frame = dump_single_frame(backend, args.elf_file, period=args.period)
        if args.json:
            print(json.dumps(serialize_frame(frame)))
        else:
            for t in frame.get("threads", []):
                wm = t.runtime.stack_watermark if t.runtime else "-"
                print(f"{t.name:30s} stack={t.stack_size:>6}  watermark={wm}")
            for h in frame.get("heaps", []):
                print(
                    f"heap {h.name:20s} free={h.free_bytes} "
                    f"alloc={h.allocated_bytes} max={h.max_allocated_bytes}"
                )
        return 0

    with backend as meta_scraper:
        z_scraper = ZScraper(meta_scraper, args.elf_file)
        if not z_scraper.has_names:
            print("NO thread names available (CONFIG_THREAD_NAME=n)", file=sys.stderr)
        if not z_scraper.has_usage:
            print("NO cpu stats available (CONFIG_THREAD_RUNTIME_STATS=n)", file=sys.stderr)
        if not z_scraper.has_heaps:
            print("NO heap stats available (CONFIG_SYS_HEAP_RUNTIME_STATS=n)", file=sys.stderr)

        curses.wrapper(tui_run, z_scraper, args.period)

    return 0


if __name__ == "__main__":
    sys.exit(main())
