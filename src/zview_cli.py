import argparse
import curses

from backend.z_scraper import (
    JLinkScraper,
    PyOCDScraper,
    ZScraper,
)
from frontend.zview_tui import tui_run


def main():
    available_runners = ["jlink", "pyocd"]

    arg_parser = argparse.ArgumentParser(
        description="ZView - A real-time thread viewer for Zephyr RTOS."
    )
    arg_parser.add_argument(
        "-e",
        "--elf-file",
        required=True,
        help="Path to the application's .elf firmware file.",
    )
    arg_parser.add_argument(
        "-r",
        "--runner",
        required=True,
        choices=available_runners,
        help="Runner to start analysis with.",
    )
    arg_parser.add_argument(
        "-t",
        "--runner-target",
        required=True,
        help="Descriptor name for the target MCU on the chosen runner",
    )
    arg_parser.add_argument(
        "--period",
        default=0.10,
        type=float,
        help="Minimum period to update system information.",
    )

    args = arg_parser.parse_args()

    scrapper_cls = PyOCDScraper if args.runner != "jlink" else JLinkScraper

    with scrapper_cls(args.runner_target) as meta_scraper:
        z_scraper = ZScraper(meta_scraper, args.elf_file)
        if not z_scraper.has_names:
            print("NO thread names available (CONFIG_THREAD_NAME=n)")
        if not z_scraper.has_usage:
            print("NO cpu stats available (CONFIG_THREAD_RUNTIME_STATS=n)")
        if not z_scraper.has_heaps:
            print("NO heap stats available (CONFIG_SYS_HEAP_RUNTIME_STATS=n)")

        curses.wrapper(tui_run, z_scraper, args.period)


if __name__ == "__main__":
    main()
