#!/usr/bin/env python3
# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
One-shot harness that records a live ZScraper session to a gzipped NDJSON
file. Intended for producing replay fixtures against a QEMU-hosted Zephyr
target exposing a GDB stub.
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backend.gdb import GDBScraper  # noqa: E402
from backend.recording import RecordingScraper  # noqa: E402
from orchestrator import ZScraper  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", required=True, help="Path to the Zephyr ELF file.")
    parser.add_argument(
        "--gdb",
        default="localhost:1234",
        help="GDB stub endpoint as host:port (default: localhost:1234).",
    )
    parser.add_argument("--out", required=True, help="Output .ndjson.gz path.")
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (default: 10).",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=0.5,
        help="Polling period in seconds (default: 0.5).",
    )
    args = parser.parse_args()

    backend = GDBScraper(args.gdb)
    recorder = RecordingScraper(backend, args.out)

    with recorder:
        scraper = ZScraper(recorder, args.elf)
        scraper.update_available_threads()
        scraper.reset_thread_pool()

        data_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        scraper.start_polling_thread(data_queue, stop_event, args.period)

        try:
            time.sleep(args.duration)
        finally:
            stop_event.set()
            scraper.finish_polling_thread()

    print(f"Recorded {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
