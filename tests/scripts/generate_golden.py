#!/usr/bin/env python3
# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

"""
Replays a recorded fixture through a real ZScraper and serializes the final
frame's thread and heap state as a golden JSON snapshot for integration tests.
"""

import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backend.replay import ReplayScraper  # noqa: E402
from orchestrator import ZScraper  # noqa: E402

GOLDEN_SCHEMA = "zview-golden/1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elf", required=True, help="Path to the matching Zephyr ELF.")
    parser.add_argument("--fixture", required=True, help="Path to .ndjson.gz recording.")
    parser.add_argument("--out", required=True, help="Output golden JSON path.")
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Seconds to wait for replay to exhaust (default: 5).",
    )
    args = parser.parse_args()

    replay = ReplayScraper(args.fixture)
    with replay:
        scraper = ZScraper(replay, args.elf)
        scraper.update_available_threads()
        scraper.reset_thread_pool()

        data_queue: queue.Queue = queue.Queue()
        stop_event = threading.Event()
        scraper.start_polling_thread(data_queue, stop_event, 0.001)

        time.sleep(args.timeout)
        stop_event.set()
        scraper.finish_polling_thread()

        frames: list[dict] = []
        while not data_queue.empty():
            frames.append(data_queue.get_nowait())

    valid = [f for f in frames if "threads" in f]
    if not valid:
        print("No valid frames captured from replay.", file=sys.stderr)
        return 1

    last = valid[-1]
    golden: dict = {
        "schema": GOLDEN_SCHEMA,
        "source": Path(args.fixture).name,
        "frame_count": len(valid),
        "threads": [
            {
                "name": t.name,
                "stack_start": t.stack_start,
                "stack_size": t.stack_size,
                "stack_watermark": t.runtime.stack_watermark if t.runtime else None,
            }
            for t in last["threads"]
        ],
    }
    if "heaps" in last:
        golden["heaps"] = [
            {
                "name": h.name,
                "address": h.address,
                "free_bytes": h.free_bytes,
                "allocated_bytes": h.allocated_bytes,
                "max_allocated_bytes": h.max_allocated_bytes,
            }
            for h in last["heaps"]
        ]

    Path(args.out).write_text(json.dumps(golden, indent=2) + "\n")
    print(f"Wrote {args.out} ({len(valid)} valid frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
