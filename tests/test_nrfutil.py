# Copyright (c) 2026 Paulo Santos (@wkhadgar)
#
# SPDX-License-Identifier: Apache-2.0

import json
import struct
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from backend.base import (
    ProbeConnectFailure,
    ProbeReadError,
    ProbeReadMalformed,
    ProbeReadTimeout,
)
from backend.nrfutil import NrfutilScraper

NRFUTIL_PATH = "/usr/bin/nrfutil"


def _success_stdout(*blocks: list) -> str:
    """Build nrfutil JSON-lines stdout for a successful ``memory-read`` task."""
    memory = [
        {"startAddress": hex(0x20000000 + index * 0x10), "values": list(values)}
        for index, values in enumerate(blocks)
    ]

    return (
        "\n".join(
            (
                json.dumps({"type": "task_begin", "data": {"description": "memory-read"}}),
                json.dumps(
                    {
                        "type": "task_end",
                        "data": {"result": "success", "data": {"memoryData": memory}},
                    }
                ),
            )
        )
        + "\n"
    )


def _fail_stdout(message: str) -> str:
    """Build nrfutil JSON-lines stdout for a failed task."""
    return json.dumps({"type": "task_end", "data": {"result": "fail", "message": message}}) + "\n"


def _completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> MagicMock:
    return MagicMock(stdout=stdout, stderr=stderr, returncode=returncode)


@pytest.fixture
def scraper():
    """A scraper that believes nrfutil is installed, without touching the filesystem."""
    with patch("backend.nrfutil.shutil.which", return_value=NRFUTIL_PATH):
        return NrfutilScraper("001057722700")


def test_selector_prefers_the_serial_number(scraper):
    """A given target_mcu selects that exact probe."""
    assert scraper._selector() == ["--serial-number", "001057722700"]


def test_selector_falls_back_to_the_jlink_trait():
    """Without a target_mcu, nrfutil is asked for the sole attached J-Link."""
    with patch("backend.nrfutil.shutil.which", return_value=NRFUTIL_PATH):
        assert NrfutilScraper(None)._selector() == ["--traits", "jlink"]


def test_read_builds_a_direct_json_read_command(scraper):
    """Reads are issued as `nrfutil --json device read --direct` for the probe."""
    with patch(
        "backend.nrfutil.subprocess.run", return_value=_completed(_success_stdout([1, 2, 3, 4]))
    ) as run:
        scraper.read_bytes(0x20000000, 4)

    cmd = run.call_args[0][0]
    assert cmd[:4] == [NRFUTIL_PATH, "--json", "device", "read"]
    assert cmd[cmd.index("--address") + 1] == "0x20000000"
    assert cmd[cmd.index("--bytes") + 1] == "4"
    # --direct keeps nrfutil out of programming mode, so any address is readable.
    assert "--direct" in cmd
    assert cmd[-2:] == ["--serial-number", "001057722700"]


def test_read_bytes_returns_the_payload(scraper):
    with patch(
        "backend.nrfutil.subprocess.run",
        return_value=_completed(_success_stdout([0xDE, 0xAD, 0xBE, 0xEF])),
    ):
        assert scraper.read_bytes(0x20000000, 4) == b"\xde\xad\xbe\xef"


def test_read_bytes_concatenates_memory_blocks(scraper):
    """nrfutil may split a read across memoryData blocks; they join in order."""
    with patch(
        "backend.nrfutil.subprocess.run",
        return_value=_completed(_success_stdout([1, 2], [3, 4])),
    ):
        assert scraper.read_bytes(0x20000000, 4) == b"\x01\x02\x03\x04"


def test_read_bytes_ignores_non_task_end_and_undecodable_lines(scraper):
    """Progress chatter and non-JSON noise must not derail parsing."""
    stdout = (
        "not json at all\n"
        "\n"
        + json.dumps({"type": "task_progress", "data": {"progressPercentage": 50}})
        + "\n"
        + _success_stdout([0xAA, 0xBB])
    )

    with patch("backend.nrfutil.subprocess.run", return_value=_completed(stdout)):
        assert scraper.read_bytes(0x20000000, 2) == b"\xaa\xbb"


def test_read_bytes_truncates_an_overlong_payload(scraper):
    """A probe returning more than requested is clipped to the asked-for length."""
    with patch(
        "backend.nrfutil.subprocess.run",
        return_value=_completed(_success_stdout([1, 2, 3, 4, 5, 6])),
    ):
        assert scraper.read_bytes(0x20000000, 4) == b"\x01\x02\x03\x04"


def test_zero_length_read_never_spawns_a_process(scraper):
    with patch("backend.nrfutil.subprocess.run") as run:
        assert scraper.read_bytes(0x20000000, 0) == b""
        assert scraper.read_bytes(0x20000000, -1) == b""

    run.assert_not_called()


def test_read8_returns_ints(scraper):
    with patch.object(NrfutilScraper, "_read_mem_raw", return_value=b"\x01\xff"):
        assert scraper.read8(0x0, 2) == [1, 255]


def test_word_reads_apply_endianness(scraper):
    """read32/read64 unpack with the ELF-derived endianness, like every backend."""
    with patch.object(NrfutilScraper, "_read_mem_raw") as raw:
        scraper.endianess = "<"
        raw.return_value = struct.pack("<2I", 1, 2)
        assert scraper.read32(0x0, 2) == (1, 2)

        raw.return_value = struct.pack("<Q", 0xAAAAAAAABBBBBBBB)
        assert scraper.read64(0x0, 1) == (0xAAAAAAAABBBBBBBB,)

        scraper.endianess = ">"
        raw.return_value = struct.pack(">2I", 1, 2)
        assert scraper.read32(0x0, 2) == (1, 2)

        raw.return_value = struct.pack(">Q", 0xAAAAAAAABBBBBBBB)
        assert scraper.read64(0x0, 1) == (0xAAAAAAAABBBBBBBB,)


def test_word_reads_request_the_right_byte_count(scraper):
    """Word reads must ask for width * amount bytes, not amount."""
    with patch.object(NrfutilScraper, "_read_mem_raw", return_value=b"\x00" * 16) as raw:
        scraper.read32(0x100, 4)
        assert raw.call_args[0] == (0x100, 16)

        scraper.read64(0x100, 2)
        assert raw.call_args[0] == (0x100, 16)


def test_timeout_maps_to_probe_read_timeout(scraper):
    with (
        patch(
            "backend.nrfutil.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="nrfutil", timeout=20.0),
        ),
        pytest.raises(ProbeReadTimeout, match="timed out"),
    ):
        scraper.read_bytes(0x20000000, 4)


def test_failed_task_maps_to_probe_read_error_and_quotes_nrfutil(scraper):
    with (
        patch(
            "backend.nrfutil.subprocess.run",
            return_value=_completed(_fail_stdout("Device is protected"), returncode=1),
        ),
        pytest.raises(ProbeReadError, match="Device is protected"),
    ):
        scraper.read_bytes(0x20000000, 4)


def test_unparseable_output_falls_back_to_stderr(scraper):
    """With no task_end record at all, stderr is the only clue worth reporting."""
    with (
        patch(
            "backend.nrfutil.subprocess.run",
            return_value=_completed("", stderr="no probe found", returncode=1),
        ),
        pytest.raises(ProbeReadError, match="no probe found"),
    ):
        scraper.read_bytes(0x20000000, 4)


def test_silent_failure_still_reports_something(scraper):
    """Empty stdout and stderr must not produce an empty error message."""
    with (
        patch("backend.nrfutil.subprocess.run", return_value=_completed("", "", 1)),
        pytest.raises(ProbeReadError, match="no data returned"),
    ):
        scraper.read_bytes(0x20000000, 4)


def test_short_read_maps_to_malformed(scraper):
    """Fewer bytes than requested is a malformed answer, never a silent short read."""
    with (
        patch("backend.nrfutil.subprocess.run", return_value=_completed(_success_stdout([1, 2]))),
        pytest.raises(ProbeReadMalformed),
    ):
        scraper.read_bytes(0x20000000, 4)


@pytest.mark.parametrize("values", ([256, 0, 0, 0], [-1, 0, 0, 0], ["ff", "ee", "dd", "cc"]))
def test_undecodable_values_map_to_malformed(scraper, values):
    """Out-of-range ints and non-int values both honor the documented contract."""
    with (
        patch("backend.nrfutil.subprocess.run", return_value=_completed(_success_stdout(values))),
        pytest.raises(ProbeReadMalformed, match="Malformed data"),
    ):
        scraper.read_bytes(0x20000000, 4)


def test_connect_requires_the_nrfutil_binary():
    with patch("backend.nrfutil.shutil.which", return_value=None):
        scraper = NrfutilScraper("001057722700")

        with pytest.raises(ProbeConnectFailure, match="not found on PATH"):
            scraper.connect()

        assert not scraper.is_connected


def test_connect_validates_reachability_with_a_small_read(scraper):
    """Having no session of its own, connect proves the probe works by reading."""
    with patch(
        "backend.nrfutil.subprocess.run", return_value=_completed(_success_stdout([0, 0, 0, 0]))
    ) as run:
        scraper.connect()

    assert scraper.is_connected
    cmd = run.call_args[0][0]
    assert cmd[cmd.index("--bytes") + 1] == "4"

    scraper.disconnect()
    assert not scraper.is_connected


def test_connect_wraps_read_failures_as_connect_failures(scraper):
    """A probe error during the reachability read is a connection problem."""
    with (
        patch(
            "backend.nrfutil.subprocess.run",
            return_value=_completed(_fail_stdout("No debuggers found"), returncode=1),
        ),
        pytest.raises(ProbeConnectFailure, match="No debuggers found"),
    ):
        scraper.connect()

    assert not scraper.is_connected


def test_connect_is_idempotent(scraper):
    """A second connect must not re-probe an already connected target."""
    with patch(
        "backend.nrfutil.subprocess.run", return_value=_completed(_success_stdout([0, 0, 0, 0]))
    ) as run:
        scraper.connect()
        scraper.connect()

    assert run.call_count == 1


def test_context_manager_connects_and_disconnects(scraper):
    with (
        patch(
            "backend.nrfutil.subprocess.run", return_value=_completed(_success_stdout([0, 0, 0, 0]))
        ),
        scraper as entered,
    ):
        assert entered is scraper
        assert scraper.is_connected

    assert not scraper.is_connected
