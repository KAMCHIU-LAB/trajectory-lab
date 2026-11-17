"""One-shot local runner: process-bound wake assertion and low-battery pause.

Run from the repository root with its virtualenv Python. The launchd job that
starts this utility must use KeepAlive=false. No pmset preferences are changed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path


def power() -> tuple[bool, int | None]:
    result = subprocess.run(
        ["/usr/bin/pmset", "-g", "batt"], capture_output=True, text=True, check=True
    )
    match = re.search(r"(\d+)%;", result.stdout)
    return "AC Power" in result.stdout, int(match.group(1)) if match else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release-temporary-label")
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    status = args.output / "runner.json"
    child = None
    stopping = False

    def record(state, **details):
        temporary = status.with_suffix(".partial.json")
        temporary.write_text(
            json.dumps({"state": state, "updated_unix": time.time(), **details}, indent=2) + "\n"
        )
        os.replace(temporary, status)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def release_temporary():
        if args.release_temporary_label:
            subprocess.run(
                ["/bin/launchctl", "remove", args.release_temporary_label],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

    try:
        while not stopping:
            external, percent = power()
            if not external and percent is not None and percent <= 20:
                release_temporary()
                record(
                    "paused-low-battery",
                    battery_percent=percent,
                    recovery="Resume automatically when external power returns and the computer wakes",
                )
                time.sleep(30)
                continue
            command = [
                "/usr/bin/caffeinate",
                "-is",
                sys.executable,
                "-u",
                "-m",
                "trajectorylab.bidding",
                "--config",
                str(args.config),
                "--output",
                str(args.output),
            ]
            child = subprocess.Popen(command, start_new_session=True)
            record("running", pid=child.pid)
            release_temporary()
            paused = False
            while child.poll() is None and not stopping:
                time.sleep(15)
                external, percent = power()
                if not external and percent is not None and percent <= 20:
                    os.killpg(child.pid, signal.SIGINT)
                    paused = True
                    break
            try:
                code = child.wait(timeout=45)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                code = child.wait(timeout=15)
            if paused:
                record("paused-low-battery", battery_percent=percent)
                continue
            if code == 0 and args.export:
                exported = subprocess.run(
                    [
                        sys.executable,
                        "scripts/export_bidding_report.py",
                        "--run",
                        str(args.output),
                        "--destination",
                        str(args.export),
                    ],
                    check=False,
                )
                if exported.returncode:
                    record("export-failed", exit_code=exported.returncode)
                    return exported.returncode
            record(
                "stopped" if stopping else ("complete" if code == 0 else "failed"), exit_code=code
            )
            return code
        record("stopped")
        return 130
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        release_temporary()


if __name__ == "__main__":
    raise SystemExit(main())
