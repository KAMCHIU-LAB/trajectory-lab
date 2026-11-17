"""One-shot controller supervisor with a main-study prerequisite and power pause.

Run with the repository virtualenv. A launchd job should use KeepAlive=false.
Wake assertions belong only to active subprocesses; system preferences stay intact.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path

from run_bidding_background import power

ROOT = Path(__file__).resolve().parents[1]
COMPLETE = {"complete", "complete-not-promoted"}


def read_if_present(path):
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config_bytes = config_path.read_bytes()
    config = tomllib.loads(config_bytes.decode())
    frozen_files = (
        Path(__file__),
        ROOT / "scripts/run_bidding_background.py",
        ROOT / "scripts/export_bidding_control.py",
        ROOT / "src/trajectorylab/bidding_control.py",
    )
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in frozen_files
    }
    output = args.output.resolve()
    if (ROOT / config["output"]).resolve() != output:
        raise ValueError("--output must match the controller configuration")
    main_output = ROOT / config["main_run"]
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    stopping = False

    def record(state, **details):
        content = {
            "state": state,
            "updated_unix": time.time(),
            "supervisor_pid": os.getpid(),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "source_sha256": source_hashes,
            **details,
        }
        fd, temporary = tempfile.mkstemp(dir=output, suffix=".runner.tmp")
        try:
            with os.fdopen(fd, "w") as handle:
                json.dump(content, handle, indent=2, sort_keys=True, allow_nan=False)
                handle.write("\n")
            os.replace(temporary, output / "runner.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def interrupt(signum=signal.SIGINT):
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        interrupt()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def main_state():
        if config_path.read_bytes() != config_bytes or any(
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected
            for path, expected in source_hashes.items()
        ):
            raise RuntimeError("Controller configuration or supervised source changed")
        runner = read_if_present(main_output / "runner.json")
        status = read_if_present(main_output / "status.json")
        if runner.get("state") in {"failed", "export-failed", "stopped"}:
            raise RuntimeError(f"Main supervisor terminated: {runner['state']}")
        if status.get("phase") == "failed":
            raise RuntimeError("Main study failed")
        return status.get("phase")

    def wait_child():
        try:
            return child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            interrupt(signal.SIGTERM)
        try:
            return child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            interrupt(signal.SIGKILL)
            return child.wait(timeout=10)

    def run_child(command, state):
        nonlocal child
        while not stopping:
            if config_path.read_bytes() != config_bytes:
                raise RuntimeError("Controller configuration changed during supervision")
            if main_state() != "complete":
                raise RuntimeError("Main study is no longer complete")
            external, percent = power()
            if not external and percent is not None and percent <= 20:
                record("paused-low-battery", battery_percent=percent, pending_stage=state)
                time.sleep(30)
                continue
            child = subprocess.Popen(
                ["/usr/bin/caffeinate", "-is", *command],
                cwd=ROOT,
                start_new_session=True,
            )
            record(state, pid=child.pid)
            paused = False
            while child.poll() is None and not stopping:
                time.sleep(15)
                if child.poll() is not None or stopping:
                    break
                main_state()
                if config_path.read_bytes() != config_bytes:
                    raise RuntimeError("Controller configuration changed during supervision")
                external, percent = power()
                if not external and percent is not None and percent <= 20:
                    interrupt()
                    paused = True
                    break
            code = wait_child()
            child = None
            if stopping:
                return 130
            if paused:
                record("paused-low-battery", battery_percent=percent, pending_stage=state)
                continue
            main_state()
            return code
        return 130

    try:
        while not stopping:
            phase = main_state()
            if phase == "complete":
                break
            record("waiting-for-main", main_phase=phase)
            time.sleep(30)
        if stopping:
            record("stopped")
            return 130
        status = read_if_present(output / "status.json")
        manifest = read_if_present(output / "manifest.json")
        if status.get("phase") not in COMPLETE or manifest.get("config") != config:
            code = run_child(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "trajectorylab.bidding_control",
                    "--config",
                    str(config_path),
                ],
                "running",
            )
            if code:
                record("stopped" if stopping else "failed", exit_code=code)
                return code
        if read_if_present(output / "status.json").get("phase") not in COMPLETE:
            raise RuntimeError("Controller exited without complete evidence")
        code = run_child(
            [
                sys.executable,
                "-u",
                str(ROOT / "scripts/export_bidding_control.py"),
                "--run",
                str(output),
                "--destination",
                str(args.export.resolve()),
            ],
            "exporting",
        )
        record(
            "stopped" if stopping else ("complete" if code == 0 else "export-failed"),
            exit_code=code,
        )
        return code
    except Exception as error:
        interrupt()
        if child is not None:
            wait_child()
        record(
            "stopped" if stopping else "failed", error_type=type(error).__name__, reason=str(error)
        )
        raise
    finally:
        if child is not None and child.poll() is None:
            interrupt()
            wait_child()


if __name__ == "__main__":
    raise SystemExit(main())
