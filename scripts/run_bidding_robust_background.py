"""Supervise one bidding development stage with process-bound wake assertions.

Run with the repository virtualenv. Low battery pauses the stage until power
recovers; no system power preferences or launchd jobs are changed.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

from run_bidding_background import power

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--module",
        choices=[
            "trajectorylab.bidding_robust",
            "trajectorylab.bidding_generative",
            "trajectorylab.bidding_generative_control",
            "trajectorylab.bidding_generative_final",
            "trajectorylab.bidding_generative_replication",
        ],
        default="trajectorylab.bidding_robust",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bidding-robust.toml")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/bidding/robust-v1")
    parser.add_argument(
        "--stage", choices=["baselines", "pilot", "train", "evaluate", "develop"], default="develop"
    )
    args = parser.parse_args()
    allowed = {
        "trajectorylab.bidding_robust": {"baselines", "pilot", "develop"},
        "trajectorylab.bidding_generative": {"pilot", "develop"},
        "trajectorylab.bidding_generative_control": {"develop"},
        "trajectorylab.bidding_generative_final": {"train", "evaluate", "develop"},
        # Replication must evaluate each seed before permitting the next fit.
        "trajectorylab.bidding_generative_replication": {"evaluate", "develop"},
    }
    if args.stage not in allowed[args.module]:
        parser.error(f"{args.module} does not support stage={args.stage}")
    config_path = args.config.resolve()
    config = tomllib.loads(config_path.read_text())
    output = args.output.resolve()
    if (ROOT / config["output"]).resolve() != output:
        raise ValueError("--output must match the selected configuration")
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    caffeinate = None
    stopping = threading.Event()

    def record(state, **details):
        status = output / "runner.json"
        temporary = status.with_suffix(".partial.json")
        temporary.write_text(
            json.dumps(
                {
                    "state": state,
                    "updated_unix": time.time(),
                    "supervisor_pid": os.getpid(),
                    "module": args.module,
                    "stage": args.stage,
                    "config": str(config_path),
                    **details,
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, status)

    def interrupt(signum=signal.SIGINT):
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    def stop(signum, frame):
        stopping.set()
        interrupt()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGHUP, stop)

    def wait_child():
        try:
            return child.wait(timeout=45)
        except subprocess.TimeoutExpired:
            interrupt(signal.SIGTERM)
        try:
            return child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            interrupt(signal.SIGKILL)
            return child.wait(timeout=10)

    def release_wake_assertion():
        nonlocal caffeinate
        if caffeinate is None:
            return
        if caffeinate.poll() is None:
            caffeinate.terminate()
        try:
            caffeinate.wait(timeout=5)
        except subprocess.TimeoutExpired:
            caffeinate.kill()
            caffeinate.wait(timeout=5)
        caffeinate = None

    try:
        while not stopping.is_set():
            external, percent = power()
            if not external and percent is not None and percent <= 20:
                release_wake_assertion()
                record("paused-low-battery", battery_percent=percent)
                stopping.wait(30)
                continue
            command = [sys.executable, "-u", "-m", args.module, "--config", str(config_path)]
            if args.module != "trajectorylab.bidding_generative_control":
                command += ["--stage", args.stage]
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                start_new_session=True,
            )
            caffeinate = subprocess.Popen(
                ["/usr/bin/caffeinate", "-is", "-w", str(child.pid)],
                start_new_session=True,
            )
            if stopping.is_set():
                interrupt()
            record("running", pid=child.pid, caffeinate_pid=caffeinate.pid)
            paused = False
            while child.poll() is None and not stopping.is_set():
                if stopping.wait(15) or child.poll() is not None:
                    break
                if caffeinate.poll() is not None and child.poll() is None:
                    raise RuntimeError("Wake assertion exited while the stage was running")
                external, percent = power()
                if not external and percent is not None and percent <= 20:
                    interrupt()
                    release_wake_assertion()
                    paused = True
                    break
            code = wait_child()
            child = None
            release_wake_assertion()
            if stopping.is_set():
                record("stopped", exit_code=code)
                return 130
            if paused:
                record("paused-low-battery", battery_percent=percent)
                stopping.wait(30)
                continue
            record("complete" if code == 0 else "failed", exit_code=code)
            return code
        record("stopped")
        return 130
    except Exception as error:
        interrupt()
        if child is not None:
            wait_child()
        release_wake_assertion()
        record(
            "stopped" if stopping.is_set() else "failed",
            error_type=type(error).__name__,
            reason=str(error),
        )
        raise
    finally:
        try:
            if child is not None and child.poll() is None:
                interrupt()
                wait_child()
        finally:
            release_wake_assertion()
            lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
