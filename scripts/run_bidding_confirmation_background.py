"""Resume a fixed confirmation with two bounded market workers and power pauses.

This operational wrapper never reads policy outcomes during market generation.
It changes no system power preferences; caffeinate belongs to active children.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from run_bidding_background import power

from trajectorylab.bidding.market import build_market
from trajectorylab.bidding_confirmation import _prepare, _relative
from trajectorylab.utils import fingerprint, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bidding-confirmation.toml")
    parser.add_argument("--one-market", type=int)
    args = parser.parse_args()
    opts, config, core = _prepare(args.config)
    identity, directory = fingerprint(core), _relative(opts["output"])
    directory.mkdir(parents=True, exist_ok=True)

    if args.one_market is not None:
        if args.one_market not in opts["markets"]:
            raise ValueError("Undeclared confirmation market")
        manifest = read_json(directory / "manifest.json")
        if manifest != {"identity": identity, "core": core}:
            raise RuntimeError("A matching locked manifest must precede generation")
        with (directory / f".market-{args.one_market}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            meta = build_market(
                directory / "markets" / f"{args.one_market}.npz",
                config["study"],
                args.one_market,
            )
            if _prepare(args.config)[2] != core:
                raise RuntimeError("Frozen inputs changed during generation")
            print(f"market={args.one_market} seconds={meta['seconds']:.1f}", flush=True)
        return 0

    with (directory / ".supervisor.lock").open("a") as supervisor_lock:
        fcntl.flock(supervisor_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with (directory / ".run.lock").open("a") as run_lock:
            fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest_path = directory / "manifest.json"
            if manifest_path.exists():
                existing = read_json(manifest_path)
                if existing.get("identity") != identity or existing.get("core") != core:
                    raise RuntimeError("Different confirmation identity")
            else:
                if (directory / "markets").exists() and any((directory / "markets").iterdir()):
                    raise RuntimeError("Generation occurred before the locked manifest")
                write_json(manifest_path, {"identity": identity, "core": core})
        own_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        active: dict[int, subprocess.Popen] = {}
        stopping = False

        def record(state, **details):
            write_json(
                directory / "runner.json",
                {
                    "identity": identity,
                    "state": state,
                    "updated_unix": time.time(),
                    "pid": os.getpid(),
                    "runner_sha256": own_hash,
                    "active": {str(k): v.pid for k, v in active.items()},
                    **details,
                },
            )

        def stop(signum, frame):
            nonlocal stopping
            stopping = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)

        def guard():
            if (
                _prepare(args.config)[2] != core
                or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != own_hash
            ):
                raise RuntimeError("Frozen confirmation inputs or operational wrapper changed")

        def terminate_children():
            for child in active.values():
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
            for child in active.values():
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
            active.clear()

        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")

        def spawn(command):
            return subprocess.Popen(
                ["/usr/bin/caffeinate", "-is", *command],
                cwd=ROOT,
                env=environment,
                start_new_session=True,
            )

        try:
            while not stopping:
                guard()
                for market, child in list(active.items()):
                    code = child.poll()
                    if code is not None:
                        del active[market]
                        if code:
                            raise RuntimeError(f"Market worker {market} exited {code}")
                completed = [
                    m
                    for m in opts["markets"]
                    if (directory / "markets" / f"{m}.json").is_file()
                    and (directory / "markets" / f"{m}.npz").is_file()
                ]
                if len(completed) == 24 and not active:
                    break
                external, percent = power()
                if not external and percent is not None and percent <= 20:
                    terminate_children()
                    record("paused-low-battery", completed_markets=len(completed), battery=percent)
                    time.sleep(15)
                    continue
                reserve = os.statvfs(directory)
                if reserve.f_bavail * reserve.f_frsize < 12 * 1024**3:
                    raise RuntimeError("Insufficient disk reserve")
                # Two single-threaded builders on AC or ample battery; one below 50%.
                limit = 2 if external or (percent is not None and percent >= 50) else 1
                pending = [m for m in opts["markets"] if m not in completed and m not in active]
                while pending and len(active) < limit:
                    market = pending.pop(0)
                    active[market] = spawn(
                        [
                            sys.executable,
                            "-u",
                            str(Path(__file__).resolve()),
                            "--config",
                            str(args.config.resolve()),
                            "--one-market",
                            str(market),
                        ]
                    )
                record("generating", completed_markets=len(completed), battery=percent)
                time.sleep(15)
            if stopping:
                terminate_children()
                record("stopped")
                return 130
            # No summary/results were read above. The module checks and seals all 24 caches.
            while not stopping:
                guard()
                external, percent = power()
                if not external and percent is not None and percent <= 20:
                    record("paused-low-battery", pending_stage="evaluation", battery=percent)
                    time.sleep(15)
                    continue
                active[-1] = spawn(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "trajectorylab.bidding_confirmation",
                        "--config",
                        str(args.config.resolve()),
                        "--stage",
                        "evaluate",
                    ]
                )
                record("evaluating", completed_markets=24)
                paused = False
                while active[-1].poll() is None and not stopping:
                    time.sleep(15)
                    guard()
                    external, percent = power()
                    if not external and percent is not None and percent <= 20:
                        paused = True
                        break
                if stopping or paused:
                    terminate_children()
                    if paused:
                        continue
                    record("stopped")
                    return 130
                code = active.pop(-1).wait()
                if code:
                    raise RuntimeError(f"Evaluation exited {code}")
                status = read_json(directory / "status.json")
                if status.get("phase") != "complete" or status.get("identity") != identity:
                    raise RuntimeError("Evaluation exited without complete evidence")
                record("complete", exit_code=0)
                return 0
            record("stopped")
            return 130
        except Exception as error:
            terminate_children()
            record("failed", error_type=type(error).__name__, reason=str(error))
            raise
        finally:
            terminate_children()


if __name__ == "__main__":
    raise SystemExit(main())
