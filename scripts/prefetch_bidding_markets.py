"""Prefetch held-out market tapes while the main job generates training markets.

Only the public market builder is used. No outcomes are inspected or used for
selection. The helper exits when training-market generation ends, so the main
job has six uncached validation markets before reaching any prefetched file.
The main runner owns the process-bound wake assertion.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
import tomllib
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--one-market", type=int)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    manifest = read(args.run / "manifest.json")
    if config != manifest["config"]:
        raise RuntimeError("Prefetch configuration differs from the active study")
    root = Path(__file__).resolve().parents[1]
    for name, expected in manifest["auction"]["sha256"].items():
        source = (
            root / "src/trajectorylab/bidding/market.py"
            if name == "market.py"
            else root / "third_party/auctionnet" / name
        )
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Prefetch source identity differs from the active study")

    def guard():
        current = read(args.run / "status.json")
        if current.get("phase") != "data" or current.get("split") != "train":
            return "main-left-training-market-generation"
        if read(args.run / "runner.json").get("state") != "running":
            return "main-runner-not-active"
        if any(
            (args.run / "markets" / f"{m}.json").exists()
            for m in config["study"]["validation_markets"]
        ):
            return "validation-buffer-not-empty"
        battery = subprocess.check_output(["/usr/bin/pmset", "-g", "batt"], text=True)
        if "AC Power" not in battery:
            return "external-power-required-for-extra-worker"
        free = os.statvfs(args.run)
        if free.f_bavail * free.f_frsize < 12 * 1024**3:
            return "disk-reserve"
        pressure = subprocess.check_output(["/usr/bin/memory_pressure", "-Q"], text=True)
        match = re.search(r"free percentage:\s*(\d+)%", pressure)
        if not match or int(match[1]) < 25:
            return "memory-reserve"
        return None

    if args.one_market is not None:
        if args.one_market not in config["study"]["test_markets"]:
            raise ValueError("Only declared held-out markets may be prefetched")
        if reason := guard():
            print(f"prefetch skipped: {reason}", flush=True)
            return 0
        from trajectorylab.bidding.market import build_market

        meta = build_market(
            args.run / "markets" / f"{args.one_market}.npz",
            config["study"],
            args.one_market,
        )
        print(
            f"prefetched market={args.one_market} seconds={meta['seconds']:.1f} "
            f"bytes={meta['bytes']}",
            flush=True,
        )
        return 0

    lock = (args.run / ".prefetch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    stopping = False
    complete = []

    def record(state, **details):
        path = args.run / "prefetch.json"
        temporary = path.with_suffix(".partial.json")
        temporary.write_text(
            json.dumps(
                {
                    "state": state,
                    "updated_unix": time.time(),
                    "pid": os.getpid(),
                    "source_identity": manifest["identity"],
                    "helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "completed_markets": complete,
                    **details,
                },
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, path)

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for market in reversed(config["study"]["test_markets"]):
            if stopping or (reason := guard()):
                record("stopped", reason="requested-stop" if stopping else reason)
                return 0
            meta_path = args.run / "markets" / f"{market}.json"
            if meta_path.exists():
                continue
            command = [
                sys.executable,
                "-u",
                str(Path(__file__).resolve()),
                "--config",
                str(args.config),
                "--run",
                str(args.run),
                "--one-market",
                str(market),
            ]
            env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            child = subprocess.Popen(command, start_new_session=True, env=env)
            record("running", market=market, worker_pid=child.pid)
            while child.poll() is None:
                time.sleep(10)
                if stopping or (reason := guard()):
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait(timeout=10)
                    record("stopped", reason="requested-stop" if stopping else reason)
                    return 0
            if child.returncode:
                record("failed", market=market, exit_code=child.returncode)
                return child.returncode
            if meta_path.exists():
                meta = read(meta_path)
                if meta["source"] != manifest["auction"]:
                    raise RuntimeError("Prefetched market source identity mismatch")
                if meta["requested_pv_count"] != config["study"]["pv_count"]:
                    raise RuntimeError("Prefetched market scale mismatch")
                if (args.run / "markets" / f"{market}.npz").stat().st_size != meta["bytes"]:
                    raise RuntimeError("Prefetched market file size mismatch")
                complete.append(market)
        record("complete")
        return 0
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)


if __name__ == "__main__":
    raise SystemExit(main())
