#!/usr/bin/env python3
"""Build a source-only public event market and optionally prove D=0 tape replay."""

from __future__ import annotations

import argparse
import json
import resource
import tempfile
import time
from pathlib import Path

import numpy as np

from trajectorylab.bidding.market import Tape, build_market
from trajectorylab.bidding_delayed_feedback import DelayQueue, EventTape, build_event_market


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", type=int, default=1001)
    parser.add_argument("--pv-count", type=int, default=10000)
    parser.add_argument("--focals", type=int, nargs="+", default=[0])
    parser.add_argument("--actions", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0, 3.0])
    parser.add_argument("--biases", type=float, nargs="+", default=[0.75, 1.0, 1.5])
    parser.add_argument("--verify-reference", action="store_true")
    args = parser.parse_args()
    study = dict(
        focal_agents=args.focals, pv_count=args.pv_count, actions=args.actions, biases=args.biases
    )
    start = time.perf_counter()
    meta = build_event_market(args.output, study, args.market)
    result = dict(
        metadata=meta,
        elapsed_seconds=time.perf_counter() - start,
        peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_rss_note="bytes on macOS; KiB on Linux",
    )
    if args.verify_reference:
        with tempfile.TemporaryDirectory() as tmp:
            reference_path = Path(tmp) / "reference.npz"
            build_market(reference_path, study, args.market)
            original, events = Tape(reference_path), EventTape(args.output)
            for k, v in original.arrays.items():
                np.testing.assert_array_equal(v, events.arrays[k], err_msg=k)
            cases = 0
            for fi, focal in enumerate(args.focals):
                queue = DelayQueue(args.market, focal, 7, "immediate")
                for tick in range(48):
                    for action in events.arrays["grid"]:
                        for budget in (0.0, float(events.arrays["budgets"][fi]), float("inf")):
                            s = events.settle_events(fi, tick, action, budget)
                            assert (
                                s.cost,
                                s.conversions,
                                s.wins,
                                s.budget_blocked,
                            ) == original.settle(fi, tick, action, budget)
                            cases += 1
                    s = events.settle_events(fi, tick, 1.0, float("inf"))
                    queue.register(tick, s)
                    assert len(queue.advance(tick)) == s.conversions
                assert queue.advance(96) == ()
            result["reference"] = dict(
                exact_arrays=True, settlement_cases=cases, immediate_arrivals_exact=True
            )
    result["total_seconds"] = time.perf_counter() - start
    result["peak_rss_native"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
