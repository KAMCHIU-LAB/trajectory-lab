#!/usr/bin/env python3
"""Offline pending-count diagnostic on the SAME frozen point-policy trajectories.

This audit never tunes a policy and is not a generative bidding method. Final
pending labels are used only after the public cohort estimator call, for scoring.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from trajectorylab import bidding_delayed_experiment as experiment
from trajectorylab.bidding.market import score
from trajectorylab.bidding_delayed_feedback import DelayQueue, EventTape
from trajectorylab.bidding_pending_response import PendingEstimator
from trajectorylab.utils import fingerprint, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
TICKS = (8, 16, 24, 32, 40, 47)
METHODS = ("naive", "mature_only", "delay_corrected", "posterior")
ARRIVAL_SEED = 11
SAMPLES = 256


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_seed(seed, market, focal, bias_group, regime, tick):
    identity = f"{seed}:{market}:{focal}:{bias_group}:{regime}:{tick}".encode("ascii")
    return int.from_bytes(hashlib.blake2b(identity, digest_size=8).digest(), "little")


def integer_crps(samples, truth):
    """Empirical CRPS, including the diagonal in the n² pairwise expectation."""
    x = np.asarray(samples)
    if (
        x.ndim != 1
        or not len(x)
        or np.any(~np.isfinite(x))
        or np.any(x != np.floor(x))
        or np.any(x < 0)
    ):
        raise ValueError("Expected nonempty nonnegative integer pending samples")
    if not np.isfinite(truth) or truth < 0 or truth != int(truth):
        raise ValueError("Expected nonnegative integer pending truth")
    ordered = np.sort(x.astype(np.float64))
    n = len(ordered)
    pair_term = np.dot(2 * np.arange(1, n + 1) - n - 1, ordered) / n**2
    return float(np.mean(np.abs(ordered - truth)) - pair_term)


def audit_trajectory(tape, fi, multiplier, regime, variant, cdf, config, seed=20261119):
    market, focal = int(tape.arrays["market_id"]), int(tape.arrays["focals"][fi])
    budget, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
    group = {0.75: "under", 1.0: "nominal", 1.5: "over"}[multiplier]
    queue = DelayQueue(market, focal, ARRIVAL_SEED, regime)
    ledger = experiment.PublicLedger(config["evaluation"]["probability_bin_edges"])
    estimators = {
        method: PendingEstimator(
            cdf, method, prior_log_sd=config["history"]["calibration_prior_log_sd"]
        )
        for method in METHODS
    }
    if variant["estimator"] not in estimators:
        raise ValueError("Audit trajectory must use a deployable frozen point-policy estimate")
    controller = estimators[variant["estimator"]]
    actions = np.asarray(config["study"]["actions"])
    spent, truth, wins = 0.0, 0, 0
    chosen, records = [], []
    for tick in range(48):
        ledger.receive(queue.advance(tick - 1))
        view = experiment.PolicyView(
            tick, spent, ledger.arrived, budget, cpa, ledger.snapshot(tick)
        )
        # Both controller and diagnostic estimators receive only this public view.
        control_estimate = controller.estimate(view.cohorts).total_mean
        if tick in TICKS:
            for method, estimator in estimators.items():
                estimate = estimator.estimate(view.cohorts)
                row = dict(
                    market=market,
                    focal=focal,
                    bias_group=group,
                    delay=regime,
                    tick=tick,
                    method=method,
                    truth_u=truth - ledger.arrived,
                    mean_u=estimate.pending_mean,
                )
                row["error"] = row["mean_u"] - row["truth_u"]
                row["absolute_error"] = abs(row["error"])
                row.update(interval_lower=None, interval_upper=None, covered=None, crps=None)
                if method == "posterior":
                    draws = estimator.sample(
                        view.cohorts,
                        SAMPLES,
                        rng=audit_seed(seed, market, focal, group, regime, tick),
                    )
                    low, high = np.quantile(draws, [0.1, 0.9], method="inverted_cdf")
                    row.update(
                        interval_lower=int(low),
                        interval_upper=int(high),
                        covered=int(low <= row["truth_u"] <= high),
                        crps=integer_crps(draws, row["truth_u"]),
                    )
                records.append(row)
        action = experiment.feedback_from_view(
            view, actions, control_estimate, variant["gain"], variant["prior"]
        )
        settled = tape.settle_events(fi, tick, action * multiplier, budget - spent)
        ledger.accept(queue.register(tick, settled, prediction_multiplier=multiplier))
        spent += settled.cost
        truth += settled.conversions
        wins += settled.wins
        chosen.append(float(action))
    ledger.receive(queue.advance(48 + config["history"]["maturity_window"]))
    if ledger.arrived != truth or spent > budget + 1e-6:
        raise RuntimeError("Audit replay maturity/budget mismatch")
    return records, dict(
        cost=spent,
        conversions=truth,
        exposures=wins,
        score=score(spent, truth, cpa),
        final_arrived=ledger.arrived,
        actions=chosen,
        violation=int(spent > cpa * truth + 1e-8),
    )


def assert_reference(actual, reference):
    for key, value in actual.items():
        if value != reference[key]:
            raise RuntimeError(f"Common trajectory differs from frozen confirmation: {key}")


def aggregate(records):
    groups = defaultdict(list)
    for row in records:
        groups[(row["bias_group"], row["delay"], row["tick"], row["method"])].append(row)
    return [
        dict(
            bias_group=g,
            delay=d,
            tick=t,
            method=m,
            decisions=len(rows),
            mean_error=float(np.mean([r["error"] for r in rows])),
            mae=float(np.mean([r["absolute_error"] for r in rows])),
            mean_truth_u=float(np.mean([r["truth_u"] for r in rows])),
            coverage80=float(np.mean([r["covered"] for r in rows])) if m == "posterior" else None,
            mean_crps=float(np.mean([r["crps"] for r in rows])) if m == "posterior" else None,
        )
        for (g, d, t, m), rows in sorted(groups.items())
    ]


def run(config_path, output, destination, seed=20261119):
    config, output, destination = (
        experiment.load_config(config_path),
        Path(output),
        Path(destination),
    )
    selection = experiment.check_frozen_selection(config, output)
    confirmation = read_json(output / "confirmation/manifest.json")
    if (
        confirmation["identity"] != fingerprint(confirmation["core"])
        or confirmation["core"]["source"] != experiment.source_record()
        or confirmation["core"]["config"] != config
        or confirmation["core"]["methods"] != selection["frozen_variants"]
        or confirmation["core"]["markets"]
        != experiment.cache_provenance(config, output, "confirmation")
    ):
        raise RuntimeError("Frozen confirmation provenance mismatch")
    if read_json(output / "confirmation/status.json")["phase"] != "complete":
        raise RuntimeError("Complete frozen confirmation before the offline audit")
    frozen_rows = read_json(output / "confirmation/results.json")
    experiment.validate_grid(frozen_rows, config, "confirmation", selection["frozen_variants"])
    variant = selection["strongest_point"]
    references = {
        (r["market"], r["focal"], r["bias_group"], r["delay"]): r
        for r in frozen_rows
        if r["method"] == variant["name"] and r["arrival_seed"] == ARRIVAL_SEED
    }
    expected = (
        len(config["study"]["confirmation_markets"])
        * len(config["study"]["focal_agents"])
        * len(config["study"]["biases"])
        * len(config["evaluation"]["regimes"])
    )
    if len(references) != expected:
        raise RuntimeError("Missing frozen strongest-point arrival-seed-11 reference rows")
    core = dict(
        script_sha256=file_sha256(__file__),
        selection_sha256=file_sha256(output / "selection.json"),
        selection_identity=selection["identity"],
        confirmation_identity=confirmation["identity"],
        input_cache_manifest=confirmation["core"]["markets"],
        results_sha256=file_sha256(output / "confirmation/results.json"),
        history_sha256=file_sha256(output / "history.json"),
        ticks=TICKS,
        methods=METHODS,
        arrival_seed=ARRIVAL_SEED,
        audit_seed=seed,
        posterior_samples=SAMPLES,
        scope="post-freeze offline inference diagnostic on common trajectories; no tuning; not a generative bidder",
    )
    # JSON roundtrip normalizes tuples for equality on resumed calls.
    import json

    manifest = json.loads(json.dumps(dict(identity=fingerprint(core), core=core)))
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise RuntimeError("Offline audit identity changed; choose a new destination")
    write_json(manifest_path, manifest)
    history = experiment.read_history(output)
    records, checked = [], 0
    for market in config["study"]["confirmation_markets"]:
        tape = EventTape(output / "markets" / f"{market}.npz")
        for fi, focal in enumerate(tape.arrays["focals"]):
            for multiplier in config["study"]["biases"]:
                group = {0.75: "under", 1.0: "nominal", 1.5: "over"}[multiplier]
                for regime in config["evaluation"]["regimes"]:
                    rows, actual = audit_trajectory(
                        tape, fi, multiplier, regime, variant, history[regime], config, seed
                    )
                    assert_reference(actual, references[market, int(focal), group, regime])
                    records.extend(rows)
                    checked += 1
        print(f"offline-pending-audit market={market} trajectories_verified={checked}", flush=True)
    if checked != expected:
        raise RuntimeError("Offline audit trajectory coverage incomplete")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "decisions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    result = dict(
        identity=manifest["identity"],
        trajectories_verified=checked,
        decisions=len(records),
        aggregate=aggregate(records),
        interpretation="Descriptive calibration/CRPS only; market-cluster uncertainty not estimated; interval coverage is conditional on frozen model and history",
    )
    write_json(destination / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/bidding-delayed-feedback.toml")
    p.add_argument("--output", type=Path, required=True, help="Frozen experiment output")
    p.add_argument("--destination", type=Path, required=True)
    p.add_argument("--audit-seed", type=int, default=20261119)
    args = p.parse_args()
    result = run(args.config, args.output, args.destination, args.audit_seed)
    print(
        f"verified={result['trajectories_verified']} scalar_rows={result['decisions']} identity={result['identity']}"
    )


if __name__ == "__main__":
    main()
