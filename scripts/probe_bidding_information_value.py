#!/usr/bin/env python3
"""Frozen winner-log information ablation; preflight never fits or writes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import probe_bidding_paired_advantage as old  # sets numerical thread limits

ROOT = Path(__file__).resolve().parents[1]
import numpy as np  # noqa: E402 - import after process-local thread limits
import torch  # noqa: E402

LOGS = ROOT / "artifacts/bidding/numerical-audits/calibration-control-v1"
OUTPUT = ROOT / "artifacts/bidding/numerical-audits/information-value-v1"
PLAN = ROOT / ".omx/plans/prd-bidding-information-value-v1.md"
MARKETS = (1002, 1006, 1010, 1014)
ARMS = ("zero_E", "actual_E", "pseudo_E")


def protocol():
    return {
        "version": "information-value-v1",
        "markets": list(MARKETS),
        "scenes": list(old.SCENES),
        "ticks": list(old.TICKS),
        "scope": "four seen training markets; one intervention then FB; descriptive feasibility only",
        "arms": list(ARMS),
        "seed": 11,
        "pseudo_seed": "20261118+held_market",
        "pseudo": "one complete pre-E vector donor per campaign, sampled with replacement from training campaigns only; diagnostic, not eligible",
        "new_feature": "raw cumulative accepted predicted probability sum before current settlement",
        "fit": "original F1 ridge weighted SUM lambda1, intercept unpenalized, original physical paired delta score",
        "guard": "original F1-only MLP2000 steps, freshly trained once per fold; shared by all arms",
        "controller": old.protocol()["controller"],
        "prediction_metric": "physical delta MSE; legal non-FB actions equally per context, contexts equally per market, markets equally",
        "gate": {
            "mse_improvement": 0.05,
            "macro_vs_FB": 1.005,
            "macro_vs_zero_E": "strictly greater",
            "positive_markets_vs_FB": 3,
            "positive_markets_vs_zero_E": 3,
            "scene_score": 0.98,
            "scene_conv": 0.97,
            "scene_violation_delta": 0,
            "nonfeedback_action": True,
        },
        "resources": {
            "s0_seconds": 900,
            "s1_wall_cpu_seconds": 2700,
            "memory_bytes": 4 * 2**30,
            "output_bytes": 256 * 2**20,
            "free_disk_bytes": 10 * 2**30,
            "threads": 1,
        },
        "independent_confirmation": False,
        "generative_contribution": False,
    }


def check_seal(folder):
    seal = old.read_json(folder / "seal.json")
    old.require(
        all(old.digest(folder / name) == sha for name, sha in seal["files"].items()),
        f"Changed sealed parent {folder.name}",
    )


def join_logs(data):
    rows, campaigns = {}, {}
    with (LOGS / "tick-logs.jsonl").open() as stream:
        for line_number, line in enumerate(stream, 1):
            item = json.loads(line)
            if item["policy"] != "FB":
                continue
            key = (item["market"], item["scene"], item["focal"])
            old.require(key[0] in MARKETS, "Unexpected FB market")
            tick = item["tick"]
            full_key = (*key, tick)
            old.require(full_key not in rows, "Duplicate FB decision")
            rows[full_key] = item, line_number
            campaigns.setdefault(key, []).append(tick)
    old.require(len(rows) == 5760 and len(campaigns) == 120, "FB coverage mismatch")
    for key, ticks in campaigns.items():
        old.require(ticks == list(range(48)), "Incomplete/noncausal FB prefix")
        cost, conv, exposure = 0.0, 0, 0.0
        for tick in ticks:
            row, _ = rows[*key, tick]
            old.require(
                row["past_C"] == conv and row["past_E"] == exposure,
                "Pre-decision C/E differs from prior settlements",
            )
            old.require(
                row["remaining_budget_before"] == row["budget"] - cost, "Pre-decision cost mismatch"
            )
            replay = row["fb_replay"]
            old.require(
                row["public_probability_valid"]
                and replay["state_equal"]
                and replay["action_equal"]
                and replay["integer_settlement_equal"]
                and replay["cost_error"] <= 1e-8,
                "Original FB replay failed",
            )
            old.require(
                row["predicted_sum"] >= 0 and row["predicted_sum"] <= row["accepted_exposures"],
                "Invalid accepted prediction sum",
            )
            cost += row["cost"]
            conv += row["conversions"]
            exposure += row["predicted_sum"]
            old.require(
                cost == row["cumulative_cost"] and conv == row["cumulative_conversions"],
                "Post-settlement accounting mismatch",
            )
    joined, audit = [], []
    for index, diag in enumerate(data["diagnostics"]):
        market, scene_index, focal, tick = map(int, diag)
        key = (market, old.SCENES[scene_index], focal)
        row, line = rows[*key, tick]
        length = int(data["lengths"][index])
        steps = data["steps"][index, :length]
        observations = np.array(
            [rows[*key, int(t)][0]["observation"] for t in steps], dtype=np.float32
        )
        old.require(
            np.array_equal(observations, data["states"][index, :length]),
            "Retained FB state prefix differs from cache",
        )
        old.require(
            row["budget"] == data["budget"][index]
            and row["cpa"] == data["cpa"][index]
            and row["past_C"] == data["conversions"][index]
            and row["action"] == old.ACTIONS[data["baseline_index"][index]],
            "Joined action/budget/CPA/C mismatch",
        )
        expected_cost = 0.0 if tick == 0 else rows[*key, tick - 1][0]["cumulative_cost"]
        old.require(expected_cost == data["spent"][index], "Joined cache cost mismatch")
        last = rows[*key, 47][0]
        base = data["baseline_index"][index]
        old.require(
            last["cumulative_conversions"] == data["terminal_conversions"][index, base]
            and abs(
                old.native_score(
                    last["cumulative_cost"], last["cumulative_conversions"], row["cpa"]
                )
                - data["terminal_score"][index, base]
            )
            <= 1e-8,
            "FB terminal differs from cached continuation",
        )
        joined.append(row["past_E"])
        prefix_hash = old.fingerprint([rows[*key, int(t)][0] for t in range(tick)])
        audit.append(
            {
                "index": index,
                "key": [market, scene_index, focal, tick],
                "source_line": line,
                "E": row["past_E"],
                "C": row["past_C"],
                "included_settlements": [0, tick],
                "end_exclusive": True,
                "prefix_sha256": prefix_hash,
                "state_prefix_exact": True,
            }
        )
    old.require(len(audit) == 360, "Expected360 matched contexts")
    return np.array(joined, dtype=float), audit


def preflight():
    start = time.monotonic()
    data, original_frozen, _ = old.preflight()
    selected = np.isin(data["diagnostics"]["market"], MARKETS)
    data = {name: values[selected] for name, values in data.items()}
    check_seal(LOGS)
    exposure, joined = join_logs(data)
    names = [
        Path(__file__).resolve(),
        PLAN,
        LOGS / "tick-logs.jsonl",
        LOGS / "seal.json",
        ROOT / "scripts/probe_bidding_paired_advantage.py",
    ]
    inputs = {str(p.relative_to(ROOT)): old.digest(p) for p in names}
    frozen = {
        "parent": original_frozen,
        "inputs": inputs,
        "protocol": protocol(),
        "join_sha256": old.fingerprint(joined),
    }
    old.require(time.monotonic() - start <= 900, "S0 resource budget exceeded")
    return data, exposure, joined, frozen


class Resources(old.Resources):
    def check(self, force=False):
        if force or time.perf_counter() - self.last_scan >= 30:
            self.bytes = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
            self.last_scan = time.perf_counter()
        state = self.snapshot()
        old.require(
            state["wall_seconds"] <= 2700 and state["cpu_seconds"] <= 2700,
            "S1 wall/CPU budget exceeded",
        )
        old.require(
            state["peak_memory_bytes"] <= 4 * 2**30
            and self.bytes <= 256 * 2**20
            and state["free_disk_bytes"] >= 10 * 2**30,
            "S1 memory/disk budget exceeded",
        )

    def save(self):
        old.write_json(
            self.output / "resources.json",
            {
                "limits": protocol()["resources"],
                "records": self.records,
                "current": self.snapshot(),
            },
        )


def pseudo_values(diag, exposure, held_market):
    keys = sorted({tuple(map(int, d))[:3] for d in diag})
    donor_keys = [k for k in keys if k[0] != held_market]
    index = {tuple(map(int, d)): i for i, d in enumerate(diag)}
    rng = np.random.default_rng(20261118 + held_market)
    result, records = np.zeros(len(diag)), []
    for key in keys:
        donor = donor_keys[int(rng.integers(len(donor_keys)))]
        old.require(donor[0] != held_market, "Pseudo donor leakage")
        vector = [float(exposure[index[*donor, t]]) for t in old.TICKS]
        old.require(
            vector[0] == 0 and vector[0] <= vector[1] <= vector[2], "Invalid donor trajectory"
        )
        for tick, value in zip(old.TICKS, vector, strict=True):
            result[index[*key, tick]] = value
        records.append({"recipient": list(key), "training_donor": list(donor), "E": vector})
    return result, records


def business(data, selected):
    actual = {
        name: data[key]
        for name, key in (
            ("score", "terminal_score"),
            ("conversions", "terminal_conversions"),
            ("violation", "terminal_violation"),
        )
    }
    markets = tuple(sorted(np.unique(data["diagnostics"]["market"]).tolist()))
    result = old.scene_metrics(
        data["diagnostics"], data["baseline_index"], selected, actual, markets
    )
    ratios = [result["scenes"][s]["score"]["ratio"] for s in old.SCENES]
    old.require(all(x is not None for x in ratios), "Undefined score ratio")
    result["five_scene_macro"] = float(np.mean(ratios))
    result["switches"] = int(np.count_nonzero(selected != data["baseline_index"]))
    result["violation_counts"] = {}
    for scene_index, scene in enumerate(old.SCENES):
        rows = np.flatnonzero(data["diagnostics"]["scene_index"] == scene_index)
        result["violation_counts"][scene] = {
            "selected": int(data["terminal_violation"][rows, selected[rows]].sum()),
            "feedback": int(data["terminal_violation"][rows, data["baseline_index"][rows]].sum()),
            "context_terminals": len(rows),
        }
    return result


def fit_fold(data, exposure, held_market, output, ledger):
    diag, baseline = data["diagnostics"], data["baseline_index"]
    train = np.flatnonzero(diag["market"] != held_market)
    held = np.flatnonzero(diag["market"] == held_market)
    old.require(len(train) == 270 and len(held) == 90, "LOMO coverage mismatch")
    train_markets = diag["market"][train]
    raw = old.features(*(data[k] for k in ("states", "steps", "lengths", "baseline_index")), 1)
    train_x, held_x, normalizer = old.normalize(raw[train], raw[held], train_markets)
    utility, advantage, conversion, delta_conv = old.targets(data)
    score_scale = max(float(np.sqrt(np.mean(utility[train] ** 2))), 0.01)
    conv_scale = max(float(np.sqrt(np.mean(conversion[train] ** 2))), 0.01)
    legal = old.legal_candidates(baseline)
    guard, metadata = old.auxiliary(
        train_x,
        held_x,
        conversion[train],
        delta_conv[train],
        data["terminal_violation"][train],
        advantage[train] > 0,
        baseline[train],
        train_markets,
        legal[train],
        conv_scale,
        ledger,
        f"market{held_market}/shared_F1_guard",
    )
    guard["delta_conv"][np.arange(len(held)), baseline[held]] = 0
    fake, donors = pseudo_values(diag, exposure, held_market)
    predictions, decisions, transformed = [], [], []
    normalizers = {}
    weight = old.context_weights(train_markets)
    for name, values in zip(ARMS, (np.zeros(len(exposure)), exposure, fake), strict=True):
        etrain, eheld, estats = old.normalize(
            values[train, None], values[held, None], train_markets
        )
        xtrain, xheld = np.column_stack((train_x, etrain)), np.column_stack((held_x, eheld))
        prediction = (
            old.ridge(
                xtrain,
                xheld,
                advantage[train] / score_scale,
                weight,
                ledger,
                f"market{held_market}/{name}",
            )
            * score_scale
        )
        prediction[np.arange(len(held)), baseline[held]] = 0
        prediction_before_guard = prediction.copy()
        decision = old.choose(
            prediction,
            guard["baseline_conv"],
            guard["delta_conv"],
            guard["violation"],
            baseline[held],
        )
        predictions.append(prediction_before_guard)
        decisions.append(decision)
        transformed.append(eheld[:, 0])
        normalizers[name] = estats
    # Check that adding a zero slot preserves the exact original model specification.
    x = np.column_stack((train_x, np.ones(len(train_x))))
    penalty = np.eye(x.shape[1])
    penalty[-1, -1] = 0
    w = weight * len(train_x)
    coefficients = np.linalg.solve(
        x.T @ (w[:, None] * x) + penalty, x.T @ (w[:, None] * (advantage[train] / score_scale))
    )
    reference = np.column_stack((held_x, np.ones(len(held_x)))) @ coefficients * score_scale
    reference[np.arange(len(held)), baseline[held]] = 0
    equivalence_error = float(np.max(np.abs(predictions[0] - reference)))
    old.require(equivalence_error <= 1e-10, "Zero-slot original ridge equivalence failed")
    filename = output / f"held-{held_market}.npz"
    np.savez_compressed(
        filename,
        held_indices=held,
        train_indices=train,
        prediction=np.stack(predictions),
        selected=np.stack(decisions),
        E=exposure[held],
        pseudo_E=fake[held],
        E_standardized=np.stack(transformed),
        guard_baseline_conv=guard["baseline_conv"],
        guard_delta_conv=guard["delta_conv"],
        guard_risk=guard["violation"],
        guard_positive=guard["positive"],
    )
    old.write_json(
        output / f"held-{held_market}.json",
        {
            "held_market": held_market,
            "training_markets": sorted(np.unique(train_markets).tolist()),
            "normalizer": normalizer,
            "E_normalizers": normalizers,
            "score_scale": score_scale,
            "conv_scale": conv_scale,
            "guard": metadata,
            "guard_prediction_sha256": old.fingerprint({k: v.tolist() for k, v in guard.items()}),
            "zero_slot_equivalence_max_error": equivalence_error,
            "pseudo_donors": donors,
            "prediction_sha256": old.digest(filename),
        },
    )
    return held, np.stack(predictions), np.stack(decisions)


def summarize(data, prediction, selected):
    _, advantage, _, _ = old.targets(data)
    legal = old.legal_candidates(data["baseline_index"])
    weight = old.candidate_weights(data["diagnostics"]["market"], legal)
    metrics = {}
    for arm, name in enumerate(ARMS):
        entry = business(data, selected[arm])
        entry["physical_delta_MSE"] = float(np.sum(weight * (prediction[arm] - advantage) ** 2))
        entry["per_market"] = {}
        for market in MARKETS:
            mask = data["diagnostics"]["market"] == market
            subset = {k: v[mask] for k, v in data.items()}
            entry["per_market"][str(market)] = business(subset, selected[arm, mask])
        metrics[name] = entry
    zero, actual = metrics["zero_E"], metrics["actual_E"]
    mse_improvement = (
        None
        if zero["physical_delta_MSE"] == 0
        else 1 - actual["physical_delta_MSE"] / zero["physical_delta_MSE"]
    )
    protections = {
        s: {
            "score": actual["scenes"][s]["score"]["ratio"] >= 0.98,
            "conversions": actual["scenes"][s]["conversions"]["ratio"] >= 0.97,
            "violation": actual["violation_counts"][s]["selected"]
            <= actual["violation_counts"][s]["feedback"],
        }
        for s in old.SCENES
    }
    positive_fb = sum(actual["per_market"][str(m)]["five_scene_macro"] > 1 for m in MARKETS)
    positive_zero = sum(
        actual["per_market"][str(m)]["five_scene_macro"]
        > zero["per_market"][str(m)]["five_scene_macro"]
        for m in MARKETS
    )
    flags = {
        "mse": mse_improvement is not None and mse_improvement >= 0.05,
        "macro_FB": actual["five_scene_macro"] >= 1.005,
        "macro_zero_E": actual["five_scene_macro"] > zero["five_scene_macro"],
        "markets_FB": positive_fb >= 3,
        "markets_zero_E": positive_zero >= 3,
        "scene_protections": all(all(v.values()) for v in protections.values()),
        "nonfeedback": actual["switches"] > 0,
    }
    return {
        "status": "complete",
        "scope": protocol()["scope"],
        "arms": metrics,
        "mse_relative_improvement": mse_improvement,
        "positive_markets_FB": positive_fb,
        "positive_markets_zero_E": positive_zero,
        "scene_protections": protections,
        "flags": flags,
        "feasibility_pass": all(flags.values()),
        "full_policy_rollout": False,
        "independent_confirmation": False,
        "generative_contribution": False,
        "independent_units": "4 markets; overlapping training folds",
        "next_action": "freeze separate full-policy protocol"
        if all(flags.values())
        else "close this fixed candidate; do not widen its guard or open heldout confirmation",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--freeze")
    args = parser.parse_args()
    data, exposure, joined, frozen = preflight()
    freeze = old.fingerprint(frozen)
    print(
        json.dumps(
            {
                "preflight": {
                    "freeze": freeze,
                    "contexts": len(exposure),
                    "market_count": 4,
                    "source_prefix_join": "exact",
                    "E_range": [float(exposure.min()), float(exposure.max())],
                    "fitted": False,
                    "new_traffic": False,
                }
            }
        ),
        flush=True,
    )
    if not args.run:
        return
    old.require(args.freeze == freeze, "Run requires current reviewed freeze")
    OUTPUT.mkdir(exist_ok=False, parents=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ledger = Resources(OUTPUT)
    ledger.probe_done = True
    old.write_json(
        OUTPUT / "manifest.json",
        {"frozen_before_fit": True, "freeze": freeze, "created_utc": old.utc(), "frozen": frozen},
    )
    old.write_json(OUTPUT / "join.json", joined)
    old.write_json(OUTPUT / "protocol.json", protocol())
    prediction = np.zeros((3, len(exposure), 11))
    selected = np.zeros((3, len(exposure)), dtype=np.int64)
    try:
        for market in MARKETS:
            held, pred, actions = fit_fold(data, exposure, market, OUTPUT, ledger)
            prediction[:, held] = pred
            selected[:, held] = actions
            print(
                json.dumps({"held_market_complete": market, "resource": ledger.snapshot()}),
                flush=True,
            )
        result = summarize(data, prediction, selected)
        result["freeze"] = freeze
        old.require(
            all(old.digest(ROOT / p) == sha for p, sha in frozen["inputs"].items()),
            "New-study input/source changed during run",
        )
        old.require(
            all(old.digest(ROOT / p) == sha for p, sha in frozen["parent"]["sources"].items()),
            "Historical source changed during run",
        )
        old.require(
            all(old.digest(ROOT / p) == sha for p, sha in frozen["parent"]["inputs"].items()),
            "Historical input changed during run",
        )
        result["input_source_integrity"] = True
        old.write_json(OUTPUT / "summary.json", result)
        print(
            json.dumps(
                {
                    "result": {
                        k: result[k]
                        for k in ("feasibility_pass", "flags", "mse_relative_improvement")
                    },
                    "macros": {k: v["five_scene_macro"] for k, v in result["arms"].items()},
                }
            ),
            flush=True,
        )
    except BaseException as error:
        old.write_json(
            OUTPUT / "failure.json",
            {
                "status": "invalid_or_incomplete",
                "type": type(error).__name__,
                "message": str(error),
                "resource": ledger.snapshot(),
            },
        )
        raise
    finally:
        ledger.save()
        old.seal(OUTPUT)


if __name__ == "__main__":
    main()
