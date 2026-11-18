"""Describe current-action outcomes and subsequent fixed-feedback compensation.

Zero fitting and zero new markets. Cost/conversion differences decompose
arithmetically; nonlinear score differences do not. Cached feedback-prefix
contexts and hindsight oracles do not establish learnability or deployed gains.
"""

from __future__ import annotations

import argparse
import fcntl
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from trajectorylab import bidding_continuation_data as continuation
from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.market import ROOT, Tape, scenario_bias, score
from trajectorylab.bidding_generative_control import keep
from trajectorylab.bidding_robust import sha
from trajectorylab.utils import fingerprint, read_json, runtime_info

BASE = ROOT / "artifacts/bidding/numerical-audits"
DATA = BASE / "continuation-feasibility-v1"
HORIZON = BASE / "generative-horizon-v1"
REPLICATION = ROOT / "artifacts/bidding/generative-replication-v1"
DEFAULT_OUTPUT = BASE / "feedback-compensation-v1"
COST_TOL, UTILITY_TOL = 1e-8, 1e-12
ZERO_TYPES = (
    "same_current_outcome",
    "different_current_same_terminal_outcome",
    "different_terminal_both_zero_conversions",
    "different_terminal_same_positive_conversions_both_on_score_plateau",
    "other_zero_utility",
)
PROTOCOL = {
    "contexts": {"train": 1440, "validation": 540},
    "scenarios": continuation.SCENARIOS,
    "ticks": [0, 16, 32],
    "radius": 1.0,
    "cost_tolerance": COST_TOL,
    "utility_tolerance": UTILITY_TOL,
    "utility_partition": "positive > tolerance; strict zero == 0; numerical near tie 0 < abs(delta) <= tolerance; negative < -tolerance",
    "conversion_accounting": "integer cached terminal conversions minus integer past and current conversions",
    "terminal_cost": "recovered from original float64 normalized endpoint and float64 budget/spent; no rounding or float32 conversion",
    "score_plateau": "positive equal integer terminal conversions and cost <= CPA*conversions on both branches; no violation tolerance substituted",
    "zero_partition_priority": list(ZERO_TYPES),
    "observed_groups": {
        "remaining_budget_state_column_2": [0.25, 0.5, 0.75],
        "CPA_debt_state_column_5": [0.0],
        "ticks": [0, 16, 32],
    },
    "oracle": "radius 1; guarded terminal conversions >= .97 baseline and terminal violation <= baseline; unprotected score upper bound also reported; baseline always legal and wins exact ties; single intervention followed by feedback only",
    "scope": "descriptive fixed-feedback intervention audit; no seed duplication, IID inference, new markets, fitting, learnability or repeated-replanning claim",
    "cpu_threads": 1,
}


def prepare(output=DEFAULT_OUTPUT, protocol_path=None):
    """Read-only validation of both caches, all 22 tapes and sealed ancestors."""
    output = Path(output).resolve()
    if protocol_path is None or not Path(protocol_path).is_file():
        raise RuntimeError("Provide the registered protocol through --protocol")
    protocol_path = Path(protocol_path)
    if not output.is_relative_to(BASE) or any(
        output.is_relative_to(p) or p.is_relative_to(output) for p in (DATA, HORIZON)
    ):
        raise RuntimeError("Use a separate numerical-audit output directory")
    manifest = read_json(DATA / "manifest.json")
    dc = manifest["core"]
    if manifest["identity"] != fingerprint(dc) or read_json(DATA / "identity.json") != {
        "identity": manifest["identity"],
        "core": dc,
    }:
        raise RuntimeError("Continuation manifest/core identity changed")
    main = ROOT / "artifacts/bidding/main-v2"
    original = read_json(main / "manifest.json")
    config = original["config"]
    if (
        fingerprint(provenance(config)) != original["identity"]
        or sha(main / "manifest.json") != dc["main_manifest_sha256"]
        or original["identity"] != dc["main_identity"]
    ):
        raise RuntimeError("Original simulation provenance changed")
    actions = np.asarray(config["study"]["actions"], dtype=np.float64)
    markets = {"train": list(range(1001, 1017)), "validation": list(range(2001, 2007))}
    if any(config["study"][split + "_markets"] != values for split, values in markets.items()):
        raise RuntimeError("Only the original 22 market tapes may be used")
    paths = {
        split: [main / "markets" / f"{m}.npz" for m in values] for split, values in markets.items()
    }
    current = continuation.identity(
        [p for group in paths.values() for p in group],
        actions,
        continuation.SCENARIOS,
        [0, 16, 32],
        24,
    )
    if current != {k: dc[k] for k in current} or dc["expected_contexts"] != PROTOCOL["contexts"]:
        raise RuntimeError("Continuation sources, auction, market bytes or protocol changed")
    files = [
        Path(__file__).resolve(),
        DATA / "manifest.json",
        DATA / "identity.json",
        main / "manifest.json",
    ]
    datasets = {}
    for split, count in PROTOCOL["contexts"].items():
        path, stamp_path = DATA / f"{split}.npz", DATA / f"{split}.identity.json"
        stamp, metadata = read_json(stamp_path), manifest["datasets"][split]
        if (
            stamp
            != {
                "identity": fingerprint({"identity": manifest["identity"], "split": split}),
                "npz_sha256": sha(path),
            }
            or metadata["npz_sha256"] != sha(path)
            or metadata["identity_sha256"] != sha(stamp_path)
            or metadata["contexts"] != count
        ):
            raise RuntimeError("Continuation cache/stamp/manifest SHA differs: " + split)
        with np.load(path, allow_pickle=False) as cache:
            data = {k: cache[k] for k in cache.files}
        continuation.validate(data)
        expected = {
            (m, si, f, t)
            for m in markets[split]
            for si in range(5)
            for f in config["study"]["focal_agents"]
            for t in (0, 16, 32)
        }
        actual = [
            tuple(int(d[k]) for k in ("market", "scene_index", "focal", "tick"))
            for d in data["diagnostics"]
        ]
        if (
            len(actual) != count
            or len(set(actual)) != count
            or set(actual) != expected
            or data["endpoints"].dtype != np.float64
            or data["spent"].dtype != np.float64
            or not np.issubdtype(data["terminal_conversions"].dtype, np.integer)
        ):
            raise RuntimeError("Exact context coverage or native accounting dtype differs")
        datasets[split] = data
        files += [path, stamp_path]
    hm, seal = read_json(HORIZON / "manifest.json"), read_json(HORIZON / "seal.json")
    if (
        hm["identity"] != fingerprint(hm["core"])
        or seal["identity"] != hm["identity"]
        or set(seal["files"]) != {"manifest.json", "contexts.json", "summary.json"}
    ):
        raise RuntimeError("Horizon manifest or complete seal differs")
    for name, digest in seal["files"].items():
        if sha(HORIZON / name) != digest:
            raise RuntimeError("Sealed horizon artifact changed")
        files.append(HORIZON / name)
    for name, digest in hm["core"]["sources"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Frozen horizon source changed: " + name)
    for name, digest in hm["core"]["inputs"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Frozen horizon input changed: " + name)
        files.append(ROOT / name)
    horizon = read_json(HORIZON / "contexts.json")
    if (
        horizon["identity"] != hm["identity"]
        or read_json(HORIZON / "summary.json")["identity"] != hm["identity"]
        or len(horizon["records"]) != 1080
    ):
        raise RuntimeError("Expected both frozen seeds on 540 validation contexts")
    records = read_json(REPLICATION / "final-training.json")
    if {r["seed"] for r in records} != {11, 23} or len(records) != 2:
        raise RuntimeError("Exactly two completed Flow seed records are required")
    for record in records:
        path = ROOT / record["best_path"]
        if (
            record["family"] != "flow"
            or sha(path) != record["best_sha256"]
            or (record["seed"] == 23 and record["fit"]["steps"] != 15000)
        ):
            raise RuntimeError("Frozen Flow training record/weights changed")
        files.append(path)
    for seed in (11, 23):
        rows = [r for r in horizon["records"] if r["model_seed"] == seed]
        if len(rows) != 540 or {r["context_index"] for r in rows} != set(range(540)):
            raise RuntimeError("Frozen horizon seed/context coverage differs")
        for r in rows:
            i, d = r["context_index"], datasets["validation"]["diagnostics"][r["context_index"]]
            if (r["market"], r["scenario"], r["focal"], r["tick"], r["baseline_index"]) != (
                int(d["market"]),
                continuation.SCENARIOS[int(d["scene_index"])],
                int(d["focal"]),
                int(d["tick"]),
                int(datasets["validation"]["baseline_index"][i]),
            ):
                raise RuntimeError("Horizon and continuation context keys differ")
    files += [HORIZON / "seal.json", REPLICATION / "final-training.json"]
    for group in paths.values():
        files += [p for path in group for p in (path, path.with_suffix(".json"))]
    core = {
        "protocol": PROTOCOL,
        "protocol_document": {"name": protocol_path.name, "sha256": sha(protocol_path)},
        "sources": continuation.sources()
        | hm["core"]["sources"]
        | {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve())},
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        "continuation_identity": manifest["identity"],
        "horizon_identity": hm["identity"],
        "runtime": runtime_info(),
        "output": str(output.relative_to(ROOT)),
    }
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != {
        "identity": fingerprint(core),
        "core": core,
    }:
        raise RuntimeError("Existing audit identity differs; cannot overwrite")
    return core, actions, paths, datasets, horizon["records"], output


def signs(values, tolerance):
    values = np.asarray(values)
    return np.where(np.abs(values) <= tolerance, 0, np.sign(values)).astype(int)


def decompose(data, i, split, tape, actions):
    d = data["diagnostics"][i]
    market, si, focal, tick = (int(d[k]) for k in ("market", "scene_index", "focal", "tick"))
    fi = int(np.flatnonzero(tape.arrays["focals"] == focal)[0])
    budget, cpa, spent = (float(data[k][i]) for k in ("budget", "cpa", "spent"))
    past, b = int(data["conversions"][i]), int(data["baseline_index"][i])
    state = data["states"][i, int(data["lengths"][i]) - 1]
    if abs(float(state[0]) * 48 - tick) > 1e-5:
        raise RuntimeError("Observable tick and cached tick differ")
    current = [
        tape.settle(
            fi, tick, float(a) * scenario_bias(continuation.SCENARIOS[si], tick), budget - spent
        )
        for a in actions
    ]
    cc = np.asarray([o[0] for o in current], dtype=np.float64)
    cv = np.asarray([o[1] for o in current], dtype=np.int64)
    tc = spent + data["endpoints"][i, :, 0] * budget
    tv = data["terminal_conversions"][i].astype(np.int64, copy=False)
    fc, fv = tc - spent - cc, tv - past - cv
    if (
        np.any(cc < 0)
        or np.any(cv < 0)
        or np.any(fc < -COST_TOL)
        or np.any(fv < 0)
        or np.any(spent + cc > budget + COST_TOL)
        or np.any(tc > budget + COST_TOL)
    ):
        raise RuntimeError("Negative remaining feedback outcome or current budget excess")
    dc, dv, dtc, dtv, dfc, dfv = (
        cc - cc[b],
        cv - cv[b],
        tc - tc[b],
        tv - tv[b],
        fc - fc[b],
        fv - fv[b],
    )
    cost_residual = float(np.max(np.abs(dtc - dc - dfc)))
    conversion_residual = int(np.max(np.abs(dtv - dv - dfv)))
    if cost_residual > COST_TOL or conversion_residual != 0:
        raise RuntimeError("Terminal delta must equal current plus subsequent feedback delta")
    utility = data["terminal_score"][i]
    recalculated_score = np.asarray([score(c, int(v), cpa) for c, v in zip(tc, tv, strict=True)])
    score_residual = float(np.max(np.abs(recalculated_score - utility)))
    recalculated_violation = tc > cpa * tv + COST_TOL
    violation_mismatches = int(
        np.count_nonzero(recalculated_violation != data["terminal_violation"][i])
    )
    if score_residual > UTILITY_TOL or violation_mismatches:
        raise RuntimeError("Reconstructed terminal score or violation differs from frozen cache")
    same_current_exact = (dc == 0) & (dv == 0)
    same_terminal_exact = (dtc == 0) & (dtv == 0)
    if np.any(same_current_exact & (~same_terminal_exact | (utility != utility[b]))):
        raise RuntimeError(
            "Identical current settlement must yield identical feedback terminal outcome"
        )
    instantaneous = np.array(
        [score(spent + c, past + int(v), cpa) for c, v in zip(cc, cv, strict=True)]
    )
    reward_delta, utility_delta = instantaneous - instantaneous[b], utility - utility[b]
    categories, equality = [], []
    for ai in range(len(actions)):
        equality.append(None)
        if utility_delta[ai] != 0:
            categories.append(None)
        elif abs(dc[ai]) <= COST_TOL and dv[ai] == 0:
            categories.append(ZERO_TYPES[0])
            equality[-1] = "exact" if dc[ai] == 0 else "cost_tolerance_only"
        elif abs(dtc[ai]) <= COST_TOL and dtv[ai] == 0:
            categories.append(ZERO_TYPES[1])
            equality[-1] = "exact" if dtc[ai] == 0 else "cost_tolerance_only"
        elif tv[ai] == 0 and tv[b] == 0:
            categories.append(ZERO_TYPES[2])
        elif tv[ai] == tv[b] and tv[ai] > 0 and tc[ai] <= cpa * tv[ai] and tc[b] <= cpa * tv[b]:
            categories.append(ZERO_TYPES[3])
        else:
            categories.append(ZERO_TYPES[4])
    radius = np.abs(actions - actions[b]) <= 1.0 + UTILITY_TOL
    mask = radius.copy()
    mask[b] = False
    eligible = (
        radius
        & (tv >= 0.97 * tv[b])
        & (data["terminal_violation"][i] <= data["terminal_violation"][i, b])
    )
    eligible[b] = True
    candidates = np.r_[b, np.flatnonzero(eligible & mask)]
    oracle = int(candidates[np.argmax(utility[candidates])])
    candidates = np.r_[b, np.flatnonzero(mask)]
    unprotected = int(candidates[np.argmax(utility[candidates])])
    budget_group = (
        "<=.25"
        if state[2] <= 0.25
        else "(.25,.5]"
        if state[2] <= 0.5
        else "(.5,.75]"
        if state[2] <= 0.75
        else ">.75"
    )
    return {
        "split": split,
        "context_index": i,
        "market": market,
        "scenario": continuation.SCENARIOS[si],
        "focal": focal,
        "tick": tick,
        "baseline_index": b,
        "budget_group": budget_group,
        "debt_group": "<=0" if state[5] <= 0 else ">0",
        "within_radius_nonbaseline": mask.tolist(),
        "spent": spent,
        "past_conversions": past,
        "current_cost": cc.tolist(),
        "current_conversions": cv.tolist(),
        "terminal_cost": tc.tolist(),
        "terminal_conversions": tv.tolist(),
        "future_cost": fc.tolist(),
        "future_conversions": fv.tolist(),
        "delta_current_cost": dc.tolist(),
        "delta_current_conversions": dv.tolist(),
        "delta_terminal_cost": dtc.tolist(),
        "delta_terminal_conversions": dtv.tolist(),
        "delta_future_cost": dfc.tolist(),
        "delta_future_conversions": dfv.tolist(),
        "current_reward_score_delta": reward_delta.tolist(),
        "terminal_utility_delta": utility_delta.tolist(),
        "terminal_utility_sign": signs(utility_delta, UTILITY_TOL).tolist(),
        "terminal_utility_class": [
            "strict_zero"
            if x == 0
            else "numeric_near_tie"
            if abs(x) <= UTILITY_TOL
            else "positive"
            if x > 0
            else "negative"
            for x in utility_delta
        ],
        "zero_utility_type": categories,
        "zero_equality_kind": equality,
        "numeric_utility_near_tie": (
            (np.abs(utility_delta) <= UTILITY_TOL) & (utility_delta != 0)
        ).tolist(),
        "CPA_boundary_near_tie": (np.abs(tc - cpa * tv) <= COST_TOL).tolist(),
        "current_future_cost_sign_reversal": (
            signs(dc, COST_TOL) * signs(dfc, COST_TOL) == -1
        ).tolist(),
        "current_future_conversion_sign_reversal": (signs(dv, 0) * signs(dfv, 0) == -1).tolist(),
        "current_terminal_reward_sign_reversal": (
            signs(reward_delta, UTILITY_TOL) * signs(utility_delta, UTILITY_TOL) == -1
        ).tolist(),
        "current_terminal_cost_sign_reversal": (
            signs(dc, COST_TOL) * signs(dtc, COST_TOL) == -1
        ).tolist(),
        "current_terminal_conversion_sign_reversal": (signs(dv, 0) * signs(dtv, 0) == -1).tolist(),
        "exact_cost_cancellation": ((dc != 0) & (dtc == 0)).tolist(),
        "exact_conversion_cancellation": ((dv != 0) & (dtv == 0)).tolist(),
        "same_current_exact": same_current_exact.tolist(),
        "same_current_cost_tolerance_only": (
            (np.abs(dc) <= COST_TOL) & (dc != 0) & (dv == 0)
        ).tolist(),
        "same_terminal_exact": same_terminal_exact.tolist(),
        "accounting_checks": {
            "max_abs_cost_delta_residual": cost_residual,
            "max_abs_integer_conversion_delta_residual": conversion_residual,
            "max_abs_recomputed_score_residual": score_residual,
            "recomputed_violation_mismatches": violation_mismatches,
            "min_current_cost": float(cc.min()),
            "min_future_cost": float(fc.min()),
            "min_current_conversions": int(cv.min()),
            "min_future_conversions": int(fv.min()),
            "max_budget_excess": float(np.max(tc - budget)),
            "same_current_exact_cells": int(same_current_exact.sum()),
            "same_current_terminal_mismatches": int(
                np.count_nonzero(same_current_exact & ~same_terminal_exact)
            ),
        },
        "oracle_index": oracle,
        "oracle_headroom": float(utility[oracle] - utility[b]),
        "unprotected_oracle_index": unprotected,
        "unprotected_oracle_headroom": float(utility[unprotected] - utility[b]),
        "oracle_only": True,
    }


def cell_summary(rows, masks):
    n, directions, types, equality = 0, Counter(), Counter(), Counter()
    values = {
        k: []
        for k in (
            "delta_current_cost",
            "delta_current_conversions",
            "delta_terminal_cost",
            "delta_terminal_conversions",
            "delta_future_cost",
            "delta_future_conversions",
            "current_reward_score_delta",
            "terminal_utility_delta",
        )
    }
    flags = Counter()
    for row, mask in zip(rows, masks, strict=True):
        for ai in np.flatnonzero(mask):
            n += 1
            directions[row["terminal_utility_class"][ai]] += 1
            if row["zero_utility_type"][ai] is not None:
                types[row["zero_utility_type"][ai]] += 1
                if row["zero_equality_kind"][ai] is not None:
                    equality[
                        row["zero_utility_type"][ai] + "/" + row["zero_equality_kind"][ai]
                    ] += 1
            for key in values:
                values[key].append(row[key][ai])
            for key in (
                "numeric_utility_near_tie",
                "CPA_boundary_near_tie",
                "current_future_cost_sign_reversal",
                "current_future_conversion_sign_reversal",
                "current_terminal_reward_sign_reversal",
                "current_terminal_cost_sign_reversal",
                "current_terminal_conversion_sign_reversal",
                "exact_cost_cancellation",
                "exact_conversion_cancellation",
                "same_current_exact",
                "same_current_cost_tolerance_only",
                "same_terminal_exact",
            ):
                flags[key] += int(row[key][ai])
    if sum(types.values()) != directions["strict_zero"] or sum(directions.values()) != n:
        raise RuntimeError("Utility partition is not exhaustive and mutually exclusive")
    return {
        "contexts": len(rows),
        "contexts_with_selected_cells": sum(bool(np.any(m)) for m in masks),
        "action_cell_denominator": n,
        "terminal_utility_counts": {
            key: directions[key]
            for key in ("positive", "strict_zero", "numeric_near_tie", "negative")
        },
        "zero_utility_denominator": directions["strict_zero"],
        "zero_utility_types": {key: types[key] for key in ZERO_TYPES},
        "zero_equality_counts": {
            key + "/" + kind: equality[key + "/" + kind]
            for key in ZERO_TYPES[:2]
            for kind in ("exact", "cost_tolerance_only")
        },
        "flags": dict(flags),
        "mean_deltas": {key: float(np.mean(v)) if v else None for key, v in values.items()},
        "current_reward_vs_terminal_direction": {
            f"{a}/{b}": sum(
                int(signs([r["current_reward_score_delta"][ai]], UTILITY_TOL)[0]) == a
                and r["terminal_utility_sign"][ai] == b
                for r, mask in zip(rows, masks, strict=True)
                for ai in np.flatnonzero(mask)
            )
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
        },
        "current_vs_future_direction": {
            metric: {
                f"{a}/{b}": sum(
                    int(
                        signs([r["delta_current_" + metric][ai]], tol)[0] == a
                        and signs([r["delta_future_" + metric][ai]], tol)[0] == b
                    )
                    for r, mask in zip(rows, masks, strict=True)
                    for ai in np.flatnonzero(mask)
                )
                for a in (-1, 0, 1)
                for b in (-1, 0, 1)
            }
            for metric, tol in (("cost", COST_TOL), ("conversions", 0))
        },
    }


def grouped(rows, masks=None, oracle=True):
    if masks is None:
        masks = [np.asarray(r["within_radius_nonbaseline"], dtype=bool) for r in rows]
    groups = {"overall": list(range(len(rows)))}
    for i, row in enumerate(rows):
        keys = [
            f"scenario/{row['scenario']}",
            f"tick/{row['tick']}",
            f"market/{row['market']}",
            f"budget/{row['budget_group']}",
            f"debt/{row['debt_group']}",
            f"observable/{row['budget_group']}/{row['debt_group']}/tick{row['tick']}",
            f"scenario/{row['scenario']}/tick/{row['tick']}",
        ]
        for key in keys:
            groups.setdefault(key, []).append(i)
    result = {}
    for key, indices in groups.items():
        selected = [rows[i] for i in indices]
        result[key] = cell_summary(selected, [masks[i] for i in indices])
        if oracle:
            for label, field in (
                ("guarded_oracle_only", "oracle_headroom"),
                ("unprotected_single_intervention_upper_bound", "unprotected_oracle_headroom"),
            ):
                result[key][label] = {
                    "mean_headroom": float(np.mean([r[field] for r in selected])),
                    "positive_contexts": sum(r[field] > UTILITY_TOL for r in selected),
                    "contexts": len(selected),
                    "by_market": {
                        str(m): {
                            "mean_headroom": float(
                                np.mean([r[field] for r in selected if r["market"] == m])
                            ),
                            "positive_contexts": sum(
                                r[field] > UTILITY_TOL for r in selected if r["market"] == m
                            ),
                            "contexts": sum(r["market"] == m for r in selected),
                        }
                        for m in sorted({r["market"] for r in selected})
                    },
                }
    return result


def run(output=DEFAULT_OUTPUT, protocol_path=None):
    torch.set_num_threads(1)
    print("feedback compensation preflight", flush=True)
    core, actions, paths, datasets, horizon, output = prepare(output, protocol_path)
    identity = fingerprint(core)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        keep(output / "manifest.json", {"identity": identity, "core": core})
        if (output / "seal.json").exists():
            seal = read_json(output / "seal.json")
            if (
                seal["identity"] != identity
                or set(seal["files"]) != {"manifest.json", "contexts.json", "summary.json"}
                or any(sha(output / name) != digest for name, digest in seal["files"].items())
            ):
                raise RuntimeError("Sealed feedback-compensation output changed")
            return output
        rows = []
        for split, data in datasets.items():
            print(
                f"feedback compensation split={split} contexts={len(data['lengths'])}", flush=True
            )
            by_market = {int(p.stem): p for p in paths[split]}
            for market, path in by_market.items():
                tape = Tape(path)
                for i in np.flatnonzero(data["diagnostics"]["market"] == market):
                    rows.append(decompose(data, int(i), split, tape, actions))
                print(
                    f"feedback compensation market={market} completed_contexts={len(rows)}",
                    flush=True,
                )
        annotations = {}
        validation = sorted(
            [r for r in rows if r["split"] == "validation"], key=lambda r: r["context_index"]
        )
        for seed in (11, 23):
            hr = {r["context_index"]: r for r in horizon if r["model_seed"] == seed}
            choices, predictions = [], []
            for row in validation:
                h = hr[row["context_index"]]
                mask = np.asarray(row["within_radius_nonbaseline"], dtype=bool)
                choice = np.zeros(len(actions), dtype=bool)
                choice[int(h["choices"]["expected"])] = True
                choices.append(choice & mask)
                utility = np.asarray(h["predictions"]["M"]["utility"])
                predictions.append(mask & (utility - utility[row["baseline_index"]] > UTILITY_TOL))
            annotations[str(seed)] = {
                "expected_nonbaseline_choices": grouped(validation, choices, oracle=False),
                "model_predicted_positive_action_cells": grouped(
                    validation, predictions, oracle=False
                ),
                "expected_baseline_fallback_contexts": sum(
                    hr[r["context_index"]]["choices"]["expected"] == r["baseline_index"]
                    for r in validation
                ),
                "scope": "annotations of the same 540 validation contexts; not additional observations or newly sampled predictions",
            }
        accounting = [r["accounting_checks"] for r in rows]
        summary = {
            "identity": identity,
            "protocol": PROTOCOL,
            "all_contexts": grouped(rows),
            "by_split": {
                split: grouped([r for r in rows if r["split"] == split]) for split in datasets
            },
            "horizon_annotations_validation_only": annotations,
            "checks": {
                "contexts": len(rows),
                "unique_contexts_without_seed_duplication": len(
                    {(r["split"], r["market"], r["scenario"], r["focal"], r["tick"]) for r in rows}
                ),
                "max_abs_cost_delta_residual": max(
                    a["max_abs_cost_delta_residual"] for a in accounting
                ),
                "max_abs_integer_conversion_delta_residual": max(
                    a["max_abs_integer_conversion_delta_residual"] for a in accounting
                ),
                "max_abs_recomputed_score_residual": max(
                    a["max_abs_recomputed_score_residual"] for a in accounting
                ),
                "recomputed_violation_mismatches": sum(
                    a["recomputed_violation_mismatches"] for a in accounting
                ),
                "same_current_exact_cells_including_baseline": sum(
                    a["same_current_exact_cells"] for a in accounting
                ),
                "same_current_terminal_mismatches": sum(
                    a["same_current_terminal_mismatches"] for a in accounting
                ),
                "min_current_cost": min(a["min_current_cost"] for a in accounting),
                "min_future_cost": min(a["min_future_cost"] for a in accounting),
                "min_current_conversions": min(a["min_current_conversions"] for a in accounting),
                "min_future_conversions": min(a["min_future_conversions"] for a in accounting),
                "max_budget_excess": max(a["max_budget_excess"] for a in accounting),
            },
            "interpretation": "Only cost and integer conversion deltas decompose additively. Immediate and terminal score signs are compared, not added. Fixed causal state groups are descriptive; simulator bias and exact outcomes are offline diagnostics, never model features. Guarded hindsight headroom and compensation do not establish causal learnability, new-market generalization, deployed gains or safety under repeated replanning.",
            "generative_value_confirmed": False,
            "authorizes_fitting": False,
        }
        if prepare(output, protocol_path)[0] != core:
            raise RuntimeError("Frozen inputs changed during feedback-compensation audit")
        keep(output / "contexts.json", {"identity": identity, "records": rows})
        keep(output / "summary.json", summary)
        if prepare(output, protocol_path)[0] != core:
            raise RuntimeError("Frozen inputs changed before output seal")
        keep(
            output / "seal.json",
            {
                "identity": identity,
                "files": {
                    name: sha(output / name)
                    for name in ("manifest.json", "contexts.json", "summary.json")
                },
            },
        )
        print(f"feedback compensation complete contexts={len(rows)}", flush=True)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.output, args.protocol))


if __name__ == "__main__":
    main()
