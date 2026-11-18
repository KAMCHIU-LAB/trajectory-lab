"""Zero-fit, fixed-prefix horizon diagnostics; no receding-policy business claim.

M: model plus proportional budget projection; P: true unbudgeted constant-action
curve plus that projection; C: exact hard-budget constant action; F: cached one
action followed by feedback. Every comparison uses its own a-minus-b endpoint.
"""

from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from trajectorylab import bidding_continuation_data as continuation
from trajectorylab import bidding_generative as gen
from trajectorylab import bidding_generative_replication as replication
from trajectorylab.bidding.market import (
    ROOT,
    Tape,
    feedback_action,
    scenario_bias,
    source_identity,
)
from trajectorylab.bidding_generative_control import keep
from trajectorylab.bidding_generative_learning import BundleSampler
from trajectorylab.bidding_robust import sha
from trajectorylab.utils import fingerprint, read_json, runtime_info, write_json

DATA = ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1"
REPLICATION = ROOT / "artifacts/bidding/generative-replication-v1"
DEFAULT_OUTPUT = ROOT / "artifacts/bidding/numerical-audits/generative-horizon-v1"
SCENARIOS = continuation.SCENARIOS
TICKS = [0, 16, 32]
EPS = 1e-12
PROTOCOL = {
    "seeds": [11, 23],
    "contexts": 540,
    "ticks": TICKS,
    "scenarios": SCENARIOS,
    "samples": 64,
    "integration_steps": 8,
    "radius": 1.0,
    "quantile": 0.5,
    "cpu_threads": 1,
    "sampling_seed": 91347,
    "randomness": "default_rng(SeedSequence([91347, context_index])); 32 normals plus antithetic negatives; 64 stratified uniforms shuffled; action shuffle uses SeedSequence([91347, context_index, 1]); shared across model seeds",
    "conditioning": "states only; restore absolute positions with excluded leading zeros; baseline from fixed causal Feedback(gain=.3,prior=8)",
    "accounting": "all M/P/C/F endpoints use cached float64 spent and integer conversions",
    "mean_endpoint": "same expected eligibility mask; only argmax utility changes to S(mean projected cost,mean projected conversions)",
    "rank_ties": "within-context unordered action pairs; score 1 for equal signs including two ties, .5 for one tie, 0 for opposite signs; tolerance 1e-12",
    "scope": "descriptive correlated-context diagnosis; no IID inference, new markets, fitting or repeated replanning",
}


def prepare(output=DEFAULT_OUTPUT, protocol_path=None):
    """Read-only provenance check; never calls a trajectory builder or fitter."""
    output = Path(output).resolve()
    if protocol_path is None or not Path(protocol_path).is_file():
        raise RuntimeError("Provide the registered protocol document explicitly")
    protocol_path = Path(protocol_path)
    if (
        output.is_relative_to(DATA)
        or DATA.is_relative_to(output)
        or output.is_relative_to(REPLICATION)
        or not output.is_relative_to(ROOT / "artifacts/bidding/numerical-audits")
    ):
        raise RuntimeError("Use a separate numerical-audit output directory")
    dm = read_json(DATA / "manifest.json")
    dc = dm["core"]
    if (
        dm["identity"] != fingerprint(dc)
        or dc["sources"] != continuation.sources()
        or dc["auction"] != source_identity()
        or dc["ticks"] != TICKS
        or dc["scenarios"] != SCENARIOS
    ):
        raise RuntimeError("Continuation source or protocol provenance changed")
    stamp = read_json(DATA / "validation.identity.json")
    dataset = dm["datasets"]["validation"]
    if (
        stamp
        != {
            "identity": fingerprint({"identity": dm["identity"], "split": "validation"}),
            "npz_sha256": sha(DATA / "validation.npz"),
        }
        or dataset["npz_sha256"] != stamp["npz_sha256"]
        or dataset["identity_sha256"] != sha(DATA / "validation.identity.json")
        or dataset["contexts"] != 540
    ):
        raise RuntimeError("Continuation validation bytes or identity changed")
    rm = read_json(REPLICATION / "manifest.json")
    rc = rm["core"]
    if rm["identity"] != fingerprint(rc) or rc["sources"] != replication.sources():
        raise RuntimeError("Frozen replication implementation changed")
    result = read_json(REPLICATION / "summary.json")
    if (
        result["identity"] != rm["identity"]
        or result["stop_seed"] != 23
        or not result["stopped"]
        or result["business_replicated"]
        or result["generative_value_confirmed"]
    ):
        raise RuntimeError("Only the completed failed seed-23 replication may be diagnosed")
    main = ROOT / rc["config"]["main_run"]
    main_manifest = read_json(main / "manifest.json")
    actions = np.asarray(main_manifest["config"]["study"]["actions"], dtype=np.float64)
    if actions.tolist() != dc["actions"] or main_manifest["config"]["study"][
        "validation_markets"
    ] != list(range(2001, 2007)):
        raise RuntimeError("Original development market/action grid changed")
    files = [
        Path(__file__).resolve(),
        DATA / "manifest.json",
        DATA / "identity.json",
        DATA / "validation.npz",
        DATA / "validation.identity.json",
        REPLICATION / "manifest.json",
        REPLICATION / "summary.json",
        REPLICATION / "final-training.json",
        main / "manifest.json",
    ]
    for name, digest in (rc["frozen_files"] | rc["relative_core"]["frozen_files"]).items():
        path = ROOT / name
        if sha(path) != digest:
            raise RuntimeError("Frozen ancestor file changed: " + name)
        files.append(path)
    paths = [main / "markets" / f"{market}.npz" for market in range(2001, 2007)]
    for path in paths:
        actual = {"npz_sha256": sha(path), "metadata_sha256": sha(path.with_suffix(".json"))}
        if (
            actual != dc["inputs"][str(path.resolve())]
            or {"npz": actual["npz_sha256"], "metadata": actual["metadata_sha256"]}
            != rc["inputs"][path.stem]
        ):
            raise RuntimeError("Development market bytes changed")
        files += [path, path.with_suffix(".json")]
    models = {}
    for seed in (11, 23):
        records = [r for r in read_json(REPLICATION / "final-training.json") if r["seed"] == seed]
        if len(records) != 1:
            raise RuntimeError("Exactly one completed Flow per registered seed is required")
        record = records[0]
        path = ROOT / record["best_path"]
        if (
            sha(path) != record["best_sha256"]
            or record["family"] != "flow"
            or (seed == 23 and record["fit"]["steps"] != 15000)
        ):
            raise RuntimeError("Frozen Flow checkpoint differs")
        if seed == 23 and sha(path.parent / "checkpoint.pt") != record["checkpoint_sha256"]:
            raise RuntimeError("Completed Flow training checkpoint changed")
        seal_dir = REPLICATION / "evaluation" / f"seed-{seed}"
        seal = read_json(seal_dir / "artifacts.json")
        if seal["identity"] != rm["identity"] or seal["weights"] != record["best_sha256"]:
            raise RuntimeError("Frozen seed evaluation identity changed")
        for name, digest in seal["files"].items():
            if sha(seal_dir / name) != digest:
                raise RuntimeError("Frozen seed evaluation bytes changed")
            files.append(seal_dir / name)
        files += [path, seal_dir / "artifacts.json"]
        if seed == 23:
            files.append(path.parent / "checkpoint.pt")
        models[seed] = record
    old_cache = (
        ROOT
        / rc["relative_core"]["config"]["parent_run"]
        / "data-cache/validation/trajectories.npz"
    )
    old_stamp = old_cache.parent / "identity.json"
    if read_json(old_stamp)["npz_sha256"] != sha(old_cache):
        raise RuntimeError("Old constant-action labels changed")
    files += [old_cache, old_stamp]
    core = {
        "protocol_document": {"name": protocol_path.name, "sha256": sha(protocol_path)},
        "focals": main_manifest["config"]["study"]["focal_agents"],
        "protocol": PROTOCOL,
        "sources": replication.sources()
        | continuation.sources()
        | {str(Path(__file__).resolve().relative_to(ROOT)): sha(Path(__file__).resolve())},
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        "continuation_identity": dm["identity"],
        "replication_identity": rm["identity"],
        "runtime": runtime_info(),
        "output": str(output.relative_to(ROOT)),
    }
    return core, actions, paths, models, old_cache, output


def endpoint(cost, conv, cpa):
    cost, conv = np.asarray(cost), np.asarray(conv)
    return {
        "cost": cost,
        "conv": conv,
        "utility": gen.expected_score(cost, conv, cpa),
        "violation": (cost > cpa * conv + 1e-8).astype(float),
    }


def project(bundles, spent, conv, budget, cpa):
    future_cost, future_conv = bundles[:, 2] * budget, bundles[:, 3] * budget / cpa
    remaining = max(0.0, budget - spent)
    fraction = np.minimum(1.0, remaining / np.maximum(future_cost, EPS))
    return endpoint(spent + np.minimum(future_cost, remaining), conv + future_conv * fraction, cpa)


def means(prediction):
    return {k: v.mean(axis=0) for k, v in prediction.items()}


def choose(prediction, baseline, actions, paired=None, ranking=None):
    p = means(prediction)
    eligible = (
        (np.abs(actions - actions[baseline]) <= 1.0 + EPS)
        & (p["utility"] > p["utility"][baseline] + EPS)
        & (p["conv"] >= 0.97 * p["conv"][baseline] - EPS)
        & (p["violation"] <= p["violation"][baseline] + 0.03 + EPS)
    )
    if paired is not None:
        eligible &= np.quantile(paired - paired[:, baseline, None], 0.5, axis=0) >= -EPS
    candidates = np.r_[baseline, np.flatnonzero(eligible)]
    utility = p["utility"] if ranking is None else ranking
    return int(candidates[np.argmax(utility[candidates])]) if len(
        candidates
    ) else baseline, eligible


def truth(data, i, tape, fi, scene, actions, old_labels):
    tick = int(data["diagnostics"][i]["tick"])
    spent, past = float(data["spent"][i]), int(data["conversions"][i])
    budget, cpa = float(data["budget"][i]), float(data["cpa"][i])
    raw_cost, raw_conv, costs, conversions = [], [], [], []
    for action in actions:
        cost, conv, unbudgeted_cost, unbudgeted_conv = spent, past, 0.0, 0
        for future in range(tick, 48):
            effective = float(action) * scenario_bias(scene, future)
            index = tape.index(effective)
            unbudgeted_cost += float(tape.arrays["costs"][fi, future, index])
            unbudgeted_conv += int(tape.arrays["conversions"][fi, future, index])
            dc, dv, _, _ = tape.settle(fi, future, effective, budget - cost)
            cost, conv = cost + dc, conv + dv
        raw_cost.append(unbudgeted_cost)
        raw_conv.append(unbudgeted_conv)
        costs.append(cost)
        conversions.append(conv)
    bundles = np.zeros((1, 4, len(actions)))
    bundles[0, 2] = np.asarray(raw_cost) / budget
    bundles[0, 3] = np.asarray(raw_conv) * cpa / budget
    if not np.allclose(bundles[0, 2:], old_labels[2:], rtol=2e-6, atol=2e-7):
        raise RuntimeError("Old nominal-action labels differ from explicit future tape sums")
    p = means(project(bundles, spent, past, budget, cpa))
    c = endpoint(np.asarray(costs), np.asarray(conversions), cpa)
    fcost = spent + data["endpoints"][i, :, 0] * budget
    f = endpoint(fcost, data["terminal_conversions"][i], cpa)
    if (
        not np.allclose(f["utility"], data["terminal_score"][i], rtol=0, atol=1e-10)
        or not np.array_equal(f["violation"], data["terminal_violation"][i])
        or np.any(np.asarray(costs) > budget + 1e-6)
    ):
        raise RuntimeError("Cached continuation or exact hard-budget accounting differs")
    return {"P": p, "C": c, "F": f}


def pair_metrics(records, left, right):
    errors, predicted, actual, risk_errors, rank, ties = [], [], [], [], [], 0
    event_errors, coverage80, coverage95, below50 = [], [], [], []
    for r in records:
        mask, b = np.asarray(r["within_radius_nonbaseline"]), r["baseline_index"]
        x, y = r["predictions"][left], r["predictions"][right]
        xd, yd = (
            np.asarray(x["utility"]) - x["utility"][b],
            np.asarray(y["utility"]) - y["utility"][b],
        )
        errors.extend((xd - yd)[mask])
        predicted.extend(xd[mask])
        actual.extend(yd[mask])
        risk_errors.extend(
            (
                (np.asarray(x["violation"]) - x["violation"][b])
                - (np.asarray(y["violation"]) - y["violation"][b])
            )[mask]
        )
        event_errors.extend((np.asarray(x["violation"]) - np.asarray(y["violation"]))[mask])
        if left == "M":
            q = {k: np.asarray(v) for k, v in r["model_paired_score_quantiles"].items()}
            coverage80.extend(((yd >= q["0.1"]) & (yd <= q["0.9"]))[mask])
            coverage95.extend(((yd >= q["0.025"]) & (yd <= q["0.975"]))[mask])
            below50.extend((yd <= q["0.5"])[mask])
        selected = np.flatnonzero(mask)
        for ii, ai in enumerate(selected):
            for aj in selected[ii + 1 :]:
                sx = int(np.sign(xd[ai] - xd[aj])) if abs(xd[ai] - xd[aj]) > EPS else 0
                sy = int(np.sign(yd[ai] - yd[aj])) if abs(yd[ai] - yd[aj]) > EPS else 0
                rank.append(1.0 if sx == sy else 0.5 if sx == 0 or sy == 0 else 0.0)
                ties += int(sx == 0 or sy == 0)
    x, y = np.asarray(predicted), np.asarray(actual)

    def sign(a):
        return np.where(np.abs(a) <= EPS, 0, np.sign(a))

    return {
        "action_context_cells": len(errors),
        "delta_score_rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mean_delta_score_error": float(np.mean(errors)),
        "positive_prediction_precision": float(np.mean(y[x > EPS] > EPS))
        if np.any(x > EPS)
        else None,
        "positive_predictions": int(np.sum(x > EPS)),
        "direction_agreement": float(np.mean(sign(x) == sign(y))),
        "violation_delta_rmse": float(np.sqrt(np.mean(np.square(risk_errors)))),
        "violation_event_brier": float(np.mean(np.square(event_errors))),
        "model_delta_interval80_coverage": float(np.mean(coverage80)) if coverage80 else None,
        "model_delta_interval95_coverage": float(np.mean(coverage95)) if coverage95 else None,
        "truth_delta_at_or_below_model_q50": float(np.mean(below50)) if below50 else None,
        "pairwise_rank_agreement": float(np.mean(rank)) if rank else None,
        "rank_pairs": len(rank),
        "rank_pairs_with_tie": ties,
    }


def summarize(records):
    result = {}
    groups = {"overall": records}
    groups |= {f"scenario/{s}": [r for r in records if r["scenario"] == s] for s in SCENARIOS}
    groups |= {f"tick/{t}": [r for r in records if r["tick"] == t] for t in TICKS}
    groups |= {
        f"scenario/{s}/tick/{t}": [r for r in records if r["scenario"] == s and r["tick"] == t]
        for s in SCENARIOS
        for t in TICKS
    }
    for group, rows in groups.items():
        comparisons = {
            f"{a}-{b}": pair_metrics(rows, a, b)
            for a, b in (("M", "P"), ("P", "C"), ("C", "F"), ("M", "C"), ("M", "F"))
        }
        methods = {}
        for method in rows[0]["choices"]:
            methods[method] = {}
            for horizon in ("P", "C", "F"):
                selected = [r["choice_outcomes"][method][horizon] for r in rows]
                methods[method][horizon] = {
                    key: float(np.mean([v[key] for v in selected])) for key in selected[0]
                }
                methods[method][horizon]["market_score_directions"] = {
                    str(market): float(
                        np.mean(
                            [
                                r["choice_outcomes"][method][horizon]["score_delta"]
                                for r in rows
                                if r["market"] == market
                            ]
                        )
                    )
                    for market in sorted({r["market"] for r in rows})
                }
            methods[method]["deviation_fraction"] = float(
                np.mean([r["choices"][method] != r["baseline_index"] for r in rows])
            )
        transfer_numerator = float(
            np.mean(
                [
                    max(
                        0.0,
                        r["choice_outcomes"]["oracle-F-guarded"]["F"]["score"]
                        - r["choice_outcomes"]["oracle-C-guarded"]["F"]["score"],
                    )
                    for r in rows
                ]
            )
        )
        transfer_denominator = float(
            np.mean(
                [
                    max(
                        0.0,
                        r["choice_outcomes"]["oracle-F-guarded"]["F"]["score"]
                        - r["choice_outcomes"]["expected"]["F"]["score"],
                    )
                    for r in rows
                ]
            )
        )
        result[group] = {
            "contexts": len(rows),
            "comparisons": comparisons,
            "methods": methods,
            "C_to_F_transfer_regret": {
                "numerator": transfer_numerator,
                "denominator": transfer_denominator,
                "ratio": transfer_numerator / transfer_denominator
                if transfer_denominator > 0
                else None,
                "undefined_reason": None if transfer_denominator > 0 else "zero denominator",
                "definition": "mean max(0,F_guarded_best-F(C_guarded_best)) / mean max(0,F_guarded_best-F(M_expected_selected))",
            },
            "mean_J_within_radius": float(
                np.mean([np.asarray(r["J"])[r["within_radius_nonbaseline"]].mean() for r in rows])
            ),
        }
    return result


def run(output=DEFAULT_OUTPUT, protocol_path=None):
    torch.set_num_threads(1)
    core, actions, paths, models, old_cache, output = prepare(output, protocol_path)
    identity = fingerprint(core)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        keep(output / "manifest.json", {"identity": identity, "core": core})
        if (output / "seal.json").exists():
            seal = read_json(output / "seal.json")
            if seal["identity"] != identity or any(
                sha(output / name) != digest for name, digest in seal["files"].items()
            ):
                raise RuntimeError("Sealed diagnostic artifacts changed")
            return output
        with np.load(DATA / "validation.npz", allow_pickle=False) as cache:
            data = {key: cache[key] for key in cache.files}
        continuation.validate(data)
        expected = {
            (m, s, f, t)
            for m in range(2001, 2007)
            for s in range(5)
            for f in core["focals"]
            for t in TICKS
        }
        actual = [
            tuple(int(d[k]) for k in ("market", "scene_index", "focal", "tick"))
            for d in data["diagnostics"]
        ]
        if len(actual) != 540 or len(set(actual)) != 540 or set(actual) != expected:
            raise RuntimeError("Fixed 540-context coverage differs")
        with np.load(old_cache, allow_pickle=False) as cache:
            labels = cache["labels"]
        tapes = {int(path.stem): Tape(path) for path in paths}
        all_records, started = [], time.perf_counter()
        samplers = {
            seed: BundleSampler(ROOT / record["best_path"], actions, 24, "cpu")
            for seed, record in models.items()
        }
        for seed, sampler in samplers.items():
            record = models[seed]
            if (
                sampler.seed != seed
                or sampler.family != "flow"
                or sampler.training_identity != record["training_identity"]
                or sampler.encoder_sha256 != record["encoder_sha256"]
            ):
                raise RuntimeError("Frozen sampler metadata differs")
        for i, d in enumerate(data["diagnostics"]):
            market, si, focal, tick = (
                int(d[k]) for k in ("market", "scene_index", "focal", "tick")
            )
            scene, tape = SCENARIOS[si], tapes[market]
            fi = int(np.flatnonzero(tape.arrays["focals"] == focal)[0])
            length = int(data["lengths"][i])
            history = np.zeros((tick + 1, 13), dtype=np.float32)
            history[-length:] = data["states"][i, :length]
            if not np.array_equal(
                data["steps"][i, :length], np.arange(tick + 1 - length, tick + 1)
            ) or length != min(24, tick + 1):
                raise RuntimeError("Absolute positions or causal context window differs")
            budget, cpa, spent, past = (
                float(data[k][i]) for k in ("budget", "cpa", "spent", "conversions")
            )
            b = int(data["baseline_index"][i])
            if (
                int(
                    np.abs(
                        actions - feedback_action(history[-1], actions, 0.3, 8.0, budget, cpa)
                    ).argmin()
                )
                != b
            ):
                raise RuntimeError("Cached feedback baseline differs from public inputs")
            label_index = (((market - 2001) * 5 + si) * 6 + fi) * 10
            truths = truth(data, i, tape, fi, scene, actions, labels[label_index, tick])
            rng = np.random.default_rng(np.random.SeedSequence([91347, i]))
            half = rng.standard_normal((32, 40)).astype(np.float32)
            normal = np.concatenate((half, -half))
            uniforms = (np.arange(64) + 0.5) / 64
            rng.shuffle(uniforms)
            radius = np.abs(actions - actions[b]) <= 1.0 + EPS
            mask = radius.copy()
            mask[b] = False
            for seed, sampler in samplers.items():
                x, steps, _ = sampler._states(history)
                if (
                    not np.array_equal(steps.cpu().numpy()[0], data["steps"][i, :length])
                    or x.shape[1] != length
                ):
                    raise RuntimeError("Sampler absolute positions differ from frozen cache")
                bundles = sampler.curves(history, normal, uniforms, 8)
                projected = project(bundles, spent, past, budget, cpa)
                quantiles = np.quantile(
                    projected["utility"] - projected["utility"][:, b, None],
                    [0.025, 0.1, 0.5, 0.9, 0.975],
                    axis=0,
                )
                mean = means(projected)
                mean_score = gen.expected_score(mean["cost"], mean["conv"], cpa)
                plugin = project(bundles.mean(axis=0, keepdims=True), spent, past, budget, cpa)
                point = project(sampler.point(history)[None], spent, past, budget, cpa)
                shuffled = bundles.copy()
                srng = np.random.default_rng(np.random.SeedSequence([91347, i, 1]))
                for ai in range(len(actions)):
                    if ai != b:
                        shuffled[:, :, ai] = bundles[srng.permutation(64), :, ai]
                shuffled_prediction = project(shuffled, spent, past, budget, cpa)
                if not np.allclose(
                    np.sort(shuffled_prediction["utility"], axis=0),
                    np.sort(projected["utility"], axis=0),
                    rtol=0,
                    atol=1e-12,
                ) or not np.array_equal(shuffled[:, :, b], bundles[:, :, b]):
                    raise RuntimeError("Action-shuffle marginal invariants differ")
                choices = {
                    "expected": choose(projected, b, actions, projected["utility"])[0],
                    "plugin": choose(plugin, b, actions)[0],
                    "point": choose(point, b, actions)[0],
                    "no-quantile": choose(projected, b, actions)[0],
                    "action-shuffled": choose(
                        shuffled_prediction, b, actions, shuffled_prediction["utility"]
                    )[0],
                    "mean-endpoint-fixed-mask": choose(
                        projected, b, actions, projected["utility"], mean_score
                    )[0],
                }
                choices |= {
                    f"oracle-{key}-guarded": choose(
                        {k: v[None] for k, v in value.items()}, b, actions
                    )[0]
                    for key, value in truths.items()
                }
                choices["oracle-F-upper"] = int(np.argmax(truths["F"]["utility"]))
                outcomes = {}
                for method, ai in choices.items():
                    outcomes[method] = {}
                    for key, value in truths.items():
                        outcomes[method][key] = {
                            "score": float(value["utility"][ai]),
                            "conv": float(value["conv"][ai]),
                            "violation": float(value["violation"][ai]),
                            "score_delta": float(value["utility"][ai] - value["utility"][b]),
                            "conv_delta": float(value["conv"][ai] - value["conv"][b]),
                            "violation_delta": float(
                                value["violation"][ai] - value["violation"][b]
                            ),
                            "radius_score_regret": float(
                                max(0.0, np.max(value["utility"][radius]) - value["utility"][ai])
                            ),
                            "risk_guard_error": float(
                                value["violation"][ai] > value["violation"][b] + 0.03
                                or value["conv"][ai] < 0.97 * value["conv"][b]
                            ),
                        }
                all_records.append(
                    {
                        "context_index": i,
                        "market": market,
                        "scenario": scene,
                        "focal": focal,
                        "tick": tick,
                        "model_seed": seed,
                        "absolute_steps": data["steps"][i, :length].tolist(),
                        "baseline_index": b,
                        "within_radius_nonbaseline": mask.tolist(),
                        "choices": choices,
                        "choice_actions": {k: float(actions[v]) for k, v in choices.items()},
                        "predictions": {
                            key: {k: v.tolist() for k, v in value.items()}
                            for key, value in ({"M": mean} | truths).items()
                        },
                        "model_plugin": {k: v.tolist() for k, v in means(plugin).items()},
                        "model_point": {k: v.tolist() for k, v in means(point).items()},
                        "model_q50_paired_score": np.quantile(
                            projected["utility"] - projected["utility"][:, b, None], 0.5, axis=0
                        ).tolist(),
                        "model_paired_score_quantiles": {
                            str(q): values.tolist()
                            for q, values in zip(
                                (0.025, 0.1, 0.5, 0.9, 0.975), quantiles, strict=True
                            )
                        },
                        "J": (mean["utility"] - mean_score).tolist(),
                        "choice_outcomes": outcomes,
                    }
                )
            if (i + 1) % 60 == 0:
                print(f"horizon audit contexts={i + 1}/540", flush=True)
        if prepare(output, protocol_path)[0] != core:
            raise RuntimeError("Frozen inputs changed during horizon diagnosis")
        summary = {
            "identity": identity,
            "protocol": PROTOCOL,
            "by_seed": {
                str(seed): summarize([r for r in all_records if r["model_seed"] == seed])
                for seed in (11, 23)
            },
            "interpretation": "M-P includes forecast uncertainty and approximation; differences and RMSE are descriptive, not a causal learning-error decomposition. P-C isolates two accounting representations on fixed tapes; C-F compares different frozen future-action horizons. No independent or IID observations are added by the shared seeds/contexts.",
            "generative_value_confirmed": False,
            "business_replicated": False,
            "wall_seconds": time.perf_counter() - started,
        }
        keep(output / "contexts.json", {"identity": identity, "records": all_records})
        write_json(output / "summary.json", summary)
        if prepare(output, protocol_path)[0] != core:
            raise RuntimeError("Frozen inputs changed before artifact seal")
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
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.output, args.protocol))


if __name__ == "__main__":
    main()
