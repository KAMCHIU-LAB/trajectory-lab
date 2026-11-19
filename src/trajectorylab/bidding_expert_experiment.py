"""Frozen, market-paired experiment for one-shot public-feedback expert choice."""

from __future__ import annotations

import argparse
import hashlib
import os
import resource
import subprocess
import sys
import time
import tomllib
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from trajectorylab.bidding.learning import Feedback
from trajectorylab.bidding.market import feedback_action
from trajectorylab.bidding_delayed_feedback import (
    EventTape,
    build_event_market,
    event_source_identity,
)
from trajectorylab.bidding_expert_selection import (
    FEATURE_NAMES,
    ExpertSelectionEnv,
    paired_continuations,
    whole_episode,
)
from trajectorylab.bidding_public_response import PublicResponse
from trajectorylab.utils import fingerprint, read_json, seed_everything, write_json

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {0.75: "under", 1.0: "nominal", 1.5: "over"}
CHEAP_PREFIXES = ("feedback-", "fixed-", "calibration-", "threshold-", "mixture-")


def load_config(path):
    return tomllib.loads(Path(path).read_text())


def source_identity(config):
    names = (
        "bidding_expert_experiment.py",
        "bidding_expert_selection.py",
        "bidding_public_response.py",
        "bidding_delayed_experiment.py",
        "bidding_delayed_feedback.py",
        "bidding_pending_response.py",
        "utils.py",
        "bidding/market.py",
        "bidding/learning.py",
        "models.py",
    )
    sources = {
        f"src/trajectorylab/{name}": hashlib.sha256(
            (ROOT / "src/trajectorylab" / name).read_bytes()
        ).hexdigest()
        for name in names
    }
    seed = config["selection"]["response_seed"]
    weights = {
        f"assets/bidding-response-v1/response-{seed}.{suffix}": hashlib.sha256(
            (ROOT / "assets/bidding-response-v1" / f"response-{seed}.{suffix}").read_bytes()
        ).hexdigest()
        for suffix in ("npz", "json")
    }
    return dict(config=config, source=sources, weights=weights, generator=event_source_identity())


def freeze(config, output):
    record = source_identity(config)
    record["protocol_sha256"] = hashlib.sha256(
        (ROOT / "docs/bidding-expert-selection-protocol.md").read_bytes()
    ).hexdigest()
    record["identity"] = fingerprint(record)
    path = output / "experiment-freeze.json"
    if path.exists() and read_json(path) != record:
        raise RuntimeError("Frozen experiment identity changed: use a new study directory")
    write_json(path, record)
    return record


def prepare(config, output, split):
    """Generate registered synthetic inputs; confirmation requires an opening seal."""
    if split == "confirmation":
        open_confirmation(config, output)
    path = output / "data-freeze.json"
    record = dict(
        config=config,
        generator=event_source_identity(),
        protocol_sha256=hashlib.sha256(
            (ROOT / "docs/bidding-expert-selection-protocol.md").read_bytes()
        ).hexdigest(),
    )
    record["identity"] = fingerprint(record)
    if not path.exists():
        write_json(path, record)
    else:
        old = read_json(path)
        if old.get("config") != config or old.get("generator") != record["generator"]:
            raise RuntimeError("Data generation identity changed")
        if old.get("identity") != fingerprint({k: v for k, v in old.items() if k != "identity"}):
            raise RuntimeError("Data-generation freeze digest mismatch")
    started = time.monotonic()
    for market in config["study"][f"{split}_markets"]:
        if time.monotonic() - started > config["limits"]["preflight_seconds"]:
            raise RuntimeError("Generation phase time budget exhausted")
        cached = sum(p.stat().st_size for p in (output / "markets").glob("*.npz"))
        if cached > config["limits"]["cache_gib"] * 2**30:
            raise RuntimeError("New cache cap exceeded")
        meta = build_event_market(output / "markets" / f"{market}.npz", config["study"], market)
        attest_market(config, output, market)
        print(f"GENERATED split={split} market={market} seconds={meta['seconds']:.1f}", flush=True)


def checked_selection(config, output, frozen):
    selection = read_json(output / "selection-freeze.json")
    core = {k: v for k, v in selection.items() if k != "selection_identity"}
    if (
        selection.get("selection_identity") != fingerprint(core)
        or selection.get("identity") != frozen["identity"]
        or selection.get("confirmation_markets") != config["study"]["confirmation_markets"]
    ):
        raise RuntimeError("Frozen selection identity, sources, or markets changed")
    return selection


def open_confirmation(config, output, *, create=False):
    frozen = freeze(config, output)
    selection = checked_selection(config, output, frozen)
    seal = dict(
        experiment_identity=frozen["identity"],
        selection_identity=selection["selection_identity"],
        markets=config["study"]["confirmation_markets"],
    )
    path = output / "confirmation-open.json"
    if path.exists():
        if read_json(path) != seal:
            raise RuntimeError("Confirmation was opened under a different immutable selection")
    elif create:
        write_json(path, seal)
    else:
        raise RuntimeError("Open confirmation with its immutable seal before generating inputs")
    return seal


def attest_market(config, output, market):
    """Bind declared generator identity and exact cache bytes before evaluation."""
    data = read_json(output / "data-freeze.json")
    core = {k: v for k, v in data.items() if k != "identity"}
    if (
        data.get("identity") != fingerprint(core)
        or data.get("config") != config
        or data.get("generator") != event_source_identity()
    ):
        raise RuntimeError("Data-generation freeze differs from the registered experiment")
    path = output / "markets" / f"{market}.npz"
    meta = read_json(path.with_suffix(".json"))
    expected = fingerprint(
        {
            "study": {
                k: config["study"][k] for k in ("focal_agents", "pv_count", "actions", "biases")
            },
            "market": market,
            "source": event_source_identity(),
        }
    )
    if meta.get("identity") != expected or meta.get("market_id") != market:
        raise RuntimeError("Market generator/configuration identity mismatch")
    record = dict(
        market=market,
        generator_identity=expected,
        data_identity=data["identity"],
        tape_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    attestation = output / "inputs" / f"{market}.json"
    if attestation.exists() and read_json(attestation) != record:
        raise RuntimeError("Previously frozen market bytes changed")
    write_json(attestation, record)
    return record


def checked_result(config, output, split, market, frozen):
    saved = read_json(output / split / f"{market}.json")
    core = {k: v for k, v in saved.items() if k != "result_digest"}
    inputs = attest_market(config, output, market)
    if (
        saved.get("identity") != frozen["identity"]
        or saved.get("market") != market
        or saved.get("split") != split
        or saved.get("result_digest") != fingerprint(core)
        or saved.get("tape_sha256") != inputs["tape_sha256"]
        or any(r["market"] != market or r["split"] != split for r in saved["records"])
    ):
        raise RuntimeError("Result provenance, digest, split, or input identity mismatch")
    if split == "confirmation":
        seal = open_confirmation(config, output)
        if saved.get("confirmation_seal") != fingerprint(seal):
            raise RuntimeError("Result does not belong to the frozen confirmation opening")
    validate_records(saved["records"], config, [market], split)
    return saved


def _constant(action):
    return lambda *_: (float(action), {})


def calibrated_episode(tape, fi, bias, config, base, prior):
    """All correction inputs are public arrived labels and accepted pCVR."""
    s = config["selection"]
    env = ExpertSelectionEnv(
        tape, fi, bias, s["regime"], s["arrival_seed"], actions=config["study"]["actions"]
    )
    while env.tick < 48:
        view = env.public_view()
        probability_sum = sum(c.public_probability * c.event_count for c in view.cohorts)
        ratio = np.clip((view.arrived + prior) / (probability_sum + prior), 0.5, 2.0)
        if base.startswith("fixed-"):
            desired = float(base.split("-")[1])
        elif base == "feedback-g0.3-p8":
            desired = feedback_action(
                env.observation(), env.actions, 0.3, 8.0, view.budget, view.cpa
            )
        else:
            raise ValueError("Undeclared calibration base")
        action = env.actions[np.argmin(abs(env.actions - desired * ratio))]
        env.run(_constant(action), stop_tick=env.tick + 1)
    return env.finish()


def evaluate_market(tape, config, split, response):
    """Features are sealed before either continuation is evaluated."""
    s, b = config["selection"], config["baselines"]
    actions = config["study"]["actions"]
    market = int(tape.arrays["market_id"])
    records = []
    for fi, focal in enumerate(tape.arrays["focals"]):
        for bias in config["study"]["biases"]:
            pair = paired_continuations(
                tape,
                fi,
                bias,
                s["regime"],
                s["arrival_seed"],
                s["prefix"],
                response,
                gain=s["prefix_gain"],
                prior=s["prefix_prior"],
                actions=actions,
            )
            outcomes = {f"prefix-{expert}": pair[expert] for expert in ("feedback", "response")}
            if split != "train":
                for gain, prior in product(b["feedback_gains"], b["feedback_priors"]):
                    name = f"feedback-g{gain:g}-p{prior:g}"
                    outcomes[name] = whole_episode(
                        tape,
                        fi,
                        bias,
                        s["regime"],
                        s["arrival_seed"],
                        Feedback(np.array(actions), gain, prior),
                        actions=actions,
                    )
                for action in b["fixed_multipliers"]:
                    outcomes[f"fixed-{action:g}"] = whole_episode(
                        tape,
                        fi,
                        bias,
                        s["regime"],
                        s["arrival_seed"],
                        _constant(action),
                        actions=actions,
                    )
                for base, prior in product(b["calibration_bases"], b["calibration_priors"]):
                    outcomes[f"calibration-{base}-p{prior:g}"] = calibrated_episode(
                        tape, fi, bias, config, base, prior
                    )
                outcomes["full-response"] = whole_episode(
                    tape,
                    fi,
                    bias,
                    s["regime"],
                    s["arrival_seed"],
                    response,
                    actions=actions,
                )
                for ratio in b["threshold_ratios"]:
                    expert = (
                        "response"
                        if pair["prefix_summary"]["calibration_ratio"] < ratio
                        else "feedback"
                    )
                    outcomes[f"threshold-{ratio:g}"] = pair[expert]
                # A random policy may hash visible history, never hidden scenario IDs.
                key = f"{b['mixture_seed']}:{pair['prefix_digest']}"
                pick = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2
                outcomes["mixture-50"] = pair[("feedback", "response")[pick]]
                best = max(("feedback", "response"), key=lambda expert: pair[expert]["score"])
                outcomes["hindsight-two-experts"] = pair[best]
            record = dict(
                market=market,
                focal=int(focal),
                bias_group=GROUPS[bias],
                split=split,
                features=pair["features"],
                feature_names=pair["feature_names"],
                prefix_digest=pair["prefix_digest"],
                prefix_summary=pair["prefix_summary"],
                target=(pair["response"]["score"] - pair["feedback"]["score"])
                / (pair["feedback"]["budget"] / pair["feedback"]["target_cpa"]),
                outcomes=outcomes,
            )
            records.append(record)
    validate_records(records, config, [market], split)
    return records


def validate_records(records, config, markets, split):
    expected = set(product(markets, config["study"]["focal_agents"], GROUPS.values()))
    actual = [(r["market"], r["focal"], r["bias_group"]) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Missing, duplicate, or unregistered campaigns")
    b = config["baselines"]
    methods = {"prefix-feedback", "prefix-response"}
    if split != "train":
        methods.update(
            f"feedback-g{g:g}-p{p:g}" for g, p in product(b["feedback_gains"], b["feedback_priors"])
        )
        methods.update(f"fixed-{a:g}" for a in b["fixed_multipliers"])
        methods.update(
            f"calibration-{base}-p{p:g}"
            for base, p in product(b["calibration_bases"], b["calibration_priors"])
        )
        methods.update(f"threshold-{r:g}" for r in b["threshold_ratios"])
        methods.update(("full-response", "mixture-50", "hindsight-two-experts"))
    for r in records:
        if tuple(r["feature_names"]) != FEATURE_NAMES or len(r["features"]) != 28:
            raise RuntimeError("Feature contract changed")
        if not np.isfinite(r["features"]).all() or not np.isfinite(r["target"]):
            raise RuntimeError("Nonfinite features/target")
        if set(r["outcomes"]) != methods:
            raise RuntimeError("Method coverage mismatch")
        for row in r["outcomes"].values():
            if not np.isfinite([row[k] for k in ("score", "cost", "conversions", "budget")]).all():
                raise RuntimeError("Nonfinite outcome")
            if row["hard_budget_violation"] or row["final_arrived"] != row["conversions"]:
                raise RuntimeError("Budget or maturity invariant failed")
        f, p = (r["outcomes"][f"prefix-{e}"] for e in ("feedback", "response"))
        target = (p["score"] - f["score"]) / (f["budget"] / f["target_cpa"])
        if not np.isclose(target, r["target"], rtol=0, atol=1e-12):
            raise RuntimeError("Target is not a full-cycle paired advantage")


def evaluate(config, output, split):
    frozen = freeze(config, output)
    seal = open_confirmation(config, output) if split == "confirmation" else None
    response = PublicResponse(seed=config["selection"]["response_seed"])
    seed_everything(config["evaluation"]["bootstrap_seed"], config["limits"]["torch_threads"])
    records = []
    for market in config["study"][f"{split}_markets"]:
        path = output / split / f"{market}.json"
        if path.exists():
            saved = checked_result(config, output, split, market, frozen)
        else:
            started = time.perf_counter()
            tape_path = output / "markets" / f"{market}.npz"
            inputs = attest_market(config, output, market)
            tape = EventTape(tape_path)
            rows = evaluate_market(tape, config, split, response)
            saved = dict(
                identity=frozen["identity"],
                market=market,
                split=split,
                records=rows,
                tape_sha256=inputs["tape_sha256"],
                confirmation_seal=fingerprint(seal) if seal else None,
                seconds=time.perf_counter() - started,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024)
                / 2**30,
            )
            saved["result_digest"] = fingerprint(saved)
            write_json(path, saved)
            del tape
            if saved["peak_rss_gib"] > config["limits"]["peak_rss_gib"]:
                raise RuntimeError("Experiment exceeded RSS cap; record preserved")
        records.extend(saved["records"])
        # Timing/coverage only. Never stream confirmation policy scores for tuning.
        print(f"EVALUATED split={split} market={market} seconds={saved['seconds']:.2f}", flush=True)
    validate_records(records, config, config["study"][f"{split}_markets"], split)
    return records


def read_records(output, split, config):
    frozen = freeze(config, output)
    if split == "confirmation":
        open_confirmation(config, output)
    records = [
        r
        for market in config["study"][f"{split}_markets"]
        for r in checked_result(config, output, split, market, frozen)["records"]
    ]
    validate_records(records, config, config["study"][f"{split}_markets"], split)
    return records


def market_values(records, method, metric="score", group="all"):
    chosen = [r for r in records if group == "all" or r["bias_group"] == group]
    markets = sorted({r["market"] for r in chosen})
    return np.array(
        [
            np.mean([r["outcomes"][method][metric] for r in chosen if r["market"] == market])
            for market in markets
        ]
    )


def compare(records, method, reference, config, *, draws=None):
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
    n = len({r["market"] for r in records})
    draws = config["evaluation"]["bootstrap_draws"] if draws is None else draws
    indices = rng.integers(n, size=(draws, n))
    result = {}
    for group in ("all", "under", "nominal", "over"):
        stats = {}
        for metric in ("score", "conversions", "violation"):
            a, b = (market_values(records, m, metric, group) for m in (method, reference))
            if metric == "violation":
                effect = float(np.mean(a - b))
                samples = (a - b)[indices].mean(axis=1)
            else:
                if b.mean() <= 0:
                    raise RuntimeError("Nonpositive reference denominator")
                effect = float(a.mean() / b.mean() - 1)
                samples = a[indices].mean(axis=1) / b[indices].mean(axis=1) - 1
            stats[metric] = dict(
                effect=effect,
                ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
                candidate_mean=float(a.mean()),
                reference_mean=float(b.mean()),
            )
        result[group] = stats
    return result


def fit_ridge(records, alpha):
    x = np.asarray([r["features"] for r in records])
    y = np.asarray([r["target"] for r in records])
    mean, std = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-8)
    z = (x - mean) / std
    intercept = float(y.mean())
    coef = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - intercept))
    return dict(
        alpha=alpha, mean=mean.tolist(), std=std.tolist(), coef=coef.tolist(), intercept=intercept
    )


def predict(model, records):
    x = np.asarray([r["features"] for r in records])
    return ((x - model["mean"]) / model["std"]) @ model["coef"] + model["intercept"]


def with_selector(records, model, threshold, name):
    predictions = predict(model, records)
    selected = []
    for row, prediction in zip(records, predictions, strict=True):
        expert = "response" if prediction > threshold else "feedback"
        selected.append(
            {**row, "outcomes": {**row["outcomes"], name: row["outcomes"][f"prefix-{expert}"]}}
        )
    return selected


def protection_point_pass(stats, config):
    e = config["evaluation"]
    return all(
        stats[g]["score"]["effect"] >= e["score_protection"] - 1
        and stats[g]["conversions"]["effect"] >= e["conversion_protection"] - 1
        and stats[g]["violation"]["effect"] <= e["violation_delta_protection"]
        for g in ("under", "nominal", "over")
    )


def planned_markets(records, method, reference, config):
    """Normal-approximation budget screen, with doubled variance sensitivity."""
    e = config["evaluation"]
    checks = []
    for group in ("all", "under", "nominal", "over"):
        for metric in ("score",) if group == "all" else ("score", "conversions", "violation"):
            a, b = (market_values(records, m, metric, group) for m in (method, reference))
            if group == "all":
                margin, alpha = 1.0, e["alpha"] / 2
            else:
                margin = {
                    "score": e["score_protection"],
                    "conversions": e["conversion_protection"],
                    "violation": e["violation_delta_protection"],
                }[metric]
                alpha = e["alpha"] / 9
            favorable = b + margin - a if metric == "violation" else a - margin * b
            mean, sd = float(favorable.mean()), float(favorable.std(ddof=1))
            required = (
                None
                if mean <= 0 or sd <= 1e-12
                else max(
                    24,
                    int(
                        np.ceil(
                            2 * sd**2 * (norm.ppf(1 - alpha) + norm.ppf(e["power"])) ** 2 / mean**2
                        )
                    ),
                )
            )
            checks.append(
                dict(
                    group=group,
                    metric=metric,
                    favorable_mean=mean,
                    sd=sd,
                    variance_multiplier=2,
                    required_markets=required,
                )
            )
    required = (
        None
        if any(c["required_markets"] is None for c in checks)
        else max(c["required_markets"] for c in checks)
    )
    return dict(
        required_markets=required,
        endpoints=checks,
        caveat="Six development markets give an unstable planning estimate. Doubled variance screens per-endpoint superiority/noninferiority, not the joint probability of a 3% point gain and nine protections. Constant differences are inconclusive.",
    )


def develop(config, output):
    freeze(config, output)
    if (output / "selection-freeze.json").exists():
        raise RuntimeError("Cannot rerun selection after confirmation freeze")
    dev = read_records(output, "validation", config)
    cheap = sorted(m for m in dev[0]["outcomes"] if m.startswith(CHEAP_PREFIXES))
    # Include the exact common-prefix feedback identity even if naming changes.
    reference = max(cheap, key=lambda m: market_values(dev, m).mean())
    oracle = compare(dev, "hindsight-two-experts", reference, config)
    summary = dict(
        status="STOP_NO_HEADROOM",
        reference=reference,
        headroom=oracle,
        methods={m: float(market_values(dev, m).mean()) for m in sorted(dev[0]["outcomes"])},
        candidate_results=[],
    )
    if oracle["all"]["score"]["effect"] < config["evaluation"]["minimum_relative_gain"]:
        write_json(output / "development.json", summary)
        return summary
    train = read_records(output, "train", config)
    candidates = []
    for alpha in config["selection"]["ridge_alphas"]:
        model = fit_ridge(train, alpha)
        cv_predictions = np.zeros(len(train))
        for market in config["study"]["train_markets"]:
            indices = [i for i, r in enumerate(train) if r["market"] == market]
            held = [train[i] for i in indices]
            fitted = fit_ridge([r for r in train if r["market"] != market], alpha)
            cv_predictions[indices] = predict(fitted, held)
        for threshold in config["selection"]["rejection_thresholds"]:
            name = f"ridge-a{alpha:g}-t{threshold:g}"
            selected = with_selector(dev, model, threshold, name)
            stats = compare(selected, name, reference, config)
            candidates.append(
                dict(
                    name=name,
                    model=model,
                    threshold=threshold,
                    stats=stats,
                    cv_mse=float(np.mean((cv_predictions - [r["target"] for r in train]) ** 2)),
                    response_fraction=float(np.mean(predict(model, dev) > threshold)),
                )
            )
    best = max(candidates, key=lambda c: c["stats"]["all"]["score"]["candidate_mean"])
    summary.update(candidate_results=candidates, selected=best["name"], status="STOP_DEV_FAILURE")
    if best["stats"]["all"]["score"]["effect"] >= config["evaluation"][
        "minimum_relative_gain"
    ] and protection_point_pass(best["stats"], config):
        selected = with_selector(dev, best["model"], best["threshold"], best["name"])
        power = planned_markets(selected, best["name"], reference, config)
        summary["power"] = power
        summary["status"] = "STOP_INSUFFICIENT_BUDGET"
        if (
            power["required_markets"] is not None
            and power["required_markets"] <= config["limits"]["confirmation_maximum_markets"]
        ):
            selection = dict(
                candidate=best,
                reference=reference,
                power=power,
                identity=read_json(output / "experiment-freeze.json")["identity"],
                confirmation_markets=config["study"]["confirmation_markets"],
            )
            selection["selection_identity"] = fingerprint(selection)
            write_json(output / "selection-freeze.json", selection)
            summary["status"] = "READY_FOR_CONFIRMATION"
    write_json(output / "development.json", summary)
    return summary


def holm_protections(records, method, reference, config):
    e, checks = config["evaluation"], []
    for group in ("under", "nominal", "over"):
        for metric in ("score", "conversions", "violation"):
            a, b = (market_values(records, m, metric, group) for m in (method, reference))
            if metric == "violation":
                differences = b + e["violation_delta_protection"] - a
            else:
                margin = e["score_protection"] if metric == "score" else e["conversion_protection"]
                differences = a - margin * b
            mean, se = float(differences.mean()), float(differences.std(ddof=1) / np.sqrt(len(a)))
            # No variance in a finite sample cannot establish a rare-event margin.
            p = float(student_t.sf(mean / se, len(a) - 1)) if se > 1e-12 else 1.0
            checks.append(dict(group=group, metric=metric, p_one_sided=p, favorable_mean=mean))
    ordered = sorted(checks, key=lambda c: c["p_one_sided"])
    running = 0.0
    for rank, check in enumerate(ordered):
        running = max(running, (len(checks) - rank) * check["p_one_sided"])
        check["holm_adjusted_p"] = min(1.0, running)
        check["pass"] = check["holm_adjusted_p"] < e["alpha"]
    return checks


def confirm(config, output):
    frozen = freeze(config, output)
    selection = checked_selection(config, output, frozen)
    seal = open_confirmation(config, output)
    records = read_records(output, "confirmation", config)
    c = selection["candidate"]
    selected = with_selector(records, c["model"], c["threshold"], c["name"])
    stats = compare(selected, c["name"], selection["reference"], config)
    protections = holm_protections(selected, c["name"], selection["reference"], config)
    passed = (
        stats["all"]["score"]["effect"] >= config["evaluation"]["minimum_relative_gain"]
        and stats["all"]["score"]["ci95"][0] > 0
        and all(c["pass"] for c in protections)
    )
    result = dict(
        status="CONFIRM_PASS" if passed else "CONFIRM_FAILURE",
        candidate=c["name"],
        reference=selection["reference"],
        stats=stats,
        protections=protections,
        confirmation_seal=fingerprint(seal),
    )
    path = output / "confirmation.json"
    if path.exists() and read_json(path) != result:
        raise RuntimeError("Immutable confirmation result changed")
    write_json(path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "freeze", "evaluate", "develop", "confirm", "open-confirmation"),
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bidding-expert-selection.toml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/bidding/expert-selection-v1"
    )
    parser.add_argument(
        "--split", choices=("train", "validation", "confirmation"), default="validation"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    # The OS releases wakefulness when this process exits, even after a crash.
    if sys.platform == "darwin":
        subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    if args.command == "prepare":
        prepare(config, args.output, args.split)
    elif args.command == "freeze":
        result = freeze(config, args.output)
        print(result["identity"])
    elif args.command == "open-confirmation":
        print(fingerprint(open_confirmation(config, args.output, create=True)))
    elif args.command == "evaluate":
        evaluate(config, args.output, args.split)
    elif args.command == "develop":
        print(develop(config, args.output)["status"])
    else:
        print(confirm(config, args.output)["status"])


if __name__ == "__main__":
    main()
