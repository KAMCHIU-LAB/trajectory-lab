"""Fixed-policy training-seed business replication on existing development tapes.

This separate study preserves the failed attribution and needs_three_seed=false
of the discovery study. It never confirms generative value or opens new markets.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from trajectorylab import bidding_generative as gen
from trajectorylab import bidding_generative_relative as rel
from trajectorylab.bidding.market import ROOT
from trajectorylab.bidding_generative_control import (
    business_gates,
    calibration_passed,
    keep,
    value_gate,
)
from trajectorylab.bidding_generative_learning import BundleSampler, fit
from trajectorylab.bidding_robust import (
    comparison,
    relative,
    sha,
    training_identity,
    validated_rows,
)
from trajectorylab.utils import fingerprint, read_json, runtime_info, write_json

SEEDS = [11, 23, 37]
SCENARIOS = ["calibrated", "over_25", "over_50", "under_25", "mid_over_50"]
PRIMARY = ["over_25", "over_50", "under_25"]
PROTOCOL = {
    "seeds": SEEDS,
    "steps": 15000,
    "scenarios": SCENARIOS,
    "primary_scenarios": PRIMARY,
    "samples": 64,
    "integration_steps": 8,
    "sampling_seed": 91347,
    "evaluation_threads": 1,
    "calibration_cases_per_market_scene": 8,
    "stress_score_min": 1.03,
    "stress_conversion_min": 0.97,
    "stress_violation_delta_max": 0.03,
    "nominal_score_min": 0.98,
    "value_mean_score_ratio_min": 1.01,
    "value_each_score_ratio_min": 0.98,
    "selected_expected": {"radius": 1.0, "quantile": 0.5},
    "selected_plugin": {"radius": 0.25},
    "selected_point": {"radius": 1.0},
}
PARAMETERS = {
    "expected": ("expected", 0.5, 1.0),
    "plugin": ("plugin", 0.0, 0.25),
    "point": ("point", 0.0, 1.0),
    "no-quantile": ("no-quantile", 0.5, 1.0),
    "action-shuffled": ("action-shuffled", 0.5, 1.0),
}
SCOPE = "fixed-policy business replication; discovery seed plus two training seeds; existing six development markets; no fresh confirmation"


def sources() -> dict:
    path = Path(__file__).resolve()
    return rel.sources() | {str(path.relative_to(ROOT)): sha(path)}


def _trace(path, policies, rows, identity):
    """Check cached traces against exact episode keys and all 48 action indices."""
    record = read_json(path)
    if record["identity"] != identity or set(record["policies"]) != {m for m, _ in policies}:
        raise RuntimeError("Trace identity or policy coverage changed")
    summaries = {}
    for (method, seed), policy in policies.items():
        item = record["policies"][method]
        expected = [
            (r["market"], r["focal"], r["scenario"])
            for r in rows
            if r["method"] == method and r["seed"] == seed
        ]
        actual, deviations = [], 0
        for episode in item["episodes"]:
            actual.append(tuple(episode[k] for k in ("market", "focal", "scenario")))
            a, b = episode["baseline_indices"], episode["chosen_indices"]
            if (
                len(a) != 48
                or len(b) != 48
                or any(type(i) is not int or not 0 <= i < len(policy.actions) for i in a + b)
            ):
                raise RuntimeError("Invalid trace action indices")
            deviations += sum(i != j for i, j in zip(a, b, strict=True))
        decisions = 48 * len(actual)
        if (
            actual != expected
            or not decisions
            or item["decisions"] != decisions
            or item["deviation_count"] != deviations
            or item["fallback_fraction"] != 1 - deviations / decisions
        ):
            raise RuntimeError("Trace coverage or fallback count changed")
        if policy.mode == "action-shuffled" and (
            not item["shuffle_invariants"] or not all(item["shuffle_invariants"].values())
        ):
            raise RuntimeError("Action-shuffle invariants failed")
        summaries[method] = {k: v for k, v in item.items() if k != "episodes"}
    return summaries


def _relative_evidence(prepared):
    opts, config, main, robust, parent, output, paths, core = prepared
    identity = fingerprint(core)
    if read_json(output / "manifest.json") != {"identity": identity, "core": core}:
        raise RuntimeError("Completed relative-study manifest differs from current provenance")
    sampler = BundleSampler(
        parent / "models/pilot-flow-11/best.pt",
        np.asarray(config["study"]["actions"]),
        config["training"]["context"],
        "cpu",
    )
    policies, parameters = {}, {}
    for mode in ("expected", "plugin", "point"):
        for radius in opts["radii"]:
            for q in opts["quantiles"] if mode == "expected" else [0.0]:
                name = f"relative-{mode}-r{radius}-q{q}"
                policies[name, 11] = rel.RelativePolicy(
                    sampler, sampler.actions, mode, q, radius, opts
                )
                parameters[name] = {"mode": mode, "radius": radius, "quantile": q}
    evaluation_opts = opts | {"relative_identity": identity, "relative_sources": rel.sources()}

    def cached(name, slots, options):
        for suffix in (".json", ".identity.json", "-policy-identity.json"):
            if not (output / f"{name}{suffix}").is_file():
                raise RuntimeError("Completed relative evaluation required: " + name)
        if read_json(output / f"{name}-policy-identity.json") != {
            "identity": gen.evaluation_identity(slots, options)
        }:
            raise RuntimeError("Relative policy/input identity changed")
        return validated_rows(output / name, paths["validation"], config["study"], SCENARIOS, slots)

    rows = cached("development-grid", policies, evaluation_opts)
    base = gen.reference_rows(robust, main, paths["validation"], config, opts)
    reference = read_json(robust / "baseline-selection.json")["feedback"]["name"]
    stats = {m: comparison(rows + base, m, reference, SCENARIOS) for m in parameters}
    choices = {
        mode: min(
            [m for m, p in parameters.items() if p["mode"] == mode],
            key=lambda m: rel.rank(
                stats[m], parameters[m]["radius"], parameters[m]["quantile"], opts
            ),
        )
        for mode in ("expected", "plugin", "point")
    }
    chosen = choices["expected"]
    p = parameters[chosen]
    extra_policies = {
        (f"relative-{mode}", 11): rel.RelativePolicy(
            sampler, sampler.actions, mode, p["quantile"], p["radius"], opts
        )
        for mode in ("action-shuffled", "no-quantile")
    }
    extra = cached("development-shuffled", extra_policies, evaluation_opts | {"chosen": chosen})
    traces = _trace(output / "grid-decision-trace.json", policies, rows, identity) | _trace(
        output / "diagnostic-decision-trace.json", extra_policies, extra, identity
    )
    selection = read_json(output / "relative-selection.json")
    gates = business_gates(stats[chosen], opts)
    business = all(all(v.values()) for v in gates.values())
    attribution = {
        mode: value_gate(comparison(rows, chosen, choices[mode], SCENARIOS), opts)
        for mode in ("plugin", "point")
    }
    expected = {
        "identity": identity,
        "checkpoint_sha256": sampler.identity,
        "parameters": parameters,
        "comparisons": stats,
        "selected": choices,
        "gates": gates,
        "business_promoted": business,
        "attribution": attribution,
        "action_shuffled": comparison(rows + extra, chosen, "relative-action-shuffled", SCENARIOS),
        "quantile_disabled": comparison(rows + extra, chosen, "relative-no-quantile", SCENARIOS),
        "decision_traces": traces,
        "needs_three_seed": business and all(v["passed"] for v in attribution.values()),
        "generative_value_confirmed": False,
        "scope": "single-seed development grid; actual hard-budget rollouts; constant-action forecast approximation",
    }
    if (
        selection != expected
        or selection["needs_three_seed"] is not False
        or all(v["passed"] for v in attribution.values())
    ):
        raise RuntimeError("Discovery results or their failed attribution changed")
    if any(
        parameters[choices[mode]]
        != {
            "mode": mode,
            "radius": PROTOCOL[f"selected_{mode}"]["radius"],
            "quantile": 0.5 if mode == "expected" else 0.0,
        }
        for mode in choices
    ):
        raise RuntimeError("Registered replication parameters differ from discovery selection")
    renamed = {choices[mode]: mode for mode in choices} | {
        f"relative-{mode}": mode for mode in ("no-quantile", "action-shuffled")
    }
    evidence = [
        dict(r, method=renamed[r["method"]]) for r in rows + extra if r["method"] in renamed
    ]
    return evidence, {renamed[m]: traces[m] for m in renamed}, base, reference


def _cache(directory, paths, actions):
    archive, stamp = directory / "trajectories.npz", directory / "identity.json"
    if not archive.is_file() or not stamp.is_file():
        raise RuntimeError("Existing parent trajectory cache required; generation is prohibited")
    data_sources = {
        k: v
        for k, v in gen.robust_sources().items()
        if not k.endswith(("bidding_robust.py", "bidding_robust_learning.py"))
    }
    expected = fingerprint(
        {
            "inputs": {
                p.stem: {"npz": sha(p), "metadata": sha(p.with_suffix(".json"))} for p in paths
            },
            "actions": actions,
            "scenarios": SCENARIOS,
            "data_sources": data_sources,
        }
    )
    record = read_json(stamp)
    if (
        record["identity"] != expected
        or record["npz_sha256"] != sha(archive)
        or record["episodes"] != len(paths) * len(SCENARIOS) * 6 * 10
    ):
        raise RuntimeError("Parent trajectory-cache identity changed")
    return [archive, stamp]


def prepare(config_path: Path):
    """Read-only preflight: validate all frozen ancestors, caches and encoders."""
    config_path = config_path.resolve()
    opts = tomllib.loads(config_path.read_text())
    if any(opts.get(k) != v for k, v in PROTOCOL.items()):
        raise RuntimeError("Replication configuration differs from its fixed protocol")
    relative_config = relative(opts["relative_config"])
    prepared = rel.prepare(relative_config)
    relative_opts, config, main, robust, parent, old_output, paths, relative_core = prepared
    output = relative(opts["output"])
    if (
        main != relative(opts["main_run"])
        or robust != relative(opts["robust_run"])
        or any(
            output.is_relative_to(p) or p.is_relative_to(output)
            for p in (main, robust, parent, old_output)
        )
    ):
        raise RuntimeError("Replication requires separate output and unchanged input runs")
    for key in (
        "scenarios",
        "primary_scenarios",
        "samples",
        "integration_steps",
        "sampling_seed",
        "stress_score_min",
        "stress_conversion_min",
        "stress_violation_delta_max",
        "nominal_score_min",
        "value_mean_score_ratio_min",
        "value_each_score_ratio_min",
    ):
        if opts[key] != relative_opts[key]:
            raise RuntimeError("Original threshold or inference protocol changed: " + key)
    evidence, traces, base, reference = _relative_evidence(prepared)
    training = copy.deepcopy(config)
    training["training"]["steps"] = opts["steps"]
    completed = read_json(robust / "final-training.json")
    encoders = {seed: robust / "models" / f"response-log-{seed}" / "best.pt" for seed in SEEDS}
    if read_json(robust / "pilot-selection.json")["loss"] != "log":
        raise RuntimeError("The original response-log encoders are required")
    for seed, encoder in encoders.items():
        records = [
            r
            for r in completed
            if r.get("family") == "response"
            and r.get("seed") == seed
            and r.get("loss") == "log"
            and r.get("steps") == opts["steps"]
        ]
        saved = torch.load(encoder, map_location="cpu", weights_only=False)
        identity = training_identity(training, main, SCENARIOS, "response", "log", seed)
        if len(records) != 1 or any(
            saved.get(k) != v
            for k, v in {
                "identity": identity,
                "family": "response",
                "kind": "response",
                "seed": seed,
                "loss": "log",
                "actions": config["study"]["actions"],
                "context": config["training"]["context"],
            }.items()
        ):
            raise RuntimeError("Original same-seed encoder training metadata changed")
    original_training = [
        r
        for r in read_json(parent / "pilot-training.json")
        if r.get("family") == "flow" and r.get("seed") == 11
    ]
    if (
        len(original_training) != 1
        or original_training[0].get("steps") != opts["steps"]
        or original_training[0].get("encoder_sha256") != sha(encoders[11])
    ):
        raise RuntimeError("Discovery Flow must have completed its original 15000-step budget")
    files = [relative_config, config_path, robust / "final-training.json", *encoders.values()]
    files += [
        p
        for p in old_output.rglob("*")
        if p.is_file()
        and p.suffix in (".json", ".csv")
        and p.name not in ("status.json", "runner.json")
    ]
    for split in ("train", "validation"):
        files += _cache(parent / "data-cache" / split, paths[split], config["study"]["actions"])
    calibration = read_json(parent / "calibration.json")["flow"]
    if calibration["cases"] != 240 or not calibration_passed(calibration):
        raise RuntimeError("Discovery Flow must retain its original 240-context calibration")
    core = {
        "config": opts,
        "sources": sources(),
        "relative_core": relative_core,
        "relative_identity": fingerprint(relative_core),
        "frozen_files": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        "inputs": relative_core["inputs"],
        "encoders": {str(seed): sha(path) for seed, path in encoders.items()},
        "training": training,
        "runtime": runtime_info(),
        "scope": SCOPE,
        "stop_rule": "each seed business, distribution, primary score direction; stop before fitting another seed on failure",
        "original_needs_three_seed": False,
        "generative_value_confirmed": False,
    }
    if (output / "manifest.json").exists() and read_json(output / "manifest.json") != {
        "identity": fingerprint(core),
        "core": core,
    }:
        raise RuntimeError("Replication inputs/source/runtime changed; cannot resume")
    return (
        opts,
        config,
        main,
        robust,
        parent,
        output,
        paths,
        core,
        evidence,
        traces,
        [r for r in base if r["method"] == reference],
        reference,
        encoders,
        calibration,
    )


def _aggregate(rows, scenarios=SCENARIOS):
    result = {}
    for method in sorted({r["method"] for r in rows}):
        result[method] = {}
        for scene in scenarios:
            group = [r for r in rows if r["method"] == method and r["scenario"] == scene]
            if not group:
                raise RuntimeError("Missing aggregate metric group")
            result[method][scene] = {
                k: float(np.mean([r[k] for r in group])) for k in ("score", "conversions", "cost")
            }
            result[method][scene] |= {
                "violation_rate": float(np.mean([r["violation"] for r in group])),
                "violation_count": sum(r["violation"] for r in group),
                "n": len(group),
            }
    return result


def _compare(rows, method, reference):
    for scene in SCENARIOS:
        group = [r for r in rows if r["method"] == reference and r["scenario"] == scene]
        if not group or any(
            float(np.mean([r[k] for r in group])) <= 0 for k in ("score", "conversions")
        ):
            raise RuntimeError("Comparison requires positive reference score and conversions")
    stats = comparison(rows, method, reference, SCENARIOS)
    for scene in SCENARIOS:
        stats[scene]["score_delta"] = stats[scene]["score"] - stats[scene]["reference_score"]
    return stats


def _result(seed, rows, base, reference, calibration, traces, opts):
    feedback = _compare(rows + base, "expected", reference)
    gates = business_gates(feedback, opts)
    direction = {s: feedback[s]["score_delta"] > 0 for s in PRIMARY}
    attribution = {
        mode: value_gate(_compare(rows, "expected", mode), opts) for mode in ("plugin", "point")
    }
    for value in attribution.values():
        value["mean_stress_score_delta"] = float(
            np.mean([value["by_scenario"][s]["score_delta"] for s in PRIMARY])
        )
        value["seed_direction_passed"] = value["mean_stress_score_delta"] > 0
        value["passed_with_seed_direction"] = bool(
            value["passed"] and value["seed_direction_passed"]
        )
    distribution = calibration_passed(calibration)
    business = all(all(v.values()) for v in gates.values())
    return {
        "seed": seed,
        "feedback": feedback,
        "business_gates": gates,
        "business_passed": business,
        "calibration": calibration,
        "distribution_passed": distribution,
        "stress_score_direction": direction,
        "passed": business and distribution and all(direction.values()),
        "attribution": attribution,
        "quantile_disabled": _compare(rows, "expected", "no-quantile"),
        "action_shuffled": _compare(rows, "expected", "action-shuffled"),
        "metrics": _aggregate(rows + base),
        "decision_traces": {
            name: values | {"fallback_count": values["decisions"] - values["deviation_count"]}
            for name, values in traces.items()
        },
        "generative_value_confirmed": False,
    }


def _policies(sampler, seed, opts):
    return {
        (name, seed): rel.RelativePolicy(sampler, sampler.actions, mode, q, radius, opts)
        for name, (mode, q, radius) in PARAMETERS.items()
    }


def run(config_path: Path, stage="develop") -> Path:
    prepared = prepare(config_path)
    (
        opts,
        config,
        main,
        robust,
        parent,
        output,
        paths,
        core,
        discovery_rows,
        discovery_traces,
        base,
        reference,
        encoders,
        discovery_calibration,
    ) = prepared
    identity = fingerprint(core)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        keep(output / "manifest.json", {"identity": identity, "core": core})

        def guard():
            if prepare(config_path)[7] != core:
                raise RuntimeError("Frozen replication inputs changed during execution")

        def status(phase, **details):
            write_json(
                output / "status.json",
                {"phase": phase, "identity": identity, "updated_unix": time.time(), **details},
            )

        records_path = output / "final-training.json"
        records = read_json(records_path) if records_path.exists() else []
        if len({r["seed"] for r in records}) != len(records) or any(
            r["seed"] not in SEEDS for r in records
        ):
            raise RuntimeError("Training record seed coverage differs")
        data = {}

        def cached_data(split):
            if split not in data:
                directory = parent / "data-cache" / split
                _cache(directory, paths[split], config["study"]["actions"])
                data[split] = gen.dataset(
                    paths[split], config["study"]["actions"], SCENARIOS, directory
                )
            return data[split]

        def model(seed, allow_fit):
            guard()
            directory = (
                parent / "models/pilot-flow-11"
                if seed == 11
                else output / "models" / f"flow-{seed}"
            )
            best = directory / "best.pt"
            train_id = fingerprint(
                {
                    "protocol_identity": identity,
                    "inputs": core["inputs"],
                    "encoder": core["encoders"][str(seed)],
                    "training": core["training"],
                    "family": "flow",
                    "seed": seed,
                    "sources": core["sources"],
                }
            )
            old = next((r for r in records if r["seed"] == seed), None)
            if old is None and seed != 11:
                if not allow_fit:
                    raise RuntimeError(f"Seed {seed} has not finished training")
                status("training", seed=seed, steps=opts["steps"])
                result = fit(
                    "flow",
                    seed,
                    core["training"],
                    cached_data("train"),
                    cached_data("validation"),
                    encoders[seed],
                    directory,
                    train_id,
                )
                guard()
            else:
                result = {} if old is None else old["fit"]
            sampler = BundleSampler(
                best, np.asarray(config["study"]["actions"]), config["training"]["context"], "cpu"
            )
            if (
                sampler.family != "flow"
                or sampler.seed != seed
                or sampler.encoder_sha256 != core["encoders"][str(seed)]
                or (seed != 11 and sampler.training_identity != train_id)
            ):
                raise RuntimeError("Fitted Flow identity differs from frozen training")
            if seed != 11:
                checkpoint = torch.load(
                    directory / "checkpoint.pt", map_location="cpu", weights_only=False
                )
                if (
                    checkpoint["identity"] != train_id
                    or checkpoint["step"] != opts["steps"]
                    or result.get("steps") != opts["steps"]
                ):
                    raise RuntimeError("Flow training did not finish its registered budget")
            record = {
                "seed": seed,
                "family": "flow",
                "reused": seed == 11,
                "training_identity": sampler.training_identity,
                "encoder_sha256": sampler.encoder_sha256,
                "best_path": str(best.relative_to(ROOT)),
                "best_sha256": sha(best),
                "checkpoint_sha256": None if seed == 11 else sha(directory / "checkpoint.pt"),
                "fit": result,
            }
            if old is not None and old != record:
                raise RuntimeError("Completed training weights or record changed")
            if old is None:
                records.append(record)
                write_json(records_path, records)
            return sampler

        def evaluate(seed, sampler):
            guard()
            torch.set_num_threads(opts["evaluation_threads"])
            directory = output / "evaluation" / f"seed-{seed}"
            directory.mkdir(parents=True, exist_ok=True)
            seal_path = directory / "artifacts.json"
            if seal_path.exists():
                seal = read_json(seal_path)
                if (
                    seal["identity"] != identity
                    or seal["weights"] != sampler.identity
                    or any(
                        sha(directory / name) != digest for name, digest in seal["files"].items()
                    )
                ):
                    raise RuntimeError("Completed seed evaluation bytes changed")
                return read_json(directory / "summary.json"), read_json(
                    directory / "development.json"
                )
            status("evaluating", seed=seed, policies=len(PARAMETERS))
            if seed == 11:
                rows, traces, calibration = discovery_rows, discovery_traces, discovery_calibration
                keep(directory / "development.json", rows)
                keep(
                    directory / "trace.json",
                    {
                        "identity": identity,
                        "source_relative_identity": core["relative_identity"],
                        "policies": traces,
                    },
                )
            else:
                policies = _policies(sampler, seed, opts)
                eval_opts = opts | {
                    "replication_identity": identity,
                    "replication_sources": sources(),
                    "seed": seed,
                    "parameters": PARAMETERS,
                }
                existed = (directory / "development.json").exists()
                rows = gen.evaluate_policies(
                    directory, "development", paths["validation"], config, eval_opts, policies
                )
                if existed and not (directory / "trace.json").exists():
                    # A kill between row save and trace save is recoverable only
                    # by an identical deterministic replay, never invented traces.
                    with tempfile.TemporaryDirectory(
                        prefix="trace-recovery-", dir=directory
                    ) as temporary:
                        replay = gen.evaluate_policies(
                            Path(temporary),
                            "development",
                            paths["validation"],
                            config,
                            eval_opts,
                            policies,
                        )
                        if replay != rows:
                            raise RuntimeError("Trace-recovery replay differs from cached rows")
                rel.decision_trace(directory / "trace.json", policies, rows, identity)
                traces = _trace(directory / "trace.json", policies, rows, identity)
                calibration = gen.calibration(sampler, cached_data("validation"), opts)
            if calibration["cases"] != 240:
                raise RuntimeError("Expected exactly the registered 240 calibration contexts")
            keep(directory / "calibration.json", calibration)
            result = _result(seed, rows, base, reference, calibration, traces, opts)
            keep(directory / "summary.json", result)
            guard()
            names = (
                ("development.json", "trace.json", "calibration.json", "summary.json")
                if seed == 11
                else (
                    "development.json",
                    "development.csv",
                    "development.identity.json",
                    "development-policy-identity.json",
                    "trace.json",
                    "calibration.json",
                    "summary.json",
                )
            )
            keep(
                seal_path,
                {
                    "identity": identity,
                    "weights": sampler.identity,
                    "files": {name: sha(directory / name) for name in names},
                },
            )
            return result, rows

        try:
            if stage == "train":
                failed_seed = any(
                    not read_json(p)["passed"]
                    for p in (output / "evaluation").glob("seed-*/summary.json")
                )
                if failed_seed or (
                    (output / "summary.json").exists()
                    and read_json(output / "summary.json").get("stopped")
                ):
                    raise RuntimeError("This branch already failed; further fitting is prohibited")
                for seed in SEEDS:
                    model(seed, True)
                guard()
                status("trained", seeds=SEEDS, generative_value_confirmed=False)
                return output
            results, all_rows, stopped = {}, [], None
            for seed in SEEDS:
                sampler = model(seed, stage == "develop")
                result, rows = evaluate(seed, sampler)
                results[str(seed)] = result
                all_rows += rows
                if not result["passed"]:
                    stopped = seed
                    break
            # Each seed has the exact same market/focal/scenario grid. Thus the
            # candidate mean is an equal seed/market/focal mean; baseline is
            # included once, never diluted by repeating other baseline methods.
            counts = {
                seed: sum(r["seed"] == seed and r["method"] == "expected" for r in all_rows)
                for seed in (int(s) for s in results)
            }
            if len(set(counts.values())) != 1 or not all(counts.values()):
                raise RuntimeError("Pooled seed weights are unbalanced")
            feedback = _compare(all_rows + base, "expected", reference)
            gates = business_gates(feedback, opts)
            attribution = {
                mode: value_gate(_compare(all_rows, "expected", mode), opts)
                for mode in ("plugin", "point")
            }
            for mode, value in attribution.items():
                value["all_seed_direction"] = {
                    seed: result["attribution"][mode]["seed_direction_passed"]
                    for seed, result in results.items()
                }
                value["passed_with_seed_direction"] = bool(
                    value["passed"] and all(value["all_seed_direction"].values())
                )
            calibration_keys = (
                "energy_score",
                "point_energy_score",
                "coverage80",
                "coverage95",
                "upper_cap_fraction",
            )
            pooled_calibration = {
                key: float(np.mean([r["calibration"][key] for r in results.values()]))
                for key in calibration_keys
            }
            pooled_calibration |= {
                "cases": 240 * len(results),
                "cases_per_seed": 240,
                "scope": "Equal means over completed seeds on the same 240 correlated development contexts; not independent observations",
                "by_scenario": {
                    scene: {
                        key: float(
                            np.mean(
                                [
                                    r["calibration"]["by_scenario"][scene][key]
                                    for r in results.values()
                                ]
                            )
                        )
                        for key in calibration_keys
                    }
                    for scene in SCENARIOS
                },
            }
            pooled_calibration["passed"] = bool(
                np.isfinite([pooled_calibration[k] for k in calibration_keys]).all()
                and pooled_calibration["energy_score"]
                <= 1.02 * pooled_calibration["point_energy_score"]
                and 0.60 <= pooled_calibration["coverage80"] <= 0.95
                and pooled_calibration["coverage95"] >= 0.80
                and pooled_calibration["upper_cap_fraction"] == 0.0
            )
            complete = set(results) == {str(s) for s in SEEDS}
            overall_gates = {
                "all_three_seeds_evaluated": complete,
                "pooled_business": all(all(v.values()) for v in gates.values()),
                "pooled_distribution": calibration_passed(pooled_calibration),
                "each_seed_business": all(r["business_passed"] for r in results.values()),
                "each_seed_distribution": all(r["distribution_passed"] for r in results.values()),
                "each_seed_primary_direction": all(
                    all(r["stress_score_direction"].values()) for r in results.values()
                ),
                "each_seed_nominal": all(
                    r["feedback"]["calibrated"]["score_ratio"] >= opts["nominal_score_min"]
                    for r in results.values()
                ),
            }
            passed = all(overall_gates.values())
            summary = {
                "identity": identity,
                "scope": SCOPE,
                "seeds": results,
                "pooled": {
                    "feedback": feedback,
                    "business_gates": gates,
                    "business_passed": all(all(v.values()) for v in gates.values()),
                    "attribution": attribution,
                    "calibration": pooled_calibration,
                    "distribution_passed": pooled_calibration["passed"],
                    "all_seed_direction": {
                        seed: r["stress_score_direction"] for seed, r in results.items()
                    },
                    "metrics": _aggregate(all_rows + base),
                    "seed_weights": {str(s): 1 / len(counts) for s in counts},
                    "baseline_copies": 1,
                },
                "business_replicated": passed,
                "overall_gates": overall_gates,
                "complete": complete,
                "stopped": stopped is not None,
                "stop_seed": stopped,
                "not_run_seeds": {
                    str(
                        s
                    ): f"Stopped after seed {stopped} failed registered business/distribution/direction gates"
                    for s in SEEDS
                    if str(s) not in results
                },
                "original_needs_three_seed": False,
                "original_attribution_preserved": {
                    "needs_three_seed": False,
                    "passed": False,
                    "relative_identity": core["relative_identity"],
                },
                "generative_value_confirmed": False,
                "needs_confirmation": False,
                "decision": "business-replicated" if passed else "not-replicated",
                "uncertainty": "Three fixed training seeds on the same six development markets; discovery seed is included. No independent market confirmation or universal retraining claim.",
            }
            guard()
            keep(output / "summary.json", summary)
            status(
                "complete" if passed else "stopped",
                business_replicated=passed,
                stop_seed=stopped,
                not_run_seeds=summary["not_run_seeds"],
                generative_value_confirmed=False,
            )
            print(
                f"Fixed business replication={passed}; stop_seed={stopped}; generative_value_confirmed=False",
                flush=True,
            )
            return output
        except Exception as error:
            status(
                "failed",
                error_type=type(error).__name__,
                reason=str(error),
                generative_value_confirmed=False,
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bidding-generative-replication.toml"
    )
    parser.add_argument("--stage", choices=("train", "evaluate", "develop"), default="develop")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
