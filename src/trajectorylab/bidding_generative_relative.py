"""Paired predicted improvement with a bounded public-feedback fallback.

This zero-refit development study preserves the constant-action forecast and
its proportional budget approximation. Only exact environment rollouts decide
business performance. Model quantiles are not confidence bounds or guarantees.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from trajectorylab import bidding_generative as gen
from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.market import ROOT, feedback_action
from trajectorylab.bidding_generative_control import (
    business_gates,
    calibration_passed,
    keep,
    value_gate,
)
from trajectorylab.bidding_generative_learning import BundleSampler
from trajectorylab.bidding_robust import comparison, relative, sha, validated_rows
from trajectorylab.utils import fingerprint, read_json, runtime_info, write_json


def sources():
    files = [Path(__file__), ROOT / "src/trajectorylab/bidding_generative_control.py"]
    return gen.sources() | {str(p.relative_to(ROOT)): sha(p) for p in files}


class RelativePolicy(gen.DistributionPolicy):
    def __init__(self, sampler, actions, mode, quantile, radius, opts):
        if mode not in ("expected", "plugin", "point", "action-shuffled", "no-quantile"):
            raise ValueError("Unknown paired-improvement mode")
        super().__init__(
            sampler,
            actions,
            "expected",
            opts["samples"],
            opts["integration_steps"],
            opts["sampling_seed"],
        )
        self.mode, self.quantile, self.radius = mode, quantile, radius
        self.episode_records = []
        self.shuffle_audit = None

    @staticmethod
    def terminal(bundles, state, budget, cpa):
        scaled = bundles * np.array([budget, budget / cpa, budget, budget / cpa])[None, :, None]
        spent, past_conv = float(state[3]) * budget, float(state[4]) * budget / cpa
        remaining = max(0.0, budget - spent)
        future_cost, future_conv = scaled[:, 2], scaled[:, 3]
        fraction = np.minimum(1.0, remaining / np.maximum(future_cost, 1e-12))
        cost = spent + np.minimum(future_cost, remaining)
        conv = past_conv + future_conv * fraction
        scores = gen.expected_score(cost, conv, cpa)
        return scaled, cost, conv, scores

    def __call__(self, states, used, rewards, budget, cpa):
        baseline_action = feedback_action(states[-1], self.actions, 0.3, 8.0, budget, cpa)
        baseline = int(np.abs(self.actions - baseline_action).argmin())
        if self.mode == "point":
            bundles = self.sampler.point(states)[None]
        else:
            half = self.rng.standard_normal((self.samples // 2, 40)).astype(np.float32)
            normal = np.concatenate((half, -half), axis=0)
            uniforms = (np.arange(self.samples) + 0.5) / self.samples
            self.rng.shuffle(uniforms)
            bundles = self.sampler.curves(states, normal, uniforms, self.steps)
            if self.mode == "action-shuffled":
                original = bundles
                bundles = bundles.copy()
                for ai in range(len(self.actions)):
                    if ai != baseline:
                        order = self.shuffle_rng.permutation(self.samples)
                        bundles[:, :, ai] = bundles[order, :, ai]
                if self.shuffle_audit is None:
                    _, c0, v0, s0 = self.terminal(original, states[-1], budget, cpa)
                    _, c1, v1, s1 = self.terminal(bundles, states[-1], budget, cpa)
                    checks = {
                        "baseline_column_exact": np.array_equal(
                            original[:, :, baseline], bundles[:, :, baseline]
                        ),
                        "score_marginals": np.allclose(
                            np.sort(s0, axis=0), np.sort(s1, axis=0), rtol=0, atol=1e-12
                        ),
                        "score_means": np.allclose(
                            s0.mean(axis=0), s1.mean(axis=0), rtol=0, atol=1e-12
                        ),
                        "conversion_means": np.allclose(
                            v0.mean(axis=0), v1.mean(axis=0), rtol=0, atol=1e-12
                        ),
                        "violation_probabilities": np.array_equal(
                            (c0 > cpa * v0 + 1e-8).mean(axis=0), (c1 > cpa * v1 + 1e-8).mean(axis=0)
                        ),
                        "baseline_paired_delta_zero": np.all(
                            s1[:, baseline] - s1[:, baseline] == 0
                        ),
                    }
                    for ai in range(len(self.actions)):
                        a, b = original[:, :, ai], bundles[:, :, ai]
                        checks[f"whole_four_head_pairs_action_{ai}"] = np.array_equal(
                            a[np.lexsort(a.T)], b[np.lexsort(b.T)]
                        )
                    if not all(checks.values()):
                        raise RuntimeError("Action shuffle changed a required marginal or pairing")
                    self.shuffle_audit = {k: bool(v) for k, v in checks.items()}
            if self.mode == "plugin":
                bundles = bundles.mean(axis=0, keepdims=True)
        if not np.isfinite(bundles).all() or np.any(bundles < 0):
            raise RuntimeError("Invalid paired response samples")
        scaled, cost, conv, scores = self.terminal(bundles, states[-1], budget, cpa)
        utility, conversion = scores.mean(axis=0), conv.mean(axis=0)
        violation = np.mean(cost > cpa * conv + 1e-8, axis=0)
        difference = scores - scores[:, baseline, None]
        eligible = (
            (np.abs(self.actions - baseline_action) <= self.radius + 1e-12)
            & (utility > utility[baseline] + 1e-12)
            & (conversion >= 0.97 * conversion[baseline] - 1e-12)
            & (violation <= violation[baseline] + 0.03 + 1e-12)
        )
        if self.mode in ("expected", "action-shuffled"):
            eligible &= np.quantile(difference, self.quantile, axis=0) >= -1e-12
        candidates = np.flatnonzero(eligible)
        ai = int(candidates[np.argmax(utility[candidates])]) if len(candidates) else baseline
        if len(states) == 1:
            self.episode_records.append({"baseline_indices": [], "chosen_indices": []})
        self.episode_records[-1]["baseline_indices"].append(baseline)
        self.episode_records[-1]["chosen_indices"].append(ai)
        return float(self.actions[ai]), {
            "predicted_tick_cost": float(scaled[:, 0, ai].mean()),
            "predicted_tick_conversions": float(scaled[:, 1, ai].mean()),
            "forecast_feasible": bool(violation[ai] <= 0.5),
        }


def prepare(config_path):
    opts = tomllib.loads(config_path.read_text())
    if opts["quantiles"] != [0.1, 0.25, 0.5] or opts["radii"] != [0.25, 0.5, 1.0]:
        raise RuntimeError("Paired-improvement grid differs from its registered protocol")
    main, robust, parent, output = [
        relative(opts[k]) for k in ("main_run", "robust_run", "parent_run", "output")
    ]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (main, robust, parent)):
        raise RuntimeError("Use a separate relative-control output directory")
    parent_config_path = relative(opts["parent_config"])
    parent_opts = tomllib.loads(parent_config_path.read_text())
    if parent_opts["pilot_steps"] != 15000 or relative(parent_opts["output"]) != parent:
        raise RuntimeError("Only the completed long-budget parent may be reused")
    for key in (
        "main_run",
        "robust_run",
        "scenarios",
        "primary_scenarios",
        "samples",
        "integration_steps",
        "sampling_seed",
    ):
        if opts[key] != parent_opts[key]:
            raise RuntimeError("Parent and relative-control protocols differ: " + key)
    parent_manifest = read_json(parent / "manifest.json")
    main_manifest = read_json(main / "manifest.json")
    config = main_manifest["config"]
    if fingerprint(provenance(config)) != main_manifest["identity"]:
        raise RuntimeError("Original simulation provenance changed")
    if config["study"]["train_markets"] != list(range(1001, 1017)) or config["study"][
        "validation_markets"
    ] != list(range(2001, 2007)):
        raise RuntimeError("This study uses only the original training and development markets")
    paths = {
        split: [main / "markets" / f"{m}.npz" for m in config["study"][split + "_markets"]]
        for split in ("train", "validation")
    }
    inputs = {
        p.stem: {"npz": sha(p), "metadata": sha(p.with_suffix(".json"))}
        for group in paths.values()
        for p in group
    }
    reference_names = (
        "development-baselines.json",
        "development-baselines.identity.json",
        "baseline-selection.json",
        "pilot-selection.json",
    )
    references = {name: sha(robust / name) for name in reference_names}
    expected_parent = {
        "sources": gen.sources(),
        "config": parent_opts,
        "inputs": inputs,
        "references": references,
        "runtime": main_manifest["runtime"],
    }
    persisted_parent = {k: v for k, v in expected_parent.items() if k != "runtime"}
    if parent_manifest != persisted_parent | {"identity": fingerprint(expected_parent)}:
        raise RuntimeError("Parent sources, raw data, references or identity changed")
    selection = read_json(parent / "pilot-selection.json")
    calibration = read_json(parent / "calibration.json")
    if (
        selection["identity"] != parent_manifest["identity"]
        or selection["calibration"] != calibration
        or not calibration_passed(calibration["flow"])
    ):
        raise RuntimeError("Completed parent Flow calibration is required")
    actions = np.asarray(config["study"]["actions"])
    samplers = {
        family: BundleSampler(
            parent / "models" / f"pilot-{family}-11" / "best.pt",
            actions,
            config["training"]["context"],
            "cpu",
        )
        for family in ("flow", "gmm")
    }
    pilot_config = copy.deepcopy(config)
    pilot_config["training"]["steps"] = parent_opts["pilot_steps"]
    encoder = (
        robust
        / "models"
        / f"response-{read_json(robust / 'pilot-selection.json')['loss']}-11"
        / "best.pt"
    )
    for family, sampler in samplers.items():
        expected_training = fingerprint(
            {
                "study_identity": parent_manifest["identity"],
                "training": pilot_config,
                "family": family,
                "seed": 11,
                "encoder": sha(encoder),
            }
        )
        if (
            sampler.family != family
            or sampler.seed != 11
            or sampler.training_identity != expected_training
            or sampler.encoder_sha256 != sha(encoder)
        ):
            raise RuntimeError("Parent generator training identity differs")
    parent_policies = {
        (f"{family}-{mode}", 11): gen.DistributionPolicy(
            sampler,
            actions,
            mode,
            opts["samples"],
            opts["integration_steps"],
            opts["sampling_seed"],
        )
        for family, sampler in samplers.items()
        for mode in ("expected", "plugin", "shuffled")
    }
    parent_policies["point-projection", 11] = gen.DistributionPolicy(
        samplers["flow"], actions, "point"
    )
    if read_json(parent / "development-pilot-policy-identity.json") != {
        "identity": gen.evaluation_identity(parent_policies, parent_opts)
    }:
        raise RuntimeError("Parent calibration/evaluation weights or policy identity differs")
    if not (parent / "development-pilot.json").exists():
        raise RuntimeError("Completed parent evaluation is required")
    validated_rows(
        parent / "development-pilot",
        paths["validation"],
        config["study"],
        opts["scenarios"],
        parent_policies,
    )
    files = [p for p in parent.rglob("*.json") if p.name not in ("runner.json", "status.json")]
    files += [
        *[parent / "models" / f"pilot-{family}-11" / "best.pt" for family in ("flow", "gmm")],
        encoder,
        main / "manifest.json",
        parent_config_path,
    ]
    files += [robust / name for name in (*reference_names, "development-manifest.json")]
    core = {
        "config": opts,
        "sources": sources(),
        "inputs": inputs,
        "parent_identity": parent_manifest["identity"],
        "frozen_files": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))},
        "runtime": runtime_info(),
        "scope": "single-seed zero-refit development; no fresh confirmation",
    }
    return opts, config, main, robust, parent, output, paths, core


def rank(stats, radius, quantile, opts):
    gates = business_gates(stats, opts)
    return (
        not all(all(v.values()) for v in gates.values()),
        -min(stats[s]["score_ratio"] for s in opts["primary_scenarios"]),
        radius,
        quantile,
    )


def decision_trace(path, policies, rows, identity):
    if all(policy.episode_records for policy in policies.values()):
        records = {}
        for (name, seed), policy in policies.items():
            episodes = [r for r in rows if r["method"] == name and r["seed"] == seed]
            if len(episodes) != len(policy.episode_records):
                raise RuntimeError("Decision trace episode coverage mismatch")
            traced = []
            for row, decisions in zip(episodes, policy.episode_records, strict=True):
                if any(len(v) != 48 for v in decisions.values()):
                    raise RuntimeError("Decision trace tick coverage mismatch")
                traced.append({k: row[k] for k in ("market", "focal", "scenario")} | decisions)
            changed = [
                sum(a != b for a, b in zip(r["baseline_indices"], r["chosen_indices"], strict=True))
                for r in traced
            ]
            records[name] = {
                "episodes": traced,
                "decisions": 48 * len(traced),
                "deviation_count": sum(changed),
                "fallback_fraction": 1 - sum(changed) / (48 * len(traced)),
                "shuffle_invariants": policy.shuffle_audit,
            }
        keep(path, {"identity": identity, "policies": records})
    else:
        if not path.exists() or read_json(path)["identity"] != identity:
            raise RuntimeError("Cached results lack their matching decision trace")
        records = read_json(path)["policies"]
    return {
        k: {field: value for field, value in v.items() if field != "episodes"}
        for k, v in records.items()
    }


def run(config_path):
    opts, config, main, robust, parent, output, paths, core = prepare(config_path)
    torch.set_num_threads(opts["evaluation_threads"])
    identity = fingerprint(core)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        keep(output / "manifest.json", {"identity": identity, "core": core})
        actions = np.asarray(config["study"]["actions"])
        sampler = BundleSampler(
            parent / "models" / "pilot-flow-11" / "best.pt",
            actions,
            config["training"]["context"],
            "cpu",
        )
        baselines = gen.reference_rows(robust, main, paths["validation"], config, opts)
        reference = read_json(robust / "baseline-selection.json")["feedback"]["name"]
        policies, parameters = {}, {}
        for mode in ("expected", "plugin", "point"):
            for radius in opts["radii"]:
                for q in opts["quantiles"] if mode == "expected" else [0.0]:
                    name = f"relative-{mode}-r{radius}-q{q}"
                    policies[name, 11] = RelativePolicy(sampler, actions, mode, q, radius, opts)
                    parameters[name] = {"mode": mode, "radius": radius, "quantile": q}
        eval_opts = opts | {"relative_identity": identity, "relative_sources": sources()}
        write_json(
            output / "status.json",
            {"phase": "grid-evaluation", "identity": identity, "updated_unix": time.time()},
        )
        rows = gen.evaluate_policies(
            output, "development-grid", paths["validation"], config, eval_opts, policies
        )
        traces = decision_trace(output / "grid-decision-trace.json", policies, rows, identity)
        stats = {
            name: comparison(rows + baselines, name, reference, opts["scenarios"])
            for name in parameters
        }
        choices = {}
        for mode in ("expected", "plugin", "point"):
            names = [name for name, p in parameters.items() if p["mode"] == mode]
            choices[mode] = min(
                names,
                key=lambda name: rank(
                    stats[name], parameters[name]["radius"], parameters[name]["quantile"], opts
                ),
            )
        chosen, p = choices["expected"], parameters[choices["expected"]]
        shuffled = {
            ("relative-action-shuffled", 11): RelativePolicy(
                sampler, actions, "action-shuffled", p["quantile"], p["radius"], opts
            ),
            ("relative-no-quantile", 11): RelativePolicy(
                sampler, actions, "no-quantile", p["quantile"], p["radius"], opts
            ),
        }
        extra = gen.evaluate_policies(
            output,
            "development-shuffled",
            paths["validation"],
            config,
            eval_opts | {"chosen": chosen},
            shuffled,
        )
        extra_traces = decision_trace(
            output / "diagnostic-decision-trace.json", shuffled, extra, identity
        )
        gates = business_gates(stats[chosen], opts)
        business_passed = all(all(v.values()) for v in gates.values())
        attribution = {
            mode: value_gate(comparison(rows, chosen, choices[mode], opts["scenarios"]), opts)
            for mode in ("plugin", "point")
        }
        result = {
            "identity": identity,
            "checkpoint_sha256": sampler.identity,
            "parameters": parameters,
            "comparisons": stats,
            "selected": choices,
            "gates": gates,
            "business_promoted": business_passed,
            "attribution": attribution,
            "action_shuffled": comparison(
                rows + extra, chosen, "relative-action-shuffled", opts["scenarios"]
            ),
            "quantile_disabled": comparison(
                rows + extra, chosen, "relative-no-quantile", opts["scenarios"]
            ),
            "decision_traces": traces | extra_traces,
            "needs_three_seed": business_passed and all(v["passed"] for v in attribution.values()),
            "generative_value_confirmed": False,
            "scope": "single-seed development grid; actual hard-budget rollouts; constant-action forecast approximation",
        }
        if prepare(config_path)[-1] != core:
            raise RuntimeError("Frozen inputs changed during paired-control evaluation")
        keep(output / "relative-selection.json", result)
        write_json(
            output / "status.json",
            {
                "phase": "complete",
                "identity": identity,
                "business_promoted": business_passed,
                "updated_unix": time.time(),
            },
        )
        print(
            f"Paired generative control complete; business={business_passed}, three_seed={result['needs_three_seed']}",
            flush=True,
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bidding-generative-relative.toml"
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
