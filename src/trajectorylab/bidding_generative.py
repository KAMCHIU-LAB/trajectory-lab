"""Conditional joint response generation: development and attribution only.

This is a stochastic response/world model, not an action generator. Every
outcome sample is integrated; no policy can observe the response tape or select
the most favorable sampled future. Fresh confirmation remains a separate stage.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import time
import tomllib
from pathlib import Path

import numpy as np

from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.learning import Feedback, Learned
from trajectorylab.bidding.market import ROOT
from trajectorylab.bidding_robust import (
    comparison,
    relative,
    sha,
    validated_rows,
)
from trajectorylab.bidding_robust import (
    sources as robust_sources,
)
from trajectorylab.bidding_robust_data import trajectories
from trajectorylab.utils import fingerprint, read_json, write_json


def sources() -> dict:
    files = [Path(__file__), ROOT / "src/trajectorylab/bidding_generative_learning.py"]
    return robust_sources() | {str(p.relative_to(ROOT)): sha(p) for p in files}


def reference_rows(robust, main, paths, config, opts):
    """Revalidate the complete old baseline cache without changing its selection."""
    record = read_json(robust / "development-manifest.json")
    if record["sources"] != robust_sources():
        raise RuntimeError("Robust baseline source revision changed")
    registered = record["config"]
    if registered["scenarios"] != opts["scenarios"]:
        raise RuntimeError("Reference scenario protocol differs")
    actions = np.asarray(config["study"]["actions"])
    policies = {("feedback-old", -1): Feedback(actions, 0.3, 8.0)}
    for gain in registered["feedback_gains"]:
        for prior in registered["feedback_priors"]:
            policies[f"feedback-g{gain}-p{prior}", -1] = Feedback(actions, gain, prior)
    for action in actions:
        policies[f"fixed-{action}", -1] = lambda s, u, r, b, c, k=float(action): (k, {})
    for seed in registered["seeds"]:
        policies["response-old", seed] = Learned(
            main / "models" / f"response-{seed}" / "best.pt",
            actions,
            config["training"]["context"],
            "cpu",
        )
    return validated_rows(
        robust / "development-baselines", paths, config["study"], opts["scenarios"], policies
    )


def dataset(paths: list[Path], actions, scenes, directory: Path) -> list[dict]:
    """Cache only generated public trajectories, bound to actual input bytes."""
    identity = fingerprint(
        {
            "inputs": {
                p.stem: {"npz": sha(p), "metadata": sha(p.with_suffix(".json"))} for p in paths
            },
            "actions": list(actions),
            "scenarios": scenes,
            "data_sources": {
                k: v
                for k, v in robust_sources().items()
                if not k.endswith(("bidding_robust.py", "bidding_robust_learning.py"))
            },
        }
    )
    directory.mkdir(parents=True, exist_ok=True)
    path, stamp = directory / "trajectories.npz", directory / "identity.json"
    if path.exists():
        record = read_json(stamp)
        if record["identity"] != identity or record["npz_sha256"] != sha(path):
            raise RuntimeError("Trajectory cache/input identity changed")
        with np.load(path, allow_pickle=False) as cache:
            arrays = {k: cache[k] for k in cache.files}
        return [{k: array[i] for k, array in arrays.items()} for i in range(len(arrays["score"]))]
    data = trajectories(paths, np.asarray(actions), scenes)
    temporary = path.with_suffix(".partial.npz")
    np.savez_compressed(
        temporary,
        **{
            k: np.stack([run[k] for run in data])
            for k in ("states", "actions", "returns", "labels", "score")
        },
    )
    temporary.replace(path)
    write_json(stamp, {"identity": identity, "npz_sha256": sha(path), "episodes": len(data)})
    return data


def expected_score(cost, conv, cpa):
    return conv * np.minimum(1.0, (cpa * conv / np.maximum(cost, 1e-12)) ** 2)


class DistributionPolicy:
    """Common controller for generated distributions and the frozen point model."""

    def __init__(
        self, sampler, actions, mode, samples=64, steps=8, sampling_seed=91347, max_violation=1.0
    ):
        if mode not in ("expected", "plugin", "shuffled", "point"):
            raise ValueError("Unknown distribution decision mode")
        if samples <= 0 or samples % 2 or not 0 <= max_violation <= 1:
            raise ValueError("Use even positive samples and a violation limit in [0, 1]")
        self.sampler, self.actions, self.mode = sampler, np.asarray(actions), mode
        self.samples, self.steps, self.sampling_seed = samples, steps, sampling_seed
        self.max_violation = max_violation
        # Expose the full policy state to the existing evaluation digest. Include
        # encoder and head weights in a dedicated signature as well (see below).
        self.context, self.kind = sampler.context, "response-distribution"
        self.corrected, self.prior, self.hard_cpa = False, 3.0, False
        self.reset()

    def reset(self):
        self.rng = np.random.default_rng(self.sampling_seed)
        self.shuffle_rng = np.random.default_rng(self.sampling_seed + 1)

    def __call__(self, states, used, rewards, budget, cpa):
        if self.mode == "point":
            bundles = self.sampler.point(states)[None]
        else:
            half = self.rng.standard_normal((self.samples // 2, 40)).astype(np.float32)
            normal = np.concatenate((half, -half), axis=0)
            uniforms = (np.arange(self.samples) + 0.5) / self.samples
            self.rng.shuffle(uniforms)
            bundles = self.sampler.curves(states, normal, uniforms, self.steps)
            if self.mode == "shuffled":
                order = self.shuffle_rng.permutation(self.samples)
                bundles = bundles.copy()
                bundles[:, (1, 3), :] = bundles[order][:, (1, 3), :]
            if self.mode == "plugin":
                bundles = bundles.mean(axis=0, keepdims=True)
        if not np.isfinite(bundles).all() or np.any(bundles < 0):
            raise RuntimeError("Invalid generated response bundle")
        scaled = bundles * np.array([budget, budget / cpa, budget, budget / cpa])[None, :, None]
        spent, past_conv = float(states[-1][3]) * budget, float(states[-1][4]) * budget / cpa
        remaining = max(0.0, budget - spent)
        future_cost, future_conv = scaled[:, 2], scaled[:, 3]
        fraction = np.minimum(1.0, remaining / np.maximum(future_cost, 1e-12))
        # This is an explicit planning approximation, shared by all four modes.
        cost = spent + np.minimum(future_cost, remaining)
        conv = past_conv + future_conv * fraction
        utility = expected_score(cost, conv, cpa).mean(axis=0)
        risk = np.mean(cost > cpa * conv + 1e-8, axis=0)
        eligible = np.flatnonzero(risk <= self.max_violation + 1e-12)
        if len(eligible):
            ai = int(eligible[np.argmax(utility[eligible])])
        else:
            debt = np.maximum(cost - cpa * conv, 0).mean(axis=0)
            ai = int(np.lexsort((-utility, debt, risk))[0])
        return float(self.actions[ai]), {
            "predicted_tick_cost": float(scaled[:, 0, ai].mean()),
            "predicted_tick_conversions": float(scaled[:, 1, ai].mean()),
            "forecast_feasible": bool(np.mean(cost[:, ai] <= cpa * conv[:, ai]) >= 0.5),
        }


def evaluation_identity(policies, opts) -> str:
    return fingerprint(
        {
            "sources": sources(),
            "options": opts,
            "weights": {f"{m}:{s}": p.sampler.identity for (m, s), p in policies.items()},
        }
    )


def evaluate_policies(output, name, paths, config, opts, policies):
    # validated_rows checks slots, coverage, hard budget and loaded model bytes.
    # These wrappers have a sampler rather than `model`; the outer filename is
    # additionally bound to all sampler checkpoint hashes and full new sources.
    identity = evaluation_identity(policies, opts)
    record = output / f"{name}-policy-identity.json"
    if record.exists() and read_json(record)["identity"] != identity:
        raise RuntimeError("Generated-policy identity changed; use a new run")
    write_json(record, {"identity": identity})
    return validated_rows(output / name, paths, config["study"], opts["scenarios"], policies)


def calibration(sampler, data, opts) -> dict:
    """Proper energy score and central coverage on fixed development contexts."""
    rng = np.random.default_rng(91348)
    # The data are ordered market, scene, focal, behavior. Sample equally within
    # each original market/scene; episodes and ticks are never treated as IID.
    per_group = 6 * 10
    cases = []
    for start in range(0, len(data), per_group):
        for _ in range(opts["calibration_cases_per_market_scene"]):
            run = data[start + int(rng.integers(per_group))]
            tick = int(rng.integers(48))
            scene = opts["scenarios"][(start // per_group) % len(opts["scenarios"])]
            cases.append((run, tick, scene))
    scores, point_scores, coverage80, coverage95, caps = [], [], [], [], []
    scene_metrics = {s: [] for s in opts["scenarios"]}
    scale = np.array([48.0, 48.0, 1.0, 1.0])[None, :, None]
    for run, tick, scene in cases:
        normal = rng.standard_normal((opts["samples"], 40)).astype(np.float32)
        uniforms = rng.uniform(size=opts["samples"])
        states = list(run["states"][: tick + 1])
        draws = sampler.curves(states, normal, uniforms, opts["integration_steps"])
        target = run["labels"][tick]
        x = np.log1p(draws[:, :, 1:] * scale).reshape(len(draws), -1)
        y = np.log1p(target[:, 1:] * scale[0]).reshape(-1)
        point = np.log1p(sampler.point(states)[:, 1:] * scale[0]).reshape(-1)
        distances = np.linalg.norm(x[:, None] - x[None, :], axis=-1)
        scores.append(float(np.linalg.norm(x - y, axis=-1).mean() - 0.5 * distances.mean()))
        point_scores.append(float(np.linalg.norm(point - y)))
        # Remaining cost and conversions, all nonzero nominal actions.
        for probability, bucket in ((0.8, coverage80), (0.95, coverage95)):
            lower, upper = np.quantile(
                draws[:, 2:, 1:], [(1 - probability) / 2, (1 + probability) / 2], axis=0
            )
            bucket.append(float(((target[2:, 1:] >= lower) & (target[2:, 1:] <= upper)).mean()))
        caps.append(sampler.last_diagnostics["transformed_upper_cap_fraction"])
        scene_metrics[scene].append(
            (scores[-1], point_scores[-1], coverage80[-1], coverage95[-1], caps[-1])
        )
    result = {
        "cases": len(cases),
        "energy_score": float(np.mean(scores)),
        "point_energy_score": float(np.mean(point_scores)),
        "coverage80": float(np.mean(coverage80)),
        "coverage95": float(np.mean(coverage95)),
        "upper_cap_fraction": float(np.mean(caps)),
        "scope": "development diagnostics, correlated contexts within original markets",
        "by_scenario": {
            s: dict(
                zip(
                    ("energy_score", "point_energy_score", "coverage80", "coverage95", "upper_cap_fraction"),
                    np.mean(values, axis=0).tolist(),
                    strict=True,
                )
            )
            for s, values in scene_metrics.items()
        },
    }
    result["passed"] = bool(
        result["energy_score"] <= 1.02 * result["point_energy_score"]
        and 0.60 <= result["coverage80"] <= 0.95
        and result["coverage95"] >= 0.80
        and result["upper_cap_fraction"] == 0.0
    )
    return result


def run(config_path: Path, stage="pilot") -> Path:
    from trajectorylab.bidding_generative_learning import BundleSampler, fit

    opts = tomllib.loads(config_path.read_text())
    main, robust, output = [relative(opts[k]) for k in ("main_run", "robust_run", "output")]
    main_manifest = read_json(main / "manifest.json")
    config = main_manifest["config"]
    if fingerprint(provenance(config)) != main_manifest["identity"]:
        raise RuntimeError("Frozen original source changed")
    if (
        opts["samples"] != 64
        or opts["samples"] % 2
        or opts["scenarios"] != config["evaluation"]["scenarios"]
    ):
        raise ValueError("Unexpected initial generative development protocol")
    import torch

    torch.set_num_threads(config["training"]["threads"])
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        paths = {
            split: [main / "markets" / f"{m}.npz" for m in config["study"][split + "_markets"]]
            for split in ("train", "validation")
        }
        inputs = {
            p.stem: {"npz": sha(p), "metadata": sha(p.with_suffix(".json"))}
            for group in paths.values()
            for p in group
        }
        baselines = reference_rows(robust, main, paths["validation"], config, opts)
        reference_files = (
            "development-baselines.json",
            "development-baselines.identity.json",
            "baseline-selection.json",
            "pilot-selection.json",
        )
        references = {name: sha(robust / name) for name in reference_files}
        identity = fingerprint(
            {
                "sources": sources(),
                "config": opts,
                "inputs": inputs,
                "references": references,
                "runtime": main_manifest["runtime"],
            }
        )
        manifest = {
            "identity": identity,
            "sources": sources(),
            "config": opts,
            "inputs": inputs,
            "references": references,
        }
        manifest_path = output / "manifest.json"
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise RuntimeError(
                "Generative study identity changed; preserve old run and register a new version"
            )
        write_json(manifest_path, manifest)

        def status(phase, **details):
            write_json(
                output / "status.json",
                {"phase": phase, "identity": identity, "updated_unix": time.time(), **details},
            )

        data = {}
        for split in paths:
            status("trajectory-cache", split=split)
            data[split] = dataset(
                paths[split],
                config["study"]["actions"],
                opts["scenarios"],
                output / "data-cache" / split,
            )
        actions = np.asarray(config["study"]["actions"])
        point_loss = read_json(robust / "pilot-selection.json")["loss"]
        training_records, samplers, policies = [], {}, {}
        pilot_config = copy.deepcopy(config)
        pilot_config["training"]["steps"] = opts["pilot_steps"]
        for family in ("flow", "gmm"):
            encoder = robust / "models" / f"response-{point_loss}-11" / "best.pt"
            # Encoders must be finished and immutable before any head is fitted.
            finished = read_json(robust / "final-training.json")
            if not any(
                r["family"] == "response" and r["seed"] == 11 and r["steps"] == 15000
                for r in finished
            ):
                raise RuntimeError("Finish the registered encoder training first")
            training_id = fingerprint(
                {
                    "study_identity": identity,
                    "training": pilot_config,
                    "family": family,
                    "seed": 11,
                    "encoder": sha(encoder),
                }
            )
            directory = output / "models" / f"pilot-{family}-11"
            status("pilot-training", family=family, seed=11)
            result = fit(
                family,
                11,
                pilot_config,
                data["train"],
                data["validation"],
                encoder,
                directory,
                training_id,
            )
            training_records.append(result)
            write_json(output / "pilot-training.json", training_records)
            sampler = BundleSampler(
                directory / "best.pt", actions, config["training"]["context"], "cpu"
            )
            samplers[family] = sampler
            for mode in ("expected", "plugin", "shuffled"):
                policies[f"{family}-{mode}", 11] = DistributionPolicy(
                    sampler,
                    actions,
                    mode,
                    opts["samples"],
                    opts["integration_steps"],
                    opts["sampling_seed"],
                )
        policies["point-projection", 11] = DistributionPolicy(samplers["flow"], actions, "point")
        status("pilot-evaluation")
        rows = evaluate_policies(
            output, "development-pilot", paths["validation"], config, opts, policies
        )
        if any(sha(robust / name) != value for name, value in references.items()):
            raise RuntimeError("Reference bytes changed during generative development")
        reference = read_json(robust / "baseline-selection.json")["feedback"]["name"]
        combined = rows + baselines
        stats = {
            family: comparison(combined, f"{family}-expected", reference, opts["scenarios"])
            for family in samplers
        }
        distributions = {}
        for family, sampler in samplers.items():
            status("distribution-calibration", family=family)
            distributions[family] = calibration(sampler, data["validation"], opts)
            write_json(output / "calibration.json", distributions)
        family = sorted(
            samplers,
            key=lambda f: (
                not distributions[f]["passed"],
                -min(stats[f][s]["score_ratio"] for s in opts["primary_scenarios"]),
                f != "gmm",
            ),
        )[0]
        gates = {
            s: {
                "score": stats[family][s]["score_ratio"] >= 1.03,
                "conversion": stats[family][s]["conversion_ratio"] >= 0.97,
                "violation": stats[family][s]["violation_delta"] <= 0.03,
            }
            for s in opts["primary_scenarios"]
        }
        gates["calibrated"] = {"score": stats[family]["calibrated"]["score_ratio"] >= 0.98}
        gates["distribution"] = {"calibrated": distributions[family]["passed"]}
        passed = all(all(g.values()) for g in gates.values())
        attribution = {
            f"{f}-{mode}": comparison(rows, f"{f}-{mode}", "point-projection", opts["scenarios"])
            for f in samplers
            for mode in ("expected", "plugin", "shuffled")
        }
        within_family = {
            f: {
                mode: comparison(rows, f"{f}-expected", f"{f}-{mode}", opts["scenarios"])
                for mode in ("plugin", "shuffled")
            }
            for f in samplers
        }
        selection = {
            "family": family,
            "gates": gates,
            "pilot_promoted": passed,
            "comparisons": stats,
            "attribution": attribution,
            "within_family": within_family,
            "calibration": distributions,
            "identity": identity,
            "scope": "development pilot; promotion permits three-seed exploration only",
            "generative_value_confirmed": False,
            "next_stage": "three-seed registered fit"
            if passed
            else "revise using development only",
        }
        write_json(output / "pilot-selection.json", selection)
        status("pilot-promoted" if passed else "pilot-not-promoted", family=family)
        print(f"generative pilot family={family} promoted={passed}", flush=True)
        if stage == "develop" and passed:
            raise RuntimeError("Register the final three-seed extension before execution")
        return output
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bidding-generative.toml")
    parser.add_argument("--stage", choices=["pilot", "develop"], default="pilot")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
