"""Registered zero-refit risk controls on the frozen generative development pilot.

Only existing development tapes and frozen CPU samplers are used. Single-seed
promotion requests three-seed exploration; it never confirms generative value.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import time
import tomllib
from pathlib import Path

import numpy as np

from trajectorylab import bidding_generative as gen
from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.market import ROOT
from trajectorylab.bidding_robust import comparison, relative, sha, validated_rows
from trajectorylab.utils import fingerprint, read_json, runtime_info, write_json

FAMILIES = ("flow", "gmm")
SCENARIOS = ["calibrated", "over_25", "over_50", "under_25", "mid_over_50"]
PRIMARY = ["over_25", "over_50", "under_25"]
PROTOCOL = {
    "seed": 11,
    "scenarios": SCENARIOS,
    "primary_scenarios": PRIMARY,
    "deltas": [1.0, 0.75, 0.50, 0.25, 0.10],
    "hard_control_delta": 0.50,
    "samples": 64,
    "integration_steps": 8,
    "sampling_seed": 91347,
    "stress_score_min": 1.03,
    "stress_conversion_min": 0.97,
    "stress_violation_delta_max": 0.03,
    "nominal_score_min": 0.98,
    "value_mean_score_ratio_min": 1.01,
    "value_each_score_ratio_min": 0.98,
}
PARENT_REQUIRED = (
    "manifest.json",
    "pilot-training.json",
    "pilot-selection.json",
    "calibration.json",
    "development-pilot.json",
    "development-pilot.identity.json",
    "development-pilot-policy-identity.json",
)
REFERENCE_FILES = (
    "development-baselines.json",
    "development-baselines.identity.json",
    "baseline-selection.json",
    "pilot-selection.json",
)


def keep(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Frozen record changed: {path.name}")
    else:
        write_json(path, value)


def sources() -> dict:
    path = Path(__file__).resolve()
    return gen.sources() | {str(path.relative_to(ROOT)): sha(path)}


def calibration_passed(value: dict) -> bool:
    numbers = [
        value[k]
        for k in (
            "energy_score",
            "point_energy_score",
            "coverage80",
            "coverage95",
            "upper_cap_fraction",
        )
    ]
    passed = bool(
        np.isfinite(numbers).all()
        and value["energy_score"] <= 1.02 * value["point_energy_score"]
        and 0.60 <= value["coverage80"] <= 0.95
        and value["coverage95"] >= 0.80
        and value["upper_cap_fraction"] == 0.0
    )
    if value.get("passed") is not passed:
        raise RuntimeError("Parent distribution decision differs from its registered diagnostics")
    return passed


def prepare(config_path: Path):
    opts = tomllib.loads(config_path.read_text())
    if any(opts.get(key) != value for key, value in PROTOCOL.items()):
        raise ValueError("Control configuration differs from the registered zero-refit protocol")
    main, robust, parent, output = [
        relative(opts[k]) for k in ("main_run", "robust_run", "parent_run", "output")
    ]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (main, robust, parent)):
        raise ValueError("The control study requires a separate output directory")
    parent_config_path = relative(opts["parent_config"])
    parent_opts = tomllib.loads(parent_config_path.read_text())
    if parent_opts["pilot_steps"] != 4000 or parent_opts["seeds"] != [11, 23, 37]:
        raise RuntimeError("Only the registered seed-11 parent pilot may be reused")
    if any(
        parent_opts.get(k) != opts[k]
        for k in (
            "main_run",
            "robust_run",
            "samples",
            "integration_steps",
            "sampling_seed",
            "scenarios",
            "primary_scenarios",
        )
    ):
        raise RuntimeError("Parent and control protocols differ")
    if relative(parent_opts["output"]) != parent:
        raise RuntimeError("Parent configuration points to a different run")
    for name in PARENT_REQUIRED:
        if not (parent / name).is_file():
            raise RuntimeError(f"The completed parent pilot is required: {name}")
    manifest = read_json(parent / "manifest.json")
    if manifest["sources"] != gen.sources() or manifest["config"] != parent_opts:
        raise RuntimeError(
            "Parent sources or configuration changed; preserve its original identity"
        )
    main_manifest = read_json(main / "manifest.json")
    config = main_manifest["config"]
    if fingerprint(provenance(config)) != main_manifest["identity"]:
        raise RuntimeError("Frozen main-study provenance changed")
    if (
        config["study"]["train_markets"] != list(range(1001, 1017))
        or config["study"]["validation_markets"] != list(range(2001, 2007))
        or config["evaluation"]["scenarios"] != SCENARIOS
    ):
        raise RuntimeError("Only the registered 22 training/development markets may be used")
    paths = {
        split: [main / "markets" / f"{m}.npz" for m in config["study"][split + "_markets"]]
        for split in ("train", "validation")
    }
    inputs = {
        p.stem: {"npz": sha(p), "metadata": sha(p.with_suffix(".json"))}
        for group in paths.values()
        for p in group
    }
    references = {name: sha(robust / name) for name in REFERENCE_FILES}
    expected_parent = fingerprint(
        {
            "sources": gen.sources(),
            "config": parent_opts,
            "inputs": inputs,
            "references": references,
            "runtime": main_manifest["runtime"],
        }
    )
    if (
        manifest["inputs"] != inputs
        or manifest["references"] != references
        or manifest["identity"] != expected_parent
    ):
        raise RuntimeError("Parent data, reference bytes, or manifest identity changed")
    selection = read_json(parent / "pilot-selection.json")
    calibration = read_json(parent / "calibration.json")
    if (
        selection["identity"] != expected_parent
        or selection["family"] not in FAMILIES
        or selection["calibration"] != calibration
        or set(calibration) != set(FAMILIES)
    ):
        raise RuntimeError("Parent selection or calibration provenance differs")
    passed = {family: calibration_passed(calibration[family]) for family in FAMILIES}
    if any(passed.values()) and not passed[selection["family"]]:
        raise RuntimeError("The frozen selected parent family failed its distribution gate")
    # Freeze all durable parent JSON, including trajectory-cache identities. The
    # supervisor's operational status is not a scientific input.
    files = [p for p in parent.rglob("*.json") if p.name not in {"runner.json", "status.json"}]
    files += [parent / "models" / f"pilot-{family}-11" / "best.pt" for family in FAMILIES]
    files += [
        robust / name
        for name in (*REFERENCE_FILES, "development-manifest.json", "final-training.json")
    ]
    files += [main / "models" / f"response-{seed}" / "best.pt" for seed in parent_opts["seeds"]]
    point_loss = read_json(robust / "pilot-selection.json")["loss"]
    encoder = robust / "models" / f"response-{point_loss}-11" / "best.pt"
    files += [encoder, main / "manifest.json", parent_config_path]
    frozen_files = {str(path.resolve()): sha(path) for path in sorted(set(files))}
    core = {
        "sources": sources(),
        "config": opts,
        "config_sha256": sha(config_path),
        "parent_identity": expected_parent,
        "parent_family": selection["family"],
        "encoder_sha256": sha(encoder),
        "inputs": inputs,
        "frozen_files": frozen_files,
        "runtime": runtime_info(),
        "scope": "zero-refit, existing development markets, one seed, no confirmation",
    }
    return (
        opts,
        parent_opts,
        config,
        main,
        robust,
        parent,
        output,
        paths,
        selection,
        calibration,
        core,
    )


def rank(stats: dict, delta: float) -> tuple:
    return (
        -min(stats[s]["score_ratio"] for s in PRIMARY),
        -float(np.mean([stats[s]["score"] for s in PRIMARY])),
        -delta,
    )


def business_gates(stats: dict, opts: dict) -> dict:
    return {
        **{
            s: {
                "score": stats[s]["score_ratio"] >= opts["stress_score_min"],
                "conversion": stats[s]["conversion_ratio"] >= opts["stress_conversion_min"],
                "violation": stats[s]["violation_delta"] <= opts["stress_violation_delta_max"],
            }
            for s in PRIMARY
        },
        "calibrated": {"score": stats["calibrated"]["score_ratio"] >= opts["nominal_score_min"]},
    }


def value_gate(stats: dict, opts: dict) -> dict:
    mean_ratio = float(np.mean([stats[s]["score_ratio"] for s in PRIMARY]))
    each = {s: stats[s]["score_ratio"] >= opts["value_each_score_ratio_min"] for s in PRIMARY}
    return {
        "by_scenario": stats,
        "mean_stress_score_ratio": mean_ratio,
        "mean_passed": mean_ratio >= opts["value_mean_score_ratio_min"],
        "each_passed": each,
        "passed": mean_ratio >= opts["value_mean_score_ratio_min"] and all(each.values()),
    }


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    prepared = prepare(config_path)
    (
        opts,
        parent_opts,
        config,
        main,
        robust,
        parent,
        output,
        paths,
        parent_selection,
        calibration,
        core,
    ) = prepared
    identity = fingerprint(core)
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        keep(output / "manifest.json", {"identity": identity, "core": core})

        def guard():
            if prepare(config_path)[-1] != core:
                raise RuntimeError("Frozen control inputs changed during evaluation")

        def status(phase, **details):
            write_json(
                output / "status.json",
                {"phase": phase, "identity": identity, "updated_unix": time.time(), **details},
            )

        family = parent_selection["family"]
        result = {
            "identity": identity,
            "family": family,
            "calibration": calibration,
            "distribution_passed": bool(calibration[family]["passed"]),
            "business_promoted": False,
            "value_promoted": False,
            "needs_three_seed": False,
            "generative_value_confirmed": False,
            "scope": "single-seed exploratory control; no new markets and no refitting",
            "selection_rule": "business guards, worst stress score ratio to feedback, mean stress score, wider delta",
            "selected": None,
            "selected_shuffled": None,
            "thresholds": [],
        }
        if not any(value["passed"] for value in calibration.values()):
            guard()
            result["decision"] = "not-promoted"
            result["reason"] = (
                "No parent family passed distribution calibration; risk filtering cannot repair calibration"
            )
            keep(output / "control-selection.json", result)
            status("not-promoted", reason=result["reason"])
            return output

        import torch

        from trajectorylab.bidding_generative_learning import BundleSampler

        torch.set_num_threads(config["training"]["threads"])
        actions = np.asarray(config["study"]["actions"])
        samplers = {
            f: BundleSampler(
                parent / "models" / f"pilot-{f}-11" / "best.pt",
                actions,
                config["training"]["context"],
                "cpu",
            )
            for f in FAMILIES
        }
        encoder_hashes = {sampler.encoder_sha256 for sampler in samplers.values()}
        if encoder_hashes != {core["encoder_sha256"]}:
            raise RuntimeError(
                "Point controls require the same frozen encoder in both parent heads"
            )
        pilot_config = copy.deepcopy(config)
        pilot_config["training"]["steps"] = parent_opts["pilot_steps"]
        training = read_json(parent / "pilot-training.json")
        encoder_hash = next(iter(encoder_hashes))
        for f, sampler in samplers.items():
            expected_training = fingerprint(
                {
                    "study_identity": core["parent_identity"],
                    "training": pilot_config,
                    "family": f,
                    "seed": 11,
                    "encoder": encoder_hash,
                }
            )
            if (
                sampler.family != f
                or sampler.seed != 11
                or sampler.training_identity != expected_training
                or not any(
                    r["family"] == f
                    and r["seed"] == 11
                    and r["steps"] == parent_opts["pilot_steps"]
                    and r["encoder_sha256"] == encoder_hash
                    for r in training
                )
            ):
                raise RuntimeError("Parent generator training identity differs")

        def policy(sampler, mode, delta=1.0):
            return gen.DistributionPolicy(
                sampler,
                actions,
                mode,
                opts["samples"],
                opts["integration_steps"],
                opts["sampling_seed"],
                max_violation=delta,
            )

        parent_policies = {
            (f"{f}-{mode}", 11): policy(samplers[f], mode)
            for f in FAMILIES
            for mode in ("expected", "plugin", "shuffled")
        }
        parent_policies["point-projection", 11] = policy(samplers["flow"], "point")
        expected_evaluation = gen.evaluation_identity(parent_policies, parent_opts)
        if read_json(parent / "development-pilot-policy-identity.json") != {
            "identity": expected_evaluation
        }:
            raise RuntimeError("Parent evaluation source, options, or weight identity changed")
        # Existence was required above: this call only reads the parent cache and
        # checks every original slot, raw input identity, coverage and budget.
        parent_rows = validated_rows(
            parent / "development-pilot",
            paths["validation"],
            config["study"],
            SCENARIOS,
            parent_policies,
        )
        baseline_rows = gen.reference_rows(robust, main, paths["validation"], config, parent_opts)
        guard()
        reference = read_json(robust / "baseline-selection.json")["feedback"]["name"]
        delta_methods = {delta: f"{family}-expected-delta-{delta:g}" for delta in opts["deltas"]}
        rows = [r for r in parent_rows if r["method"] in {f"{family}-plugin", "point-projection"}]
        rows += [
            dict(r, method=delta_methods[1.0])
            for r in parent_rows
            if r["method"] == f"{family}-expected"
        ]
        sampler = samplers[family]

        def evaluate(name, policies):
            guard()
            suffixes = (".json", ".csv", ".identity.json", "-policy-identity.json")
            files = [output / f"{name}{suffix}" for suffix in suffixes]
            seal_path = output / f"{name}-artifacts.json"
            if seal_path.exists():
                expected = {"identity": identity, "files": {p.name: sha(p) for p in files}}
                if read_json(seal_path) != expected:
                    raise RuntimeError("Control evaluation bytes changed")
            elif files[0].exists():
                raise RuntimeError("Control cache exists without a complete byte seal")
            evaluation_opts = dict(
                opts,
                control_identity=identity,
                controls={
                    f"{m}:{s}": {"mode": p.mode, "max_violation": p.max_violation}
                    for (m, s), p in policies.items()
                },
            )
            evaluated = gen.evaluate_policies(
                output, name, paths["validation"], config, evaluation_opts, policies
            )
            guard()
            keep(seal_path, {"identity": identity, "files": {p.name: sha(p) for p in files}})
            return evaluated

        policies = {
            (delta_methods[delta], 11): policy(sampler, "expected", delta)
            for delta in opts["deltas"]
            if delta != 1.0
        }
        policies["plugin-hard", 11] = policy(sampler, "plugin", opts["hard_control_delta"])
        policies["point-hard", 11] = policy(sampler, "point", opts["hard_control_delta"])
        status("control-grid", family=family, slots=len(policies))
        rows += evaluate("development-controls", policies)
        combined = rows + baseline_rows
        controls = {}
        for kind, names in {
            "point": ("point-projection", "point-hard"),
            "plugin": (f"{family}-plugin", "plugin-hard"),
        }.items():
            records = [
                {
                    "method": name,
                    "delta": 1.0 if i == 0 else opts["hard_control_delta"],
                    "feedback": comparison(combined, name, reference, SCENARIOS),
                }
                for i, name in enumerate(names)
            ]
            winner = min(records, key=lambda r: rank(r["feedback"], r["delta"]))
            controls[kind] = {"selected": winner["method"], "variants": records}
        result["deterministic_controls"] = controls
        for delta, method in delta_methods.items():
            stats = comparison(combined, method, reference, SCENARIOS)
            gates = business_gates(stats, opts)
            result["thresholds"].append(
                {
                    "method": method,
                    "delta": delta,
                    "feedback": stats,
                    "business_gates": gates,
                    "business_passed": all(all(g.values()) for g in gates.values()),
                    "worst_stress_score_ratio": min(stats[s]["score_ratio"] for s in PRIMARY),
                    "mean_stress_score": float(np.mean([stats[s]["score"] for s in PRIMARY])),
                    "deterministic_attribution": {
                        kind: value_gate(
                            comparison(rows, method, control["selected"], SCENARIOS), opts
                        )
                        for kind, control in controls.items()
                    },
                    "all_control_comparisons": {
                        variant["method"]: comparison(rows, method, variant["method"], SCENARIOS)
                        for control in controls.values()
                        for variant in control["variants"]
                    },
                }
            )
        diagnostic = min(result["thresholds"], key=lambda r: rank(r["feedback"], r["delta"]))
        result["best_diagnostic"] = diagnostic
        eligible = [r for r in result["thresholds"] if r["business_passed"]]
        if not eligible:
            result["decision"] = "not-promoted"
            result["reason"] = "No risk threshold passed every registered business guard"
        else:
            selected = min(eligible, key=lambda r: rank(r["feedback"], r["delta"]))
            result["selected"] = selected
            result["business_promoted"] = True
            result["value_promoted"] = all(
                v["passed"] for v in selected["deterministic_attribution"].values()
            )
            result["needs_three_seed"] = result["value_promoted"]
            result["decision"] = (
                "exploratory-value-promoted" if result["value_promoted"] else "business-only"
            )
            result["reason"] = (
                "Three-seed paired exploration is required"
                if result["value_promoted"]
                else "Business guards passed; generative increment failed a deterministic-control gate"
            )
            shuffled = f"{family}-shuffled-delta-{selected['delta']:g}"
            status("selected-shuffled", family=family, delta=selected["delta"])
            shuffled_rows = evaluate(
                "development-selected-shuffled",
                {(shuffled, 11): policy(sampler, "shuffled", selected["delta"])},
            )
            result["selected_shuffled"] = {
                "method": shuffled,
                "delta": selected["delta"],
                "expected_vs_shuffled": comparison(
                    rows + shuffled_rows, selected["method"], shuffled, SCENARIOS
                ),
                "feedback": comparison(combined + shuffled_rows, shuffled, reference, SCENARIOS),
                "deterministic_attribution": {
                    kind: comparison(rows + shuffled_rows, shuffled, control["selected"], SCENARIOS)
                    for kind, control in controls.items()
                },
                "scope": "cost/conversion pairing diagnostic; not a distribution-calibration repair",
            }
        guard()
        keep(output / "control-selection.json", result)
        status(
            "complete" if result["business_promoted"] else "not-promoted",
            decision=result["decision"],
            family=family,
        )
        print(
            f"generative control family={family} business={result['business_promoted']} value={result['value_promoted']}",
            flush=True,
        )
        return output
    except Exception as error:
        write_json(
            output / "status.json",
            {
                "phase": "failed",
                "identity": identity,
                "updated_unix": time.time(),
                "error_type": type(error).__name__,
                "reason": str(error),
            },
        )
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bidding-generative-control.toml"
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
