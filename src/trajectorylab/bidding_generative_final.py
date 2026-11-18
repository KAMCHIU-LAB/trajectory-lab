"""Registered three-seed generative development, gated by the completed pilot.

Uses only the existing 16 training and six development markets. Business,
distribution and incremental generative value gates are recorded separately;
this entry point neither generates fresh markets nor performs confirmation.
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

from trajectorylab import bidding_generative as generative
from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.market import ROOT, effective_actions, source_identity
from trajectorylab.bidding_control import metrics
from trajectorylab.bidding_generative_learning import BundleSampler, fit
from trajectorylab.bidding_robust import relative, sha, training_identity
from trajectorylab.utils import fingerprint, read_json, write_json

SEEDS = [11, 23, 37]
FAMILIES = ("flow", "gmm")
SCENARIOS = ["calibrated", "over_25", "over_50", "under_25", "mid_over_50"]
PRIMARY = ["over_25", "over_50", "under_25"]


def _freeze(path: Path, record: dict) -> None:
    if path.exists():
        if read_json(path) != record:
            raise RuntimeError(f"Registered generative final identity changed: {path.name}")
    else:
        write_json(path, record)


def _finite(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, list):
        for child in value:
            _finite(child)
    elif isinstance(value, (int, float)) and not np.isfinite(value):
        raise RuntimeError("Nonfinite development evidence")


def _distribution_gates(record: dict) -> dict:
    _finite(record)
    return {
        "proper_score": 0 <= record["energy_score"] <= 1.02 * record["point_energy_score"],
        "coverage80": 0.60 <= record["coverage80"] <= 0.95,
        "coverage95": 0.80 <= record["coverage95"] <= 1.0,
        "no_upper_cap": record["upper_cap_fraction"] == 0.0,
    }


def _inputs(paths: dict, study: dict) -> dict:
    auction = source_identity()
    inputs = {}
    for group in paths.values():
        for path in group:
            meta = read_json(path.with_suffix(".json"))
            expected = fingerprint(
                {
                    "study": {
                        k: study[k] for k in ("focal_agents", "pv_count", "actions", "biases")
                    },
                    "market": int(path.stem),
                    "source": auction,
                }
            )
            if (
                meta["identity"] != expected
                or meta["market_id"] != int(path.stem)
                or meta["source"] != auction
                or meta["requested_pv_count"] != study["pv_count"]
                or meta["focal_agents"] != study["focal_agents"]
                or meta["effective_actions"] != effective_actions(study).tolist()
                or meta["bytes"] != path.stat().st_size
            ):
                raise RuntimeError(
                    "Raw market metadata differs from the original frozen environment"
                )
            inputs[path.stem] = {"npz": sha(path), "metadata": sha(path.with_suffix(".json"))}
    return inputs


def _prepare(config_path: Path):
    opts = tomllib.loads(config_path.read_text())
    main, robust, pilot, output = [
        relative(opts[k]) for k in ("main_run", "robust_run", "pilot_run", "output")
    ]
    if len({p.resolve() for p in (main, robust, pilot, output)}) != 4:
        raise ValueError("Final output must be independent of original, robust and pilot runs")
    main_manifest = read_json(main / "manifest.json")
    config = main_manifest["config"]
    current_provenance = provenance(config)
    if fingerprint(current_provenance) != main_manifest["identity"]:
        raise RuntimeError("Frozen original provenance changed")
    if (
        opts["seeds"] != SEEDS
        or opts["scenarios"] != SCENARIOS
        or opts["primary_scenarios"] != PRIMARY
        or opts["final_steps"] != 15000
        or opts["encoder_loss"] != "log"
        or opts["samples"] != 64
        or opts["integration_steps"] != 8
        or opts["sampling_seed"] != 91347
        or opts["calibration_cases_per_market_scene"] != 8
        or config["study"]["train_markets"] != list(range(1001, 1017))
        or config["study"]["validation_markets"] != list(range(2001, 2007))
        or config["evaluation"]["scenarios"] != SCENARIOS
    ):
        raise ValueError("Unexpected registered three-seed development protocol")
    pilot_manifest = read_json(pilot / "manifest.json")
    selection = read_json(pilot / "pilot-selection.json")
    pilot_status = read_json(pilot / "status.json")
    calibration = read_json(pilot / "calibration.json")
    if (
        selection.get("pilot_promoted") is not True
        or selection.get("identity") != pilot_manifest["identity"]
        or selection.get("family") not in FAMILIES
        or pilot_status.get("phase") != "pilot-promoted"
        or pilot_status.get("identity") != pilot_manifest["identity"]
        or pilot_manifest["sources"] != generative.sources()
        or selection.get("calibration") != calibration
        or not all(_distribution_gates(calibration[selection["family"]]).values())
    ):
        raise RuntimeError("A completed, current-source, promoted generative pilot is required")
    pilot_opts = pilot_manifest["config"]
    for key in (
        "main_run",
        "robust_run",
        "scenarios",
        "primary_scenarios",
        "samples",
        "integration_steps",
        "sampling_seed",
        "calibration_cases_per_market_scene",
    ):
        if pilot_opts[key] != opts[key]:
            raise RuntimeError(f"Final protocol differs from pilot: {key}")
    expected_pilot = fingerprint(
        {
            "sources": pilot_manifest["sources"],
            "config": pilot_opts,
            "inputs": pilot_manifest["inputs"],
            "references": pilot_manifest["references"],
            "runtime": main_manifest["runtime"],
        }
    )
    if expected_pilot != pilot_manifest["identity"] or relative(pilot_opts["output"]) != pilot:
        raise RuntimeError("Pilot manifest identity is invalid")
    paths = {
        split: [main / "markets" / f"{m}.npz" for m in config["study"][split + "_markets"]]
        for split in ("train", "validation")
    }
    inputs = _inputs(paths, config["study"])
    if inputs != pilot_manifest["inputs"]:
        raise RuntimeError("Training/development data changed since the pilot")
    if read_json(robust / "pilot-selection.json")["loss"] != "log":
        raise RuntimeError("This extension requires the registered response-log encoders")
    completed = read_json(robust / "final-training.json")
    encoders = {seed: robust / "models" / f"response-log-{seed}" / "best.pt" for seed in SEEDS}
    training_config = copy.deepcopy(config)
    training_config["training"]["steps"] = 15000
    for seed, encoder in encoders.items():
        if (
            len(
                [
                    r
                    for r in completed
                    if r["family"] == "response"
                    and r["seed"] == seed
                    and r["loss"] == "log"
                    and r["steps"] == 15000
                ]
            )
            != 1
        ):
            raise RuntimeError("All three corresponding response-log encoders must finish first")
        saved = torch.load(encoder, map_location="cpu", weights_only=False)
        expected = training_identity(training_config, main, SCENARIOS, "response", "log", seed)
        if (
            saved.get("identity") != expected
            or saved.get("seed") != seed
            or saved.get("family") != "response"
            or saved.get("loss") != "log"
        ):
            raise RuntimeError("Frozen encoder metadata differs from registered same-seed training")
    pilot_config = copy.deepcopy(config)
    pilot_config["training"]["steps"] = pilot_opts["pilot_steps"]
    for family in FAMILIES:
        sampler = BundleSampler(
            pilot / "models" / f"pilot-{family}-11" / "best.pt",
            np.asarray(config["study"]["actions"]),
            config["training"]["context"],
            "cpu",
        )
        expected = fingerprint(
            {
                "study_identity": pilot_manifest["identity"],
                "training": pilot_config,
                "family": family,
                "seed": 11,
                "encoder": sha(encoders[11]),
            }
        )
        if (
            sampler.training_identity != expected
            or sampler.family != family
            or sampler.seed != 11
            or sampler.encoder_sha256 != sha(encoders[11])
        ):
            raise RuntimeError("Pilot checkpoint differs from the promoted pilot identity")
    for name, digest in pilot_manifest["references"].items():
        if sha(robust / name) != digest:
            raise RuntimeError("Pilot baseline reference changed")
    # Do not let reference_rows manufacture missing old baseline evidence.
    for name in ("development-baselines.json", "development-baselines.identity.json"):
        if not (robust / name).is_file():
            raise RuntimeError("Existing baseline evidence is required")
    baselines = generative.reference_rows(robust, main, paths["validation"], config, opts)
    frozen_files = [config_path.resolve()]
    frozen_files += [
        main / f"{name}.json" for name in ("manifest", "selection", "status", "training")
    ]
    frozen_files += [
        pilot / f"{name}.json"
        for name in (
            "manifest",
            "pilot-selection",
            "calibration",
            "pilot-training",
            "status",
            "development-pilot",
            "development-pilot.identity",
            "development-pilot-policy-identity",
        )
    ]
    frozen_files += [robust / name for name in pilot_manifest["references"]]
    frozen_files += [robust / "development-manifest.json", robust / "final-training.json"]
    frozen_files += list(encoders.values())
    frozen_files += [pilot / "models" / f"pilot-{f}-11" / "best.pt" for f in FAMILIES]
    frozen_files += [main / "models" / f"response-{s}" / "best.pt" for s in SEEDS]
    for split in paths:
        for name in ("trajectories.npz", "identity.json"):
            cache_file = pilot / "data-cache" / split / name
            if not cache_file.is_file():
                raise RuntimeError(
                    "Final development requires the existing pilot trajectory caches"
                )
            frozen_files.append(cache_file)
    sources = generative.sources() | {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            Path(__file__),
            ROOT / "src/trajectorylab/__init__.py",
            ROOT / "third_party/auctionnet/BiddingEnv.py",
            ROOT / "third_party/auctionnet/NeurIPSPvGen.py",
        )
    }
    core = {
        "sources": sources,
        "config": opts,
        "training_config": training_config,
        "runtime": current_provenance["runtime"],
        "inputs": inputs,
        "files": {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(frozen_files))},
        "pilot_identity": pilot_manifest["identity"],
        "pilot_family": selection["family"],
        "encoder_sha256": {str(s): sha(p) for s, p in encoders.items()},
        "scope": "three-seed development selection; 16 training and six previously used development markets only",
        "slots": [
            [f"{f}-{mode}", seed]
            for f in FAMILIES
            for seed in SEEDS
            for mode in ("expected", "plugin")
        ]
        + [["point-projection", s] for s in SEEDS]
        + [[f"{selection['family']}-shuffled", 11]],
    }
    return opts, config, training_config, paths, encoders, pilot, output, baselines, core


def _guard(core: dict) -> None:
    for name, digest in (core["sources"] | core["files"]).items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Frozen final-development input changed: {name}")


def _compare(rows, method, reference, scenes=SCENARIOS):
    result = {}
    for scene in scenes:
        a, b = metrics(rows, method, scene), metrics(rows, reference, scene)
        result[scene] = {
            "score": a["score"],
            "reference_score": b["score"],
            "score_delta": a["score"] - b["score"],
            "score_ratio": a["score"] / b["score"] if b["score"] > 0 else None,
            "conversion_ratio": a["conversions"] / b["conversions"]
            if b["conversions"] > 0
            else None,
            "violation_delta": a["violation"] - b["violation"],
            "seed_score_deltas": {
                str(s): metrics([r for r in rows if r["seed"] == s], method, scene)["score"]
                - (
                    metrics([r for r in rows if r["seed"] == s], reference, scene)["score"]
                    if any(r["method"] == reference and r["seed"] == s for r in rows)
                    else b["score"]
                )
                for s in sorted({r["seed"] for r in rows if r["method"] == method})
            },
        }
    return result


def _at_least(value, threshold) -> bool:
    return value is not None and value >= threshold


def _finished(records: list[dict], output: Path) -> None:
    if read_json(output / "final-training.json") != records:
        raise RuntimeError("Final training record changed during execution")
    for record in records:
        directory = output / "models" / f"{record['family']}-{record['seed']}"
        if record["best_sha256"] != sha(directory / "best.pt") or record[
            "checkpoint_sha256"
        ] != sha(directory / "checkpoint.pt"):
            raise RuntimeError("Finished generator weights changed during execution")


def _value_gates(rows, family: str, reference: str) -> dict:
    method = f"{family}-expected"
    stats = _compare(rows, method, reference, PRIMARY)
    seed_deltas = {}
    for seed in SEEDS:
        paired = [r for r in rows if r["seed"] == seed]
        seed_deltas[str(seed)] = float(
            np.mean(
                [
                    metrics(paired, method, s)["score"] - metrics(paired, reference, s)["score"]
                    for s in PRIMARY
                ]
            )
        )
    ratios = [stats[s]["score_ratio"] for s in PRIMARY]
    mean_ratio = float(np.mean(ratios)) if all(v is not None for v in ratios) else None
    gates = {
        "equal_scene_mean_ratio_at_least_1.01": _at_least(mean_ratio, 1.01),
        "every_scene_ratio_at_least_0.98": all(_at_least(v, 0.98) for v in ratios),
        "every_paired_seed_mean_positive": all(v > 0 for v in seed_deltas.values()),
    }
    return {
        "gates": gates,
        "passed": all(gates.values()),
        "equal_scene_mean_score_ratio": mean_ratio,
        "paired_seed_mean_score_deltas": seed_deltas,
        "comparisons": stats,
    }


def _select(rows, baselines, calibration: dict, core: dict, reference: str) -> dict:
    combined = rows + baselines
    stats = {f: _compare(combined, f"{f}-expected", reference) for f in FAMILIES}
    records = {}
    for family in FAMILIES:
        business = {
            scene: {
                "score": _at_least(stats[family][scene]["score_ratio"], 1.03),
                "conversion": _at_least(stats[family][scene]["conversion_ratio"], 0.97),
                "violation": stats[family][scene]["violation_delta"] <= 0.03,
                "all_seeds_positive": set(stats[family][scene]["seed_score_deltas"])
                == {str(s) for s in SEEDS}
                and all(v > 0 for v in stats[family][scene]["seed_score_deltas"].values()),
            }
            for scene in PRIMARY
        }
        business["calibrated"] = {
            "score": _at_least(stats[family]["calibrated"]["score_ratio"], 0.98)
        }
        distribution = {
            str(s): _distribution_gates(calibration[f"{family}:{s}"]["result"]) for s in SEEDS
        }
        value = {
            name: _value_gates(rows, family, method)
            for name, method in (("point", "point-projection"), ("plugin", f"{family}-plugin"))
        }
        records[family] = {
            "business_gates": business,
            "business_passed": all(all(g.values()) for g in business.values()),
            "distribution_gates": distribution,
            "distribution_passed": all(all(g.values()) for g in distribution.values()),
            "generative_value": value,
            "generative_value_passed": all(v["passed"] for v in value.values()),
            "worst_primary_score_ratio": min(stats[family][s]["score_ratio"] for s in PRIMARY)
            if all(stats[family][s]["score_ratio"] is not None for s in PRIMARY)
            else None,
        }
    eligible = [
        f for f in FAMILIES if records[f]["business_passed"] and records[f]["distribution_passed"]
    ]
    family = (
        sorted(eligible, key=lambda f: (-records[f]["worst_primary_score_ratio"], f != "gmm"))[0]
        if eligible
        else None
    )
    promoted = family is not None
    value_promoted = promoted and records[family]["generative_value_passed"]
    return {
        "family": family,
        "business_promoted": promoted,
        "value_promoted": value_promoted,
        "needs_confirmation": value_promoted,
        "generative_value_confirmed": False,
        "eligible_families": eligible,
        "family_records": records,
        "comparisons": stats,
        "flow_vs_gmm": _compare(rows, "flow-expected", "gmm-expected"),
        "expected_vs_plugin": {f: _compare(rows, f"{f}-expected", f"{f}-plugin") for f in FAMILIES},
        "expected_vs_point": {
            f: _compare(rows, f"{f}-expected", "point-projection") for f in FAMILIES
        },
        "pilot_family_shuffled_seed11": _compare(
            [r for r in rows if r["seed"] == 11],
            f"{core['pilot_family']}-expected",
            f"{core['pilot_family']}-shuffled",
        ),
        "scope": core["scope"],
        "limits": [
            "Development screening only; all six markets were previously used for selection.",
            "Seed results describe three fitted encoder/head pairs, not future retraining or online performance.",
            "A distributional contribution gate does not establish an effect unique to flow; GMM remains an eligible family.",
            "Shuffling is a seed-11 development diagnostic, not a separate causal proof.",
        ],
    }


def run(config_path: Path, stage: str = "develop") -> Path:
    if stage not in ("train", "evaluate", "develop"):
        raise ValueError("Expected stage=train|evaluate|develop")
    opts = tomllib.loads(config_path.read_text())
    output = relative(opts["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        opts, config, training_config, paths, encoders, pilot, output, baselines, core = _prepare(
            config_path
        )
        identity = fingerprint(core)
        manifest = output / "manifest.json"
        if not manifest.exists() and any(p.name != ".run.lock" for p in output.iterdir()):
            raise RuntimeError(
                "Register the final run before creating training or evaluation artifacts"
            )
        _freeze(manifest, {"identity": identity, "core": core})

        def status(phase, **details):
            write_json(
                output / "status.json",
                {
                    "identity": identity,
                    "phase": phase,
                    "updated_unix": time.time(),
                    "scope": core["scope"],
                    **details,
                },
            )

        _guard(core)
        # dataset validates both the existing cache bytes and current raw inputs.
        data = {
            split: generative.dataset(
                group, config["study"]["actions"], SCENARIOS, pilot / "data-cache" / split
            )
            for split, group in paths.items()
        }
        torch.set_num_threads(config["training"]["threads"])
        records_path = output / "final-training.json"
        records = read_json(records_path) if records_path.exists() else []
        if len({(r["family"], r["seed"]) for r in records}) != len(records) or any(
            r["family"] not in FAMILIES or r["seed"] not in SEEDS for r in records
        ):
            raise RuntimeError("Unexpected final training record coverage")
        existing = {(r["family"], r["seed"]): r for r in records}
        samplers, policies = {}, {}
        for family in FAMILIES:
            for seed in SEEDS:
                _guard(core)
                training_id = fingerprint(
                    {
                        "run_identity": identity,
                        "training_config": training_config,
                        "family": family,
                        "seed": seed,
                        "encoder_sha256": core["encoder_sha256"][str(seed)],
                    }
                )
                directory = output / "models" / f"{family}-{seed}"
                record = existing.get((family, seed))
                if record is None:
                    if stage == "evaluate":
                        raise RuntimeError("Finish all six registered models before evaluation")
                    status("final-training", family=family, seed=seed)
                    result = fit(
                        family,
                        seed,
                        training_config,
                        data["train"],
                        data["validation"],
                        encoders[seed],
                        directory,
                        training_id,
                    )
                    if (
                        result["steps"] != 15000
                        or result["family"] != family
                        or result["seed"] != seed
                    ):
                        raise RuntimeError("Final training did not complete the registered budget")
                    record = result | {
                        "training_identity": training_id,
                        "best_sha256": sha(directory / "best.pt"),
                        "checkpoint_sha256": sha(directory / "checkpoint.pt"),
                    }
                    existing[family, seed] = record
                    records = [existing[k] for k in sorted(existing)]
                    write_json(records_path, records)
                if (
                    record["training_identity"] != training_id
                    or record["steps"] != 15000
                    or record["encoder_sha256"] != core["encoder_sha256"][str(seed)]
                    or record["best_sha256"] != sha(directory / "best.pt")
                    or record["checkpoint_sha256"] != sha(directory / "checkpoint.pt")
                ):
                    raise RuntimeError(
                        "Finished model weights, budget or training identity changed"
                    )
                sampler = BundleSampler(
                    directory / "best.pt",
                    np.asarray(config["study"]["actions"]),
                    config["training"]["context"],
                    "cpu",
                )
                if (
                    sampler.training_identity != training_id
                    or sampler.encoder_sha256 != core["encoder_sha256"][str(seed)]
                    or sampler.family != family
                    or sampler.seed != seed
                ):
                    raise RuntimeError(
                        "Final sampler does not match its registered same-seed encoder/head"
                    )
                samplers[family, seed] = sampler
                for mode in ("expected", "plugin"):
                    policies[f"{family}-{mode}", seed] = generative.DistributionPolicy(
                        sampler,
                        config["study"]["actions"],
                        mode,
                        opts["samples"],
                        opts["integration_steps"],
                        opts["sampling_seed"],
                    )
        _guard(core)
        _finished(records, output)
        if stage == "train":
            if _inputs(paths, config["study"]) != core["inputs"]:
                raise RuntimeError("Frozen raw training/development markets changed during fitting")
            status("final-training-complete", models=6)
            return output
        for seed in SEEDS:
            policies["point-projection", seed] = generative.DistributionPolicy(
                samplers["flow", seed],
                config["study"]["actions"],
                "point",
                opts["samples"],
                opts["integration_steps"],
                opts["sampling_seed"],
            )
        pilot_family = core["pilot_family"]
        policies[f"{pilot_family}-shuffled", 11] = generative.DistributionPolicy(
            samplers[pilot_family, 11],
            config["study"]["actions"],
            "shuffled",
            opts["samples"],
            opts["integration_steps"],
            opts["sampling_seed"],
        )
        if {tuple(s) for s in core["slots"]} != set(policies) or len(policies) != 16:
            raise RuntimeError("Final policy coverage differs from registered 16 slots")
        evaluation_opts = opts | {"final_manifest_identity": identity}
        status("final-development-evaluation", policy_slots=16)
        rows = generative.evaluate_policies(
            output, "development-final", paths["validation"], config, evaluation_opts, policies
        )
        _finished(records, output)
        calibration_path = output / "calibration.json"
        calibrated = (
            read_json(calibration_path)
            if calibration_path.exists()
            else {"identity": identity, "models": {}}
        )
        if calibrated["identity"] != identity or not set(calibrated["models"]).issubset(
            {f"{f}:{s}" for f in FAMILIES for s in SEEDS}
        ):
            raise RuntimeError("Calibration belongs to a different final run")
        for (family, seed), sampler in samplers.items():
            _guard(core)
            key = f"{family}:{seed}"
            if key not in calibrated["models"]:
                status("distribution-calibration", family=family, seed=seed)
                result = generative.calibration(sampler, data["validation"], evaluation_opts)
                calibrated["models"][key] = {
                    "checkpoint_sha256": sampler.identity,
                    "result": result,
                }
                write_json(calibration_path, calibrated)
            record = calibrated["models"][key]
            if (
                record["checkpoint_sha256"] != sampler.identity
                or record["result"]["cases"] != 6 * 5 * opts["calibration_cases_per_market_scene"]
                or record["result"]["passed"] != all(_distribution_gates(record["result"]).values())
            ):
                raise RuntimeError("Calibration checkpoint, coverage or gate record changed")
        if _prepare(config_path)[-1] != core:
            raise RuntimeError(
                "Frozen data, references or sources changed during final development"
            )
        _finished(records, output)
        reference = read_json(relative(opts["robust_run"]) / "baseline-selection.json")["feedback"][
            "name"
        ]
        selection = _select(rows, baselines, calibrated["models"], core, reference)
        selection |= {
            "identity": identity,
            "reference": reference,
            "rows": len(rows),
            "policy_slots": 16,
            "calibration": calibrated["models"],
            "training": records,
            "development_evidence_sha256": {
                name: sha(output / name)
                for name in (
                    "development-final.json",
                    "development-final.identity.json",
                    "development-final-policy-identity.json",
                    "calibration.json",
                )
            },
        }
        _finite(selection)
        _freeze(output / "candidate-selection.json", selection)
        status(
            "development-promoted"
            if selection["needs_confirmation"]
            else "development-not-promoted",
            family=selection["family"],
            business_promoted=selection["business_promoted"],
            value_promoted=selection["value_promoted"],
            needs_confirmation=selection["needs_confirmation"],
        )
        print(
            f"generative final family={selection['family']} business={selection['business_promoted']} value={selection['value_promoted']} needs_confirmation={selection['needs_confirmation']}",
            flush=True,
        )
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bidding-generative-final.toml"
    )
    parser.add_argument("--stage", choices=["train", "evaluate", "develop"], default="develop")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
