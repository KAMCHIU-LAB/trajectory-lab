"""Resumable, split-aware execution of the public bidding study."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import platform
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from trajectorylab.utils import fingerprint, read_json, write_json

from .learning import Feedback, Learned, fit, rollout, trajectories
from .market import ROOT, Tape, build_market, kernel_equivalence, scenario_bias, source_identity


def provenance(config: dict) -> dict:
    files = list(Path(__file__).parent.glob("*.py")) + [ROOT / "src/trajectorylab/models.py"]
    return {
        "config": config,
        "auction": source_identity(),
        "implementation": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": config["training"]["device"],
        },
    }


def validate(config: dict) -> None:
    study, training = config["study"], config["training"]
    groups = [study[k] for k in ("train_markets", "validation_markets", "test_markets")]
    joined = sum(groups, [])
    if not all(groups) or len(joined) != len(set(joined)):
        raise ValueError("Market splits must be nonempty, disjoint, and unique")
    actions = np.asarray(study["actions"])
    if actions[0] != 0 or actions[-1] != 3 or np.any(np.diff(actions) <= 0):
        raise ValueError("Actions must increase strictly from zero to three")
    if not study["training_seeds"] or len(set(study["training_seeds"])) != len(
        study["training_seeds"]
    ):
        raise ValueError("Training seeds must be nonempty and unique")
    if len(set(study["focal_agents"])) != len(study["focal_agents"]) or not all(
        0 <= f < 48 for f in study["focal_agents"]
    ):
        raise ValueError("Invalid focal advertiser IDs")
    if training["context"] < 1 or training["context"] > 48 or training["steps"] < 1:
        raise ValueError("Invalid context or training budget")
    for scenario in config["evaluation"]["scenarios"]:
        if any(scenario_bias(scenario, t) not in study["biases"] for t in range(48)):
            raise ValueError("Stress scenario missing from the declared exact action grid")


def evaluate(
    paths: list[Path], study: dict, scenarios: list[str], policies: dict, progress=None
) -> list[dict]:
    rows = []
    actions = np.asarray(study["actions"])
    for path in paths:
        tape = Tape(path)
        for fi, focal in enumerate(tape.arrays["focals"]):
            for scenario in scenarios:
                for (method, seed), policy in policies.items():
                    if hasattr(policy, "reset"):
                        policy.reset()
                    run = rollout(tape, fi, actions, scenario, policy)
                    rows.append(
                        {
                            "market": int(path.stem),
                            "focal": int(focal),
                            "scenario": scenario,
                            "method": method,
                            "seed": seed,
                            **run,
                        }
                    )
        if progress:
            progress(path.stem, len(rows))
    return rows


def tune(paths, config, directory):
    selection_path = directory / "selection.json"
    if selection_path.exists():
        return read_json(selection_path)
    study, opts = config["study"], config["evaluation"]
    actions = np.asarray(study["actions"])
    scenarios = opts["scenarios"]
    candidates = {}
    for gain in opts["feedback_gains"]:
        for prior in opts["correction_priors"]:
            candidates[(f"g{gain}-p{prior}", -1)] = Feedback(actions, gain, prior)
    rows = evaluate(paths, study, scenarios, candidates)
    scores = {
        key: float(np.mean([r["score"] for r in rows if r["method"] == key]))
        for key, _ in candidates
    }
    winner = max(scores, key=scores.get)
    chosen = next(policy for (name, _), policy in candidates.items() if name == winner)
    selection = {
        "feedback": {"gain": chosen.gain, "prior": chosen.prior},
        "feedback_validation_scores": scores,
        "correction": {},
        "selection_split": "validation only; equal weight per scenario/campaign",
    }
    fixed = {(str(float(k)), -1): (lambda s, u, r, b, c, k=float(k): (k, {})) for k in actions}
    fixed_rows = evaluate(paths, study, scenarios, fixed)
    fixed_scores = {
        key: float(np.mean([r["score"] for r in fixed_rows if r["method"] == key]))
        for key, _ in fixed
    }
    selection["fixed"] = {
        "action": float(max(fixed_scores, key=fixed_scores.get)),
        "validation_scores": fixed_scores,
    }
    for kind in ("dt", "response"):
        policies = {}
        for seed in study["training_seeds"]:
            for prior in opts["correction_priors"]:
                policies[(str(prior), seed)] = Learned(
                    directory / "models" / f"{kind}-{seed}" / "best.pt",
                    actions,
                    config["training"]["context"],
                    "cpu",
                    True,
                    prior,
                )
        rows = evaluate(paths, study, scenarios, policies)
        scores = {
            str(p): float(np.mean([r["score"] for r in rows if r["method"] == str(p)]))
            for p in opts["correction_priors"]
        }
        prior = float(max(scores, key=scores.get))
        selection["correction"][kind] = {"prior": prior, "validation_scores": scores}
        print(f"selection kind={kind} prior={prior} validation_scores={scores}", flush=True)
    write_json(selection_path, selection)
    return selection


def run(config_path: Path, output: Path | None = None, stage: str = "all") -> Path:
    config = tomllib.loads(config_path.read_text())
    validate(config)
    manifest = provenance(config)
    identity = fingerprint(manifest)
    directory = output or ROOT / "artifacts/bidding" / f"{config['study']['name']}-{identity[:10]}"
    directory.mkdir(parents=True, exist_ok=True)
    lock = (directory / ".run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock.close()
        raise RuntimeError("Another process already owns this experiment output") from error
    if (directory / "manifest.json").exists():
        if read_json(directory / "manifest.json")["identity"] != identity:
            raise RuntimeError(
                "Output belongs to different sources/configuration; select a new directory"
            )
    write_json(directory / "manifest.json", manifest | {"identity": identity})
    print(
        f"study={config['study']['name']} identity={identity[:10]} output={directory}", flush=True
    )
    started = time.perf_counter()

    def status(phase, **details):
        write_json(
            directory / "status.json",
            {
                "phase": phase,
                "elapsed_this_session_seconds": time.perf_counter() - started,
                "updated_unix": time.time(),
                **details,
            },
        )

    try:
        if stage in ("all", "data"):
            status("kernel-audit")
            write_json(directory / "kernel-audit.json", kernel_equivalence())
            for split in ("train", "validation", "test"):
                for market in config["study"][split + "_markets"]:
                    status("data", split=split, market=market)
                    meta = build_market(
                        directory / "markets" / f"{market}.npz", config["study"], market
                    )
                    print(
                        f"market={market} pv={meta['pv_count']} seconds={meta['seconds']:.1f} bytes={meta['bytes']}",
                        flush=True,
                    )
            if stage == "data":
                status("data-complete")
                return directory
        paths = {
            split: [directory / "markets" / f"{m}.npz" for m in config["study"][split + "_markets"]]
            for split in ("train", "validation", "test")
        }
        if not all(p.exists() for group in paths.values() for p in group):
            raise RuntimeError("Generate every declared market before training/evaluation")
        if stage in ("all", "train"):
            status("building-trajectories")
            actions = np.asarray(config["study"]["actions"])
            train = trajectories(paths["train"], actions)
            validation = trajectories(paths["validation"], actions)
            write_json(
                directory / "dataset-summary.json",
                {
                    "train_trajectories": len(train),
                    "validation_trajectories": len(validation),
                    "behavior_count_per_campaign": 10,
                    "ticks_per_trajectory": 48,
                    "train_behavior_mean_conversions": float(
                        np.mean([x["conversions"] for x in train])
                    ),
                    "validation_behavior_mean_conversions": float(
                        np.mean([x["conversions"] for x in validation])
                    ),
                    "label_source": "simulator interventions; dense counterfactual supervision for response model",
                },
            )
            results = []
            for seed in config["study"]["training_seeds"]:
                for kind in ("dt", "response"):
                    status("training", kind=kind, seed=seed)
                    result = fit(
                        kind,
                        seed,
                        config,
                        train,
                        validation,
                        directory / "models" / f"{kind}-{seed}",
                        identity,
                    )
                    results.append(result)
                    write_json(directory / "training.json", results)
            del train, validation
            if stage == "train":
                status("training-complete")
                return directory
        status("validation-selection")
        selection = tune(paths["validation"], config, directory)
        actions = np.asarray(config["study"]["actions"])
        policies = {("feedback", -1): Feedback(actions, **selection["feedback"])}
        for name, k in (("fixed-1", 1.0), ("fixed-tuned", selection["fixed"]["action"])):
            policies[(name, -1)] = lambda s, u, r, b, c, k=k: (k, {})
        for seed in config["study"]["training_seeds"]:
            for kind in ("dt", "response"):
                for corrected in (False, True):
                    name = kind + ("+correction" if corrected else "")
                    policies[(name, seed)] = Learned(
                        directory / "models" / f"{kind}-{seed}" / "best.pt",
                        actions,
                        config["training"]["context"],
                        "cpu",
                        corrected,
                        selection["correction"][kind]["prior"],
                    )
            policies[("response-hard", seed)] = Learned(
                directory / "models" / f"response-{seed}" / "best.pt",
                actions,
                config["training"]["context"],
                "cpu",
                False,
                hard_cpa=True,
            )
        rows = evaluate(
            paths["test"],
            config["study"],
            config["evaluation"]["scenarios"],
            policies,
            lambda market, n: status("evaluation", market=int(market), rows=n),
        )
        expected = (
            len(paths["test"])
            * len(config["study"]["focal_agents"])
            * len(config["evaluation"]["scenarios"])
            * len(policies)
        )
        keys = {(r["market"], r["focal"], r["scenario"], r["method"], r["seed"]) for r in rows}
        if len(rows) != expected or len(keys) != expected:
            raise RuntimeError("Evaluation is incomplete or contains duplicate identities")
        for row in rows:
            if any(not np.isfinite(v) for v in row.values() if isinstance(v, (float, int))):
                raise RuntimeError("Evaluation contains nonfinite evidence")
        if fingerprint(provenance(config)) != identity:
            raise RuntimeError("Sources changed while the study was running")
        write_json(directory / "results.json", rows)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (directory / "results.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        from .report import render

        render(directory)
        status("complete", rows=len(rows), purpose=config["study"]["purpose"])
        print(
            f"complete rows={len(rows)} elapsed_seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
    except BaseException as error:
        status(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
        )
        raise
    return directory
