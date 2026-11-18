"""Development-only robust response study; frozen studies remain read-only.

Scenario names belong to the offline data builder and evaluator. Policies receive
only the original observation/feedback interface. No fresh market is generated
until the development promotion and complete configuration freeze succeed.
"""

from __future__ import annotations

import argparse
import copy
import csv
import fcntl
import hashlib
import io
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from trajectorylab.bidding.experiment import evaluate, provenance
from trajectorylab.bidding.learning import Feedback, Learned
from trajectorylab.bidding.market import ROOT
from trajectorylab.bidding_control import metrics
from trajectorylab.utils import fingerprint, read_json, write_json


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Study paths must be repository-relative")
    return ROOT / path


def sources() -> dict:
    files = list((ROOT / "src/trajectorylab/bidding").glob("*.py"))
    files += [
        ROOT / "src/trajectorylab" / name
        for name in ("bidding_robust.py", "bidding_robust_data.py", "bidding_robust_learning.py")
    ]
    files += [ROOT / "src/trajectorylab/models.py", ROOT / "src/trajectorylab/utils.py"]
    files += [ROOT / "src/trajectorylab/bidding_control.py"]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def setup(config_path: Path) -> tuple[dict, dict, Path, Path]:
    opts = tomllib.loads(config_path.read_text())
    main, output = relative(opts["main_run"]), relative(opts["output"])
    manifest = read_json(main / "manifest.json")
    config = manifest["config"]
    if fingerprint(provenance(config)) != manifest["identity"]:
        raise RuntimeError("Frozen main provenance changed")
    if opts["scenarios"] != config["evaluation"]["scenarios"]:
        raise ValueError("Stress set differs from frozen environment")
    if opts["seeds"] != [11, 23, 37] or opts["primary_scenarios"] != [
        "over_25",
        "over_50",
        "under_25",
    ]:
        raise ValueError("Unexpected seeds or co-primary scenes")
    used = sum(
        [config["study"][k] for k in ("train_markets", "validation_markets", "test_markets")], []
    ) + list(range(3301, 3325))
    fresh = opts["confirmation_markets"]
    if len(fresh) != 24 or len(set(fresh)) != 24 or set(fresh).intersection(used):
        raise ValueError("Confirmation requires 24 distinct unused markets")
    output.mkdir(parents=True, exist_ok=True)
    return opts, config, main, output


def save_rows(path: Path, rows: list[dict]) -> None:
    write_json(path.with_suffix(".json"), rows)
    buffer = io.StringIO()
    keys = sorted(set().union(*(r.keys() for r in rows)))
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    temp = path.with_suffix(".partial.csv")
    temp.write_text(buffer.getvalue())
    temp.replace(path.with_suffix(".csv"))


def policy_digest(policy) -> dict:
    if not hasattr(policy, "model"):
        return {"gain": getattr(policy, "gain", None), "prior": getattr(policy, "prior", None)}
    digest = hashlib.sha256()
    for name, value in sorted(policy.model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    for name in ("mean", "std"):
        digest.update(np.asarray(getattr(policy, name)).tobytes())
    return {
        "weights_normalizer_sha256": digest.hexdigest(),
        "context": policy.context,
        "kind": policy.kind,
        "corrected": policy.corrected,
        "prior": policy.prior,
        "hard_cpa": policy.hard_cpa,
    }


def validated_rows(path: Path, paths, study, scenarios, policies) -> list[dict]:
    """Reuse only results with exact coverage and matching policy/data identities."""
    identity = fingerprint(
        {
            "markets": {
                p.stem: {"metadata": sha(p.with_suffix(".json")), "npz": sha(p)} for p in paths
            },
            "scenarios": scenarios,
            "slots": sorted([list(k) for k in policies]),
            "policies": {f"{m}:{s}": policy_digest(p) for (m, s), p in policies.items()},
            "sources": sources(),
        }
    )
    stamp = path.with_suffix(".identity.json")
    if path.with_suffix(".json").exists():
        if not stamp.exists() or read_json(stamp)["identity"] != identity:
            raise RuntimeError("Cached evaluation identity differs; use a new result path")
        rows = read_json(path.with_suffix(".json"))
    else:
        rows = evaluate(
            paths,
            study,
            scenarios,
            policies,
            lambda market, n: print(f"evaluate market={market} rows={n}", flush=True),
        )
        save_rows(path, rows)
        write_json(stamp, {"identity": identity})
    expected = {
        (int(p.stem), int(f), s, m, seed)
        for p in paths
        for f in study["focal_agents"]
        for s in scenarios
        for m, seed in policies
    }
    actual = [(r["market"], r["focal"], r["scenario"], r["method"], r["seed"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Evaluation coverage mismatch")
    for row in rows:
        if row["cost"] > row["budget"] + 1e-6:
            raise RuntimeError("Hard budget violated")
        if any(not np.isfinite(v) for v in row.values() if isinstance(v, (int, float))):
            raise RuntimeError("Nonfinite evaluation metric")
    return rows


def comparison(rows, method, reference, scenarios) -> dict:
    result = {}
    for scene in scenarios:
        a, b = metrics(rows, method, scene), metrics(rows, reference, scene)
        result[scene] = {
            "score": a["score"],
            "reference_score": b["score"],
            "score_ratio": a["score"] / max(b["score"], 1e-12),
            "conversion_ratio": a["conversions"] / max(b["conversions"], 1e-12),
            "violation_delta": a["violation"] - b["violation"],
            "seed_score_deltas": {
                str(seed): float(
                    np.mean(
                        [
                            r["score"]
                            for r in rows
                            if r["method"] == method
                            and r["scenario"] == scene
                            and r["seed"] == seed
                        ]
                    )
                )
                - b["score"]
                for seed in sorted({r["seed"] for r in rows if r["method"] == method})
            },
        }
    return result


def baselines(opts, config, main, output, paths) -> tuple[list[dict], dict]:
    actions = np.asarray(config["study"]["actions"])
    policies = {("feedback-old", -1): Feedback(actions, 0.3, 8.0)}
    for gain in opts["feedback_gains"]:
        for prior in opts["feedback_priors"]:
            policies[(f"feedback-g{gain}-p{prior}", -1)] = Feedback(actions, gain, prior)
    for action in actions:
        policies[(f"fixed-{action}", -1)] = lambda s, u, r, b, c, k=float(action): (k, {})
    for seed in opts["seeds"]:
        policies[("response-old", seed)] = Learned(
            main / "models" / f"response-{seed}" / "best.pt",
            actions,
            config["training"]["context"],
            "cpu",
        )
    rows = validated_rows(
        output / "development-baselines", paths, config["study"], opts["scenarios"], policies
    )
    eligible = []
    records = []
    for gain in opts["feedback_gains"]:
        for prior in opts["feedback_priors"]:
            name = f"feedback-g{gain}-p{prior}"
            stats = comparison(rows, name, "feedback-old", opts["scenarios"])
            minimum = min(stats[s]["score_ratio"] for s in opts["primary_scenarios"])
            pooled = float(np.mean([stats[s]["score"] for s in opts["primary_scenarios"]]))
            record = {
                "name": name,
                "gain": gain,
                "prior": prior,
                "worst_ratio": minimum,
                "pooled_score": pooled,
                "stats": stats,
            }
            records.append(record)
            if stats["calibrated"]["score_ratio"] >= 0.98:
                eligible.append(record)
    winner = sorted(
        eligible, key=lambda r: (-r["worst_ratio"], -r["pooled_score"], r["gain"], r["prior"])
    )[0]
    fixed = sorted(
        (float(a) for a in actions),
        key=lambda a: (
            -float(np.mean([r["score"] for r in rows if r["method"] == f"fixed-{a}"])),
            a,
        ),
    )[0]
    selection = {
        "feedback": {k: winner[k] for k in ("name", "gain", "prior")},
        "feedback_grid": records,
        "fixed": fixed,
        "selection_split": "development only; one setting across all scenes",
    }
    write_json(output / "baseline-selection.json", selection)
    return rows, selection


def training_identity(config, main, training_scenes, family, loss, seed) -> str:
    code = sources()
    # The coordinator does not affect fitted weights, but its revision is stored
    # with every result and is frozen for confirmation separately.
    code.pop("src/trajectorylab/bidding_robust.py", None)
    markets = {
        str(m): sha(main / "markets" / f"{m}.json")
        for split in ("train_markets", "validation_markets")
        for m in config["study"][split]
    }
    return fingerprint(
        {
            "config": config,
            "family": family,
            "loss": loss,
            "seed": seed,
            "scenarios": training_scenes,
            "source": code,
            "markets": markets,
        }
    )


def develop(config_path: Path, stage: str) -> Path:
    from trajectorylab.bidding_robust_data import audit, trajectories
    from trajectorylab.bidding_robust_learning import RobustLearned, fit

    opts, config, main, output = setup(config_path)
    torch.set_num_threads(config["training"]["threads"])
    lock = (output / ".development.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    revision = fingerprint({"sources": sources(), "config": opts})
    write_json(
        output / "development-manifest.json",
        {
            "identity": revision,
            "sources": sources(),
            "config": opts,
            "frozen_main_identity": read_json(main / "manifest.json")["identity"],
            "started_unix": time.time(),
            "scope": "development only, not confirmation",
        },
    )

    def status(phase, **details):
        write_json(
            output / "status.json",
            {"phase": phase, "updated_unix": time.time(), "revision": revision, **details},
        )

    paths = {
        split: [main / "markets" / f"{m}.npz" for m in config["study"][split + "_markets"]]
        for split in ("train", "validation")
    }
    try:
        status("development-baselines")
        base_rows, selection = baselines(opts, config, main, output, paths["validation"])
        if stage == "baselines":
            status("development-baselines-complete")
            return output
        status("label-audit")
        write_json(
            output / "label-audit.json",
            audit(paths["train"][0], np.asarray(config["study"]["actions"])),
        )
        datasets = {}
        for label, scenes in (("nominal", ["calibrated"]), ("augmented", opts["scenarios"])):
            for split in ("train", "validation"):
                status("building-trajectories", support=label, split=split)
                datasets[(label, split)] = trajectories(
                    paths[split], np.asarray(config["study"]["actions"]), scenes
                )
        write_json(
            output / "dataset-summary.json",
            {
                f"{support}-{split}": {
                    "trajectories": len(data),
                    "steps": sum(len(x["states"]) for x in data),
                }
                for (support, split), data in datasets.items()
            },
        )
        pilot_config = copy.deepcopy(config)
        pilot_config["training"]["steps"] = opts["pilot_steps"]
        actions = np.asarray(config["study"]["actions"])
        pilot_policies, training_records = {}, []
        for support in ("nominal", "augmented"):
            for loss in ("log", "mean"):
                name = f"pilot-{support}-{loss}"
                directory = output / "models" / name
                status("pilot-training", candidate=name, seed=11)
                identity = training_identity(
                    pilot_config,
                    main,
                    ["calibrated"] if support == "nominal" else opts["scenarios"],
                    "response",
                    loss,
                    11,
                )
                result = fit(
                    "response",
                    loss,
                    11,
                    pilot_config,
                    datasets[(support, "train")],
                    datasets[(support, "validation")],
                    directory,
                    identity,
                )
                training_records.append({"candidate": name, **result})
                write_json(output / "pilot-training.json", training_records)
                pilot_policies[(name, 11)] = RobustLearned(
                    directory / "best.pt", actions, config["training"]["context"], "cpu"
                )
        status("pilot-evaluation")
        pilot_rows = validated_rows(
            output / "development-pilot",
            paths["validation"],
            config["study"],
            opts["scenarios"],
            pilot_policies,
        )
        rows = base_rows + pilot_rows
        reference = selection["feedback"]["name"]
        pilot_stats = {
            name: comparison(rows, name, reference, opts["scenarios"]) for name, _ in pilot_policies
        }
        loss = sorted(
            ("log", "mean"),
            key=lambda loss: (
                -min(
                    pilot_stats[f"pilot-augmented-{loss}"][s]["score_ratio"]
                    for s in opts["primary_scenarios"]
                ),
                loss != "log",
            ),
        )[0]
        write_json(
            output / "pilot-selection.json",
            {
                "loss": loss,
                "comparisons": pilot_stats,
                "limitation": "single seed, development selection; no confirmatory inference",
            },
        )
        print(f"pilot selected_loss={loss}", flush=True)
        if stage == "pilot":
            status("pilot-complete", loss=loss)
            return output
        final_config = copy.deepcopy(config)
        final_config["training"]["steps"] = opts["final_steps"]
        final_policies, final_records = {}, []
        for family in ("response", "mlp"):
            for seed in opts["seeds"]:
                directory = output / "models" / f"{family}-{loss}-{seed}"
                status("final-training", family=family, loss=loss, seed=seed)
                result = fit(
                    family,
                    loss,
                    seed,
                    final_config,
                    datasets[("augmented", "train")],
                    datasets[("augmented", "validation")],
                    directory,
                    training_identity(final_config, main, opts["scenarios"], family, loss, seed),
                )
                final_records.append({"candidate": family, **result})
                write_json(output / "final-training.json", final_records)
                final_policies[(family, seed)] = RobustLearned(
                    directory / "best.pt", actions, config["training"]["context"], "cpu"
                )
        status("final-development-evaluation")
        final_rows = validated_rows(
            output / "development-final",
            paths["validation"],
            config["study"],
            opts["scenarios"],
            final_policies,
        )
        rows += final_rows
        stats = {f: comparison(rows, f, reference, opts["scenarios"]) for f in ("response", "mlp")}
        family = sorted(
            stats,
            key=lambda f: (
                -min(stats[f][s]["score_ratio"] for s in opts["primary_scenarios"]),
                f != "response",
            ),
        )[0]
        gates = {
            scene: {
                "score": stats[family][scene]["score_ratio"] >= 1.03,
                "conversion": stats[family][scene]["conversion_ratio"] >= 0.97,
                "violation": stats[family][scene]["violation_delta"] <= 0.03,
                "all_seeds_positive": all(
                    v > 0 for v in stats[family][scene]["seed_score_deltas"].values()
                ),
            }
            for scene in opts["primary_scenarios"]
        }
        gates["calibrated"] = {"score": stats[family]["calibrated"]["score_ratio"] >= 0.98}
        promoted = all(all(g.values()) for g in gates.values())
        record = {
            "family": family,
            "loss": loss,
            "reference": reference,
            "gates": gates,
            "promoted": promoted,
            "comparisons": stats,
            "revision": revision,
            "scope": "development selection; independent confirmation not yet run",
        }
        write_json(output / "candidate-selection.json", record)
        status(
            "development-promoted" if promoted else "development-not-promoted",
            family=family,
            loss=loss,
            promoted=promoted,
        )
        print(f"development family={family} loss={loss} promoted={promoted}", flush=True)
        return output
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bidding-robust.toml")
    parser.add_argument("--stage", choices=["baselines", "pilot", "develop"], default="develop")
    args = parser.parse_args()
    develop(args.config, args.stage)


if __name__ == "__main__":
    main()
