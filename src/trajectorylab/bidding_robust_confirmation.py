"""Fixed-sample confirmation E; every entry verifies the frozen development inputs.

Build is sequential. A resource supervisor may call ``--stage build --market ID``;
evaluation opens only after all 24 market tapes have been sealed. No intermediate
effect, selection, sample-size change, or early stopping is implemented.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import time
from pathlib import Path

import numpy as np
import torch

from trajectorylab.bidding.learning import Feedback, Learned
from trajectorylab.bidding.market import ROOT, build_market, effective_actions, source_identity
from trajectorylab.bidding_control import metrics
from trajectorylab.bidding_robust import (
    save_rows,
    setup,
    sha,
    sources,
    training_identity,
    validated_rows,
)
from trajectorylab.bidding_robust_learning import RobustLearned
from trajectorylab.utils import fingerprint, read_json, write_json

MARKETS = list(range(4101, 4125))
SEEDS = [11, 23, 37]
SCENARIOS = ["calibrated", "over_25", "over_50", "under_25", "mid_over_50"]
PRIMARY = ["over_25", "over_50", "under_25"]
SLOTS = [(m, s) for m in ("response", "mlp", "response-old") for s in SEEDS] + [
    (m, -1) for m in ("feedback-new", "feedback-old", "fixed-tuned", "fixed-1")
]
MEASURES = ("score", "conversions", "violation", "cost", "budget_utilization", "zero_conversions")


def _keep(path: Path, value: dict) -> None:
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Frozen record changed: {path.name}")
    else:
        write_json(path, value)


def _cache(path: Path, study: dict, auction: dict) -> dict:
    meta = read_json(path.with_suffix(".json"))
    expected = fingerprint(
        {
            "study": {k: study[k] for k in ("focal_agents", "pv_count", "actions", "biases")},
            "market": int(path.stem),
            "source": auction,
        }
    )
    if (
        meta.get("identity") != expected
        or meta.get("market_id") != int(path.stem)
        or meta.get("source") != auction
        or meta.get("focal_agents") != study["focal_agents"]
        or meta.get("requested_pv_count") != study["pv_count"]
        or meta.get("effective_actions") != effective_actions(study).tolist()
        or meta.get("bytes") != path.stat().st_size
    ):
        raise RuntimeError(f"Market metadata differs from frozen construction: {path.stem}")
    return {
        "identity": expected,
        "npz_sha256": sha(path),
        "metadata_sha256": sha(path.with_suffix(".json")),
    }


def _weight(path: Path, identity: str, family: str, loss: str, seed: int, config: dict) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    architecture = (
        {k: config["training"][k] for k in ("width", "layers", "heads")}
        if family != "mlp"
        else {"width": 448, "layers": 3}
    )
    if (
        saved.get("identity") != identity
        or saved.get("seed") != seed
        or saved.get("kind") != "response"
        or not 0 < saved.get("step", 0) <= 15000
        or saved.get("architecture") != architecture
        or any(np.shape(saved[k]) != (13,) for k in ("mean", "std"))
        or not np.isfinite(
            np.r_[
                saved["mean"],
                saved["std"],
                saved["return_scale"],
                saved["target_return"],
                saved["validation_loss"],
            ]
        ).all()
        or np.any(saved["std"] <= 0)
        or saved["return_scale"] <= 0
        or any(not torch.isfinite(v).all() for v in saved["model"].values())
        or (
            family != "response-old"
            and (
                saved.get("family") != family
                or saved.get("loss") != loss
                or saved.get("actions") != config["study"]["actions"]
                or saved.get("context") != config["training"]["context"]
            )
        )
    ):
        raise RuntimeError(
            f"Checkpoint metadata or numerical values differ from registered training: {path.parent.name}"
        )
    return {
        "identity": identity,
        "family": family,
        "seed": seed,
        "step": saved["step"],
        "architecture": saved["architecture"],
    }


def _prepare(config_path: Path) -> tuple[dict, dict, Path, Path, dict]:
    opts, config, main, output = setup(config_path)
    study = config["study"]
    candidate = read_json(output / "candidate-selection.json")
    baseline = read_json(output / "baseline-selection.json")
    status = read_json(output / "status.json")
    revision = fingerprint({"sources": sources(), "config": opts})
    gate_keys = {s: {"score", "conversion", "violation", "all_seeds_positive"} for s in PRIMARY} | {
        "calibrated": {"score"}
    }
    if (
        candidate.get("promoted") is not True
        or status.get("phase") != "development-promoted"
        or candidate.get("family") not in ("response", "mlp")
        or candidate.get("loss") not in ("log", "mean")
        or candidate.get("reference") != baseline["feedback"]["name"]
        or any(record.get("revision") != revision for record in (candidate, status))
        or read_json(output / "development-manifest.json").get("identity") != revision
        or {s: set(g) for s, g in candidate.get("gates", {}).items()} != gate_keys
        or not all(v is True for gates in candidate["gates"].values() for v in gates.values())
        or status.get("family") != candidate["family"]
        or status.get("loss") != candidate["loss"]
    ):
        raise RuntimeError("A current, completed development promotion is required")
    if (
        opts["confirmation_markets"] != MARKETS
        or opts["seeds"] != SEEDS
        or opts["scenarios"] != SCENARIOS
        or opts["primary_scenarios"] != PRIMARY
        or opts["bootstrap_samples"] != 20000
        or opts["bootstrap_rng"] != 29118
        or opts["alpha"] != 0.025
        or opts["final_steps"] != 15000
        or opts["feedback_gains"] != [0.1, 0.3, 0.7, 1.2]
        or opts["feedback_priors"] != [1.0, 3.0, 8.0, 16.0]
        or study["pv_count"] != 500000
        or study["focal_agents"] != [0, 8, 16, 24, 32, 40]
        or study["train_markets"] != list(range(1001, 1017))
        or study["validation_markets"] != list(range(2001, 2007))
        or baseline["fixed"] not in study["actions"]
        or baseline["feedback"]["gain"] not in opts["feedback_gains"]
        or baseline["feedback"]["prior"] not in opts["feedback_priors"]
    ):
        raise ValueError("Configuration differs from the registered confirmation E design")
    if any(any((main / "markets").glob(f"{m}.*")) for m in MARKETS):
        raise RuntimeError("A confirmation market already exists in the historical main run")
    records = [
        output / f"{name}.json"
        for name in (
            "baseline-selection",
            "candidate-selection",
            "development-manifest",
            "status",
            "development-baselines",
            "development-baselines.identity",
            "development-final",
            "development-final.identity",
            "pilot-selection",
            "pilot-training",
            "final-training",
            "dataset-summary",
            "label-audit",
        )
    ] + [main / f"{name}.json" for name in ("manifest", "selection", "status", "training")]
    final = read_json(output / "final-training.json")
    if (
        len(final) != 6
        or {(r["family"], r["seed"]) for r in final}
        != {(f, s) for f in ("response", "mlp") for s in SEEDS}
        or any(r["steps"] != 15000 or r["loss"] != candidate["loss"] for r in final)
    ):
        raise RuntimeError(
            "Both final families must complete all three registered training budgets"
        )
    old = [r for r in read_json(main / "training.json") if r["kind"] == "response"]
    if (
        len(old) != 3
        or {r["seed"] for r in old} != set(SEEDS)
        or any(r["steps"] != config["training"]["steps"] for r in old)
    ):
        raise RuntimeError("Original response training records are incomplete")
    weights = {
        f"{family}:{seed}": str(
            (
                (main / "models" / f"response-{seed}")
                if family == "response-old"
                else (output / "models" / f"{family}-{candidate['loss']}-{seed}")
            )
            / "best.pt"
        )
        for family in ("response", "mlp", "response-old")
        for seed in SEEDS
    }
    final_config = copy.deepcopy(config)
    final_config["training"]["steps"] = 15000
    main_identity = read_json(main / "manifest.json")["identity"]
    weight_identities = {
        f"{family}:{seed}": _weight(
            Path(weights[f"{family}:{seed}"]),
            main_identity
            if family == "response-old"
            else training_identity(
                final_config, main, opts["scenarios"], family, candidate["loss"], seed
            ),
            family,
            candidate["loss"],
            seed,
            config,
        )
        for family in ("response", "mlp", "response-old")
        for seed in SEEDS
    }
    records += [Path(p).parent / "history.json" for p in weights.values()]
    records += list(output.glob("development-*.csv")) + list(output.glob("development-pilot*.json"))
    execution = sources() | {
        str(p.relative_to(ROOT)): sha(p)
        for p in (
            Path(__file__),
            ROOT / "src/trajectorylab/__init__.py",
            ROOT / "third_party/auctionnet/BiddingEnv.py",
            ROOT / "third_party/auctionnet/NeurIPSPvGen.py",
            ROOT / "scripts/run_bidding_robust_background.py",
            ROOT / "scripts/run_bidding_background.py",
        )
    }
    files = {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(records))}
    files[str(config_path.resolve().relative_to(ROOT))] = sha(config_path)
    files.update({str(Path(p).relative_to(ROOT)): sha(Path(p)) for p in weights.values()})
    core = {
        "options": opts,
        "config": config,
        "sources": execution,
        "files": files,
        "candidate": candidate,
        "baseline": baseline,
        "weights": weights,
        "weight_identities": weight_identities,
        "auction": source_identity(),
        "slots": [list(s) for s in SLOTS],
        "training_development_data": {
            str(m): _cache(main / "markets" / f"{m}.npz", study, source_identity())
            for split in ("train_markets", "validation_markets")
            for m in study[split]
        },
        "design": "confirmation E; fixed 24 markets; no intermediate inference or early stopping",
        "uncertainty_scope": "Market uncertainty conditional on three fixed trained models; no inference over future retraining, online deployment, or arbitrary unknown bias.",
    }
    return opts, config, main, output / "confirmation", core


def _live(core: dict) -> None:
    for filename, expected in (core["sources"] | core["files"]).items():
        if sha(ROOT / filename) != expected:
            raise RuntimeError(
                f"Frozen source, record, configuration or weight changed: {filename}"
            )


def _ledger(directory: Path, name: str, identity: str) -> dict:
    path = directory / name
    value = read_json(path) if path.exists() else {"identity": identity, "markets": {}}
    if value.get("identity") != identity or not set(value["markets"]).issubset(
        {str(m) for m in MARKETS}
    ):
        raise RuntimeError("Ledger belongs to a different confirmation")
    return value


def _built(directory: Path, config: dict, core: dict, identity: str) -> dict:
    ledger = _ledger(directory, "build-data.json", identity)
    expected = {f"{m}{ext}" for m in ledger["markets"] for ext in (".npz", ".json")}
    cache_dir = directory / "markets"
    actual = (
        {p.name for p in cache_dir.iterdir() if p.suffix in (".npz", ".json")}
        if cache_dir.exists()
        else set()
    )
    if actual != expected:
        raise RuntimeError(
            "Unregistered or missing market files; refusing to adopt preexisting tapes"
        )
    for market, record in ledger["markets"].items():
        if _cache(cache_dir / f"{market}.npz", config["study"], core["auction"]) != record:
            raise RuntimeError(f"Previously built market changed: {market}")
    return ledger


def _policies(core: dict) -> dict:
    config = core["config"]
    actions = np.asarray(config["study"]["actions"])
    policies = {}
    for family, seed in SLOTS[:9]:
        path = Path(core["weights"][f"{family}:{seed}"])
        loader = Learned if family == "response-old" else RobustLearned
        policies[family, seed] = loader(path, actions, config["training"]["context"], "cpu")
    feedback = core["baseline"]["feedback"]
    policies["feedback-new", -1] = Feedback(actions, feedback["gain"], feedback["prior"])
    policies["feedback-old", -1] = Feedback(actions, 0.3, 8.0)
    for name, action in (("fixed-tuned", core["baseline"]["fixed"]), ("fixed-1", 1.0)):
        policies[name, -1] = lambda states, used, rewards, budget, cpa, a=action: (a, {})
    return policies


def _rows(rows: list[dict], markets: list[int], study: dict) -> None:
    expected = {
        (m, f, s, method, seed)
        for m in markets
        for f in study["focal_agents"]
        for s in SCENARIOS
        for method, seed in SLOTS
    }
    actual = [(r["market"], r["focal"], r["scenario"], r["method"], r["seed"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise RuntimeError("Confirmation campaign coverage mismatch")
    for row in rows:
        if any(type(row[k]) is not int for k in ("market", "focal", "seed")) or any(
            k not in row for k in (*MEASURES, "budget")
        ):
            raise RuntimeError("Missing or invalid campaign identities or metrics")
        for key, value in row.items():
            if key in ("scenario", "method") or (
                key == "cpa" and value is None and row["conversions"] == 0
            ):
                continue
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RuntimeError("Nonfinite or unsupported campaign evidence")
        if (
            row["budget"] <= 0
            or row["cost"] < 0
            or row["cost"] > row["budget"] + 1e-6
            or row["conversions"] < 0
            or row["violation"] not in (0, 1)
        ):
            raise RuntimeError("Invalid budget settlement or campaign metric")


def _summary(rows: list[dict], identity: str, core: dict) -> dict:
    rng = np.random.default_rng(29118)
    samples = rng.integers(0, 24, size=(20000, 24))
    quantiles = [0.025 / 3 / 2, 1 - 0.025 / 3 / 2]
    methods = list(dict.fromkeys(m for m, _ in SLOTS))

    def vector(method, scene, key):
        return np.array(
            [
                np.mean(
                    [
                        r[key]
                        for r in rows
                        if r["market"] == m and r["scenario"] == scene and r["method"] == method
                    ]
                )
                for m in MARKETS
            ]
        )

    def estimate(a, b, ratio=False):
        a_boot, b_boot = a[samples].mean(axis=1), b[samples].mean(axis=1)
        if ratio and (b.mean() <= 0 or np.any(b_boot <= 0)):
            raise RuntimeError("Nonpositive bootstrap ratio denominator; confirmation is undefined")
        values = a_boot / b_boot if ratio else a_boot - b_boot
        if not np.isfinite(values).all():
            raise RuntimeError("Nonfinite bootstrap estimate")
        return {
            "mean": float(a.mean() / b.mean() if ratio else (a - b).mean()),
            "interval": np.quantile(values, quantiles).tolist(),
        }

    effects = {}
    for method in methods:
        if method == "feedback-new":
            continue
        effects[method] = {}
        for scene in SCENARIOS:
            a = {key: vector(method, scene, key) for key in ("score", "conversions", "violation")}
            b = {key: vector("feedback-new", scene, key) for key in a}
            effects[method][scene] = {
                "score_delta": estimate(a["score"], b["score"]),
                "score_ratio": estimate(a["score"], b["score"], True),
                "conversion_ratio": estimate(a["conversions"], b["conversions"], True),
                "violation_delta": estimate(a["violation"], b["violation"]),
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
                        - b["score"].mean()
                    )
                    for name, seed in SLOTS
                    if name == method
                },
            }
    selected = effects[core["candidate"]["family"]]
    gates = {
        scene: {
            "score_lower_positive": selected[scene]["score_delta"]["interval"][0] > 0,
            "conversion_ratio_lower_at_least_0.95": selected[scene]["conversion_ratio"]["interval"][
                0
            ]
            >= 0.95,
            "violation_delta_upper_at_most_0.03": selected[scene]["violation_delta"]["interval"][1]
            <= 0.03,
            "all_three_seed_means_positive": set(selected[scene]["seed_score_deltas"])
            == {str(s) for s in SEEDS}
            and all(v > 0 for v in selected[scene]["seed_score_deltas"].values()),
        }
        for scene in PRIMARY
    }
    gates["calibrated"] = {
        "score_ratio_lower_at_least_0.95": selected["calibrated"]["score_ratio"]["interval"][0]
        >= 0.95
    }
    return {
        "identity": identity,
        "decision": "confirmed"
        if all(all(g.values()) for g in gates.values())
        else "not-confirmed",
        "family": core["candidate"]["family"],
        "reference": "feedback-new",
        "gates": gates,
        "primary_scenarios": PRIMARY,
        "secondary_scenarios": ["calibrated", "mid_over_50"],
        "effects": effects,
        "groups": [
            {"method": m, "scenario": s, **metrics(rows, m, s)} for s in SCENARIOS for m in methods
        ],
        "bootstrap": {
            "samples": 20000,
            "rng": 29118,
            "alpha": 0.025,
            "per_scene_alpha": 0.025 / 3,
            "two_sided_quantiles": quantiles,
            "unit": "market; advertiser and fixed-model means first",
            "joint_metric_coverage_claimed": False,
        },
        "sample_size": {
            "markets": 24,
            "focals": 6,
            "scenarios": 5,
            "policy_slots": 13,
            "rows": len(rows),
        },
        "training_records": read_json(ROOT / core["options"]["output"] / "final-training.json"),
        "limits": [
            core["uncertainty_scope"],
            "All prespecified gates form an intersection; the guard intervals do not imply simultaneous coverage of all metrics.",
            "Fixed-sample failure is retained; this confirmation data cannot be reused for tuning.",
        ],
    }


def run(config_path: Path, stage: str = "all", market: int | None = None) -> Path:
    if stage not in ("freeze", "build", "evaluate", "all") or (
        market is not None and (stage != "build" or market not in MARKETS)
    ):
        raise ValueError("--market is permitted only for one registered --stage build market")
    opts, config, _, directory, core = _prepare(config_path)
    identity = fingerprint(core)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = directory / "manifest.json"
        if not manifest.exists() and any(p.name != ".run.lock" for p in directory.iterdir()):
            raise RuntimeError("Freeze must precede every confirmation artifact")
        _keep(manifest, {"identity": identity, "core": core})
        built = _built(directory, config, core, identity)
        seal_path = directory / "sealed-data.json"
        if seal_path.exists():
            _keep(seal_path, built)
        results = _ledger(directory, "evaluation-ledger.json", identity)
        result_dir = directory / "per-market"
        expected = {
            f"{m}{suffix}"
            for m in results["markets"]
            for suffix in (".json", ".csv", ".identity.json")
        }
        actual = {p.name for p in result_dir.iterdir()} if result_dir.exists() else set()
        if actual != expected:
            raise RuntimeError("Unregistered or missing per-market evaluation artifacts")
        for record in results["markets"].values():
            if any(sha(result_dir / name) != digest for name, digest in record.items()):
                raise RuntimeError("A completed per-market result changed")
        if stage == "freeze":
            return directory
        if stage in ("build", "all"):
            for number in [market] if market is not None else MARKETS:
                _live(core)
                if str(number) in built["markets"]:
                    continue
                if seal_path.exists() or results["markets"]:
                    raise RuntimeError("Cannot add markets after sealing or evaluation")
                path = directory / "markets" / f"{number}.npz"
                build_market(path, config["study"], number)
                _live(core)
                built["markets"][str(number)] = _cache(path, config["study"], core["auction"])
                write_json(directory / "build-data.json", built)
        if set(built["markets"]) != {str(m) for m in MARKETS}:
            if stage == "build":
                return directory
            raise RuntimeError("All 24 frozen market caches are required before any evaluation")
        _keep(seal_path, _built(directory, config, core, identity))
        if stage == "build":
            return directory
        _live(core)
        torch.set_num_threads(config["training"]["threads"])
        policies = _policies(core)
        all_rows = []
        for number in MARKETS:
            _live(core)
            path = directory / "markets" / f"{number}.npz"
            if _cache(path, config["study"], core["auction"]) != built["markets"][str(number)]:
                raise RuntimeError("Sealed confirmation market changed")
            rows = validated_rows(
                result_dir / str(number), [path], config["study"], SCENARIOS, policies
            )
            _rows(rows, [number], config["study"])
            _live(core)
            results["markets"][str(number)] = {
                f"{number}{suffix}": sha(result_dir / f"{number}{suffix}")
                for suffix in (".json", ".csv", ".identity.json")
            }
            write_json(directory / "evaluation-ledger.json", results)
            all_rows.extend(rows)
        _rows(all_rows, MARKETS, config["study"])
        if _prepare(config_path)[4] != core:
            raise RuntimeError("Frozen inputs changed during confirmation")
        _keep(seal_path, _built(directory, config, core, identity))
        summary = _summary(all_rows, identity, core)
        save_rows(directory / "results", all_rows)
        _keep(directory / "summary.json", summary)
        report = (
            "# Independent confirmation E\n\n"
            + f"Decision: **{summary['decision']}**. Candidate: {summary['family']}; reference: feedback-new.\n\n"
        )
        report += "24 fixed markets, 6 advertisers, 3 fixed models per learned family; all five scenarios reported.\n\n"
        report += (
            "\n\n".join(summary["limits"])
            + "\n\n```json\n"
            + json.dumps(summary, indent=2, ensure_ascii=False)
            + "\n```\n"
        )
        temporary = directory / "report.partial.md"
        temporary.write_text(report)
        temporary.replace(directory / "report.md")
        write_json(
            directory / "status.json",
            {
                "identity": identity,
                "phase": "complete",
                "decision": summary["decision"],
                "updated_unix": time.time(),
            },
        )
        return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bidding-robust.toml")
    parser.add_argument("--stage", choices=["freeze", "build", "evaluate", "all"], default="all")
    parser.add_argument("--market", type=int)
    args = parser.parse_args()
    run(args.config, args.stage, args.market)


if __name__ == "__main__":
    main()
