"""Development-only one-step deviations with exact feedback continuation.

The offline endpoints are simulator counterfactuals. Selecting an action uses
only causal observations and model predictions, never those endpoint labels.
Single-deviation results do not establish safety under repeated replanning.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import numpy as np

from trajectorylab.bidding.learning import Feedback, rollout
from trajectorylab.bidding.market import ROOT, Tape, feedback_action
from trajectorylab.bidding_robust import sha
from trajectorylab.utils import fingerprint, read_json, write_json


def source_hashes():
    files = [
        Path(__file__),
        ROOT / "src/trajectorylab/bidding_continuation_data.py",
        ROOT / "src/trajectorylab/bidding_continuation_learning.py",
        *sorted((ROOT / "src/trajectorylab/bidding").glob("*.py")),
        ROOT / "src/trajectorylab/models.py",
        ROOT / "src/trajectorylab/utils.py",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def choose(prediction, baseline):
    """Apply identical predicted guards to direct utility and joint means."""
    utility = np.asarray(prediction["utility"])
    conv, violation = prediction["conv"], prediction["violation"]
    baseline = np.asarray(baseline, dtype=int)
    row = np.arange(len(baseline))
    eligible = (
        (utility > utility[row, baseline, None] + 1e-12)
        & (conv >= 0.97 * conv[row, baseline, None])
        & (violation <= violation[row, baseline, None] + 0.03)
    )
    candidate = np.where(eligible, utility, -np.inf).argmax(axis=1)
    return np.where(eligible.any(axis=1), candidate, baseline)


def summarize(data, selected, scenes, ticks):
    """Read exact outcomes only after predicted actions have been selected."""
    row = np.arange(len(selected))
    baseline = data["baseline_index"].astype(int)
    diagnostics = data["diagnostics"]
    result = {}
    for tick in ticks:
        by_scene = {}
        for si, scene in enumerate(scenes):
            mask = (diagnostics["scene_index"] == si) & (diagnostics["tick"] == tick)
            if not mask.any():
                raise RuntimeError("Missing fixed-time evaluation cell")
            fields = {}
            for key in ("terminal_score", "terminal_conversions", "terminal_violation"):
                values = data[key]
                actual = values[row, selected][mask]
                reference = values[row, baseline][mask]
                fields[key] = {
                    "candidate": float(actual.mean()),
                    "reference": float(reference.mean()),
                    "difference": float((actual - reference).mean()),
                    "ratio": float(actual.mean() / reference.mean())
                    if reference.mean() > 0
                    else None,
                }
            fields["contexts"] = int(mask.sum())
            fields["deviation_fraction"] = float((selected[mask] != baseline[mask]).mean())
            by_scene[scene] = fields
        result[str(tick)] = by_scene
    return result


def business_gate(comparison, opts):
    gates = {}
    for scene in opts["primary_scenarios"]:
        cell = comparison[scene]
        score_ratio = cell["terminal_score"]["ratio"]
        conv_ratio = cell["terminal_conversions"]["ratio"]
        gates[scene] = {
            "score": score_ratio is not None and score_ratio >= 1.03,
            "conversions": conv_ratio is not None and conv_ratio >= 0.97,
            "violation": cell["terminal_violation"]["difference"] <= 0.03,
        }
    nominal = comparison["calibrated"]["terminal_score"]["ratio"]
    gates["calibrated"] = {"score": nominal is not None and nominal >= 0.98}
    return {"passed": all(all(x.values()) for x in gates.values()), "gates": gates}


class ContinuationPolicy:
    """Public baseline with model-selected deviations at registered times."""

    def __init__(self, model, actions, ticks):
        self.model, self.actions, self.ticks = model, np.asarray(actions), set(ticks)

    def __call__(self, states, used, rewards, budget, cpa):
        baseline_action = feedback_action(states[-1], self.actions, 0.3, 8.0, budget, cpa)
        tick = len(states) - 1
        if tick not in self.ticks:
            return baseline_action, {}
        baseline = int(np.abs(self.actions - baseline_action).argmin())
        start = max(0, len(states) - 24)
        length = len(states) - start
        values = np.zeros((1, 24, 13), dtype=np.float32)
        steps = np.zeros((1, 24), dtype=np.int64)
        values[0, :length] = states[start:]
        steps[0, :length] = np.arange(start, len(states))
        prediction = self.model.predict(values, steps, np.array([length]), np.array([baseline]))
        selected = int(choose(prediction, [baseline])[0])
        return float(self.actions[selected]), {"model_deviation": int(selected != baseline)}


def run(config_path, stage="fit"):
    from trajectorylab.bidding_continuation_learning import ContinuationModel, diagnostic, fit

    opts = tomllib.loads(Path(config_path).read_text())
    directory, output = ROOT / opts["data"], ROOT / opts["output"]
    output.mkdir(parents=True, exist_ok=True)
    # Persist input bytes and sources before fitting. A changed experiment gets
    # its own directory; it cannot silently reuse a different model or cache.
    inputs = {
        str(p.relative_to(ROOT)): sha(p)
        for p in [
            directory / "manifest.json",
            directory / "train.npz",
            directory / "validation.npz",
        ]
    }
    encoder = ROOT / opts["encoder"]
    dataset_manifest = read_json(directory / "manifest.json")
    core = dataset_manifest["core"]
    if fingerprint(core) != dataset_manifest["identity"]:
        raise RuntimeError("Counterfactual dataset manifest identity mismatch")
    for name, digest in core["sources"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError("Counterfactual data source changed")
    for name, digests in core["inputs"].items():
        path = Path(name)
        if (
            sha(path) != digests["npz_sha256"]
            or sha(path.with_suffix(".json")) != digests["metadata_sha256"]
        ):
            raise RuntimeError("Counterfactual raw market bytes changed")
    if core["scenarios"] != opts["scenarios"] or core["ticks"] != opts["ticks"]:
        raise RuntimeError("Counterfactual data protocol differs from this study")
    if sha(ROOT / opts["main_run"] / "manifest.json") != core["main_manifest_sha256"]:
        raise RuntimeError("Original study manifest bytes changed")
    if core["context"] != opts["training"]["context"]:
        raise RuntimeError("Counterfactual context length differs")
    for split in ("train", "validation"):
        if sha(directory / f"{split}.npz") != dataset_manifest["datasets"][split]["npz_sha256"]:
            raise RuntimeError("Sealed counterfactual endpoint bytes changed")
    record = {
        "config": opts,
        "data": inputs,
        "encoder_sha256": sha(encoder),
        "sources": source_hashes(),
        "scope": "development feasibility; no independent confirmation",
    }
    identity = fingerprint(record)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
        raise RuntimeError("Continuation experiment changed; use a new output directory")
    write_json(manifest_path, record | {"identity": identity})
    data = {}
    for split in ("train", "validation"):
        with np.load(directory / f"{split}.npz", allow_pickle=False) as saved:
            data[split] = {k: saved[k] for k in saved.files}
    main = read_json(ROOT / opts["main_run"] / "manifest.json")
    actions = np.asarray(main["config"]["study"]["actions"])
    if actions.tolist() != core["actions"]:
        raise RuntimeError("Counterfactual action grid differs from original study")
    model_config = {"training": opts["training"], "study": {"actions": actions.tolist()}}
    families = ("utility", "joint_mean")
    training, reports = {}, {}
    for family in families:
        location = output / "models" / f"{family}-{opts['seed']}"
        if stage == "fit":
            training[family] = fit(
                family,
                opts["seed"],
                model_config,
                data["train"],
                data["validation"],
                encoder,
                location,
                fingerprint({"study": identity, "family": family}),
            )
            write_json(output / "training.json", training)
        model = ContinuationModel(location / "best.pt", actions, "cpu")
        if (
            model.training_identity != fingerprint({"study": identity, "family": family})
            or model.family != family
            or model.seed != opts["seed"]
        ):
            raise RuntimeError("Continuation fitted model does not match this experiment")
        validation = data["validation"]
        prediction = model.predict(
            validation["states"],
            validation["steps"],
            validation["lengths"],
            validation["baseline_index"],
        )
        selected = choose(prediction, validation["baseline_index"])
        comparisons = summarize(validation, selected, opts["scenarios"], opts["ticks"])
        reports[family] = {
            "checkpoint_sha256": model.identity,
            "diagnostics": diagnostic(model, validation),
            "single_deviation": comparisons,
            "gates": {k: business_gate(v, opts) for k, v in comparisons.items()},
        }
        if stage == "receding":
            rows = []
            for market in main["config"]["study"]["validation_markets"]:
                tape = Tape(ROOT / opts["main_run"] / "markets" / f"{market}.npz")
                for scene in opts["scenarios"]:
                    for fi, focal in enumerate(tape.arrays["focals"]):
                        for name, policy in (
                            (family, ContinuationPolicy(model, actions, opts["ticks"])),
                            ("feedback", Feedback(actions, 0.3, 8.0)),
                        ):
                            rows.append(
                                rollout(tape, fi, actions, scene, policy)
                                | {
                                    "market": market,
                                    "scene": scene,
                                    "focal": int(focal),
                                    "method": name,
                                }
                            )
            write_json(output / f"receding-{family}.json", {"identity": identity, "rows": rows})
        write_json(output / "feasibility.json", {"identity": identity, "models": reports})
    print("Continuation feasibility complete; no fresh confirmation used", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bidding-continuation.toml")
    parser.add_argument("--stage", choices=("fit", "evaluate", "receding"), default="fit")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
