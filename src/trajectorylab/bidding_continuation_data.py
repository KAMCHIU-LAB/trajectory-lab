"""Exact one-action interventions followed by the fixed public feedback policy.

States are float32 observations, RIGHT padded: valid states occupy ``:length``.
``steps`` stores their original tick IDs. Only states, lengths, steps and the
baseline action index are public model inputs. All other fields are targets or
offline diagnostics. Branch state uses original float64 spending and integer
conversions, never values reconstructed from a normalized observation.

The feasibility CLI uses only 22 existing training/development markets. Its
oracle headroom is an offline upper bound, not evidence of a deployed gain.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import platform
import time
from pathlib import Path

import numpy as np

from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.learning import Feedback, rollout
from trajectorylab.bidding.market import (
    ROOT,
    Tape,
    observation,
    scenario_bias,
    score,
    source_identity,
)
from trajectorylab.utils import fingerprint, read_json, write_json

SCENARIOS = ["calibrated", "over_25", "over_50", "under_25", "mid_over_50"]
TICKS = [0, 16, 32]
DIAGNOSTICS = np.dtype(
    [("market", "<i8"), ("scene_index", "<i8"), ("focal", "<i8"), ("tick", "<i8")]
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sources() -> dict:
    files = list((ROOT / "src/trajectorylab/bidding").glob("*.py"))
    files += [
        Path(__file__).resolve(),
        ROOT / "src/trajectorylab/models.py",
        ROOT / "src/trajectorylab/utils.py",
    ]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def identity(paths, actions, scenarios, ticks, context=24) -> dict:
    return {
        "sources": sources(),
        "auction": source_identity(),
        "inputs": {
            str(Path(p).resolve()): {
                "npz_sha256": sha(Path(p)),
                "metadata_sha256": sha(Path(p).with_suffix(".json")),
            }
            for p in paths
        },
        "actions": np.asarray(actions, dtype=np.float64).tolist(),
        "scenarios": list(scenarios),
        "ticks": list(ticks),
        "context": context,
        "behavior": {"class": "Feedback", "gain": 0.3, "prior": 8.0},
        "continuation": "one intervention at current tick, then the same fixed Feedback through tick 47",
        "padding": "right; valid states and actual tick IDs occupy :length; padding is zero",
        "endpoint_units": [
            "remaining cost / budget",
            "remaining conversions * target CPA / budget",
        ],
        "endpoint_horizon": "current tick through tick 47, under actual hard-budget settlement",
        "endpoint_constraints": "nonnegative; no action monotonicity or action-zero constraint",
        "public_condition_fields": ["states", "lengths", "steps", "baseline_index"],
        "diagnostics_fields": ["market", "scene_index", "focal", "tick"],
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }


def _protocol(paths, actions, scenarios, ticks, context):
    paths = [Path(p).resolve() for p in paths]
    actions = np.asarray(actions, dtype=np.float64)
    if (
        not paths
        or len(set(paths)) != len(paths)
        or actions.ndim != 1
        or len(actions) != 11
        or not np.isfinite(actions).all()
        or actions[0] != 0
        or np.any(np.diff(actions) <= 0)
        or not scenarios
        or len(set(scenarios)) != len(scenarios)
        or not ticks
        or list(ticks) != sorted(set(ticks))
        or any(not isinstance(t, (int, np.integer)) or not 0 <= t < 48 for t in ticks)
        or not isinstance(context, int)
        or not 1 <= context <= 48
    ):
        raise ValueError(
            "Use unique markets/scenes, 11 increasing actions, sorted ticks in [0,47], and a positive context"
        )
    for scene in scenarios:
        scenario_bias(scene, 0)
    return paths, actions


def _exact_prefixes(tape, fi, actions, scene, ticks):
    """Replay native accounting alongside an independent original rollout."""
    chooser = Feedback(actions, 0.3, 8.0)
    reference = rollout(tape, fi, actions, scene, Feedback(actions, 0.3, 8.0), retain=True)
    budget, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
    states, used, rewards = [], [], []
    spent, conv, previous_score = 0.0, 0, 0.0
    previous = (0.0, 0, 0.0, 0)
    prefixes = {}
    for tick in range(48):
        bias = scenario_bias(scene, tick)
        states.append(observation(tape, fi, tick, spent, conv, previous, bias))
        action, _ = chooser(states, used, rewards, budget, cpa)
        if not np.array_equal(states[-1], reference["states"][tick]) or action != float(
            reference["actions"][tick]
        ):
            raise RuntimeError("Exact replay differs from the retained original rollout")
        if tick in ticks:
            prefixes[tick] = {
                "spent": spent,
                "conversions": conv,
                "previous": previous,
                "states": states.copy(),
                "used": used.copy(),
                "rewards": rewards.copy(),
                "previous_score": previous_score,
                "baseline_action": action,
            }
        tick_cost, tick_conv, tick_wins, _ = tape.settle(fi, tick, action * bias, budget - spent)
        spent += tick_cost
        conv += tick_conv
        current_score = score(spent, conv, cpa)
        rewards.append(current_score - previous_score)
        previous_score = current_score
        previous = (tick_cost, tick_conv, action, tick_wins)
        used.append(action)
    if (spent, conv, previous_score) != (
        reference["cost"],
        reference["conversions"],
        reference["score"],
    ):
        raise RuntimeError("Exact replay accounting differs from independent feedback rollout")
    return prefixes, reference


def _branch(tape, fi, actions, scene, tick, prefix, action, reference):
    budget, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
    states, used, rewards = prefix["states"].copy(), prefix["used"].copy(), prefix["rewards"].copy()
    spent, conv, previous = prefix["spent"], prefix["conversions"], prefix["previous"]
    previous_score = prefix["previous_score"]
    chooser = Feedback(actions, 0.3, 8.0)
    baseline = action == prefix["baseline_action"]
    for future in range(tick, 48):
        bias = scenario_bias(scene, future)
        if future > tick:
            states.append(observation(tape, fi, future, spent, conv, previous, bias))
            action, _ = chooser(states, used, rewards, budget, cpa)
        if baseline and (
            not np.array_equal(states[-1], reference["states"][future])
            or action != float(reference["actions"][future])
        ):
            raise RuntimeError("a=b continuation differs from independent feedback states/actions")
        tick_cost, tick_conv, tick_wins, _ = tape.settle(fi, future, action * bias, budget - spent)
        spent += tick_cost
        conv += tick_conv
        current_score = score(spent, conv, cpa)
        rewards.append(current_score - previous_score)
        previous_score = current_score
        previous = (tick_cost, tick_conv, action, tick_wins)
        used.append(action)
        if spent > budget + 1e-6:
            raise RuntimeError("A branch exceeded the original hard budget")
    violation = int(spent > cpa * conv + 1e-8)
    if baseline and (spent, conv, previous_score, violation) != (
        reference["cost"],
        reference["conversions"],
        reference["score"],
        reference["violation"],
    ):
        raise RuntimeError("a=b endpoints differ from independent feedback continuation")
    return spent, conv, previous_score, violation


def build(paths, actions, scenarios, ticks, context=24) -> dict[str, np.ndarray]:
    """Build exact feedback-prefix branches; never generate a market or fit a model.

    ``diagnostics`` is a numeric structured array of shape (N,), with fields
    market, scene_index (index in ``scenarios``), focal, and context tick.
    ``spent``/``conversions`` are exact pre-intervention accounting diagnostics.
    ``endpoints`` includes the current intervention tick and fixed continuation.
    """
    paths, actions = _protocol(paths, actions, scenarios, ticks, context)
    frozen = identity(paths, actions, scenarios, ticks, context)
    records = []
    for path in paths:
        tape = Tape(path)
        for scene_index, scene in enumerate(scenarios):
            for fi, focal in enumerate(tape.arrays["focals"]):
                budget, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
                if not np.isfinite([budget, cpa]).all() or budget <= 0 or cpa <= 0:
                    raise RuntimeError("Positive finite budget and CPA are required")
                prefixes, reference = _exact_prefixes(tape, fi, actions, scene, ticks)
                for tick in ticks:
                    prefix = prefixes[tick]
                    begin = max(0, tick + 1 - context)
                    length = tick + 1 - begin
                    states = np.zeros((context, 13), dtype=np.float32)
                    steps = np.zeros(context, dtype=np.int64)
                    states[:length] = prefix["states"][begin:]
                    steps[:length] = np.arange(begin, tick + 1)
                    baseline_index = int(np.flatnonzero(actions == prefix["baseline_action"])[0])
                    outcomes = [
                        _branch(tape, fi, actions, scene, tick, prefix, float(action), reference)
                        for action in actions
                    ]
                    terminal_cost = np.array([o[0] for o in outcomes], dtype=np.float64)
                    terminal_conv = np.array([o[1] for o in outcomes], dtype=np.int64)
                    endpoints = np.stack(
                        (
                            (terminal_cost - prefix["spent"]) / budget,
                            (terminal_conv - prefix["conversions"]) * cpa / budget,
                        ),
                        axis=-1,
                    )
                    records.append(
                        {
                            "states": states,
                            "lengths": length,
                            "steps": steps,
                            "baseline_index": baseline_index,
                            "budget": budget,
                            "cpa": cpa,
                            "spent": prefix["spent"],
                            "conversions": prefix["conversions"],
                            "endpoints": endpoints,
                            "terminal_score": np.array([o[2] for o in outcomes], dtype=np.float64),
                            "terminal_conversions": terminal_conv,
                            "terminal_violation": np.array([o[3] for o in outcomes], dtype=np.int8),
                            "diagnostics": (int(path.stem), scene_index, int(focal), tick),
                        }
                    )
        print(f"continuation market={path.stem} contexts={len(records)}", flush=True)
    arrays = {
        key: np.stack([record[key] for record in records])
        for key in records[0]
        if key != "diagnostics"
    }
    arrays["diagnostics"] = np.array(
        [record["diagnostics"] for record in records], dtype=DIAGNOSTICS
    )
    validate(arrays, len(actions), context)
    if identity(paths, actions, scenarios, ticks, context) != frozen:
        raise RuntimeError("Data sources or actual market bytes changed during branch generation")
    return arrays


def validate(data, action_count=11, context=24):
    n = len(data["lengths"])
    shapes = {"states": (n, context, 13), "steps": (n, context), "endpoints": (n, action_count, 2)}
    shapes |= {
        k: (n, action_count)
        for k in ("terminal_score", "terminal_conversions", "terminal_violation")
    }
    shapes |= {
        k: (n,)
        for k in (
            "lengths",
            "baseline_index",
            "budget",
            "cpa",
            "spent",
            "conversions",
            "diagnostics",
        )
    }
    if set(data) != set(shapes) or any(data[k].shape != shape for k, shape in shapes.items()):
        raise RuntimeError("Continuation array schema or coverage changed")
    if any(not np.isfinite(v).all() for k, v in data.items() if k != "diagnostics"):
        raise RuntimeError("Nonfinite continuation data")
    if (
        np.any(data["lengths"] < 1)
        or np.any(data["lengths"] > context)
        or np.any(data["baseline_index"] < 0)
        or np.any(data["baseline_index"] >= action_count)
        or np.any(data["endpoints"] < 0)
        or np.any(data["spent"] < 0)
        or np.any(data["spent"] > data["budget"] + 1e-6)
        or np.any(
            data["endpoints"][:, :, 0] * data["budget"][:, None] + data["spent"][:, None]
            > data["budget"][:, None] + 1e-6
        )
    ):
        raise RuntimeError("Invalid accounting or hard-budget endpoint")
    for i, length in enumerate(data["lengths"]):
        tick = int(data["diagnostics"][i]["tick"])
        if (
            not np.array_equal(data["steps"][i, :length], np.arange(tick + 1 - length, tick + 1))
            or np.any(data["states"][i, length:] != 0)
            or np.any(data["steps"][i, length:] != 0)
        ):
            raise RuntimeError("Right padding or original tick IDs changed")


def _summary(data, mask):
    indices = np.flatnonzero(mask)
    row = np.arange(len(data["lengths"]))
    b = data["baseline_index"]
    baseline_score = data["terminal_score"][row, b]
    baseline_conv = data["terminal_conversions"][row, b]
    baseline_violation = data["terminal_violation"][row, b]
    oracle = np.argmax(data["terminal_score"], axis=1)
    eligible = (data["terminal_conversions"] >= 0.97 * baseline_conv[:, None]) & (
        data["terminal_violation"] <= baseline_violation[:, None]
    )
    guarded = np.argmax(np.where(eligible, data["terminal_score"], -np.inf), axis=1)
    if not eligible[row, b].all():
        raise RuntimeError("Feedback action disappeared from its own oracle guard")

    def policy(choices):
        scores = data["terminal_score"][row, choices][indices]
        conv = data["terminal_conversions"][row, choices][indices]
        violation = data["terminal_violation"][row, choices][indices]
        return {
            "mean_score": float(scores.mean()),
            "score_ratio_to_feedback": float(
                scores.mean() / max(baseline_score[indices].mean(), 1e-12)
            ),
            "mean_score_delta": float((scores - baseline_score[indices]).mean()),
            "mean_conversions": float(conv.mean()),
            "conversion_ratio_to_feedback": float(
                conv.mean() / max(baseline_conv[indices].mean(), 1e-12)
            ),
            "violation_rate": float(violation.mean()),
            "violation_delta": float(
                (violation.astype(np.float64) - baseline_violation[indices]).mean()
            ),
            "changed_action_fraction": float((choices[indices] != b[indices]).mean()),
            "mean_remaining_endpoint": data["endpoints"][row, choices][indices]
            .mean(axis=0)
            .tolist(),
        }

    return {
        "contexts": len(indices),
        "feedback": policy(b),
        "oracle": policy(oracle),
        "guarded_oracle": policy(guarded),
    }


def oracle_headroom(datasets, scenarios=SCENARIOS, ticks=TICKS):
    result = {
        "scope": "ORACLE ONLY: one hindsight action at each context, then fixed feedback; no deployed gain",
        "guard": "same context terminal conversions >= .97 * feedback and terminal violation <= feedback",
        "dependence": "contexts at different ticks share an episode; no IID or confidence claim",
        "feedback_endpoints": "endpoints[np.arange(N), baseline_index], current tick through end",
        "splits": {},
    }
    for split, data in datasets.items():
        diag = data["diagnostics"]
        result["splits"][split] = {
            "overall": _summary(data, np.ones(len(diag), dtype=bool)),
            "by_scenario": {
                scene: _summary(data, diag["scene_index"] == si)
                for si, scene in enumerate(scenarios)
            },
            "by_tick": {str(tick): _summary(data, diag["tick"] == tick) for tick in ticks},
            "by_scenario_tick": {
                scene: {
                    str(tick): _summary(data, (diag["scene_index"] == si) & (diag["tick"] == tick))
                    for tick in ticks
                }
                for si, scene in enumerate(scenarios)
            },
            "checks": {
                "exact_original_rollout_states_actions_and_accounting": True,
                "baseline_branch_equals_independent_feedback_continuation": True,
                "hard_budget_and_finiteness": True,
                "nonmonotone_endpoint_action_differences": int(
                    np.count_nonzero(np.diff(data["endpoints"], axis=1) < 0)
                ),
                "action_zero_nonzero_endpoint_contexts": int(
                    np.count_nonzero(np.any(data["endpoints"][:, 0] != 0, axis=1))
                ),
            },
        }
    return result


def _keep(path, value):
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Artifact identity changed: {path.name}")
    else:
        write_json(path, value)


def feasibility(output: Path) -> Path:
    started = time.perf_counter()
    main = ROOT / "artifacts/bidding/main-v2"
    main_manifest = read_json(main / "manifest.json")
    config = main_manifest["config"]
    if fingerprint(provenance(config)) != main_manifest["identity"]:
        raise RuntimeError("The frozen original implementation or main-study identity changed")
    expected = {"train": list(range(1001, 1017)), "validation": list(range(2001, 2007))}
    if any(config["study"][split + "_markets"] != markets for split, markets in expected.items()):
        raise RuntimeError("Feasibility uses exactly the original 22 train/development markets")
    actions = np.asarray(config["study"]["actions"], dtype=np.float64)
    paths = {
        split: [main / "markets" / f"{market}.npz" for market in markets]
        for split, markets in expected.items()
    }
    core = identity([p for group in paths.values() for p in group], actions, SCENARIOS, TICKS)
    core |= {
        "main_manifest_sha256": sha(main / "manifest.json"),
        "main_identity": main_manifest["identity"],
        "expected_contexts": {"train": 1440, "validation": 540},
        "oracle_only": True,
    }
    run_identity = fingerprint(core)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _keep(output / "identity.json", {"identity": run_identity, "core": core})
        datasets, dataset_artifacts = {}, {}
        for split, group in paths.items():
            path, stamp = output / f"{split}.npz", output / f"{split}.identity.json"
            split_identity = fingerprint({"identity": run_identity, "split": split})
            if path.exists():
                record = read_json(stamp)
                if record["identity"] != split_identity or record["npz_sha256"] != sha(path):
                    raise RuntimeError("Continuation cache identity or output bytes changed")
                with np.load(path, allow_pickle=False) as cache:
                    data = {k: cache[k] for k in cache.files}
                validate(data)
            else:
                if stamp.exists():
                    raise RuntimeError("A sealed continuation cache is missing")
                data = build(group, actions, SCENARIOS, TICKS)
                temporary = path.with_suffix(".partial.npz")
                np.savez_compressed(temporary, **data)
                temporary.replace(path)
                write_json(stamp, {"identity": split_identity, "npz_sha256": sha(path)})
            if len(data["lengths"]) != core["expected_contexts"][split]:
                raise RuntimeError("Feasibility context coverage mismatch")
            datasets[split] = data
            dataset_artifacts[split] = {
                "npz_sha256": sha(path),
                "identity_sha256": sha(stamp),
                "contexts": len(data["lengths"]),
                "arrays": {
                    k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()
                },
            }
        audit = oracle_headroom(datasets)
        audit["identity"] = run_identity
        _keep(output / "oracle-headroom.json", audit)
        current = identity(
            [p for group in paths.values() for p in group], actions, SCENARIOS, TICKS
        )
        if (
            current != {k: core[k] for k in current}
            or sha(main / "manifest.json") != core["main_manifest_sha256"]
        ):
            raise RuntimeError("Feasibility sources or actual inputs changed")
        _keep(
            output / "manifest.json",
            {
                "identity": run_identity,
                "core": core,
                "datasets": dataset_artifacts,
                "oracle_headroom_sha256": sha(output / "oracle-headroom.json"),
            },
        )
        seconds = time.perf_counter() - started
        metadata = output / "execution.json"
        if not metadata.exists():
            write_json(
                metadata,
                {
                    "identity": run_identity,
                    "wall_seconds": seconds,
                    "settle_calls": 760320,
                    "exact_contexts": 1980,
                    "fitted_models": 0,
                    "device": "CPU-only NumPy replay",
                },
            )
        print(
            f"continuation feasibility complete seconds={seconds:.3f} output={output}", flush=True
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1",
    )
    args = parser.parse_args()
    feasibility(args.output)


if __name__ == "__main__":
    main()
