#!/usr/bin/env python3
"""No-fit consistency audit of tick and fixed-action open-loop response curves."""

from __future__ import annotations

import argparse
import gc
import resource
import shutil
import time
from pathlib import Path

import audit_bidding_winner_belief as a
import numpy as np
import torch

from trajectorylab.bidding.learning import Learned
from trajectorylab.bidding_robust_learning import RobustLearned

ROOT = a.ROOT
PLAN = ROOT / ".omx/plans/prd-bidding-current-response-consistency-v1.md"
OUTPUT = ROOT / "artifacts/bidding/numerical-audits/current-response-consistency-v1"
MARKETS = (1002, 1006, 1010, 1014)
TICKS = (0, 16, 24, 32)
SCALES = (1.0, 0.75, 1.25, 1.5)
SEEDS = (11, 23, 37)
START = time.monotonic()


def pairs():
    return [
        (si, ai, bi)
        for si, c in enumerate(SCALES)
        if si
        for ai, x in enumerate(a.ACTIONS)
        if x > 0
        for bi, y in enumerate(a.ACTIONS)
        if x == c * y
    ]


def checkpoints():
    return {
        f"{family}-{seed}": (
            a.MAIN / f"models/response-{seed}/best.pt"
            if family == "nominal"
            else ROOT / f"artifacts/bidding/robust-v1/models/response-log-{seed}/best.pt"
        )
        for family in ("nominal", "robust")
        for seed in SEEDS
    }


def stratum(tick, si):
    return (
        "initial_shift"
        if tick == 0
        else ("registered_midshift" if tick == 24 and SCALES[si] == 1.5 else "additional_shock")
    )


def limits():
    a.require(time.monotonic() - START <= 900, "Wall limit exceeded")
    a.require(time.process_time() <= 900, "CPU limit exceeded")
    a.require(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 2 * 2**30,
        "Memory limit exceeded (macOS bytes)",
    )
    a.require(shutil.disk_usage(ROOT).free >= 10 * 2**30, "Disk reserve")


def preflight():
    old = a.read(ROOT / ".omx/state/bidding-robust/frozen-sources.json")
    a.require(all(a.sha(ROOT / p) == h for p, h in old.items()), "Protected source changed")
    source_names = [
        Path(__file__).resolve(),
        PLAN,
        ROOT / "scripts/audit_bidding_winner_belief.py",
        ROOT / "src/trajectorylab/bidding_robust_learning.py",
    ]
    sources = old | {str(p.relative_to(ROOT)): a.sha(p) for p in source_names}
    parent = a.read(
        ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1/manifest.json"
    )
    inputs = {}
    for market in MARKETS:
        path = a.MAIN / f"markets/{market}.npz"
        expected = parent["core"]["inputs"][str(path)]
        for p, key in ((path, "npz_sha256"), (path.with_suffix(".json"), "metadata_sha256")):
            h = a.sha(p)
            a.require(h == expected[key], "Old market input changed")
            inputs[str(p.relative_to(ROOT))] = h
    registered = a.tomllib.loads((ROOT / "configs/bidding-confirmation.toml").read_text())
    for name, p in checkpoints().items():
        h = a.sha(p)
        if name.startswith("nominal"):
            a.require(
                h == registered["checkpoint_sha256"][name.split("-")[-1]],
                "Nominal checkpoint changed",
            )
        inputs[str(p.relative_to(ROOT))] = h
    protocol = {
        "version": OUTPUT.name,
        "markets": MARKETS,
        "focals": a.FOCALS,
        "ticks": TICKS,
        "scales_including_base": SCALES,
        "context": 24,
        "history": "nominal FB gain0.3/prior3",
        "pairs": pairs(),
        "models": list(checkpoints()),
        "fit": False,
        "new_data": False,
        "labels": "raw tick and persistent-shock fixed-action open-loop tail, cost/B and conversions*CPA/B",
        "interpretation": "same physical environment only; not cross-world or closed-loop invariance",
        "training_authorized": False,
    }
    a.require(len(pairs()) == 7, "Pair coverage changed")
    return {"sources": sources, "inputs": inputs, "protocol": protocol}


def collect():
    contexts, histories, steps_all, masks, labels, indices, settlements = [], [], [], [], [], [], []
    for market in MARKETS:
        limits()
        tape = a.Tape(a.MAIN / f"markets/{market}.npz")
        for fi, focal in enumerate(a.FOCALS):
            budget, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
            cost, conv, history, previous = 0.0, 0, [], (0.0, 0, 0.0, 0)
            for tick in range(48):
                state = a.observation(tape, fi, tick, cost, conv, previous, 1.0)
                history.append(state)
                if tick in TICKS:
                    start = max(0, tick + 1 - 24)
                    length = tick + 1 - start
                    x = np.zeros((4, 24, 13), dtype=np.float32)
                    steps = np.zeros(24, dtype=np.int64)
                    mask = np.zeros(24, dtype=bool)
                    steps[:length], mask[:length] = np.arange(start, tick + 1), True
                    lab = np.zeros((4, 4, 11), dtype=np.float64)
                    idx = np.zeros((4, 11), dtype=np.int64)
                    settled = np.zeros((4, 11, 4), dtype=np.float64)
                    for si, c in enumerate(SCALES):
                        x[si, :length] = np.asarray(history[start:])
                        x[si, length - 1] = a.observation(tape, fi, tick, cost, conv, previous, c)
                        a.require(
                            np.array_equal(
                                np.delete(x[si, :length], 9, axis=1),
                                np.delete(x[0, :length], 9, axis=1),
                            ),
                            "Unexpected transformed feature",
                        )
                        a.require(
                            np.array_equal(x[si, : length - 1], x[0, : length - 1]),
                            "Past observation was transformed",
                        )
                        for ai, action in enumerate(a.ACTIONS):
                            idx[si, ai] = tape.index(action * c)
                            future_cost = tape.arrays["costs"][fi, tick:, idx[si, ai]]
                            future_conv = tape.arrays["conversions"][fi, tick:, idx[si, ai]]
                            lab[si, :, ai] = (
                                future_cost[0] / budget,
                                future_conv[0] * cpa / budget,
                                future_cost[::-1].cumsum()[-1] / budget,
                                future_conv[::-1].cumsum()[-1] * cpa / budget,
                            )
                            settled[si, ai] = tape.settle(fi, tick, action * c, budget - cost)
                        a.require(
                            np.all(lab[si, :, 0] == 0) and np.all(settled[si, 0, :3] == 0),
                            "Nonzero structural zero label",
                        )
                    for si, ai, bi in pairs():
                        a.require(idx[0, ai] == idx[si, bi], "Unequal exact tape indices")
                        for future_tick in range(tick, 48):
                            a.require(
                                tape.index(a.ACTIONS[ai]) == tape.index(a.ACTIONS[bi] * SCALES[si]),
                                f"Unequal fixed-action future index at {future_tick}",
                            )
                        a.require(
                            np.array_equal(lab[0, :, ai], lab[si, :, bi]), "Raw labels differ"
                        )
                        a.require(
                            np.array_equal(settled[0, ai], settled[si, bi]), "Settlements differ"
                        )
                    contexts.append(
                        {
                            "market": market,
                            "focal": focal,
                            "tick": tick,
                            "budget": budget,
                            "cpa": cpa,
                            "past_cost": cost,
                            "past_C": conv,
                            "length": length,
                        }
                    )
                    histories.append(x)
                    steps_all.append(steps)
                    masks.append(mask)
                    labels.append(lab)
                    indices.append(idx)
                    settlements.append(settled)
                action, _ = a.Feedback(a.ACTIONS, 0.3, 3.0)([state], [], [], budget, cpa)
                tc, cv, wins, _ = tape.settle(fi, tick, action, budget - cost)
                cost, conv = cost + tc, conv + cv
                previous = tc, cv, action, wins
            a.require(cost <= budget + 1e-6, "History exceeded budget")
        del tape
        gc.collect()
    a.require(len(contexts) == 96, "Context count")
    return contexts, dict(
        states=np.array(histories),
        steps=np.array(steps_all),
        mask=np.array(masks),
        labels=np.array(labels),
        tape_indices=np.array(indices),
        settlements=np.array(settlements),
    )


@torch.inference_mode()
def predict(data, contexts):
    result = []
    torch.set_num_threads(1)
    for name, path in checkpoints().items():
        limits()
        loader = RobustLearned if name.startswith("robust") else Learned
        model = loader(path, a.ACTIONS, 24, "cpu")
        predictions = []
        for i, context in enumerate(contexts):
            length = context["length"]
            x = torch.as_tensor((data["states"][i, :, :length] - model.mean) / model.std)
            steps = torch.as_tensor(np.broadcast_to(data["steps"][i, :length], (4, length)).copy())
            mask = torch.ones((4, length), dtype=torch.bool)
            p = model.model(x, steps, mask)[:, -1].numpy()
            a.require(np.isfinite(p).all() and (p >= 0).all(), "Invalid predictions")
            a.require(np.all(p[:, :, 0] == 0), "Nonzero predicted structural zero")
            predictions.append(p)
        result.append(predictions)
        del model
        gc.collect()
        print(f"inference complete: {name}", flush=True)
    return np.asarray(result)


def summarize(contexts, data, predictions):
    cells = {}
    for mi, name in enumerate(checkpoints()):
        for i, context in enumerate(contexts):
            for si, ai, bi in pairs():
                key = (name, context["market"], stratum(context["tick"], si))
                row = cells.setdefault(
                    key,
                    {
                        "pairs": 0,
                        "prediction_gap_sq": np.zeros(4),
                        "base_error_sq": np.zeros(4),
                        "transformed_error_sq": np.zeros(4),
                        "label_sq": np.zeros(4),
                        "zero_labels": np.zeros(4, dtype=np.int64),
                    },
                )
                base, changed = predictions[mi, i, 0, :, ai], predictions[mi, i, si, :, bi]
                label = data["labels"][i, 0, :, ai]
                row["pairs"] += 1
                row["prediction_gap_sq"] += (changed - base) ** 2
                row["base_error_sq"] += (base - label) ** 2
                row["transformed_error_sq"] += (changed - label) ** 2
                row["label_sq"] += label**2
                row["zero_labels"] += label == 0
    rows = []
    for (model, market, group), cell in cells.items():
        row = {"model": model, "market": market, "stratum": group, "pairs": cell["pairs"]}
        for key, values in cell.items():
            if key != "pairs":
                row[key if key == "zero_labels" else key + "_mean"] = (
                    values.tolist() if key == "zero_labels" else (values / cell["pairs"]).tolist()
                )
        rows.append(row)
    return {
        "status": "complete",
        "physical_contexts": len(contexts),
        "physical_pairs": 672,
        "model_pair_evaluations": 4032,
        "head_order": [
            "tick_cost/B",
            "tick_conversions*CPA/B",
            "fixed_action_raw_tail_cost/B",
            "fixed_action_raw_tail_conversions*CPA/B",
        ],
        "rows": rows,
        "independent_units": "4 already-fitted training markets",
        "policy_gain_measured": False,
        "generative_contribution": False,
        "training_authorized": False,
        "next_boundary": "review diagnostic pattern; no automatic fit",
    }


def run(frozen, freeze):
    OUTPUT.mkdir(parents=True, exist_ok=False)
    a.write(
        OUTPUT / "manifest.json",
        {"freeze": freeze, "frozen_before_inference": True, "frozen": frozen},
    )
    try:
        contexts, data = collect()
        a.write(OUTPUT / "contexts.json", contexts)
        np.savez_compressed(OUTPUT / "observations.npz", **data)
        predictions = predict(data, contexts)
        np.savez_compressed(OUTPUT / "predictions.npz", predictions=predictions)
        result = summarize(contexts, data, predictions)
        a.require(preflight() == frozen, "Sources or inputs changed during run")
        limits()
        result["freeze"] = freeze
        result["resources"] = {
            "wall_seconds": time.monotonic() - START,
            "cpu_seconds": time.process_time(),
            "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        a.write(OUTPUT / "summary.json", result)
        a.require(sum(p.stat().st_size for p in OUTPUT.iterdir()) <= 100 * 2**20, "Output limit")
        files = {p.name: a.sha(p) for p in OUTPUT.iterdir() if p.is_file()}
        a.write(OUTPUT / "seal.json", {"files": files, "identity": a.identity(files)})
        print(
            f"complete contexts=96 physical_pairs=672 model_pairs=4032 freeze={freeze}", flush=True
        )
    except BaseException as exc:
        a.write(
            OUTPUT / "failure.json",
            {"error": type(exc).__name__, "message": str(exc), "freeze": freeze},
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--freeze")
    args = parser.parse_args()
    frozen = preflight()
    freeze = a.identity(frozen)
    if args.run:
        a.require(args.freeze == freeze, "Required freeze does not match preflight")
        run(frozen, freeze)
    else:
        print(f"preflight PASS: freeze={freeze}; no inference, fit, or output written")
