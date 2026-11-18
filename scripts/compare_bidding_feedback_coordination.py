#!/usr/bin/env python3
"""One frozen, no-fit ablation of product versus minimum feedback signals."""
from __future__ import annotations

import argparse
import gc
import json
import resource
import shutil
import sys
import time
from pathlib import Path

import audit_bidding_winner_belief as a

START = time.monotonic()
import numpy as np  # noqa: E402 - thread limits set by audit import

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / ".omx/plans/prd-bidding-feedback-coordination-v1.md"
OUTPUT = ROOT / "artifacts/bidding/numerical-audits/feedback-coordination-v1"
SPLITS = {"train": (1002, 1006, 1010, 1014), "seen_development": tuple(range(2001, 2007))}
ARMS = ("FB", "minimum", "pacing_only", "quality_only")


def protocol():
    return {"version": "feedback-coordination-v1", "splits": SPLITS, "arms": ARMS,
            "candidate": "minimum only; other non-FB arms diagnostic",
            "signals": "original P/R from float32 observable state, gain.3/prior8",
            "actions": a.ACTIONS.tolist(), "scenes": a.SCENES,
            "gate": {"over50": 1.03, "under25": 1.03, "five_scene_macro": 1.03,
                     "scene_score": .98, "scene_conv": .97, "integer_violations": "not increased",
                     "positive_train_markets": 3, "positive_seen_markets": 4},
            "fit": False, "new_traffic": False, "independent_confirmation": False,
            "limits": {"seconds": 900, "memory_bytes": 2 * 2**30,
                       "output_bytes": 100 * 2**20, "free_disk_bytes": 10 * 2**30}}


def preflight():
    protected = a.read(ROOT / ".omx/state/bidding-robust/frozen-sources.json")
    sources = {name: a.sha(ROOT / name) for name in protected}
    a.require(sources == protected, "Protected native sources changed")
    names = [Path(__file__).resolve(), PLAN, ROOT / "scripts/audit_bidding_winner_belief.py",
             ROOT / "src/trajectorylab/bidding/market.py", ROOT / "src/trajectorylab/bidding/learning.py"]
    sources.update({str(p.relative_to(ROOT)): a.sha(p) for p in names})
    parent = a.read(ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1/manifest.json")
    inputs = {}
    for markets in SPLITS.values():
        for market in markets:
            path = a.MAIN / f"markets/{market}.npz"
            metadata = path.with_suffix(".json")
            expected = parent["core"]["inputs"][str(path)]
            for p, key in ((path, "npz_sha256"), (metadata, "metadata_sha256")):
                inputs[str(p.relative_to(ROOT))] = a.sha(p)
                a.require(inputs[str(p.relative_to(ROOT))] == expected[key], "Old tape identity changed")
            m = a.read(metadata)
            a.require(m["market_id"] == market and m["focal_agents"] == list(a.FOCALS), "Tape metadata changed")
    return {"protocol": protocol(), "sources": sources, "inputs": inputs}


def decision(state, budget, cpa, arm):
    elapsed, spend, conv = float(state[0]), float(state[3]) * budget, float(state[4]) * budget / cpa
    pacing = ((elapsed + .05) * budget / (spend + .05 * budget)) ** .3
    quality = ((conv + 8.) * cpa / (spend + 8. * cpa)) ** .3
    desired = {"FB": pacing * quality, "minimum": min(pacing, quality),
               "pacing_only": pacing, "quality_only": quality}[arm]
    action = float(a.ACTIONS[np.abs(a.ACTIONS - np.clip(desired, .05, 3.)).argmin()])
    return action, pacing, quality


def aggregate(terminals):
    result = {}
    for split, markets in SPLITS.items():
        rows = [r for r in terminals if r["split"] == split]
        arms = {}
        for arm in ARMS:
            scene_cells, per_market = {}, {}
            for scene in a.SCENES:
                chosen = [r for r in rows if r["arm"] == arm and r["scene"] == scene]
                base = [r for r in rows if r["arm"] == "FB" and r["scene"] == scene]
                a.require(len(chosen) == len(base) == len(markets) * 6, "Terminal coverage changed")
                cell = {}
                for metric in ("score", "conversions", "cost", "budget_utilization"):
                    x, b = np.mean([r[metric] for r in chosen]), np.mean([r[metric] for r in base])
                    cell[metric] = {"value": float(x), "FB": float(b), "ratio": float(x / b) if b != 0 else None}
                cell["violations"] = {"value": sum(r["violation"] for r in chosen),
                                      "FB": sum(r["violation"] for r in base), "campaigns": len(chosen)}
                scene_cells[scene] = cell
            for market in markets:
                ratios = []
                for scene in a.SCENES:
                    x = [r["score"] for r in rows if r["arm"] == arm and r["scene"] == scene and r["market"] == market]
                    b = [r["score"] for r in rows if r["arm"] == "FB" and r["scene"] == scene and r["market"] == market]
                    a.require(len(x) == len(b) == 6 and np.mean(b) != 0, "Undefined per-market ratio")
                    ratios.append(float(np.mean(x) / np.mean(b)))
                per_market[str(market)] = float(np.mean(ratios))
            arms[arm] = {"scenes": scene_cells, "per_market_macro": per_market,
                         "five_scene_macro": float(np.mean([scene_cells[s]["score"]["ratio"] for s in a.SCENES]))}
        candidate = arms["minimum"]
        protections = {s: {"score": c["score"]["ratio"] >= .98,
                            "conversions": c["conversions"]["ratio"] >= .97,
                            "violations": c["violations"]["value"] <= c["violations"]["FB"]}
                       for s, c in candidate["scenes"].items()}
        flags = {"over50": candidate["scenes"]["over_50"]["score"]["ratio"] >= 1.03,
                 "under25": candidate["scenes"]["under_25"]["score"]["ratio"] >= 1.03,
                 "macro": candidate["five_scene_macro"] >= 1.03,
                 "protections": all(all(c.values()) for c in protections.values()),
                 "market_direction": sum(x > 1 for x in candidate["per_market_macro"].values()) >= (3 if split == "train" else 4)}
        result[split] = {"arms": arms, "scene_protections": protections, "flags": flags,
                         "split_pass": all(flags.values())}
    return result


def run(frozen, freeze):
    OUTPUT.mkdir(parents=True, exist_ok=False)
    a.write(OUTPUT / "manifest.json", {"freeze": freeze, "frozen_before_rollout": True, "frozen": frozen})
    terminals, switch_count, max_memory = [], 0, 0
    try:
        with (OUTPUT / "tick-logs.jsonl").open("w") as stream:
            for split, markets in SPLITS.items():
                for market in markets:
                    tape = a.Tape(a.MAIN / f"markets/{market}.npz")
                    for scene in a.SCENES:
                        for fi, focal in enumerate(a.FOCALS):
                            budget, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
                            for arm in ARMS:
                                cost, conv, previous = 0., 0, (0., 0, 0., 0)
                                for tick in range(48):
                                    a.require(time.monotonic() - START <= 900, "Wall resource limit")
                                    bias = a.scenario_bias(scene, tick)  # environment only
                                    state = a.observation(tape, fi, tick, cost, conv, previous, bias)
                                    action, pacing, quality = decision(state, budget, cpa, arm)
                                    native, _ = a.Feedback(a.ACTIONS, .3, 8.)([state], [], [], budget, cpa)
                                    if arm == "FB":
                                        a.require(action == native, "Native FB action replay failed")
                                    if arm == "minimum":
                                        switch_count += int(action != native)
                                    before_cost, before_conv = cost, conv
                                    settlement = tape.settle(fi, tick, action * bias, budget - cost)
                                    tick_cost, tick_conv, wins, blocked = settlement
                                    cost += tick_cost
                                    conv += tick_conv
                                    previous = tick_cost, tick_conv, action, wins
                                    stream.write(json.dumps({"split": split, "market": market, "scene": scene,
                                        "focal": focal, "tick": tick, "arm": arm, "state": state.tolist(),
                                        "budget": budget, "cpa": cpa, "past_cost": before_cost, "past_C": before_conv,
                                        "P": pacing, "R": quality, "action": action, "same_state_FB": native,
                                        "settlement": list(settlement), "cost": cost, "C": conv}, allow_nan=False) + "\n")
                                a.require(cost <= budget + 1e-6, "Hard budget violated")
                                if arm == "FB":
                                    from trajectorylab.bidding.learning import rollout
                                    reference = rollout(tape, fi, a.ACTIONS, scene, a.Feedback(a.ACTIONS, .3, 8.))
                                    a.require(reference["cost"] == cost and reference["conversions"] == conv,
                                              "Native FB full trajectory differs")
                                terminals.append({"split": split, "market": market, "scene": scene,
                                                  "focal": focal, "arm": arm, "cost": cost,
                                                  "conversions": conv, "budget": budget, "cpa": cpa,
                                                  "budget_utilization": cost / budget, "score": a.score(cost, conv, cpa),
                                                  "violation": int(cost > cpa * conv + 1e-8)})
                    del tape
                    gc.collect()
                    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    max_memory = max(max_memory, int(peak if sys.platform == "darwin" else peak * 1024))
                    a.require(max_memory <= 2 * 2**30 and time.process_time() <= 900, "CPU/memory limit")
                    a.require(shutil.disk_usage(OUTPUT).free >= 10 * 2**30, "Disk resource limit")
                    print(json.dumps({"market_complete": market, "seconds": time.monotonic() - START}), flush=True)
        a.require(len(terminals) == 1200, "Incomplete terminal coverage")
        a.require(sum(p.stat().st_size for p in OUTPUT.iterdir()) <= 100 * 2**20, "Output limit")
        result = aggregate(terminals)
        a.require(all(a.sha(ROOT / p) == h for p, h in (frozen["sources"] | frozen["inputs"]).items()), "Source/input changed")
        summary = {"status": "complete", "freeze": freeze, "splits": result,
                   "candidate_pass": all(r["split_pass"] for r in result.values()) and switch_count > 0,
                   "minimum_non_FB_decisions": switch_count, "input_source_integrity": True,
                   "independent_confirmation": False, "generative_contribution": False,
                   "resources": {"wall_seconds": time.monotonic() - START, "cpu_seconds": time.process_time(),
                                 "peak_memory_bytes": max_memory}}
        a.write(OUTPUT / "terminals.json", terminals)
        a.write(OUTPUT / "summary.json", summary)
        print(json.dumps({"candidate_pass": summary["candidate_pass"], "splits": {
            k: {"flags": v["flags"], "macro": v["arms"]["minimum"]["five_scene_macro"]} for k, v in result.items()}}), flush=True)
    except BaseException as error:
        a.write(OUTPUT / "failure.json", {"status": "invalid_or_incomplete", "error": str(error)})
        raise
    finally:
        files = {p.name: a.sha(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != "seal.json"}
        a.write(OUTPUT / "seal.json", {"files": files, "identity": a.identity(files)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--freeze")
    args = parser.parse_args()
    frozen = preflight()
    freeze = a.identity(frozen)
    print(json.dumps({"preflight": True, "freeze": freeze, "fitted": False, "new_traffic": False}), flush=True)
    if args.run:
        a.require(args.freeze == freeze, "Requires current frozen identity")
        run(frozen, freeze)


if __name__ == "__main__":
    main()
