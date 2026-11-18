#!/usr/bin/env python3
"""Compare fixed feedback scales, frozen response models and causal routing.

Preflight reads identities and the sealed training reference only. It generates
no traffic and creates no study output. Running requires the reviewed freeze.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import math
import platform
import resource
import shutil
import signal
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

PROCESS_START = time.monotonic()
import audit_bidding_winner_belief as audit  # noqa: E402 - sets thread limits
import numpy as np  # noqa: E402

from trajectorylab.bidding.learning import Learned  # noqa: E402
from trajectorylab.bidding_calibration_router import (  # noqa: E402
    branch,
    diagnose,
    scaled_feedback_action,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/bidding-calibration-router.json"
MAIN = ROOT / "artifacts/bidding/main-v2"
OLD = ROOT / "artifacts/bidding/numerical-audits/calibration-control-v1"
PARENT = ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1/manifest.json"
VERIFICATION = ROOT / ".omx/state/bidding-v2/calibration-control-independent-verification.json"
TRAIN = (1002, 1006, 1010, 1014)
EVALUATION = (2001, 2002, 2003, 2004, 2005, 2006)
SCENES, FOCALS, ACTIONS = audit.SCENES, audit.FOCALS, audit.ACTIONS
SEEDS = (11, 23, 37)
SCALES = (.5, 2 / 3, .75, 1., 1.25, 4 / 3, 1.5)
SCALE_ARMS = tuple(f"scale_{i}" for i in range(len(SCALES)))
RESPONSE_ARMS = tuple(f"response_{s}" for s in SEEDS)
ROUTER_ARMS = tuple(f"router_{s}" for s in SEEDS)
EVAL_ARMS = SCALE_ARMS + RESPONSE_ARMS + ROUTER_ARMS + ("gate_scaled", "known_inverse")
BASE = SCALE_ARMS[SCALES.index(1.)]
PHASES = {"train": (TRAIN, SCALE_ARMS), "evaluation": (EVALUATION, EVAL_ARMS)}
REQUESTED = sum(len(ms) * len(ps) * len(SCENES) * len(FOCALS) * 48 for ms, ps in PHASES.values())
TERMINALS = REQUESTED // 48
LIMIT, RESERVE = 1800, 60
MIN_FREE, MAX_OUTPUT, MAX_MEMORY = 10 * 2**30, 2**30, 10 * 2**30


def configuration():
    config = audit.read(CONFIG)
    expected = {
        "version": "calibration-router-v1", "train_markets": list(TRAIN),
        "evaluation_markets": list(EVALUATION), "scenarios": list(SCENES),
        "scales": list(SCALES), "seeds": list(SEEDS), "minimum_expected": 8.,
        "probability": .95, "margin": .1, "wall_limit_seconds": LIMIT,
        "gate_scale_selection_scenarios": ["over_25", "over_50"],
        "output": "artifacts/bidding/numerical-audits/calibration-router-v1",
    }
    audit.require(config == expected, "Configuration differs from the fixed comparison")
    return config


def protocol(config):
    return {
        "version": config["version"], "configuration": config,
        "phases": {k: {"markets": list(ms), "arms": list(ps)} for k, (ms, ps) in PHASES.items()},
        "scenarios": list(SCENES), "focals": list(FOCALS), "actions": ACTIONS.tolist(),
        "requested_cells": REQUESTED, "requested_terminals": TERMINALS,
        "order": "train four markets then freeze both scales; evaluation six seen validation markets; tick,scene,focal,arm in fixed order; release each market",
        "scaled_feedback": "native continuous P*Q**.3 times scale, then clip[.05,3] and nearest original11 grid, first tie; scale1 is native feedback gain.3 prior8",
        "scale_selection": "global: mean terminal score over four training markets/all five scenes/six focals; gate: same but over25/over50 only; exact ties favor scale nearest1 then smaller scale; no evaluation selection",
        "response": "unchanged main-v2 response best.pt seeds11/23/37,context24,CPU; Learned corrected=False,hard_cpa=False; separate causal state/action/reward histories, shared read-only model weights",
        "reference": "rho0=sum accepted integer C/sum accepted public p only for FB/calibrated rows of sealed old calibration-control-v1 in four training markets; selected-winner nominal reference, not true p calibration",
        "diagnosis": "own past accepted C/E only; Gamma(shape1+C,rate1+rho0*E); expected<8 unknown; Pr(theta<.9)>=.95 over; Pr(theta>1.1)>=.95 under; probabilities are unvalidated working-model diagnostics",
        "router": "over uses unchanged response model on own entire causal history; under/unknown native FB; no scenario,bias,current outcome,future traffic,unexposed labels in policy API",
        "gate_scaled": "same diagnose on own past accepted C/E; over uses train-over-selected scaled FB; under/unknown native FB; matched mechanism comparator with independently evolving history",
        "known_inverse": "diagnostic only: native FB selects effective grid action from own realized cost/conversions; evaluator sets nominal alpha=effective/scenario_bias without new clip/grid; execute exact effective directly; allowed nominalalpha>3; restore nominal FB terminal every scene; no optimality claim",
        "public_prediction": "adapter sums generator public p*scenario_bias over actually paid/exposed/budget-accepted PV prefix in original order; no latent p access; finite[0,1] required; update C/E only after settlement",
        "kernel": "same public generator episode=market,pv_num500000,one prepared CRN kernel/tick and original fixed sinusoidal opponents; memo only untruncated chosen(focal,effective action) prefix, own budget truncation per arm",
        "replay": "every actual arm action settlement compared to immutable exact tape at same effective action and own remaining budget; integers exact,cost abs<=1e-8; independent native FB replay states/actions/integer settlement exact,cumulativecost<=1e-8; oracle compared to nominal FB per tick and terminal",
        "aggregation": "six-focal market means then market means; scenario ratio=ratio of means; macro=equal five-scene mean of score ratios; router group averages three frozen seeds; violations exact integer counts, reference replicated three times for group; no epsilon for zero denominator",
        "gate": {
            "eligible": "three-seed router group only", "macro_ratio_vs_FB_global_scale_gate_scaled_min": 1.03,
            "every_scene_score_vs_FB_min": .98, "over50_score_vs_FB_min": 1.03,
            "every_scene_conversion_vs_FB_min": .97, "every_scene_violation_count_increase_max": 0,
            "every_seed_macro_vs_global_scale_and_gate_scaled_strictly_above": 1.,
            "positive_markets_vs_global_scale_min": 4, "complete_coverage_required": True,
        },
        "diagnostics": "per scene/seed label occupancy/confusion against injected direction evaluator-only; firstover tick and spent fraction; midshift pre24 falseover separate and firstpost24 over delay; actual response branch count and disagreement with same-state FB; all descriptive,no diagnosis accuracy or causal mechanism proof",
        "resources": {"wall_cpu_seconds_from_process_start": LIMIT, "seal_reserve_seconds": RESERVE,
            "threads": 1, "memory_bytes_max": MAX_MEMORY, "artifact_bytes_max": MAX_OUTPUT,
            "free_disk_bytes_min": MIN_FREE, "stop": "resource/integrity only; no effect-dependent stopping or parameter rescue; real/profiling timers and per-cell checks"},
        "claim": "Exploratory comparison on six historically seen validation markets, not independent confirmation, generative contribution, guaranteed CPA safety or industrial deployment evidence.",
        "followup": "any evaluable scientific gate failure stops this candidate; pass permits separate new-market confirmation preregistration only; no automatic fits/thresholds/seeds",
    }


def training_reference():
    selected, conversions, prediction = 0, 0, 0.
    with (OLD / "tick-logs.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["market"] in TRAIN and row["policy"] == "FB" and row["scene"] == "calibrated":
                selected += 1
                conversions += row["conversions"]
                prediction += row["predicted_sum"]
    audit.require(selected == len(TRAIN) * len(FOCALS) * 48 and prediction > 0 and conversions > 0,
                  "Incomplete nominal training reference")
    return {"rho0": conversions / prediction, "conversions": conversions,
            "prediction_sum": prediction, "cells": selected,
            "origin": "sealed old train FB calibrated accepted-winner logs only"}


def preflight():
    """Hash sources, models and existing tapes; never load tape arrays/generate PVs."""
    config = configuration()
    old_manifest, seal = audit.read(OLD / "manifest.json"), audit.read(OLD / "seal.json")
    audit.require(audit.identity(seal["files"]) == seal["content_identity"] and
                  all(audit.sha(OLD / p) == h for p, h in seal["files"].items()), "Old control seal changed")
    verified = audit.read(VERIFICATION)
    audit.require(verified["integrity_verdict"] == "PASS" and verified["failed_checks"] == [],
                  "Old independent integrity verification did not pass")
    old_summary = audit.read(OLD / "summary.json")
    audit.require(old_summary["status"] == "complete" and old_summary["full_FB_replay_pass"],
                  "Old native replay incomplete")
    sources = old_manifest["frozen"]["sources"].copy()
    audit.require(audit.after_hashes(sources) == sources, "Historical frozen source changed")
    old_inputs_frozen = old_manifest["frozen"]["inputs"]
    audit.require(audit.after_hashes(old_inputs_frozen) == old_inputs_frozen, "Historical frozen input changed")
    for name in (str(Path(__file__).resolve().relative_to(ROOT)), "src/trajectorylab/bidding_calibration_router.py"):
        sources[name] = audit.sha(ROOT / name)
    main = audit.read(MAIN / "manifest.json")
    study = main["config"]["study"]
    audit.require(study["actions"] == ACTIONS.tolist() and study["focal_agents"] == list(FOCALS)
                  and main["config"]["training"]["context"] == 24 and study["pv_count"] == 500000
                  and set(TRAIN) <= set(study["train_markets"])
                  and list(EVALUATION) == study["validation_markets"], "Original study contract changed")
    old_inputs = audit.read(PARENT)["core"]["inputs"]
    paths = [CONFIG, PARENT, MAIN / "manifest.json", MAIN / "training.json", MAIN / "selection.json",
             OLD / "manifest.json", OLD / "seal.json", OLD / "summary.json", OLD / "tick-logs.jsonl", VERIFICATION,
             ROOT / "configs/bidding-confirmation.toml"]
    for market in TRAIN + EVALUATION:
        path = MAIN / "markets" / f"{market}.npz"
        metadata, expected = audit.read(path.with_suffix(".json")), old_inputs[str(path)]
        audit.require(metadata["market_id"] == market and metadata["focal_agents"] == list(FOCALS)
                      and metadata["source"] == main["auction"]
                      and audit.sha(path) == expected["npz_sha256"]
                      and audit.sha(path.with_suffix(".json")) == expected["metadata_sha256"],
                      "Existing train/seen-validation tape identity changed")
        paths.extend([path, path.with_suffix(".json")])
    checkpoints = audit.tomllib.loads((ROOT / "configs/bidding-confirmation.toml").read_text())["checkpoint_sha256"]
    for seed in SEEDS:
        path = MAIN / "models" / f"response-{seed}" / "best.pt"
        audit.require(audit.sha(path) == checkpoints[str(seed)], "Original response checkpoint changed")
        paths.append(path)
    frozen = {"sources": sources, "inputs": {str(p.relative_to(ROOT)): audit.sha(p) for p in paths},
        "protocol": protocol(config), "reference": training_reference(), "study": study,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": audit.scipy.__version__, "torch": audit.torch.__version__, "platform": platform.platform()}}
    return frozen, {"freeze_identity": audit.identity(frozen), "protocol_sha256": audit.identity(frozen["protocol"]),
        "source_sha256": sources[str(Path(__file__).resolve().relative_to(ROOT))],
        "requested_cells": REQUESTED, "requested_terminals": TERMINALS,
        "read_only": True, "pv_generated": False, "output_created": False, "reference": frozen["reference"]}


class ResourceStop(Exception):
    pass


class Resources:
    def __init__(self, output):
        self.output, self.ticks, self.markets = output, [], []
        self.kernel_attempted, self.kernel_completed = 0, 0
        self.generation, self.last_scan, self.bytes = None, -math.inf, 0

    def snapshot(self):
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"wall_seconds": time.monotonic() - PROCESS_START, "cpu_seconds": time.process_time(),
                "peak_memory_bytes": int(memory if audit.sys.platform == "darwin" else memory * 1024),
                "free_disk_bytes": shutil.disk_usage(self.output).free, "artifact_bytes": self.bytes}

    def check(self, next_phase=None):
        if time.monotonic() - self.last_scan >= 15:
            self.bytes = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
            self.last_scan = time.monotonic()
        state = self.snapshot()
        if state["wall_seconds"] >= LIMIT - RESERVE or state["cpu_seconds"] >= LIMIT - RESERVE:
            raise ResourceStop("Time boundary reached; reserve60s for seal")
        if state["peak_memory_bytes"] > MAX_MEMORY or state["free_disk_bytes"] < MIN_FREE or self.bytes > MAX_OUTPUT:
            raise ResourceStop("Memory/disk/artifact resource boundary reached")
        previous = [r for r in self.markets if r["phase"] == next_phase]
        if next_phase and previous:
            for key in ("wall_seconds", "cpu_seconds"):
                estimate = max(previous[-1][key], float(np.mean([r[key] for r in previous])))
                if state[key] + estimate >= LIMIT - RESERVE:
                    raise ResourceStop(f"Next complete market exceeds remaining {key} budget")

    def save(self):
        audit.write(self.output / "resources.json", {"current": self.snapshot(), "generation": self.generation,
                    "ticks": self.ticks, "markets": self.markets, "kernel_calls_attempted": self.kernel_attempted,
                    "kernel_calls_completed": self.kernel_completed, "requested_cells": REQUESTED,
                    "limits": {"seconds": LIMIT, "reserve": RESERVE, "memory": MAX_MEMORY, "output": MAX_OUTPUT}})


def decide(arm, track, state, budget, cpa, models, rho0, config, selection):
    """Policy boundary: own history and fixed train-derived constants only."""
    feedback = scaled_feedback_action(state, ACTIONS, budget, cpa)
    diagnostic = diagnose(track["C"], track["E"], rho0, config["minimum_expected"],
                          config["probability"], config["margin"])
    chosen_branch, learned = "feedback", {}
    if arm in SCALE_ARMS:
        chosen_branch = "scaled_feedback"
        action = scaled_feedback_action(state, ACTIONS, budget, cpa, SCALES[SCALE_ARMS.index(arm)])
    elif arm == "gate_scaled":
        chosen_branch = "scaled_feedback" if diagnostic["label"] == "over" else "feedback"
        scale = selection["over"]["scale"] if diagnostic["label"] == "over" else 1.
        action = scaled_feedback_action(state, ACTIONS, budget, cpa, scale)
    elif arm == "known_inverse":
        # Returns the desired effective action. Privileged inverse is adapter-only.
        action, chosen_branch = feedback, "known_inverse_diagnostic"
    else:
        seed = int(arm.rsplit("_", 1)[1])
        chosen_branch = "response" if arm.startswith("response_") else branch(diagnostic["label"])
        if chosen_branch == "response":
            action, learned = models[seed](track["states"], track["actions"], track["rewards"], budget, cpa)
        else:
            action = feedback
    return {"action": float(action), "same_state_FB_action": feedback,
            "branch": chosen_branch, "diagnosis": diagnostic, "model_diagnostic": learned,
            "action_differs_same_state_FB": action != feedback}


def replay_check(actual, expected):
    return {"expected_cost": float(expected[0]), "expected_conversions": int(expected[1]),
            "expected_exposures": int(expected[2]), "expected_blocked": bool(expected[3]),
            "cost_error": abs(actual[0] - expected[0]), "exact_cost_equal": actual[0] == expected[0],
            "integers_equal": actual[1:] == expected[1:]}


def market_run(phase, market, stream, rows, resources, models, reference, config, selection):
    begin = resources.snapshot()
    arms = PHASES[phase][1]
    resources.generation = {"phase": phase, "market": market, "status": "running", "start": begin}
    resources.save()
    tape = audit.Tape(MAIN / "markets" / f"{market}.npz")
    Generator = audit._load("router_public_generator", "NeurIPSPvGen.py").NeurIPSPvGen
    Env = audit._load("router_public_kernel", "BiddingEnv.py").BiddingEnv
    pv, env = Generator(episode=market, pv_num=500000), audit._cached_env(Env, market)
    generated = {"counts": np.array([len(v) for v in pv.pv_values]),
        "pmeans": np.array([[v[:, f].mean() for v in pv.pv_values] for f in FOCALS]),
        "smeans": np.array([[v[:, f].mean() for v in pv.pValueSigmas] for f in FOCALS]),
        "budgets": audit.BUDGET[list(FOCALS)].astype(float), "cpas": audit.CPA[list(FOCALS)].astype(float)}
    resources.generation.update(status="generated", actual_pv_count=int(generated["counts"].sum()))
    audit.require(all(np.array_equal(generated[k], tape.arrays[k]) for k in generated), "Public generator/tape statistics mismatch")
    public = SimpleNamespace(arrays=generated)
    rng = np.random.default_rng(market + 730001)
    alpha, offset = rng.uniform(15, 145, 48), rng.uniform(0, 2 * np.pi, 48)
    tracks = {(p, s, f): audit.new_track() for p in arms for s in SCENES for f in FOCALS}
    native = {(s, f): audit.new_track() for s in SCENES for f in FOCALS}
    chooser = audit.Feedback(ACTIONS, .3, 8.)
    for tick, (values, sigmas) in enumerate(zip(pv.pv_values, pv.pValueSigmas, strict=True)):
        resources.check()
        tick_start = resources.snapshot()
        env.prepare(values, sigmas)
        bids = values * (alpha * (1 + .15 * np.sin(2 * np.pi * tick / 48 + offset)))
        memo = {}
        for scene in SCENES:
            bias = audit.scenario_bias(scene, tick)  # evaluator adapter only
            for fi, focal in enumerate(FOCALS):
                budget, cpa = float(generated["budgets"][fi]), float(generated["cpas"][fi])
                for arm in arms:
                    resources.check()
                    track = tracks[arm, scene, focal]
                    state = audit.observation(public, fi, tick, track["cost"], track["conv"], track["previous"], bias)
                    track["states"].append(state)
                    decision = decide(arm, track, state, budget, cpa, models, reference["rho0"], config, selection)
                    effective = decision["action"] if arm == "known_inverse" else decision["action"] * bias
                    action = effective / bias if arm == "known_inverse" else decision["action"]
                    audit.require(arm == "known_inverse" or action in ACTIONS, "Normal policy returned off-grid action")
                    key = (fi, float(effective))
                    cached = key in memo
                    if not cached:
                        original = bids[:, focal].copy()
                        try:
                            bids[:, focal] = values[:, focal] * audit.CPA[focal] * effective
                            resources.kernel_attempted += 1
                            result = env.simulate_ad_bidding(values, sigmas, bids)
                            resources.kernel_completed += 1
                            memo[key] = audit.budget_prefix(result[2][focal] * result[3][focal], result[4][focal])
                            del result
                        finally:
                            bids[:, focal] = original
                    log = audit.accepted_log(memo[key], values[:, focal] * bias, budget - track["cost"])
                    settlement = (log["cost"], log["conversions"], log["accepted_exposures"], log["budget_blocked"])
                    expected = tape.settle(fi, tick, effective, budget - track["cost"])
                    replay = replay_check(settlement, expected)
                    audit.require(replay["integers_equal"] and replay["cost_error"] <= 1e-8, "Actual action/tape prefix replay mismatch")
                    native_replay, nominal_replay = None, None
                    if arm == BASE:
                        old = native[scene, focal]
                        old_state = audit.observation(tape, fi, tick, old["cost"], old["conv"], old["previous"], bias)
                        old["states"].append(old_state)
                        old_action, _ = chooser(old["states"], old["actions"], old["rewards"], budget, cpa)
                        old_settlement = tape.settle(fi, tick, old_action * bias, budget - old["cost"])
                        native_replay = replay_check(settlement, old_settlement)
                        native_replay.update(state_equal=bool(np.array_equal(state, old_state)), action_equal=action == old_action)
                        audit.advance(old, old_settlement, old_action, cpa)
                        audit.require(native_replay["state_equal"] and native_replay["action_equal"]
                                      and native_replay["integers_equal"] and native_replay["cost_error"] <= 1e-8,
                                      "Independent native FB replay mismatch")
                    if arm == "known_inverse":
                        nominal = tracks[BASE, "calibrated", focal]
                        nominal_replay = {"effective_action_equal": effective == nominal["actions"][tick],
                            "cost_after_error": abs(track["cost"] + settlement[0] - nominal["cost"]),
                            "conversions_after_equal": track["conv"] + settlement[1] == nominal["conv"]}
                        audit.require(nominal_replay["effective_action_equal"] and nominal_replay["cost_after_error"] <= 1e-8
                                      and nominal_replay["conversions_after_equal"], "Inverse failed to restore nominal FB")
                    log.update(decision)
                    log.update(phase=phase, market=market, policy=arm, scene=scene, focal=focal, tick=tick,
                        action=float(action), effective_action=float(effective), observation=state.tolist(),
                        action_differs_same_state_FB=action != decision["same_state_FB_action"],
                        budget=budget, cpa=cpa, past_C=track["C"], past_E=track["E"],
                        remaining_budget_before=budget - track["cost"], spent_fraction_before=track["cost"] / budget,
                        memoized_kernel=cached, tape_replay=replay, native_FB_replay=native_replay,
                        nominal_FB_replay=nominal_replay, privileged_bias=bias if arm == "known_inverse" else None,
                        injected_direction_evaluator_only="over" if bias > 1 else "under" if bias < 1 else "nominal",
                        outcome_available_at=f"end_of_tick_{tick}")
                    audit.advance(track, settlement, action, cpa)
                    track["C"] += log["conversions"]
                    track["E"] += log["predicted_sum"]
                    log.update(cumulative_cost=track["cost"], cumulative_conversions=track["conv"],
                               remaining_budget_after=budget - track["cost"], cumulative_E=track["E"])
                    if native_replay is not None:
                        native_replay["cumulative_cost_error"] = abs(track["cost"] - native[scene, focal]["cost"])
                        audit.require(native_replay["cumulative_cost_error"] <= 1e-8, "Native cumulative cost mismatch")
                    audit.require(log["public_probability_valid"] and 0 <= track["cost"] <= budget + 1e-6
                                  and track["C"] == track["conv"] and track["conv"] >= 0,
                                  "Public probability/accounting invariant failed")
                    rows.append({k: log[k] for k in (
                        "phase", "market", "policy", "scene", "focal", "tick", "diagnosis",
                        "injected_direction_evaluator_only", "branch", "action_differs_same_state_FB",
                        "spent_fraction_before")})
                    stream.write(json.dumps(log, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
        end = resources.snapshot()
        resources.ticks.append({"phase": phase, "market": market, "tick": tick, "completed_cells": len(rows),
            "wall_seconds": end["wall_seconds"] - tick_start["wall_seconds"],
            "cpu_seconds": end["cpu_seconds"] - tick_start["cpu_seconds"]})
        resources.save()
    terminals = []
    for arm in arms:
        for scene in SCENES:
            for fi, focal in enumerate(FOCALS):
                track = tracks[arm, scene, focal]
                budget, cpa = float(generated["budgets"][fi]), float(generated["cpas"][fi])
                terminal = {"phase": phase, "market": market, "policy": arm, "scene": scene, "focal": focal,
                    "cost": track["cost"], "conversions": track["conv"], "score": audit.score(track["cost"], track["conv"], cpa),
                    "violation": int(track["cost"] > cpa * track["conv"] + 1e-8), "zero_conversions": int(track["conv"] == 0),
                    "budget": budget, "target_cpa": cpa, "remaining_budget": budget - track["cost"],
                    "realized_cpa": track["cost"] / track["conv"] if track["conv"] else None,
                    "C": track["C"], "E": track["E"], "nominal_FB_terminal_match": None}
                if arm == "known_inverse":
                    nominal = tracks[BASE, "calibrated", focal]
                    terminal["nominal_FB_terminal_match"] = (abs(track["cost"] - nominal["cost"]) <= 1e-8
                        and track["conv"] == nominal["conv"]
                        and abs(terminal["score"] - audit.score(nominal["cost"], nominal["conv"], cpa)) <= 1e-8)
                    audit.require(terminal["nominal_FB_terminal_match"], "Inverse nominal terminal mismatch")
                terminals.append(terminal)
    end = resources.snapshot()
    resources.markets.append({"phase": phase, "market": market, "cells": len(arms) * 1440,
        "wall_seconds": end["wall_seconds"] - begin["wall_seconds"], "cpu_seconds": end["cpu_seconds"] - begin["cpu_seconds"]})
    resources.save()
    print(json.dumps({"phase": phase, "market": market, "completed_cells": len(rows), "resources": end}), flush=True)
    return terminals


def select_scales(terminals, config):
    training = [r for r in terminals if r["phase"] == "train"]
    audit.require(len(training) == len(TRAIN) * len(SCALE_ARMS) * len(SCENES) * len(FOCALS), "Training selection coverage incomplete")
    choices = {}
    for name, scenes in (("global", SCENES), ("over", config["gate_scale_selection_scenarios"])):
        scores = [float(np.mean([r["score"] for r in training if r["policy"] == arm and r["scene"] in scenes]))
                  for arm in SCALE_ARMS]
        chosen = max(range(len(SCALES)), key=lambda i: (scores[i], -abs(SCALES[i] - 1), -SCALES[i]))
        choices[name] = {"policy": SCALE_ARMS[chosen], "scale": SCALES[chosen], "training_mean_score": scores[chosen],
                         "scene_selection": list(scenes), "all_scale_scores": dict(zip(SCALE_ARMS, scores, strict=True))}
    return {**choices, "selection_split": "four training markets only", "evaluation_read_before_selection": False}


def coverage(rows):
    ticks = defaultdict(list)
    for row in rows:
        key = (row["phase"], row["market"], row["policy"], row["scene"], row["focal"])
        ticks[key].append(row["tick"])
    prefixes, missing = {}, []
    for phase, (markets, arms) in PHASES.items():
        for market in markets:
            for arm in arms:
                for scene in SCENES:
                    for focal in FOCALS:
                        key = (phase, market, arm, scene, focal)
                        present = ticks.pop(key, [])
                        audit.require(present == list(range(len(present))), "Duplicate/noncausal trajectory cells")
                        prefixes["/".join(map(str, key))] = len(present)
                        if len(present) != 48:
                            missing.append({"trajectory": list(key), "missing_tick_range": [len(present), 48]})
    audit.require(not ticks, "Unknown study cell")
    return {"requested": REQUESTED, "completed": len(rows), "complete": len(rows) == REQUESTED and not missing,
            "trajectory_prefix_lengths": prefixes, "incomplete_trajectories": missing}


def comparison(selected, reference):
    return {"selected": float(selected), "reference": float(reference), "difference": float(selected - reference),
            "ratio": float(selected / reference) if reference != 0 else None}


def summarize(terminals, selection):
    rows = [r for r in terminals if r["phase"] == "evaluation"]
    groups = {p: [p] for p in EVAL_ARMS}
    groups.update(router_mean=list(ROUTER_ARMS), response_mean=list(RESPONSE_ARMS))
    fields = ("score", "conversions", "violation", "zero_conversions", "cost")
    indexed = defaultdict(list)
    for row in rows:
        indexed[row["policy"], row["scene"], row["market"]].append(row)
    aggregates = {}
    for name, policies in groups.items():
        for scene in SCENES:
            for market in EVALUATION:
                selected = [r for p in policies for r in indexed[p, scene, market]]
                audit.require(len(selected) == len(policies) * len(FOCALS), "Summary focal/seed coverage incomplete")
                values = {f: float(np.mean([r[f] for r in selected])) for f in fields}
                values.update(violation_count=sum(r["violation"] for r in selected),
                              zero_count=sum(r["zero_conversions"] for r in selected), sample_count=len(selected))
                aggregates[name, scene, market] = values

    def pair(name, reference, market=None):
        markets = [market] if market is not None else EVALUATION
        scenes = {}
        for scene in SCENES:
            item = {f: comparison(np.mean([aggregates[name, scene, m][f] for m in markets]),
                                  np.mean([aggregates[reference, scene, m][f] for m in markets])) for f in fields}
            count = sum(aggregates[name, scene, m]["violation_count"] for m in markets)
            ref_count = sum(aggregates[reference, scene, m]["violation_count"] for m in markets)
            denominator = len(markets) * len(FOCALS) * len(groups[name])
            ref_denominator = len(markets) * len(FOCALS) * len(groups[reference])
            item["violation"] = comparison(count / denominator, ref_count / ref_denominator)
            item["violation"].update(selected_count=count, reference_count=ref_count,
                selected_sample_count=denominator, reference_sample_count=ref_denominator,
                nonworsening=count * ref_denominator <= ref_count * denominator)
            scenes[scene] = item
        ratios = [scenes[s]["score"]["ratio"] for s in SCENES]
        return {"scenes": scenes, "macro_score_ratio": None if any(v is None for v in ratios) else float(np.mean(ratios))}

    references = {"FB": BASE, "global_scale": selection["global"]["policy"], "gate_scaled": "gate_scaled"}
    comparisons = {name: {label: pair(name, ref) for label, ref in references.items()} for name in groups}
    paired = {name: {label: [{"market": m, **pair(name, ref, m)} for m in EVALUATION]
                    for label, ref in references.items()} for name in ("router_mean", "response_mean", "gate_scaled") + ROUTER_ARMS}
    router = comparisons["router_mean"]
    positive_markets = sum(r["macro_score_ratio"] is not None and r["macro_score_ratio"] > 1
                           for r in paired["router_mean"]["global_scale"])
    flags = {"complete": True,
        "macro_vs_FB_global_scale_gate_scaled": all(router[label]["macro_score_ratio"] is not None
            and router[label]["macro_score_ratio"] >= 1.03 for label in references),
        "every_scene_score_vs_FB": all(router["FB"]["scenes"][s]["score"]["ratio"] is not None
            and router["FB"]["scenes"][s]["score"]["ratio"] >= .98 for s in SCENES),
        "over50_score_vs_FB": router["FB"]["scenes"]["over_50"]["score"]["ratio"] is not None
            and router["FB"]["scenes"]["over_50"]["score"]["ratio"] >= 1.03,
        "every_scene_conversion_vs_FB": all(router["FB"]["scenes"][s]["conversions"]["ratio"] is not None
            and router["FB"]["scenes"][s]["conversions"]["ratio"] >= .97 for s in SCENES),
        "every_scene_violation_vs_FB": all(router["FB"]["scenes"][s]["violation"]["nonworsening"] for s in SCENES),
        "every_seed_positive_vs_global_scale_and_gate_scaled": all(
            comparisons[p][label]["macro_score_ratio"] is not None and comparisons[p][label]["macro_score_ratio"] > 1
            for p in ROUTER_ARMS for label in ("global_scale", "gate_scaled")),
        "positive_markets_vs_global_scale": positive_markets >= 4}
    passed = all(flags.values())
    return {"comparisons": comparisons, "paired_market_comparisons": paired, "selection": selection,
        "gate": {"flags": flags, "conjunction": passed, "positive_markets_vs_global_scale": positive_markets},
        "primary_router_pass": passed,
        "next_action": "separate new-market preregistration only" if passed else "stop this router candidate; no automatic rescue",
        "independent_confirmation": False, "generative_contribution": False,
        "known_inverse_eligible": False, "other_arms_eligible": False}


def diagnostics(rows):
    grouped, trajectories = defaultdict(list), defaultdict(list)
    for row in rows:
        if row["phase"] == "evaluation" and row["policy"] in ROUTER_ARMS + ("gate_scaled",):
            grouped[row["policy"], row["scene"]].append(row)
            trajectories[row["policy"], row["scene"], row["market"], row["focal"]].append(row)
    result = {}
    for (policy, scene), selected in grouped.items():
        confusion = Counter((r["injected_direction_evaluator_only"], r["diagnosis"]["label"]) for r in selected)
        result[f"{policy}/{scene}"] = {
            "ticks": len(selected), "occupancy": dict(Counter(r["diagnosis"]["label"] for r in selected)),
            "confusion_counts_evaluator_only": {f"truth_{t}/signal_{p}": n for (t, p), n in sorted(confusion.items())},
            "response_branch_count": sum(r["branch"] == "response" for r in selected),
            "action_disagreement_same_state_FB": sum(r["action_differs_same_state_FB"] for r in selected),
            "response_action_disagreement_same_state_FB": sum(r["branch"] == "response" and r["action_differs_same_state_FB"] for r in selected)}
    first = []
    for (policy, scene, market, focal), selected in trajectories.items():
        over = [r for r in selected if r["diagnosis"]["label"] == "over"]
        post = [r for r in over if r["tick"] >= 24]
        first.append({"policy": policy, "scene": scene, "market": market, "focal": focal,
            "first_over_tick": over[0]["tick"] if over else None,
            "spent_fraction_at_first_over": over[0]["spent_fraction_before"] if over else None,
            "midshift_pre24_falseover_count": sum(r["tick"] < 24 for r in over) if scene == "mid_over_50" else None,
            "midshift_first_post24_over_delay": post[0]["tick"] - 24 if scene == "mid_over_50" and post else None,
            "midshift_no_post24_over": not post if scene == "mid_over_50" else None})
    return {"groups": result, "first_over_by_trajectory": first,
        "interpretation": "Signals are relative to a selected-winner nominal reference; injected-truth confusion is descriptive, not validated coverage or identification. Delays include persistent pre-shift signals; see separate falseover counts."}


def run(output, expected):
    output = Path(output).resolve()
    audit.require(not output.exists(), "Output exists; preserve it and use a fresh registered run identity")
    frozen, checks = preflight()
    audit.require(audit.identity(frozen) == expected, "Reviewed source/protocol/input freeze changed")
    audit.require(output.parent.is_dir() and shutil.disk_usage(output.parent).free >= MIN_FREE, "Output parent missing or less than10GiB free disk")
    config = frozen["protocol"]["configuration"]
    with Path(__file__).open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        audit.write(output / "protocol.json", frozen["protocol"])
        audit.write(output / "manifest.json", {"freeze_identity": expected, "frozen": frozen,
            "frozen_before_pv_generation": True, "created_utc": datetime.now(UTC).isoformat()})
        audit.write(output / "preflight.json", checks)
        audit.write(output / "summary.json", {"status": "running", "scientific_outcome_evaluable": False, "primary_router_pass": None})
        resources, handlers = Resources(output), {}
        rows, terminals, selection = [], [], None
        status, reason, code = "complete", None, 0

        def stop(signum, _frame):
            raise ResourceStop(f"Interrupt/deadline signal{signum}")

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM, signal.SIGPROF):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, LIMIT - RESERVE - (time.monotonic() - PROCESS_START)))
        signal.setitimer(signal.ITIMER_PROF, max(.001, LIMIT - RESERVE - time.process_time()))
        try:
            audit.torch.set_num_threads(1)
            models = {seed: Learned(MAIN / "models" / f"response-{seed}" / "best.pt", ACTIONS, 24, "cpu") for seed in SEEDS}
            audit.require(all(p.kind == "response" and not p.corrected and not p.hard_cpa for p in models.values()), "Response model policy changed")
            with (output / "tick-logs.jsonl").open("w") as stream:
                for phase, (markets, _arms) in PHASES.items():
                    for market in markets:
                        resources.check(next_phase=phase)
                        terminals.extend(market_run(phase, market, stream, rows, resources, models,
                                                    frozen["reference"], config, selection))
                        audit.write(output / "trajectories.json", terminals)
                        gc.collect()
                    if phase == "train":
                        selection = select_scales(terminals, config)
                        audit.write(output / "selection.json", {**selection, "selected_before_evaluation_pv_generation": True})
                        print(json.dumps({"selection_frozen": selection}), flush=True)
            resources.check()
        except ResourceStop as error:
            status, reason, code = "incomplete_resource", str(error), 2
        except Exception as error:
            status, reason, code = "invalid_run", f"{type(error).__name__}: {error}", 3
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.setitimer(signal.ITIMER_PROF, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            after = {"sources": audit.after_hashes(frozen["sources"]), "inputs": audit.after_hashes(frozen["inputs"])}
            unchanged = after["sources"] == frozen["sources"] and after["inputs"] == frozen["inputs"]
            unchanged = unchanged and audit.identity(audit.read(output / "protocol.json")) == audit.identity(frozen["protocol"])
            if not unchanged:
                status, reason, code = "invalid_identity", "Source/protocol/input changed after freeze", 3
            covered = coverage(rows)
            if status == "complete" and (not covered["complete"] or len(terminals) != TERMINALS):
                status, reason, code = "incomplete_resource", "Fixed comparison coverage incomplete", 2
            if time.monotonic() - PROCESS_START > LIMIT or time.process_time() > LIMIT:
                status, reason, code = "incomplete_resource", "Total process wall/CPU budget exceeded", 2
            summary = summarize(terminals, selection) if status == "complete" else {"primary_router_pass": None}
            summary.update(status=status, reason=reason, freeze_identity=expected,
                scientific_outcome_evaluable=status == "complete", coverage_complete=covered["complete"],
                completed_cells=len(rows), completed_terminals=len(terminals),
                full_actual_action_tape_replay_pass=status == "complete", full_native_FB_replay_pass=status == "complete",
                all_inverse_nominal_terminals_match=status == "complete", automatic_followup_authorized=False,
                claim_scope=frozen["protocol"]["claim"])
            audit.write(output / "coverage.json", covered)
            audit.write(output / "trajectories.json", terminals)
            audit.write(output / "diagnostics.json", diagnostics(rows))
            audit.write(output / "summary.json", summary)
            resources.save()
            audit.write(output / "execution.json", {"status": status, "hashes_after": after,
                "identities_unchanged": unchanged, "resources": resources.snapshot(), "power_changes": False})
            files = {p.name: audit.sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "seal.json"}
            audit.write(output / "seal.json", {"files": files, "content_identity": audit.identity(files)})
            audit.require(all(audit.sha(output / name) == digest for name, digest in files.items()), "Seal reread failed")
            print(json.dumps({"status": status, "reason": reason, "completed_cells": len(rows),
                "primary_router_pass": summary["primary_router_pass"], "output": str(output)}, sort_keys=True), flush=True)
        return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--preflight", action="store_true", help="read-only source/input freeze; no PV generation")
    modes.add_argument("--run", action="store_true", help="fixed training selection then full 15-arm evaluation")
    parser.add_argument("--freeze", help="reviewed preflight identity, required for --run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.preflight:
        frozen, checks = preflight()
        print(json.dumps({"frozen": frozen, "checks": checks}, indent=2, sort_keys=True))
        return 0
    if not args.freeze:
        parser.error("--run requires --freeze")
    output = args.output or ROOT / configuration()["output"]
    return run(output, args.freeze)


if __name__ == "__main__":
    raise SystemExit(main())
