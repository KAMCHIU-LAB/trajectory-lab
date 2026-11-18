#!/usr/bin/env python3
"""Frozen four-training-market FB/R/N control experiment, without model fitting.

Only R is eligible for the registered gate. Preflight/self-check generate no PVs
and create no study output. The authoritative run requires a reviewed freeze.
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
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

PROCESS_START = time.monotonic()
import audit_bidding_winner_belief as audit  # noqa: E402 - capture startup before heavy imports
import numpy as np  # noqa: E402 - audit sets process-local thread limits first

ROOT = Path(__file__).resolve().parents[1]
MARKETS = (1002, 1006, 1010, 1014)
POLICIES = ("FB", "R", "N")
SCENES, FOCALS, ACTIONS = audit.SCENES, audit.FOCALS, audit.ACTIONS
PRIMARY = ("over_25", "over_50", "under_25")
REQUESTED = 4 * 3 * 5 * 6 * 48
LIMIT, RESERVE = 1800, 60
MIN_FREE, MAX_OUTPUT, MAX_MEMORY = 10 * 2**30, 2**30, 10 * 2**30
MAIN = ROOT / "artifacts/bidding/main-v2"
OUTPUT = ROOT / "artifacts/bidding/numerical-audits/calibration-control-v1"
PLAN = ROOT / ".omx/plans/astra-winner-belief-next-decision.md"
HANDOFF = ROOT / ".omx/state/bidding-v2/calibration-control-handoff.json"
B_OUTPUT = ROOT / "artifacts/bidding/numerical-audits/winner-belief-audit-v1"
B_SOURCE = "481505aa9526774adf615671b0cd6d4ee5125888b2a8e1541e953c77715533df"


def control(state, budget, cpa, past_C, past_E, policy):
    """Policy accepts only its own public observation and completed C/E."""
    elapsed = float(state[0])
    spend = float(state[3]) * budget
    conversions = float(state[4]) * budget / cpa
    pacing = ((elapsed + .05) * budget / (spend + .05 * budget)) ** .3
    quality = ((conversions + 8) * cpa / (spend + 8 * cpa))
    k = (1 + past_C) / (1 + past_E)
    base = pacing * quality ** .3
    u = base if policy == "FB" else base * k ** (.7 if policy == "R" else 1.)
    clipped = float(np.clip(u, .05, 3.))
    index = int(np.abs(ACTIONS - clipped).argmin())
    return {"P": pacing, "Q": quality, "Q_over_k": quality / k, "k_before": k,
            "u_FB": base, "u_continuous": u, "u_clipped": clipped,
            "clip_applied": u != clipped, "projected_index": index,
            "action": float(ACTIONS[index]), "projection_changed_value": clipped != float(ACTIONS[index]),
            "elapsed_from_state": elapsed, "spend_from_state": spend,
            "conversions_from_state": conversions}


def protocol():
    return {
        "version": "calibration-control-v1", "markets": list(MARKETS), "policies": list(POLICIES),
        "primary_candidate": "R", "diagnostic_only": "N", "requested_cells": REQUESTED,
        "scenes": list(SCENES), "focals": list(FOCALS), "actions": ACTIONS.tolist(),
        "order": "markets in fixed order; each tick0..47,scene,focal,FB/R/N; release one market before generating next",
        "control": "from native float32 state reconstruct elapsed=float(s0),spend=float(s3)*B,C=float(s4)*B/CPA; P=(((elapsed+.05)*B)/(spend+.05*B))**.3; Q=((C+8)*CPA)/(spend+8*CPA); u_FB=P*Q**.3; u_R=u_FB*k**.7; u_N=u_FB*k; multiply continuous then native clip[.05,3] and nearest original11 grid(first tie)",
        "belief": "Gamma1,1 independently for each market/policy/scene/focal; k=(1+past accepted integer C)/(1+past accepted public-p sum E); commit current tick after settlement; no pooling/forgetting/fits",
        "policy_allowlist": ["own current float32 public state", "public budget/CPA/action grid", "own completed accepted C/E", "fixed policy identifier"],
        "prediction_origin": "environment adapter only: generator values for accepted paid/exposed/budget-accepted PVs times public scenario perturbation; never cached latent values; no new clipping",
        "kernel": "same native generator/_cached_env CRN/opponents as immutable B; each market episode=market,pv_num500000; prepare once per tick; alter one focal bid column at a time,restore",
        "memoization": "within one tick keyed(focal,effective action); cache untruncated pc/cv/paid PV indices only; each trajectory selects own budget prefix",
        "accepted_prefix": "immutable B budget_prefix/accepted_log helpers; original PV order/right search; accepted public p must be finite within[0,1], otherwise invalidate control study",
        "replay": "FB independently compared to old tape each tick: float32 state/action/integer conv/exposure/blocked exact; settlement/cumulative cost abs<=1e-8 with nonexact reported; terminal score abs<=1e-8,violation exact",
        "accounting": "every arm own exact cost and integer conv; hard budget<=B+1e-6; terminal native score,violation cost>CPA*C+1e-8; zero-conversion and absolute risk reported",
        "aggregation": "terminal market/scene mean of six focals; scene ratio=ratio of four-market means; binary violation/zero-conversion means use integer count/6 per market,count/24 across four markets; violation gate compares integer counts; headline equal mean of over25,over50,under25 scene score ratios; keep all per-market/focal outcomes; zero denominator ->None and difference,no epsilon",
        "gate": {"R_only": True, "under25_score_min": 1.03, "under25_positive_markets_min": 3,
                 "primary_headline_min": 1.03, "over25_over50_calibrated_midshift_score_min": .98,
                 "every_scene_conversion_min": .97, "every_scene_violation_delta_max": 0,
                 "requires_non_FB_action": True, "requires_all17280_cells": True},
        "diagnostics": "own k/Q/Q_over_k/clip/projection/actions and realized accepted E/C; conditional Poisson deviance uses B no-epsilon branches, descriptive only,no selection",
        "resources": {"wall_cpu_limit_seconds_from_process_start": LIMIT, "seal_reserve_seconds": RESERVE,
                      "threads": 1, "peak_memory_bytes_max": MAX_MEMORY, "output_bytes_max": MAX_OUTPUT,
                      "free_disk_bytes_min": MIN_FREE, "stop": "resource/integrity only; no effect-dependent stopping; real/profiling timers plus per-cell checks; first complete1002 is part of fixed job"},
        "claim": "exploratory four-training-market screening; no independent confirmation,generative contribution or guaranteed safety",
        "followup": "R pass permits only separate future preregistration; any evaluable R gate failure stops branch; N never substitutes",
    }


def preflight():
    """Identity-only inspection; never instantiate a generator or load tape arrays."""
    handoff = audit.read(HANDOFF)
    audit.require(handoff["authorizes_implementation"] and handoff["authorizes_run_after_source_protocol_freeze"]
                  and handoff["markets"] == list(MARKETS) and handoff["policies"] == list(POLICIES)
                  and handoff["plan_sha256"] == audit.sha(PLAN), "Approved calibration handoff mismatch")
    b_manifest, b_seal = audit.read(B_OUTPUT / "manifest.json"), audit.read(B_OUTPUT / "seal.json")
    audit.require(audit.sha(ROOT / "scripts/audit_bidding_winner_belief.py") == B_SOURCE, "Immutable B source changed")
    audit.require(audit.identity(b_seal["files"]) == b_seal["content_identity"]
                  and all(audit.sha(B_OUTPUT / name) == digest for name, digest in b_seal["files"].items()), "B seal changed")
    b_summary = audit.read(B_OUTPUT / "summary.json")
    audit.require(b_summary["status"] == "complete" and b_summary["full_replay_pass"]
                  and b_summary["legal_observability"], "B prerequisites did not pass")
    old_sources = b_manifest["frozen"]["sources"]
    sources = {name: audit.sha(ROOT / name) for name in old_sources}
    audit.require(sources == old_sources, "Frozen B/native/A source changed")
    audit.require(all(audit.sha(ROOT / name) == digest for name, digest in b_manifest["frozen"]["inputs"].items()),
                  "Frozen B input changed")
    verification = audit.read(ROOT / ".omx/state/bidding-v2/winner-independent-verification.json")
    audit.require(verification["integrity_verdict"] == "PASS" and verification["failed_checks"] == []
                  and verification["source_sha256"] == B_SOURCE, "Independent B verification did not pass")
    source_name = str(Path(__file__).resolve().relative_to(ROOT))
    sources[source_name] = audit.sha(Path(__file__))
    main = audit.read(MAIN / "manifest.json")
    study = main["config"]["study"]
    audit.require(study["focal_agents"] == list(FOCALS) and study["actions"] == ACTIONS.tolist()
                  and study["pv_count"] == 500000 and all(m in study["train_markets"] for m in MARKETS),
                  "Original study contract changed")
    cache_parent = ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1/manifest.json"
    old_inputs = audit.read(cache_parent)["core"]["inputs"]
    paths = [PLAN, HANDOFF, MAIN / "manifest.json", ROOT / "configs/bidding-main.toml", cache_parent,
             B_OUTPUT / "manifest.json", B_OUTPUT / "seal.json", B_OUTPUT / "summary.json",
             ROOT / ".omx/state/bidding-v2/winner-independent-verification.json"]
    for market in MARKETS:
        path = MAIN / "markets" / f"{market}.npz"
        meta = audit.read(path.with_suffix(".json"))
        audit.require(meta["market_id"] == market and meta["focal_agents"] == list(FOCALS)
                      and meta["source"] == main["auction"], "Market metadata mismatch")
        expected = old_inputs[str(path)]
        audit.require(audit.sha(path) == expected["npz_sha256"]
                      and audit.sha(path.with_suffix(".json")) == expected["metadata_sha256"], "Old training tape changed")
        paths += [path, path.with_suffix(".json")]
    frozen = {"sources": sources, "inputs": {str(p.relative_to(ROOT)): audit.sha(p) for p in paths},
              "protocol": protocol(), "study": study,
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": audit.scipy.__version__, "torch": audit.torch.__version__,
                          "platform": platform.platform()}, "generator_external_resources": []}
    return frozen, {"freeze_identity": audit.identity(frozen), "protocol_sha256": audit.identity(protocol()),
                    "source_sha256": sources[source_name], "read_only": True, "pv_generated": False,
                    "requested_cells": REQUESTED, "markets": list(MARKETS)}


class ResourceStop(Exception):
    pass


class Resources:
    def __init__(self, output):
        self.output, self.ticks, self.markets = output, [], []
        self.kernel_attempted, self.kernel_completed = 0, 0
        self.generation = None
        self.last_scan, self.bytes = -math.inf, 0

    def snapshot(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"wall_seconds": time.monotonic() - PROCESS_START, "cpu_seconds": time.process_time(),
                "peak_memory_bytes": int(peak if audit.sys.platform == "darwin" else peak * 1024),
                "free_disk_bytes": shutil.disk_usage(self.output).free, "artifact_bytes": self.bytes}

    def check(self, next_market=False):
        state = self.snapshot()
        if time.monotonic() - self.last_scan >= 15:
            self.bytes = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
            self.last_scan = time.monotonic()
        if state["wall_seconds"] >= LIMIT - RESERVE or state["cpu_seconds"] >= LIMIT - RESERVE:
            raise ResourceStop("Time boundary reached; reserve60s for seal")
        if state["peak_memory_bytes"] > MAX_MEMORY or state["free_disk_bytes"] < MIN_FREE or self.bytes > MAX_OUTPUT:
            raise ResourceStop("Memory/disk/output resource boundary reached")
        if next_market and self.markets:
            for key in ("wall_seconds", "cpu_seconds"):
                expected = max(self.markets[-1][key], float(np.mean([r[key] for r in self.markets])))
                if state[key] + expected >= LIMIT - RESERVE:
                    raise ResourceStop(f"Next complete market exceeds remaining {key} budget")

    def save(self):
        audit.write(self.output / "resources.json", {"current": self.snapshot(), "generation": self.generation,
                    "ticks": self.ticks, "markets": self.markets,
                    "kernel_calls_attempted": self.kernel_attempted, "kernel_calls_completed": self.kernel_completed,
                    "requested_cells": REQUESTED, "limits": protocol()["resources"]})


def coverage(rows):
    cells = {(r["market"], r["policy"], r["scene"], r["focal"], r["tick"]) for r in rows}
    audit.require(len(cells) == len(rows), "Duplicate study cell")
    prefixes, missing = {}, []
    for market in MARKETS:
        for policy in POLICIES:
            for scene in SCENES:
                for focal in FOCALS:
                    ticks = [t for t in range(48) if (market, policy, scene, focal, t) in cells]
                    audit.require(ticks == list(range(len(ticks))), "Noncausal trajectory gap")
                    prefixes[f"{market}/{policy}/{scene}/{focal}"] = len(ticks)
                    missing += [[market, policy, scene, focal, t] for t in range(len(ticks), 48)]
    return {"requested": REQUESTED, "completed": len(rows), "complete": len(rows) == REQUESTED,
            "trajectory_prefix_lengths": prefixes, "missing_cells": missing,
            "completed_markets": [m for m in MARKETS if all(v == 48 for k, v in prefixes.items() if k.startswith(str(m) + "/"))]}


def market_run(market, stream, rows, resources):
    begin = resources.snapshot()
    resources.generation = {"market": market, "status": "running", "start": begin}
    resources.save()
    tape = audit.Tape(MAIN / "markets" / f"{market}.npz")
    Generator = audit._load("calibration_public_generator", "NeurIPSPvGen.py").NeurIPSPvGen
    Env = audit._load("calibration_public_kernel", "BiddingEnv.py").BiddingEnv
    pv = Generator(episode=market, pv_num=500000)
    env = audit._cached_env(Env, market)
    generated = {"counts": np.array([len(v) for v in pv.pv_values]),
        "pmeans": np.array([[v[:, f].mean() for v in pv.pv_values] for f in FOCALS]),
        "smeans": np.array([[v[:, f].mean() for v in pv.pValueSigmas] for f in FOCALS]),
        "budgets": audit.BUDGET[list(FOCALS)].astype(float), "cpas": audit.CPA[list(FOCALS)].astype(float)}
    resources.generation.update(status="generated", end=resources.snapshot(), actual_pv_count=int(generated["counts"].sum()))
    resources.save()
    audit.require(all(np.array_equal(generated[k], tape.arrays[k]) for k in generated), "Public generator/tape statistics mismatch")
    public_tape = SimpleNamespace(arrays=generated)
    rng = np.random.default_rng(market + 730001)
    alpha, phase = rng.uniform(15, 145, 48), rng.uniform(0, 2 * np.pi, 48)
    tracks = {(p, s, f): audit.new_track() for p in POLICIES for s in SCENES for f in FOCALS}
    reference = {(s, f): audit.new_track() for s in SCENES for f in FOCALS}
    chooser = audit.Feedback(ACTIONS, .3, 8.)
    for tick, (values, sigmas) in enumerate(zip(pv.pv_values, pv.pValueSigmas, strict=True)):
        resources.check()
        tick_start = resources.snapshot()
        env.prepare(values, sigmas)
        bids = values * (alpha * (1 + .15 * np.sin(2 * np.pi * tick / 48 + phase)))
        memo = {}
        for scene in SCENES:
            for fi, focal in enumerate(FOCALS):
                fb_action = None
                budget, cpa = float(generated["budgets"][fi]), float(generated["cpas"][fi])
                bias = audit.scenario_bias(scene, tick)  # adapter only, never passed to control()
                for policy in POLICIES:
                    resources.check()
                    track = tracks[policy, scene, focal]
                    state = audit.observation(public_tape, fi, tick, track["cost"], track["conv"], track["previous"], bias)
                    track["states"].append(state)
                    decision = control(state, budget, cpa, track["C"], track["E"], policy)
                    action = decision["action"]
                    context_fb = control(state, budget, cpa, track["C"], track["E"], "FB")["action"]
                    if policy == "FB":
                        fb_action = action
                    audit.require(tick != 0 or action == fb_action, "Initial k1 arms disagree")
                    key = (fi, float(action * bias))
                    cached = key in memo
                    if not cached:
                        original = bids[:, focal].copy()
                        try:
                            bids[:, focal] = values[:, focal] * audit.CPA[focal] * (action * bias)
                            resources.kernel_attempted += 1
                            out = env.simulate_ad_bidding(values, sigmas, bids)
                            resources.kernel_completed += 1
                            memo[key] = audit.budget_prefix(out[2][focal] * out[3][focal], out[4][focal])
                            del out
                        finally:
                            bids[:, focal] = original
                    log = audit.accepted_log(memo[key], values[:, focal] * bias, budget - track["cost"])
                    log.update(decision)
                    log.update(market=market, policy=policy, scene=scene, focal=focal, tick=tick,
                        observation=state.tolist(), budget=budget, cpa=cpa, remaining_budget_before=budget - track["cost"],
                        past_C=track["C"], past_E=track["E"], context_feedback_action=context_fb,
                        actual_FB_action=fb_action, action_differs_same_context_FB=action != context_fb,
                        action_differs_actual_FB=action != fb_action, memoized_kernel=cached,
                        prediction_origin="accepted generator public p from environment adapter",
                        belief_origin="own completed accepted C/E; Gamma1,1; no pooling",
                        settlement_timestamp=f"end_of_tick_{tick}", fb_replay=None)
                    settlement = (log["cost"], log["conversions"], log["accepted_exposures"], log["budget_blocked"])
                    expected = None
                    if policy == "FB":
                        old = reference[scene, focal]
                        old_state = audit.observation(tape, fi, tick, old["cost"], old["conv"], old["previous"], bias)
                        old["states"].append(old_state)
                        old_action, _ = chooser(old["states"], old["actions"], old["rewards"], budget, cpa)
                        expected = tape.settle(fi, tick, old_action * bias, budget - old["cost"])
                        log["fb_replay"] = {"state_equal": bool(np.array_equal(state, old_state)),
                            "action_equal": action == old_action, "expected_cost": float(expected[0]),
                            "expected_conversions": int(expected[1]), "expected_exposures": int(expected[2]),
                            "expected_blocked": bool(expected[3]), "exact_cost_equal": settlement[0] == expected[0],
                            "cost_error": abs(settlement[0] - expected[0]), "integer_settlement_equal": settlement[1:] == expected[1:]}
                        audit.advance(old, expected, old_action, cpa)
                    audit.advance(track, settlement, action, cpa)
                    track["C"] += log["conversions"]
                    track["E"] += log["predicted_sum"]
                    log.update(cumulative_cost=track["cost"], cumulative_conversions=track["conv"],
                        remaining_budget_after=budget - track["cost"], k_after=(1 + track["C"]) / (1 + track["E"]),
                        mu=decision["k_before"] * log["predicted_sum"])
                    log["deviance_original"] = audit.deviance(log["conversions"], log["predicted_sum"])
                    log["deviance_belief"] = audit.deviance(log["conversions"], log["mu"])
                    if expected is not None:
                        log["fb_replay"]["cumulative_cost_error"] = abs(track["cost"] - reference[scene, focal]["cost"])
                        log["fb_replay"]["cumulative_conversions_equal"] = track["conv"] == reference[scene, focal]["conv"]
                    rows.append(log)
                    stream.write(json.dumps(log, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    audit.require(log["public_probability_valid"] and 0 <= track["cost"] <= budget + 1e-6
                        and track["conv"] == track["C"] and track["conv"] >= 0, "Public probability/accounting invariant failed")
                    if expected is not None:
                        replay = log["fb_replay"]
                        audit.require(replay["state_equal"] and replay["action_equal"] and replay["integer_settlement_equal"]
                            and replay["cost_error"] <= 1e-8 and replay["cumulative_cost_error"] <= 1e-8
                            and replay["cumulative_conversions_equal"], "Independent native FB replay mismatch")
                    resources.check()
        end = resources.snapshot()
        resources.ticks.append({"market": market, "tick": tick, "completed_cells": len(rows),
            "wall_seconds": end["wall_seconds"] - tick_start["wall_seconds"],
            "cpu_seconds": end["cpu_seconds"] - tick_start["cpu_seconds"]})
        resources.save()
    trajectories = []
    for policy in POLICIES:
        for scene in SCENES:
            for fi, focal in enumerate(FOCALS):
                track = tracks[policy, scene, focal]
                cpa, budget = float(generated["cpas"][fi]), float(generated["budgets"][fi])
                violation = int(track["cost"] > cpa * track["conv"] + 1e-8)
                terminal_score = audit.score(track["cost"], track["conv"], cpa)
                native_match = None
                if policy == "FB":
                    old = reference[scene, focal]
                    native_match = (abs(terminal_score - audit.score(old["cost"], old["conv"], cpa)) <= 1e-8
                                    and violation == int(old["cost"] > cpa * old["conv"] + 1e-8))
                    audit.require(native_match, "Native FB terminal mismatch")
                trajectories.append({"market": market, "policy": policy, "scene": scene, "focal": focal,
                    "cost": track["cost"], "conversions": track["conv"], "score": terminal_score,
                    "violation": violation, "zero_conversions": int(track["conv"] == 0),
                    "realized_cpa": track["cost"] / track["conv"] if track["conv"] else None,
                    "budget": budget, "target_cpa": cpa, "remaining_budget": budget - track["cost"],
                    "C": track["C"], "E": track["E"], "k_final": (1 + track["C"]) / (1 + track["E"]),
                    "native_FB_terminal_match": native_match})
    end = resources.snapshot()
    resources.markets.append({"market": market, "completed_cells": 4320,
        "wall_seconds": end["wall_seconds"] - begin["wall_seconds"], "cpu_seconds": end["cpu_seconds"] - begin["cpu_seconds"],
        "projected_four_market_wall_seconds": (end["wall_seconds"] - begin["wall_seconds"]) * 4})
    resources.save()
    return trajectories


def comparison(selected, reference):
    return {"selected": float(selected), "FB": float(reference), "difference": float(selected - reference),
            "ratio": float(selected / reference) if reference != 0 else None}


def risk_comparison(selected_count, reference_count, sample_count):
    result = comparison(selected_count / sample_count, reference_count / sample_count)
    result.update(selected_count=selected_count, FB_count=reference_count, sample_count=sample_count)
    return result


def gate(metrics, positive_under_markets, switches, complete):
    def minimum(scene, field, bound):
        value = metrics[scene][field]["ratio"]
        return value is not None and value >= bound

    ratios = [metrics[s]["score"]["ratio"] for s in PRIMARY]
    headline = None if any(r is None for r in ratios) else float(np.mean(ratios))
    flags = {"complete": complete, "under25_score": minimum("under_25", "score", 1.03),
        "under25_positive_markets": positive_under_markets >= 3,
        "primary_headline": headline is not None and headline >= 1.03,
        "other_scene_score": all(minimum(s, "score", .98) for s in SCENES if s != "under_25"),
        "all_scene_conversion": all(minimum(s, "conversions", .97) for s in SCENES),
        "all_scene_violation": all(metrics[s]["violation"]["selected_count"] <= metrics[s]["violation"]["FB_count"] for s in SCENES),
        "non_FB_action": switches > 0}
    return {"flags": flags, "conjunction": all(flags.values()), "headline_score_ratio": headline}


def summarize(trajectories, rows):
    fields = ("score", "conversions", "violation", "zero_conversions", "cost")
    binary = ("violation", "zero_conversions")
    market_means = {(p, s, m): {field: float(np.mean([r[field] for r in trajectories
                    if r["policy"] == p and r["scene"] == s and r["market"] == m])) for field in fields}
                    for p in POLICIES for s in SCENES for m in MARKETS}
    market_counts = {(p, s, m): {field: sum(int(r[field]) for r in trajectories
                     if r["policy"] == p and r["scene"] == s and r["market"] == m) for field in binary}
                     for p in POLICIES for s in SCENES for m in MARKETS}
    result = {}
    for policy in POLICIES:
        scenes, paired = {}, []
        for scene in SCENES:
            for market in MARKETS:
                item = {"market": market, "scene": scene, **{field: comparison(
                    market_means[policy, scene, market][field], market_means["FB", scene, market][field]) for field in fields}}
                for field in binary:
                    item[field] = risk_comparison(market_counts[policy, scene, market][field],
                                                 market_counts["FB", scene, market][field], len(FOCALS))
                paired.append(item)
            scenes[scene] = {field: comparison(np.mean([market_means[policy, scene, m][field] for m in MARKETS]),
                np.mean([market_means["FB", scene, m][field] for m in MARKETS])) for field in fields}
            for field in binary:
                scenes[scene][field] = risk_comparison(sum(market_counts[policy, scene, m][field] for m in MARKETS),
                    sum(market_counts["FB", scene, m][field] for m in MARKETS), len(MARKETS) * len(FOCALS))
        selected_rows = [r for r in rows if r["policy"] == policy]
        switches = sum(r["action_differs_actual_FB"] for r in selected_rows)
        positive = sum(market_means[policy, "under_25", m]["score"] > market_means["FB", "under_25", m]["score"] for m in MARKETS)
        gating = gate(scenes, positive, switches, True)
        result[policy] = {"scenes": scenes, "paired_market_results": paired,
            "under25_positive_markets": positive, "action_difference_count": switches,
            "same_actual_FB_action_fraction": 1 - switches / len(selected_rows),
            "same_context_FB_action_fraction": float(np.mean([not r["action_differs_same_context_FB"] for r in selected_rows])),
            "clip_fraction": float(np.mean([r["clip_applied"] for r in selected_rows])),
            "projection_changed_fraction": float(np.mean([r["projection_changed_value"] for r in selected_rows])),
            "conditional_deviance_difference": None if any(r["deviance_original"] is None or r["deviance_belief"] is None for r in selected_rows)
                else float(sum(r["deviance_belief"] - r["deviance_original"] for r in selected_rows)),
            "gate": gating, "eligible": policy == "R", "advance": policy == "R" and gating["conjunction"]}
    return {"policies": result, "primary_R_pass": result["R"]["advance"], "N_eligible": False,
            "branch_next_action": "distinct future preregistration" if result["R"]["advance"] else "stop winner-belief control branch",
            "claim_scope": protocol()["claim"], "independent_confirmation": False, "generative_contribution": False}


def run(output, expected):
    output = Path(output).resolve()
    audit.require(not output.exists(), "Output exists; use a fresh run identity/directory")
    frozen, checks = preflight()
    audit.require(audit.identity(frozen) == expected, "Reviewed freeze identity changed")
    audit.require(shutil.disk_usage(output.parent).free >= MIN_FREE, "Less than10GiB free disk")
    with Path(__file__).open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        audit.write(output / "protocol.json", protocol())
        audit.write(output / "manifest.json", {"freeze_identity": expected, "frozen": frozen,
                    "frozen_before_pv_generation": True, "created_utc": datetime.now(UTC).isoformat()})
        audit.write(output / "preflight.json", checks)
        audit.write(output / "summary.json", {"status": "running", "scientific_outcome_evaluable": False, "primary_R_pass": None})
        rows, trajectories, status, reason, code = [], [], "complete", None, 0
        resources, handlers = Resources(output), {}

        def stop(signum, _frame):
            raise ResourceStop(f"Interrupt/deadline signal{signum}")

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM, signal.SIGPROF):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, LIMIT - RESERVE - (time.monotonic() - PROCESS_START)))
        signal.setitimer(signal.ITIMER_PROF, max(.001, LIMIT - RESERVE - time.process_time()))
        try:
            audit.torch.set_num_threads(1)
            with (output / "tick-logs.jsonl").open("w") as stream:
                for market in MARKETS:
                    resources.check(next_market=True)
                    trajectories.extend(market_run(market, stream, rows, resources))
                    audit.write(output / "trajectories.json", trajectories)
                    gc.collect()  # market_run locals/PVs are released before next generator
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
            try:
                unchanged = unchanged and audit.identity(audit.read(output / "protocol.json")) == audit.identity(frozen["protocol"])
            except (OSError, ValueError):
                unchanged = False
            if not unchanged:
                status, reason, code = "invalid_identity", "Source/protocol/input changed after freeze", 3
            covered = coverage(rows)
            if status == "complete" and (not covered["complete"] or len(trajectories) != 360):
                status, reason, code = "incomplete_resource", "Fixed study coverage incomplete", 2
            if time.monotonic() - PROCESS_START > LIMIT or time.process_time() > LIMIT:
                status, reason, code = "incomplete_resource", "Total process wall/CPU budget exceeded", 2
            summary = summarize(trajectories, rows) if status == "complete" else {"primary_R_pass": None, "N_eligible": False}
            summary.update(status=status, reason=reason, freeze_identity=expected,
                scientific_outcome_evaluable=status == "complete", coverage_complete=covered["complete"],
                completed_cells=len(rows), full_FB_replay_pass=status == "complete", automatic_followup_authorized=False)
            audit.write(output / "coverage.json", covered)
            audit.write(output / "trajectories.json", trajectories)
            audit.write(output / "summary.json", summary)
            resources.save()
            audit.write(output / "execution.json", {"status": status, "hashes_after": after,
                        "identities_unchanged": unchanged, "resources": resources.snapshot(), "power_changes": False})
            files = {p.name: audit.sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "seal.json"}
            audit.write(output / "seal.json", {"files": files, "content_identity": audit.identity(files)})
            audit.require(all(audit.sha(output / name) == digest for name, digest in files.items()), "Seal reread failed")
            print(json.dumps({"status": status, "completed_cells": len(rows), "primary_R_pass": summary["primary_R_pass"],
                              "output": str(output)}, sort_keys=True), flush=True)
        return code


def self_check():
    state = np.array([.4, .6, .7, .3, .24] + [0.] * 8, dtype=np.float32)
    fb = control(state, 2000., 100., 2, .5, "FB")
    r = control(state, 2000., 100., 2, .5, "R")
    n = control(state, 2000., 100., 2, .5, "N")
    audit.require(np.isclose(r["u_continuous"], r["P"] * r["k_before"] * r["Q_over_k"] ** .3), "Residual algebra failed")
    audit.require(np.isclose(n["u_continuous"], fb["u_continuous"] * 2), "Naive algebra failed")
    equal = [control(state, 2000., 100., 0, 0, p) for p in POLICIES]
    audit.require(all(d["action"] == equal[0]["action"] and d["u_continuous"] == equal[0]["u_continuous"] for d in equal), "Initial k1 arms differ")
    native = audit.Feedback(ACTIONS, .3, 8.)
    expected, _ = native([state], [], [], 2000., 100.)
    audit.require(fb["action"] == expected and fb["spend_from_state"] == float(state[3]) * 2000,
                  "Float32-native feedback mismatch")
    prefix = audit.budget_prefix(np.array([1., 2., 4.]), np.array([1, 0, 1]))
    small = audit.accepted_log(prefix, np.array([.1, .2, .3]), 1)
    large = audit.accepted_log(prefix, np.array([.1, .2, .3]), 3)
    audit.require(small["accepted_exposures"] == 1 and large["accepted_exposures"] == 2
                  and small["predicted_sum"] != large["predicted_sum"], "Memoized prefix incorrectly shares budget truncation")
    original = audit.new_track()
    other = audit.new_track()
    other["C"], other["E"] = 100, .2
    audit.require(control(state, 2000., 100., original["C"], original["E"], "R") == equal[1], "Other trajectory contaminated policy")
    audit.require(comparison(1, 0)["ratio"] is None and comparison(0, 0)["ratio"] is None, "Zero denominator epsilon found")
    metrics = {s: {"score": comparison(1.04, 1), "conversions": comparison(1, 1), "violation": risk_comparison(0, 0, 24)} for s in SCENES}
    audit.require(gate(metrics, 3, 1, True)["conjunction"], "Synthetic passing gate failed")
    audit.require(not gate(metrics, 3, 0, True)["conjunction"] and not gate(metrics, 3, 1, False)["conjunction"], "Fallback/incomplete passed")
    metrics["under_25"]["score"] = comparison(1, 0)
    audit.require(not gate(metrics, 3, 1, True)["conjunction"], "Undefined headline passed")
    partial = [{"market": MARKETS[0], "policy": p, "scene": s, "focal": f, "tick": 0}
               for s in SCENES for f in FOCALS for p in POLICIES]
    audit.require(not coverage(partial)["complete"] and coverage(partial)["completed"] == 90, "Partial coverage failed")
    return {"status": "passed", "checks": ["residual/naive algebra", "k1 initial equality", "float32 native FB reconstruction",
        "own budget truncation under shared prefix", "trajectory isolation", "undefined ratios", "gate conjunction/fallback/incomplete",
        "partial17280 coverage", "N diagnostic-only protocol"], "pv_generated": False, "output_created": False,
        "protocol_sha256": audit.identity(protocol())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="read-only four-tape/source freeze; no PV generation")
    commands.add_parser("self-check", help="synthetic controller/accounting/gate checks")
    runner = commands.add_parser("run", help="all four markets, all three policies; fresh output required")
    runner.add_argument("--expect-freeze", required=True)
    runner.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.command == "run":
        return run(args.output, args.expect_freeze)
    value = self_check() if args.command == "self-check" else dict(zip(("frozen", "checks"), preflight(), strict=True))
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
