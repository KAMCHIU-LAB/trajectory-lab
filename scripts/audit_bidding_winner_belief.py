#!/usr/bin/env python3
"""Market1001 accepted-winner public-p audit; no new policy or model fit.

Preflight/self-check do not generate PVs or write study output. Run requires the
reviewed preflight freeze identity and creates a fresh directory. The only
auction calls are actual feedback choices, with memoization within each tick.
"""
from __future__ import annotations

import argparse
import ast
import fcntl
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import signal
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

START_WALL = time.monotonic()
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trajectorylab.bidding.learning import Feedback  # noqa: E402
from trajectorylab.bidding.market import (  # noqa: E402
    BUDGET,
    CPA,
    REVISION,
    Tape,
    _cached_env,
    _load,
    observation,
    scenario_bias,
    score,
)

MAIN = ROOT / "artifacts/bidding/main-v2"
PLAN = ROOT / ".omx/plans/prd-bidding-winner-belief-audit.md"
OUTPUT = ROOT / "artifacts/bidding/numerical-audits/winner-belief-audit-v1"
SCENES = ("calibrated", "over_25", "over_50", "under_25", "mid_over_50")
FOCALS = (0, 8, 16, 24, 32, 40)
ACTIONS = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3], dtype=float)
MARKET, REQUESTED = 1001, 1440
LIMIT, RESERVE = 1800, 60
MIN_FREE, MAX_OUTPUT, MAX_MEMORY = 10 * 2**30, 2**30, 10 * 2**30
A_SOURCE = "8879cc50dc959b589f23e438c4e9511251a2ace10ff263edb86c645a8dd5f46a"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def protocol():
    return {
        "version": "winner-belief-audit-v1", "market": MARKET, "requested_cells": REQUESTED,
        "scenes": list(SCENES), "focals": list(FOCALS), "actions": ACTIONS.tolist(),
        "order": "tick0..47, then fixed scene order, then focal order; no complete action grid",
        "behavior": {"class": "Feedback", "gain": .3, "prior": 8.0, "changes": False},
        "generator": {"episode": MARKET, "requested_pv_count": 500000,
                      "actual_pv_count": "sum of48 generated integer traffic counts; compare old tape",
                      "external_resource_files": [],
                      "resource_evidence": "both generator resource loaders return inline np.array; AST rejects file-reading calls"},
        "kernel": "native _load/_cached_env, reset1001,prepare once per tick; original cached CRN",
        "opponents": "default_rng(731002): uniform(15,145,48),then uniform(0,2*pi,48); native sinusoidal bids; replace only one focal column and restore",
        "memoization": "within current tick only, key(focal index,effective action); cache selected focal budget-prefix arrays, not unselected action outputs",
        "accepted_prefix": "paid=cost*exposed; mask paid>0; original PV order; pc=[0,cumsum(paid[mask],float64)],cv=[0,cumsum(conversion[mask],int32)]; n=max(0,searchsorted(pc,max(0,remaining),right)-1)",
        "observable_prediction": "sum(generator public values[flatnonzero(mask)[:n],focal]*scenario_bias); no latent cached_values or new clipping; empty prefix sum0",
        "probability_semantics": "all accepted public p must be finite and within[0,1]; failure disables calibration, replay continues",
        "belief_allowlist": ["past accepted public probability sum", "past accepted integer conversions"],
        "belief": "independent Gamma(shape1,rate1) per scene/focal; k_before=(1+Cpast)/(1+Epast),mu=k_before*current realized accepted public-p sum; update after tick settlement; no forgetting/fitting/action change",
        "scoring": "conditional on realized accepted exposures; not pre-bid flow forecasting or policy utility",
        "deviance": "2*(C*log(C/mu)-C+mu); C0 ->2mu; mu0/C0 ->0; mu0/Cpositive ->None with invalid flag, no epsilon",
        "windows": {"all": [0, 48], "early": [0, 16], "middle": [16, 32], "late": [32, 48],
                    "midshift_before": [0, 24], "midshift_after": [24, 48]},
        "aggregation": "each trajectory first; scene mean across all6 focal summaries; no intervals/significance or bias recovery claims; windows do not reset belief",
        "replay": "compare native tape settlement at chosen effective action; exact action/n/conversion/blocked/state; cost and cumulative cost exact preferred else abs<=1e-8 reported; terminal score abs<=1e-8 and exact violation",
        "resources": {"wall_seconds_from_process_start": LIMIT, "cpu_seconds_from_process_start": LIMIT,
                      "seal_reserve_seconds": RESERVE, "threads": 1, "artifact_bytes_max": MAX_OUTPUT,
                      "free_disk_bytes_min": MIN_FREE, "peak_memory_bytes_max": MAX_MEMORY,
                      "stop": "real/profiling timers at1740s plus per-cell checks; forecast next complete tick using max(last tick,mean tick) wall/CPU; stop before next tick if insufficient budget",
                      "power_changes": False},
        "claims": "legal observability, exact replay completeness, descriptive calibration are separate; no business or generalization pass",
        "output": "aggregate per-cell JSONL only; no full PV/public/latent matrices or action tapes",
    }


def preflight():
    config_path = ROOT / "configs/bidding-main.toml"
    config = tomllib.loads(config_path.read_text())
    main, metadata = read(MAIN / "manifest.json"), read(MAIN / "markets/1001.json")
    study = config["study"]
    require(config == main["config"] and study["focal_agents"] == list(FOCALS)
            and study["actions"] == ACTIONS.tolist() and study["pv_count"] == 500000
            and MARKET in study["train_markets"], "Original study configuration mismatch")
    expected_market = identity({"study": {k: study[k] for k in ("focal_agents", "pv_count", "actions", "biases")},
                                "market": MARKET, "source": metadata["source"]})
    require(metadata["market_id"] == MARKET and metadata["identity"] == expected_market
            and metadata["focal_agents"] == list(FOCALS) and metadata["source"] == main["auction"]
            and metadata["source"]["upstream_revision"] == REVISION, "Market metadata identity mismatch")
    protected_path = ROOT / ".omx/state/bidding-robust/frozen-sources.json"
    protected = read(protected_path)
    sources = {name: sha(ROOT / name) for name in protected}
    require(len(protected) == 10 and sources == protected, "Historical frozen source changed")
    names = [str(Path(__file__).resolve().relative_to(ROOT)), "src/trajectorylab/bidding/market.py",
             "src/trajectorylab/bidding/learning.py", "src/trajectorylab/utils.py", "src/trajectorylab/models.py",
             "third_party/auctionnet/NeurIPSPvGen.py", "third_party/auctionnet/BiddingEnv.py",
             "scripts/probe_bidding_paired_advantage.py"]
    sources.update({name: sha(ROOT / name) for name in names})
    require(sources["scripts/probe_bidding_paired_advantage.py"] == A_SOURCE, "Sealed A source changed")
    for filename, digest in metadata["source"]["sha256"].items():
        key = "src/trajectorylab/bidding/market.py" if filename == "market.py" else "third_party/auctionnet/" + filename
        require(sources[key] == digest, "Public kernel/generator identity changed")
    # These installed generator loaders use inline constants, not external CSV resources.
    tree = ast.parse((ROOT / "third_party/auctionnet/NeurIPSPvGen.py").read_text())
    readers = {"open", "read_csv", "read_json", "read_text", "read_bytes", "loadtxt", "genfromtxt", "load"}
    require(not any(isinstance(node, ast.Call) and
                    (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) in readers
                    for node in ast.walk(tree)), "Generator gained external file resources; re-register freeze")
    parent = ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1/manifest.json"
    parent_manifest = read(parent)
    old = parent_manifest["core"]["inputs"][str(MAIN / "markets/1001.npz")]
    paths = [config_path, PLAN, MAIN / "manifest.json", MAIN / "markets/1001.json",
             MAIN / "markets/1001.npz", protected_path, parent]
    inputs = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    require(inputs[str((MAIN / "markets/1001.npz").relative_to(ROOT))] == old["npz_sha256"]
            and inputs[str((MAIN / "markets/1001.json").relative_to(ROOT))] == old["metadata_sha256"],
            "Market1001 tape/metadata changed")
    frozen = {"sources": sources, "inputs": inputs, "protocol": protocol(),
              "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "torch": torch.__version__, "platform": platform.platform()},
              "public_revision": REVISION, "study": study, "market_identity": expected_market}
    return frozen, {"freeze_identity": identity(frozen), "protocol_sha256": identity(protocol()),
                    "source_sha256": sources[names[0]], "pv_generation": False, "read_only": True,
                    "market": MARKET, "requested_cells": REQUESTED, "generator_external_resources": [],
                    "elapsed_wall_seconds": time.monotonic() - START_WALL}


def budget_prefix(paid, conversion):
    mask = np.flatnonzero(paid > 0)
    return (np.r_[0., np.cumsum(paid[mask], dtype=np.float64)],
            np.r_[0, np.cumsum(conversion[mask], dtype=np.int32)], mask)


def accepted_log(prefix, public_probability, remaining):
    pc, cv, indices = prefix
    n = max(0, int(np.searchsorted(pc, max(0, remaining), side="right")) - 1)
    accepted = public_probability[indices[:n]]
    return {"cost": float(pc[n]), "conversions": int(cv[n]), "accepted_exposures": n,
            "budget_blocked": bool(n < len(pc) - 1), "predicted_sum": float(accepted.sum()),
            "public_probability_valid": bool(np.isfinite(accepted).all() and np.all((accepted >= 0) & (accepted <= 1)))}


def forecast(past_conversion, past_exposure, current_exposure):
    k = (1 + past_conversion) / (1 + past_exposure)
    return float(k), float(k * current_exposure)


def deviance(conversion, mean):
    require(conversion >= 0 and mean >= 0 and math.isfinite(mean), "Invalid Poisson inputs")
    if mean == 0:
        return 0.0 if conversion == 0 else None
    return float(2 * (mean if conversion == 0 else conversion * math.log(conversion / mean) - conversion + mean))


class AuditStop(Exception):
    pass


class Resources:
    def __init__(self, output):
        self.output, self.ticks, self.kernel_attempted, self.kernel_completed = output, [], 0, 0
        self.generation = None
        self.last_scan, self.bytes = -math.inf, 0

    def snapshot(self):
        memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"wall_seconds": time.monotonic() - START_WALL, "cpu_seconds": time.process_time(),
                "peak_memory_bytes": int(memory if sys.platform == "darwin" else memory * 1024),
                "free_disk_bytes": shutil.disk_usage(self.output).free, "artifact_bytes": self.bytes}

    def check(self, forecast_tick=False):
        state = self.snapshot()
        now = time.monotonic()
        if now - self.last_scan >= 15:
            self.bytes = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
            self.last_scan = now
        if state["wall_seconds"] >= LIMIT - RESERVE or state["cpu_seconds"] >= LIMIT - RESERVE:
            raise AuditStop("Time budget reached; reserve60s for seal")
        if state["peak_memory_bytes"] > MAX_MEMORY or state["free_disk_bytes"] < MIN_FREE or self.bytes > MAX_OUTPUT:
            raise AuditStop("Memory/disk/artifact resource boundary reached")
        if forecast_tick and self.ticks:
            for key in ("wall_seconds", "cpu_seconds"):
                estimate = max(self.ticks[-1][key], float(np.mean([t[key] for t in self.ticks])))
                if state[key] + estimate >= LIMIT - RESERVE:
                    raise AuditStop(f"Next complete tick exceeds remaining {key} budget")

    def save(self):
        write(self.output / "resources.json", {"current": self.snapshot(), "generation": self.generation,
              "ticks": self.ticks, "kernel_calls_attempted": self.kernel_attempted,
              "kernel_calls_completed": self.kernel_completed, "requested_cells": REQUESTED,
              "limits": protocol()["resources"]})


def coverage(rows):
    cells = {(r["scene"], r["focal"], r["tick"]) for r in rows}
    require(len(cells) == len(rows), "Duplicate completed audit cell")
    prefixes, missing = {}, []
    for scene in SCENES:
        for focal in FOCALS:
            ticks = sorted(t for s, f, t in cells if s == scene and f == focal)
            require(ticks == list(range(len(ticks))), "Audit trajectory has a noncausal gap")
            prefixes[f"{scene}/{focal}"] = len(ticks)
            missing += [{"scene": scene, "focal": focal, "tick": t} for t in range(48) if t not in ticks]
    return {"requested": REQUESTED, "completed": len(rows), "complete": len(rows) == REQUESTED,
            "trajectory_prefix_lengths": prefixes, "missing_cells": missing,
            "maximum_covered_tick": max((r["tick"] for r in rows), default=None),
            "complete_ticks": min(prefixes.values())}


def diagnostics(rows, probability_valid):
    if not probability_valid:
        return {"status": "not_evaluable_probability_semantics", "scenes": {}}
    result = {}
    for scene in SCENES:
        windows = {k: v for k, v in protocol()["windows"].items()
                   if scene == "mid_over_50" or not k.startswith("midshift")}
        result[scene] = {}
        for name, (start, end) in windows.items():
            trajectories = []
            for focal in FOCALS:
                selected = [r for r in rows if r["scene"] == scene and r["focal"] == focal and start <= r["tick"] < end]
                if not selected:
                    trajectories.append({"focal": focal, "ticks": 0})
                    continue
                valid = all(r["deviance_original"] is not None and r["deviance_belief"] is not None for r in selected)
                trajectories.append({"focal": focal, "ticks": len(selected),
                    "E": float(sum(r["predicted_sum"] for r in selected)),
                    "C": int(sum(r["conversions"] for r in selected)),
                    "zero_exposure_fraction": float(np.mean([r["accepted_exposures"] == 0 for r in selected])),
                    "zero_conversion_fraction": float(np.mean([r["conversions"] == 0 for r in selected])),
                    "last_k_after": selected[-1]["k_after"], "deviance_valid": valid,
                    "deviance_original": float(sum(r["deviance_original"] for r in selected)) if valid else None,
                    "deviance_belief": float(sum(r["deviance_belief"] for r in selected)) if valid else None,
                    "absolute_error_original": float(np.mean([abs(r["conversions"] - r["predicted_sum"]) for r in selected])),
                    "absolute_error_belief": float(np.mean([abs(r["conversions"] - r["mu"]) for r in selected]))})
            complete_focals = all(r["ticks"] > 0 for r in trajectories)
            means = {}
            if complete_focals:
                for key in ("E", "C", "zero_exposure_fraction", "zero_conversion_fraction", "last_k_after",
                            "deviance_original", "deviance_belief", "absolute_error_original", "absolute_error_belief"):
                    means[key] = None if any(r[key] is None for r in trajectories) else float(np.mean([r[key] for r in trajectories]))
                means["deviance_difference_belief_minus_original"] = (None if means["deviance_belief"] is None
                    else means["deviance_belief"] - means["deviance_original"])
                means["absolute_error_difference_belief_minus_original"] = means["absolute_error_belief"] - means["absolute_error_original"]
            result[scene][name] = {"focals": trajectories, "equal_focal_means": means,
                                  "all_six_focals_have_observations": complete_focals}
    valid = all(r["deviance_original"] is not None and r["deviance_belief"] is not None for r in rows)
    return {"status": "descriptive_only" if valid else "not_evaluable_zero_mean_positive_conversion", "scenes": result,
            "claim": "conditional on realized accepted exposures, one market; no confidence or business claim"}


def new_track():
    return {"cost": 0., "conv": 0, "previous": (0., 0, 0., 0), "score": 0.,
            "states": [], "actions": [], "rewards": [], "E": 0., "C": 0}


def advance(track, settlement, action, cpa):
    track["cost"] += settlement[0]
    track["conv"] += settlement[1]
    current_score = score(track["cost"], track["conv"], cpa)
    track["rewards"].append(current_score - track["score"])
    track["score"] = current_score
    track["previous"] = (*settlement[:2], action, settlement[2])
    track["actions"].append(action)


def replay(rows, output, resources):
    tape = Tape(MAIN / "markets/1001.npz")
    pv_start = resources.snapshot()
    resources.generation = {"status": "running", "start": pv_start}
    resources.save()
    Generator = _load("winner_audit_public_generator", "NeurIPSPvGen.py").NeurIPSPvGen
    Env = _load("winner_audit_public_kernel", "BiddingEnv.py").BiddingEnv
    pv = Generator(episode=MARKET, pv_num=500000)
    resources.generation.update(status="generated", end=resources.snapshot(),
                                actual_pv_count=sum(len(v) for v in pv.pv_values))
    env = _cached_env(Env, MARKET)
    generated = {"counts": np.array([len(v) for v in pv.pv_values]),
                 "pmeans": np.array([[v[:, f].mean() for v in pv.pv_values] for f in FOCALS]),
                 "smeans": np.array([[v[:, f].mean() for v in pv.pValueSigmas] for f in FOCALS]),
                 "budgets": BUDGET[list(FOCALS)].astype(float), "cpas": CPA[list(FOCALS)].astype(float)}
    require(all(np.array_equal(generated[k], tape.arrays[k]) for k in generated), "Generated public statistics/tape mismatch")
    resources.generation = {"start": pv_start, "end": resources.snapshot(),
                            "actual_pv_count": int(generated["counts"].sum())}
    resources.save()
    resources.check()
    public_tape = SimpleNamespace(arrays=generated)
    rng = np.random.default_rng(MARKET + 730001)
    alpha, phase = rng.uniform(15, 145, 48), rng.uniform(0, 2 * np.pi, 48)
    actual, reference = {(s, f): new_track() for s in SCENES for f in FOCALS}, {(s, f): new_track() for s in SCENES for f in FOCALS}
    chooser, probability_valid = Feedback(ACTIONS, .3, 8.), True
    with (output / "tick-logs.jsonl").open("w") as stream:
        for tick, (values, sigmas) in enumerate(zip(pv.pv_values, pv.pValueSigmas, strict=True)):
            resources.check(forecast_tick=tick > 0)
            begin = resources.snapshot()
            env.prepare(values, sigmas)
            bids = values * (alpha * (1 + .15 * np.sin(2 * np.pi * tick / 48 + phase)))
            memo = {}
            for scene in SCENES:
                for fi, focal in enumerate(FOCALS):
                    resources.check()
                    track, old = actual[scene, focal], reference[scene, focal]
                    budget, cpa = generated["budgets"][fi], generated["cpas"][fi]
                    bias = scenario_bias(scene, tick)
                    state = observation(public_tape, fi, tick, track["cost"], track["conv"], track["previous"], bias)
                    old_state = observation(tape, fi, tick, old["cost"], old["conv"], old["previous"], bias)
                    track["states"].append(state)
                    old["states"].append(old_state)
                    action, _ = chooser(track["states"], track["actions"], track["rewards"], budget, cpa)
                    old_action, _ = chooser(old["states"], old["actions"], old["rewards"], budget, cpa)
                    key = (fi, float(action * bias))
                    cached = key in memo
                    cell_start = resources.snapshot()
                    if not cached:
                        original = bids[:, focal].copy()
                        try:
                            bids[:, focal] = values[:, focal] * CPA[focal] * (action * bias)
                            resources.kernel_attempted += 1
                            out = env.simulate_ad_bidding(values, sigmas, bids)
                            resources.kernel_completed += 1
                            memo[key] = budget_prefix(out[2][focal] * out[3][focal], out[4][focal])
                            del out
                        finally:
                            bids[:, focal] = original
                    log = accepted_log(memo[key], values[:, focal] * bias, budget - track["cost"])
                    expected = tape.settle(fi, tick, old_action * bias, budget - old["cost"])
                    cost_error = abs(log["cost"] - expected[0])
                    settlement = (log["cost"], log["conversions"], log["accepted_exposures"], log["budget_blocked"])
                    match = (action == old_action and np.array_equal(state, old_state) and cost_error <= 1e-8
                             and settlement[1:] == expected[1:])
                    k, mu = forecast(track["C"], track["E"], log["predicted_sum"])
                    probability_valid = probability_valid and log["public_probability_valid"]
                    log.update(scene=scene, focal=focal, tick=tick, action=float(action),
                        remaining_budget_before=float(budget - track["cost"]),
                        k_before=k if probability_valid else None, mu=mu if probability_valid else None,
                        deviance_original=deviance(log["conversions"], log["predicted_sum"]) if probability_valid else None,
                        deviance_belief=deviance(log["conversions"], mu) if probability_valid else None,
                        expected_cost=float(expected[0]), expected_conversions=int(expected[1]),
                        expected_exposures=int(expected[2]), expected_budget_blocked=bool(expected[3]),
                        exact_cost_equal=log["cost"] == expected[0], cost_absolute_error=cost_error,
                        state_array_equal=bool(np.array_equal(state, old_state)), action_equal=bool(action == old_action),
                        memoized_kernel=cached, kernel_wall_seconds=resources.snapshot()["wall_seconds"] - cell_start["wall_seconds"],
                        settlement_timestamp=f"end_of_tick_{tick}", replay_match=bool(match))
                    advance(track, settlement, action, cpa)
                    advance(old, expected, old_action, cpa)
                    log["cumulative_cost_error"] = abs(track["cost"] - old["cost"])
                    log["cumulative_conversions_equal"] = track["conv"] == old["conv"]
                    if probability_valid:
                        track["E"] += log["predicted_sum"]
                        track["C"] += log["conversions"]
                    log["k_after"] = forecast(track["C"], track["E"], 0)[0] if probability_valid else None
                    rows.append(log)
                    stream.write(json.dumps(log, sort_keys=True, allow_nan=False) + "\n")
                    stream.flush()
                    require(match and log["cumulative_cost_error"] <= 1e-8 and log["cumulative_conversions_equal"],
                            f"Replay mismatch at{scene}/{focal}/{tick}")
                    resources.check()
            elapsed = resources.snapshot()
            resources.ticks.append({"tick": tick, "wall_seconds": elapsed["wall_seconds"] - begin["wall_seconds"],
                                    "cpu_seconds": elapsed["cpu_seconds"] - begin["cpu_seconds"],
                                    "completed_cells": len(rows), "kernel_calls_completed": resources.kernel_completed})
            if tick == 0:
                resources.ticks[-1]["projected_1440_cell_seconds"] = resources.ticks[-1]["wall_seconds"] * 48
            resources.save()
    terminals = []
    for scene in SCENES:
        for fi, focal in enumerate(FOCALS):
            track, old = actual[scene, focal], reference[scene, focal]
            violation = int(track["cost"] > generated["cpas"][fi] * track["conv"] + 1e-8)
            old_violation = int(old["cost"] > generated["cpas"][fi] * old["conv"] + 1e-8)
            require(abs(track["score"] - old["score"]) <= 1e-8 and violation == old_violation,
                    "Terminal score/violation mismatch")
            terminals.append({"scene": scene, "focal": focal, "cost": track["cost"], "conversions": track["conv"],
                              "score": track["score"], "violation": violation, "reference_score": old["score"],
                              "reference_violation": old_violation})
    return terminals


def after_hashes(names):
    result = {}
    for name in names:
        try:
            result[name] = sha(ROOT / name)
        except OSError:
            result[name] = None
    return result


def run(output, expected):
    output = Path(output).resolve()
    require(not output.exists(), "Output exists; use a fresh run identity/directory")
    frozen, checks = preflight()
    require(identity(frozen) == expected, "Reviewed source/protocol/input freeze changed")
    require(shutil.disk_usage(output.parent).free >= MIN_FREE, "Less than10GiB free disk")
    with Path(__file__).open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        write(output / "protocol.json", protocol())
        write(output / "manifest.json", {"freeze_identity": expected, "frozen": frozen,
              "frozen_before_pv_generation": True, "created_utc": datetime.now(UTC).isoformat()})
        write(output / "preflight.json", checks)
        write(output / "summary.json", {"status": "running", "scientific_outcome_evaluable": False})
        rows, terminals, status, reason, code = [], [], "complete", None, 0
        resources, handlers = Resources(output), {}

        def stop(signum, _frame):
            raise AuditStop(f"Interrupted/deadline signal{signum}")

        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM, signal.SIGPROF):
            handlers[sig] = signal.signal(sig, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.001, LIMIT - RESERVE - (time.monotonic() - START_WALL)))
        signal.setitimer(signal.ITIMER_PROF, max(.001, LIMIT - RESERVE - time.process_time()))
        try:
            torch.set_num_threads(1)
            resources.check()
            terminals = replay(rows, output, resources)
            resources.check()
        except AuditStop as error:
            status, reason, code = "incomplete_resource", str(error), 2
        except Exception as error:
            status, reason, code = "invalid_replay", f"{type(error).__name__}: {error}", 3
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.setitimer(signal.ITIMER_PROF, 0)
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            hashes = {"sources": after_hashes(frozen["sources"]), "inputs": after_hashes(frozen["inputs"])}
            unchanged = hashes["sources"] == frozen["sources"] and hashes["inputs"] == frozen["inputs"]
            try:
                unchanged = unchanged and identity(read(output / "protocol.json")) == identity(frozen["protocol"])
            except (OSError, ValueError):
                unchanged = False
            if not unchanged:
                status, reason, code = "invalid_identity", "Source/protocol/input changed after freeze", 3
            covered = coverage(rows)
            if status == "complete" and not covered["complete"]:
                status, reason, code = "incomplete_resource", "Fixed1440 cell coverage incomplete", 2
            if time.monotonic() - START_WALL > LIMIT or time.process_time() > LIMIT:
                status, reason, code = "incomplete_resource", "Total process wall/CPU budget exceeded", 2
            probability_valid = all(r["public_probability_valid"] for r in rows)
            calibration = diagnostics(rows, probability_valid)
            calibration["coverage_complete"] = covered["complete"]
            if not covered["complete"] and calibration["status"] == "descriptive_only":
                calibration["status"] = "partial_descriptive_only"
            replay_pass = status == "complete" and covered["complete"] and all(r["replay_match"] for r in rows)
            write(output / "coverage.json", covered)
            write(output / "calibration.json", calibration)
            write(output / "replay.json", {"terminals": terminals, "full_replay_pass": replay_pass,
                  "maximum_cost_error": max((r["cost_absolute_error"] for r in rows), default=None),
                  "maximum_cumulative_cost_error": max((r["cumulative_cost_error"] for r in rows), default=None),
                  "nonexact_cost_cells": sum(not r["exact_cost_equal"] for r in rows)})
            resources.save()
            write(output / "execution.json", {"status": status, "source_and_input_hashes_after": hashes,
                  "identities_unchanged": unchanged, "resources": resources.snapshot(), "power_changes": False})
            summary = {"status": status, "reason": reason, "freeze_identity": expected,
                       "coverage_complete": covered["complete"], "completed_cells": covered["completed"],
                       "legal_observability": covered["complete"] and probability_valid and status == "complete",
                       "observed_cells_public_probability_valid": probability_valid if rows else None,
                       "full_replay_pass": replay_pass, "calibration_status": calibration["status"],
                       "scientific_outcome_evaluable": status == "complete" and probability_valid and calibration["status"] == "descriptive_only",
                       "business_improvement_proven": False, "cross_market_generalization_proven": False,
                       "automatic_followup_authorized": False}
            write(output / "summary.json", summary)
            files = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "seal.json"}
            write(output / "seal.json", {"files": files, "content_identity": identity(files)})
            require(all(sha(output / name) == digest for name, digest in files.items()), "Seal reread failed")
            print(json.dumps(summary, sort_keys=True), flush=True)
        return code


def self_check():
    paid = np.array([0., 2., 1., 4.])
    prefix = budget_prefix(paid, np.array([1, 1, 0, 1]))
    p = np.array([.9, .2, .3, .7])
    exact = accepted_log(prefix, p, 3)
    require(exact["accepted_exposures"] == 2 and exact["cost"] == 3 and exact["conversions"] == 1
            and exact["predicted_sum"] == .5 and exact["budget_blocked"], "Budget prefix/order boundary failed")
    require(accepted_log(prefix, p, 2)["accepted_exposures"] == 1, "Exactly payable prefix failed")
    require(accepted_log(prefix, p, 0)["predicted_sum"] == 0, "Zero budget is not empty")
    changed = p.copy()
    changed[[0, 3]] = 99
    require(accepted_log(prefix, changed, 3)["predicted_sum"] == .5, "Unaccepted probability leaked")
    empty = accepted_log(budget_prefix(np.zeros(4), np.ones(4)), p, 100)
    require(empty["cost"] == empty["conversions"] == empty["predicted_sum"] == 0, "Empty exposures failed")
    k, mu = forecast(2, .5, .25)
    require(k == 2 and mu == .5 and forecast(0, 0, .25) == (1., .25), "Gamma update/reset failed")
    require(forecast(2, .5, .25)[0] == forecast(2, .5, 99)[0], "Current exposure changed k_before")
    require(deviance(0, 0) == 0 and deviance(1, 0) is None and deviance(0, .5) == 1
            and deviance(1, 1) == 0, "Poisson zero branches failed")
    require(not accepted_log(prefix, np.array([0., 2., .3, .7]), 3)["public_probability_valid"], "Invalid public probability clipped")
    partial = [{"scene": s, "focal": f, "tick": 0} for s in SCENES for f in FOCALS]
    require(not coverage(partial)["complete"] and coverage(partial)["complete_ticks"] == 1, "Partial coverage failed")
    full = [{"scene": s, "focal": f, "tick": t} for t in range(48) for s in SCENES for f in FOCALS]
    require(coverage(full)["complete"] and not coverage(full)["missing_cells"], "Full coverage failed")
    return {"status": "passed", "checks": ["original paid-PV order", "exact right-search boundary",
            "zero budget/empty exposures", "only accepted public p affects sum", "Gamma update/reset/prequential timing",
            "Poisson zero branches/no epsilon", "probability-range rejection", "partial/full fixed coverage"],
            "pv_generated": False, "output_created": False, "protocol_sha256": identity(protocol())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="read-only identity/resource check; no PV generation")
    commands.add_parser("self-check", help="synthetic prefix/belief/coverage checks; no audit")
    runner = commands.add_parser("run", help="frozen market1001 audit only; fresh output required")
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
