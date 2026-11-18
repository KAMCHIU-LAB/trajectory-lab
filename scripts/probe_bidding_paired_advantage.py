#!/usr/bin/env python3
"""Frozen, train-only market CV for one-intervention paired advantage.

This driver never loads a validation cache, encoder, or raw market array. Raw
training tapes are hashed, not decoded. ``preflight`` and ``self-check`` write
nothing. ``run`` requires the freeze identity printed by preflight and owns a
fresh directory. A stopped/invalid run cannot report scientific success.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import platform
import resource
import shutil
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Set numerical-library limits before importing either library; process-local only.
for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                         "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_variable] = "1"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from trajectorylab.bidding.market import score as native_score  # noqa: E402

CACHE = ROOT / "artifacts/bidding/numerical-audits/continuation-feasibility-v1"
OUTPUT = ROOT / "artifacts/bidding/numerical-audits/paired-advantage-v2"
PROTECTED = ROOT / ".omx/state/bidding-robust/frozen-sources.json"
MARKETS = tuple(range(1001, 1017))
SCENES = ("calibrated", "over_25", "over_50", "under_25", "mid_over_50")
PRIMARY = ("over_25", "over_50", "under_25")
TICKS = (0, 16, 32)
ACTIONS = np.array([0, .25, .5, .75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3], dtype=float)
CONFIGS = (
    "F0_ridge_delta", "F1_ridge_delta", "F1_linear_three_class",
    "F1_mlp_delta", "F1_mlp_absolute", "F1_knn_delta",
    "F1_ridge_shuffled_control",
)
FAMILIES = (0, 1, 1, 1, 1, 1, 1)
DIRECT = (0, 1, 2, 3, 5)
SEED, UPDATES, BATCH, WIDTH = 11, 2000, 64, 64
WALL_LIMIT, PREFLIGHT_LIMIT = 7200, 1800
MIN_FREE, MAX_ARTIFACT, MAX_MEMORY = 10 * 2**30, 2 * 2**30, 10 * 2**30
HISTORY_COLUMNS = (6, 7, 12, 8)
STATE_NAMES = (
    "elapsed", "remaining_time", "remaining_budget", "spent", "conversions",
    "cpa_debt", "last_cost", "last_conversion", "previous_action",
    "public_predicted_pmean", "public_value_ratio", "public_traffic", "last_win_fraction",
)


def utc():
    return datetime.now(UTC).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint(value):
    # This serialization also matches the historical manifest identity function.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def hashes_after_run(names):
    result = {}
    for name in names:
        try:
            result[name] = digest(ROOT / name)
        except OSError:
            result[name] = None
    return result


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def feature_names(family):
    names = list(STATE_NAMES) + [f"feedback_is_action_{i}" for i in range(11)]
    if family:
        for window in (4, 8):
            for col in HISTORY_COLUMNS:
                names += [f"window{window}_{STATE_NAMES[col]}_mean",
                          f"window{window}_{STATE_NAMES[col]}_last_minus_first"]
            names += [f"window{window}_available_length_ratio", f"window{window}_debt_change"]
        names += ["consecutive_no_win_completed_settlements_capped8"]
    return names


def protocol():
    """All numerical choices are data-independent; no CLI tuning surface."""
    return {
        "version": "paired-advantage-v2.1",
        "scope": "exploratory training-market CV; one intervention then fixed feedback to tick47",
        "input_allowlist": ["train.npz", "train.identity.json", "identity metadata and source hashes"],
        "policy_input_allowlist": ["states", "steps", "lengths", "baseline_index"],
        "forbidden": ["validation labels", "fresh markets", "encoder", "legacy feasibility/fit"],
        "markets": list(MARKETS), "scenes": list(SCENES), "ticks": list(TICKS),
        "actions": ACTIONS.tolist(),
        "folds": [list(MARKETS[i:i + 4]) for i in range(0, 16, 4)],
        "features": {
            "F0": feature_names(0), "F1": feature_names(1),
            "action_axis": "eleven fixed outputs in original action order; feedback is one-hot",
            "history": "last min(window,length) valid observations; right padding ignored",
            "tick0": "one public initial observation; zero completed settlements",
            "no_win": "count trailing column12==0 among last8 observations with actual step>0; cap8",
            "normalizer": "per feature train-context mean and population std; std floor1e-6; no IDs",
        },
        "targets": {
            "utility": "terminal_score * cpa / budget (physical normalized unit)",
            "advantage": "(terminal_score[action]-terminal_score[feedback])*cpa/budget; subtract exact score labels before scaling; force feedback=0 exactly",
            "shared_score_scale": "max(sqrt(mean(training utility**2 across contexts/actions)),0.01); no centering; shared across all rankers, direct and absolute",
            "conversion_unit": "terminal_conversions * cpa / budget",
            "shared_conversion_scale": "max(sqrt(mean(training normalized_conversion**2)),0.01)",
            "classes": "negative iff exact delta<0; zero iff delta==0; positive iff delta>0; no tolerance",
        },
        "training": {
            "seed": SEED, "updates": UPDATES, "batch": BATCH, "lr": .001,
            "device": "cpu", "threads": 1, "deterministic_algorithms": True,
            "optimizer": "Adam defaults betas=(0.9,0.999),eps=1e-8,weight_decay=0",
            "batch_stream": "numpy default_rng(11).integers(0,n_train,(2000,64)); identical for matched MLPs",
            "initialization": "torch.manual_seed(11) immediately before each model; torch Linear defaults",
            "hidden": [WIDTH, WIDTH], "activation": "SiLU", "early_stopping": False,
            "neural_fits": 20,
            "probe": "first100 updates of fold0 F0 auxiliary fit; continue same fit to2000; no restart",
        },
        "configurations": [
            {"name": CONFIGS[0], "family": "F0", "target": "paired", "model": "ridge"},
            {"name": CONFIGS[1], "family": "F1", "target": "paired", "model": "ridge"},
            {"name": CONFIGS[2], "family": "F1", "target": "paired", "model": "linear33 logits"},
            {"name": CONFIGS[3], "family": "F1", "target": "paired", "model": "MLP64x2,11 unconstrained outputs"},
            {"name": CONFIGS[4], "family": "F1", "target": "absolute", "model": "MLP64x2,11 unconstrained outputs; subtract predicted feedback"},
            {"name": CONFIGS[5], "family": "F1", "target": "paired", "model": "Euclidean kNN32,uniform neighbors; ties by training context index"},
        ],
        "ridge": "lambda1; weighted residual SUM with context weights normalized to n_train; intercept unpenalized; float64 solve",
        "three_class": {
            "loss": "weighted cross entropy over legal nonfeedback candidates; same context/market weighting as Brier; 2000 Adam updates",
            "magnitudes": "per action and exact sign class, training-weighted mean physical advantage; zero class=0; absent class=0, no smoothing",
            "decode": "sum(class_probability * corresponding signed class magnitude)",
        },
        "auxiliary": {
            "fits": "one F0 and one F1 fit per fold; reused identically by every ranker in that family",
            "architecture": "MLP64x2 SiLU,34 unconstrained linear outputs",
            "outputs": "baseline normalized conv(1), candidate normalized conv increment(11), violation logits(11), positive logits(11)",
            "loss": "sum of four equally weighted terms: scaled-baseline MSE, scaled-delta-conv MSE(action mean), violation BCE(action mean), positive BCE(legal nonfeedback candidate mean); context/market weighted means",
            "scale": "both conversion MSE targets divided by shared training conversion scalar; logits unscaled",
            "decode": "baseline conv=max(raw*conv_scale,0); delta conv=raw*conv_scale; sigmoid risks/positive; baseline delta=0, positive=0; baseline risk is candidate risk at feedback",
            "brier_attribution": "shared positive head establishes feature-family signal, not unique ranker calibration",
        },
        "shuffled_control": "one F1 ridge specification in all4 folds; permute complete target vectors using default_rng(11) within training fold; same guards; never eligible winner",
        "controller": {
            "radius": 1.0,
            "switch": "abs(action-feedback)<=1 AND predicted delta_score>0 AND predicted delta_conv>=-0.03*max(predicted baseline conv,0) AND predicted candidate risk<=predicted feedback risk",
            "zero_baseline": "predicted baseline conv==0 requires predicted delta_conv>=0",
            "ties": "feedback default; greatest strictly positive advantage; equal candidate advantages use original action order",
            "identity": "feedback score/conv delta exactly0; candidate/baseline risk equal; no true feedback outcome consulted",
        },
        "weighting": {
            "candidate": "radius-legal nonfeedback candidates equally within context; context equally within market; markets equally",
            "prior": "training-fold weighted positive frequency using identical weights; never held fold",
            "empty_candidates": "record count and return undefined calibration, no silent context removal",
            "business": "per market and scene average six focals x three ticks; scene ratio is ratio of equal-market means; headline equal mean of three primary scene ratios",
            "zero_denominator": "ratio=None; retain market and absolute difference; undefined headline cannot pass",
        },
        "g1": {
            "brier_relative_improvement_min": .05, "headline_ratio_min": 1.005,
            "strictly_positive_folds_min": 3, "primary_score_ratio_min": .98,
            "primary_conversion_ratio_min": .97, "primary_violation_delta_max": 0,
            "requires_switch": True, "selection": "all6 exploratory; shuffled control excluded",
        },
        "resources": {
            "s0_seconds": PREFLIGHT_LIMIT, "s1_seconds": WALL_LIMIT,
            "artifact_bytes_max": MAX_ARTIFACT, "free_disk_bytes_min": MIN_FREE,
            "process_peak_memory_bytes_max": MAX_MEMORY, "one_fit_at_a_time": True,
            "resource_check": "every100 updates and task boundary; artifact scan every30s and fold boundary",
            "stop": "incomplete for resource/interruption; invalid for identity/calculation failure; no success claim",
        },
    }


def features(states, steps, lengths, baseline, family):
    """Only four allowlisted public arrays enter this function."""
    result = []
    for observations, tick_ids, length, base in zip(states, steps, lengths, baseline, strict=True):
        prefix = observations[:int(length)]
        values = list(prefix[-1].astype(float)) + np.eye(11)[int(base)].tolist()
        if family:
            for window in (4, 8):
                visible = prefix[-window:]
                for col in HISTORY_COLUMNS:
                    values += [float(visible[:, col].mean()), float(visible[-1, col] - visible[0, col])]
                values += [len(visible) / window, float(visible[-1, 5] - visible[0, 5])]
            completed = prefix[np.asarray(tick_ids[:int(length)]) > 0][-8:]
            no_win = 0
            for state in completed[::-1]:
                if state[12] != 0:
                    break
                no_win += 1
            values.append(no_win)
        result.append(values)
    result = np.asarray(result, dtype=np.float64)
    require(result.shape == (len(states), len(feature_names(family))), "Feature schema changed")
    require(np.isfinite(result).all(), "Nonfinite policy features")
    return result


def legal_candidates(baseline):
    legal = np.abs(ACTIONS[None] - ACTIONS[np.asarray(baseline), None]) <= 1.0
    legal[np.arange(len(baseline)), baseline] = False
    return legal


def context_weights(markets):
    unique, inverse, counts = np.unique(markets, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse])


def candidate_weights(markets, legal):
    counts = legal.sum(axis=1)
    return context_weights(markets)[:, None] * legal / np.maximum(counts[:, None], 1)


def weighted_brier(probability, labels, markets, legal):
    empty = int(np.count_nonzero(legal.sum(axis=1) == 0))
    if empty:
        return {"value": None, "empty_contexts": empty}
    value = np.sum(candidate_weights(markets, legal) * (probability - labels) ** 2)
    return {"value": float(value), "empty_contexts": 0}


def positive_prior(labels, markets, legal):
    require(np.all(legal.sum(axis=1) > 0), "Empty training candidate set; prior not evaluable")
    return float(np.sum(candidate_weights(markets, legal) * labels))


def identities():
    protected = read_json(PROTECTED)
    require(len(protected) == 10, "Expected ten historically protected sources")
    source_names = set(protected) | {
        str(Path(__file__).resolve().relative_to(ROOT)),
        "src/trajectorylab/bidding/market.py", "src/trajectorylab/utils.py",
        "src/trajectorylab/bidding_continuation_data.py",
    }
    sources = {name: digest(ROOT / name) for name in sorted(source_names)}
    require(all(sources[k] == v for k, v in protected.items()), "Historically frozen source changed")
    manifest = read_json(CACHE / "manifest.json")
    core, training = manifest["core"], manifest["datasets"]["train"]
    stamp = read_json(CACHE / "train.identity.json")
    require(fingerprint(core) == manifest["identity"], "Parent manifest identity mismatch")
    require(stamp["identity"] == fingerprint({"identity": manifest["identity"], "split": "train"}),
            "Training split identity mismatch")
    require(core["actions"] == ACTIONS.tolist() and core["scenarios"] == list(SCENES)
            and core["ticks"] == list(TICKS) and core["context"] == 24, "Cache protocol changed")
    for name, expected in core["sources"].items():
        actual = digest(ROOT / name)
        require(actual == expected, f"Historical cache source changed: {name}")
        sources[name] = actual
    inputs = {}
    fixed_inputs = [CACHE / "manifest.json", CACHE / "train.identity.json", CACHE / "train.npz",
                    PROTECTED, ROOT / "artifacts/bidding/main-v2/manifest.json",
                    ROOT / ".omx/plans/prd-bidding-advantage-v2.md",
                    ROOT / ".omx/plans/test-spec-bidding-advantage-v2.md",
                    ROOT / ".omx/state/bidding-v2/planning-handoff.json"]
    for path in fixed_inputs:
        inputs[str(path.relative_to(ROOT))] = digest(path)
    require(inputs[str((CACHE / "train.npz").relative_to(ROOT))] == training["npz_sha256"]
            == stamp["npz_sha256"], "Training cache hash mismatch")
    require(inputs[str((CACHE / "train.identity.json").relative_to(ROOT))]
            == training["identity_sha256"], "Training identity metadata hash mismatch")
    main_path = ROOT / "artifacts/bidding/main-v2/manifest.json"
    require(inputs[str(main_path.relative_to(ROOT))] == core["main_manifest_sha256"],
            "Original main manifest changed")
    main = read_json(main_path)
    require(main["identity"] == core["main_identity"]
            and main["config"]["study"]["train_markets"] == list(MARKETS)
            and main["config"]["study"]["actions"] == ACTIONS.tolist(), "Original training identities changed")
    handoff = read_json(ROOT / ".omx/state/bidding-v2/planning-handoff.json")
    require(handoff.get("authorizes_new_scoped_implementation") is True
            and handoff.get("authorizes_new_scoped_fitting_after_freeze") is True,
            "Missing terminal direct execution handoff")
    reviews = handoff["ordered_alternative_reviews"]
    require([r["role"] for r in reviews] == ["architect", "critic"]
            and all(r["verdict"] == "APPROVE S0/S1" for r in reviews), "Ordered reviews missing")
    for review in reviews:
        inputs[review["artifact"]] = digest(ROOT / review["artifact"])
    raw_training = {int(Path(p).stem): (Path(p), hashes) for p, hashes in core["inputs"].items()
                    if int(Path(p).stem) in MARKETS}
    require(sorted(raw_training) == list(MARKETS), "Missing declared training market identity")
    focals = {}
    for market, (path, expected) in sorted(raw_training.items()):
        # Hashing does not decode or use raw tape labels. Validation paths are never opened.
        for item, expected_hash in ((path, expected["npz_sha256"]),
                                    (path.with_suffix(".json"), expected["metadata_sha256"])):
            actual = digest(item)
            require(actual == expected_hash, f"Training market bytes changed: {item.name}")
            inputs[str(item.relative_to(ROOT))] = actual
        metadata = read_json(path.with_suffix(".json"))
        require(metadata["market_id"] == market and metadata["source"] == core["auction"],
                "Training market metadata/source mismatch")
        require(len(metadata["focal_agents"]) == 6 and len(set(metadata["focal_agents"])) == 6,
                "Expected six unique focal agents")
        focals[market] = metadata["focal_agents"]
    for basename, expected in core["auction"]["sha256"].items():
        path = ROOT / ("src/trajectorylab/bidding/market.py" if basename == "market.py"
                       else f"third_party/auctionnet/{basename}")
        require(digest(path) == expected, "Auction source identity changed")
        sources[str(path.relative_to(ROOT))] = expected
    return {"sources": sources, "inputs": inputs, "protected_sources": protected,
            "parent_identity": manifest["identity"], "training_schema": training["arrays"]}, focals


def validate_data(data, schema, focals):
    require(set(data) == set(schema), "Training array schema changed")
    for name, array in data.items():
        require(list(array.shape) == schema[name]["shape"] and str(array.dtype) == schema[name]["dtype"],
                f"Training array shape/dtype changed: {name}")
        if name != "diagnostics":
            require(np.isfinite(array).all(), f"Nonfinite array: {name}")
    n, base, diag = len(data["lengths"]), data["baseline_index"], data["diagnostics"]
    require(n == 1440 and data["states"].shape == (n, 24, 13), "Expected 1440 train contexts")
    require(diag.dtype.names == ("market", "scene_index", "focal", "tick"), "Diagnostics schema changed")
    expected = {(m, s, f, t) for m in MARKETS for s in range(5) for f in focals[m] for t in TICKS}
    require({tuple(map(int, row)) for row in diag} == expected and len(expected) == n,
            "Incomplete/duplicate market-scene-focal-tick coverage")
    require(np.all((base >= 0) & (base < 11)), "Invalid feedback action")
    require(np.all(data["budget"] > 0) and np.all(data["cpa"] > 0), "Invalid budget/CPA")
    require(np.all(data["spent"] >= 0) and np.all(data["conversions"] >= 0)
            and np.all(data["terminal_conversions"] >= 0) and np.all(data["endpoints"] >= 0),
            "Negative accounting label")
    require(np.issubdtype(data["terminal_conversions"].dtype, np.integer)
            and np.issubdtype(data["conversions"].dtype, np.integer), "Conversions must be integers")
    require(np.isin(data["terminal_violation"], [0, 1]).all(), "Nonbinary violation label")
    for i, length in enumerate(data["lengths"]):
        tick = int(diag[i]["tick"])
        require(length == min(tick + 1, 24), "Retained observation length changed")
        steps = np.arange(tick + 1 - length, tick + 1)
        require(np.array_equal(data["steps"][i, :length], steps)
                and np.all(data["states"][i, length:] == 0)
                and np.all(data["steps"][i, length:] == 0), "Right padding/original ticks changed")
        require(np.allclose(data["states"][i, :length, 0], steps / 48, rtol=0, atol=4e-8),
                "Public observation time differs from actual tick IDs")
    current = data["states"][np.arange(n), data["lengths"] - 1]
    require(np.allclose(current[:, 3], data["spent"] / data["budget"], rtol=1e-6, atol=1e-7)
            and np.allclose(current[:, 4], data["conversions"] * data["cpa"] / data["budget"],
                            rtol=1e-6, atol=1e-7), "Public cumulative accounting mismatch")
    cost = data["spent"][:, None] + data["endpoints"][:, :, 0] * data["budget"][:, None]
    conv = data["conversions"][:, None] + data["endpoints"][:, :, 1] * (data["budget"] / data["cpa"])[:, None]
    require(np.all(cost >= 0) and np.all(cost <= data["budget"][:, None] + 1e-6), "Hard budget violated")
    require(np.allclose(conv, data["terminal_conversions"], rtol=0, atol=1e-8),
            "Endpoint conversions do not reconstruct exact integer labels")
    recomputed = np.array([[native_score(float(c), int(v), float(cpa))
                            for c, v in zip(row, conversions, strict=True)]
                           for row, conversions, cpa in zip(cost, data["terminal_conversions"],
                                                            data["cpa"], strict=True)])
    require(np.allclose(recomputed, data["terminal_score"], rtol=1e-10, atol=1e-9),
            "Native terminal score reconstruction failed")
    violation = cost > data["cpa"][:, None] * data["terminal_conversions"] + 1e-8
    require(np.array_equal(violation, data["terminal_violation"]), "Native violation branch mismatch")
    utility, advantage, conversion, delta_conv = targets(data)
    rows = np.arange(n)
    raw_advantage = data["terminal_score"] - data["terminal_score"][rows, base, None]
    require(np.array_equal(np.sign(raw_advantage), np.sign(advantage)),
            "Physical normalization changed an exact raw advantage sign")
    require(np.all(advantage[rows, base] == 0) and np.all(delta_conv[rows, base] == 0),
            "a=b identity changed")
    # Each baseline continuation must reproduce the same feedback terminal label across ticks.
    for market in MARKETS:
        for scene in range(5):
            for focal in focals[market]:
                mask = (diag["market"] == market) & (diag["scene_index"] == scene) & (diag["focal"] == focal)
                selected_rows = np.flatnonzero(mask)
                for name in ("terminal_score", "terminal_conversions", "terminal_violation"):
                    values = data[name][selected_rows, base[selected_rows]]
                    require(np.all(values == values[0]), "Feedback branch differs across intervention ticks")
    legal = legal_candidates(base)
    require(np.all(legal.sum(axis=1) > 0), "No legal nonfeedback candidate")
    fold = (diag["market"] - 1001) // 4
    require(set(fold) == set(range(4)) and all(np.count_nonzero(fold == f) == 360 for f in range(4)),
            "Fold coverage failed")
    return {
        "contexts": n, "markets": list(MARKETS), "fold_contexts": [360] * 4,
        "observation_lengths_by_tick": {str(t): sorted(np.unique(data["lengths"][diag["tick"] == t]).tolist()) for t in TICKS},
        "maximum_score_reconstruction_error": float(np.max(np.abs(recomputed - data["terminal_score"]))),
        "maximum_conversion_reconstruction_error": float(np.max(np.abs(conv - data["terminal_conversions"]))),
        "legal_nonfeedback_candidate_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(legal.sum(axis=1), return_counts=True), strict=True)},
        "exact_advantage_class_counts": [int(np.count_nonzero(advantage[legal] < 0)),
                                         int(np.count_nonzero(advantage[legal] == 0)),
                                         int(np.count_nonzero(advantage[legal] > 0))],
        "finite_features": True, "raw_and_physical_advantage_signs_identical": True,
        "encoder_loaded": False, "validation_loaded": False,
        "actual_outcome_unit": "one intervention, then original fixed feedback",
    }


def targets(data):
    physical = (data["cpa"] / data["budget"])[:, None]
    utility = data["terminal_score"] * physical
    conversion = data["terminal_conversions"] * physical
    rows, base = np.arange(len(utility)), data["baseline_index"]
    advantage = (data["terminal_score"] - data["terminal_score"][rows, base, None]) * physical
    delta_conv = (data["terminal_conversions"] - data["terminal_conversions"][rows, base, None]) * physical
    advantage[rows, base], delta_conv[rows, base] = 0, 0
    return utility, advantage, conversion, delta_conv


def preflight():
    started = time.perf_counter()
    frozen, focals = identities()
    with np.load(CACHE / "train.npz", allow_pickle=False) as cache:
        data = {name: cache[name] for name in cache.files}
    checks = validate_data(data, frozen["training_schema"], focals)
    public = [data[k] for k in ("states", "steps", "lengths", "baseline_index")]
    for family in (0, 1):
        features(*public, family)
    require(time.perf_counter() - started <= PREFLIGHT_LIMIT, "S0 preflight time budget exceeded")
    frozen["protocol"] = protocol()
    frozen["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                         "torch": torch.__version__, "platform": platform.platform()}
    return data, frozen, checks | {
        "preflight_seconds": time.perf_counter() - started,
        "source_sha256": frozen["sources"][str(Path(__file__).resolve().relative_to(ROOT))],
        "protocol_sha256": fingerprint(frozen["protocol"]), "freeze_identity": fingerprint(frozen),
        "read_only": True, "training_fitted": False,
    }


def choose(advantage, baseline_conv, delta_conv, risk, baseline):
    """Prediction-only controller; there is intentionally no outcome argument."""
    rows = np.arange(len(baseline))
    eligible = (legal_candidates(baseline) & (advantage > 0)
                & (delta_conv >= -.03 * np.maximum(baseline_conv, 0)[:, None])
                & (risk <= risk[rows, baseline, None]))
    candidate = np.where(eligible, advantage, -np.inf).argmax(axis=1)
    return np.where(eligible.any(axis=1), candidate, baseline).astype(np.int64)


class StudyStop(Exception):
    """Resource/user interruption: an incomplete comparison, not a result."""


class Resources:
    def __init__(self, output):
        self.output = output
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        self.records = []
        self.last_scan = -float("inf")
        self.bytes = 0
        self.probe_done = False
        self.probe = None

    def snapshot(self):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return {"wall_seconds": time.perf_counter() - self.started,
                "cpu_seconds": time.process_time() - self.cpu_started,
                "peak_memory_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
                "free_disk_bytes": shutil.disk_usage(self.output).free,
                "artifact_bytes": self.bytes}

    def save(self):
        write_json(self.output / "resources.json", {"started_utc": self.started_utc,
                   "limits": protocol()["resources"], "records": self.records,
                   "throughput_probe": self.probe, "current": self.snapshot()})

    def check(self, force=False):
        now = time.perf_counter()
        if force or now - self.last_scan >= 30:
            self.bytes = sum(p.stat().st_size for p in self.output.rglob("*") if p.is_file())
            self.last_scan = now
        state = self.snapshot()
        for condition, message in (
            (state["wall_seconds"] <= WALL_LIMIT, "S1 two-hour wall budget exceeded"),
            (state["peak_memory_bytes"] <= MAX_MEMORY, "Process memory budget exceeded"),
            (state["free_disk_bytes"] >= MIN_FREE, "Less than10GiB free disk"),
            (self.bytes <= MAX_ARTIFACT, "New artifacts exceed2GiB"),
        ):
            if not condition:
                raise StudyStop(message)

    def start(self, name, **details):
        self.check()
        record = {"task": name, "status": "running", "start": self.snapshot(), **details}
        self.records.append(record)
        self.save()
        return record

    def finish(self, record, **details):
        record.update(status="complete", end=self.snapshot(), **details)
        self.check()
        self.save()


def mlp(input_size, output_size):
    return nn.Sequential(nn.Linear(input_size, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH),
                         nn.SiLU(), nn.Linear(WIDTH, output_size))


def state_digest(model):
    result = hashlib.sha256()
    for key, value in model.state_dict().items():
        result.update(key.encode())
        result.update(value.numpy().tobytes())
    return result.hexdigest()


def neural_fit(name, train_x, held_x, loss_function, output_size, ledger, linear=False):
    torch.manual_seed(SEED)
    model = nn.Linear(train_x.shape[1], output_size) if linear else mlp(train_x.shape[1], output_size)
    batches = np.random.default_rng(SEED).integers(0, len(train_x), (UPDATES, BATCH))
    record = ledger.start(name, parameters=sum(p.numel() for p in model.parameters()),
                          initialization_sha256=state_digest(model),
                          batch_stream_sha256=hashlib.sha256(batches.tobytes()).hexdigest(),
                          updates_completed=0)
    x = torch.tensor(train_x, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    begin = time.perf_counter()
    for step, batch in enumerate(batches, 1):
        indices = torch.as_tensor(batch)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_function(model(x[indices]), indices)
        require(torch.isfinite(loss).item(), f"Nonfinite loss: {name}")
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            record["updates_completed"] = step
            record["last_training_loss"] = float(loss.detach())
            ledger.check()
            if not ledger.probe_done:
                ledger.probe = {"task": name, "updates": step, "seconds": time.perf_counter() - begin,
                                "is_part_of_existing_fit": True, "restarted": False,
                                "projected_total_neural_seconds": (time.perf_counter() - begin) * (20 * UPDATES) / step,
                                "resources": ledger.snapshot()}
                ledger.probe_done = True
                ledger.save()
                print(json.dumps({"throughput_probe": ledger.probe}), flush=True)
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite fitted parameters")
    with torch.no_grad():
        prediction = model(torch.tensor(held_x, dtype=torch.float32)).numpy().astype(float)
    ledger.finish(record, updates_completed=UPDATES, last_training_loss=float(loss.detach()))
    return prediction, {k: record[k] for k in ("parameters", "initialization_sha256", "batch_stream_sha256")}


def normalize(train, held, training_markets):
    mean, std = train.mean(axis=0), np.maximum(train.std(axis=0), 1e-6)
    return (train - mean) / std, (held - mean) / std, {
        "mean": mean.tolist(), "std": std.tolist(), "training_markets": sorted(np.unique(training_markets).tolist())}


def ridge(train_x, held_x, target, row_weight, ledger, name):
    record = ledger.start(name, solver="float64 ridge lambda1", updates_completed=0)
    x = np.column_stack((train_x, np.ones(len(train_x))))
    held = np.column_stack((held_x, np.ones(len(held_x))))
    penalty = np.eye(x.shape[1])
    penalty[-1, -1] = 0
    weights = row_weight * len(train_x)
    coefficients = np.linalg.solve(x.T @ (weights[:, None] * x) + penalty,
                                   x.T @ (weights[:, None] * target))
    result = held @ coefficients
    require(np.isfinite(result).all(), "Nonfinite ridge predictions")
    ledger.finish(record)
    return result


def auxiliary(train_x, held_x, conversion, delta_conv, violation, positive, baseline,
              train_markets, legal, scale, ledger, name):
    rows = np.arange(len(train_x))
    base_target = torch.tensor(conversion[rows, baseline] / scale, dtype=torch.float32)
    delta_target = torch.tensor(delta_conv / scale, dtype=torch.float32)
    risk_target = torch.tensor(violation, dtype=torch.float32)
    positive_target = torch.tensor(positive, dtype=torch.float32)
    row_weight = torch.tensor(context_weights(train_markets) * len(train_x), dtype=torch.float32)
    candidate_weight = torch.tensor(legal / legal.sum(axis=1, keepdims=True), dtype=torch.float32)

    def loss(raw, ix):
        baseline_loss = (raw[:, 0] - base_target[ix]) ** 2
        delta_loss = ((raw[:, 1:12] - delta_target[ix]) ** 2).mean(dim=1)
        risk_loss = F.binary_cross_entropy_with_logits(raw[:, 12:23], risk_target[ix], reduction="none").mean(dim=1)
        positive_loss = (F.binary_cross_entropy_with_logits(raw[:, 23:34], positive_target[ix], reduction="none") * candidate_weight[ix]).sum(dim=1)
        return ((baseline_loss + delta_loss + risk_loss + positive_loss) * row_weight[ix]).mean()

    raw, metadata = neural_fit(name, train_x, held_x, loss, 34, ledger)
    return {"baseline_conv": np.maximum(raw[:, 0] * scale, 0),
            "delta_conv": raw[:, 1:12] * scale,
            "violation": torch.sigmoid(torch.tensor(raw[:, 12:23])).numpy(),
            "positive": torch.sigmoid(torch.tensor(raw[:, 23:34])).numpy()}, metadata


def regression_mlp(name, train_x, held_x, target, row_weight, ledger):
    y = torch.tensor(target, dtype=torch.float32)
    weight = torch.tensor(row_weight * len(train_x), dtype=torch.float32)

    def loss(raw, ix):
        return (((raw - y[ix]) ** 2).mean(dim=1) * weight[ix]).mean()

    return neural_fit(name, train_x, held_x, loss, 11, ledger)


def three_class(train_x, held_x, advantage, train_markets, legal, ledger, name):
    classes = (advantage > 0).astype(np.int64) + 1
    classes[advantage < 0] = 0
    y = torch.tensor(classes)
    row_weight = torch.tensor(context_weights(train_markets) * len(train_x), dtype=torch.float32)
    candidate_weight = torch.tensor(legal / legal.sum(axis=1, keepdims=True), dtype=torch.float32)

    def loss(raw, ix):
        element = F.cross_entropy(raw.reshape(-1, 3), y[ix].reshape(-1), reduction="none").reshape(-1, 11)
        return ((element * candidate_weight[ix]).sum(dim=1) * row_weight[ix]).mean()

    raw, metadata = neural_fit(name, train_x, held_x, loss, 33, ledger, linear=True)
    probabilities = torch.softmax(torch.tensor(raw.reshape(-1, 11, 3)), dim=-1).numpy()
    magnitudes, mass = np.zeros((11, 3)), np.zeros((11, 3))
    weights = candidate_weights(train_markets, legal)
    for action in range(11):
        for cls in range(3):
            selected = classes[:, action] == cls
            mass[action, cls] = weights[selected, action].sum()
            if cls != 1 and mass[action, cls] > 0:
                magnitudes[action, cls] = np.sum(weights[selected, action] * advantage[selected, action]) / mass[action, cls]
    return (probabilities * magnitudes[None]).sum(axis=2), probabilities, metadata | {
        "magnitudes": magnitudes.tolist(), "class_mass": mass.tolist(),
        "absent_class_magnitude": 0, "smoothing": "none"}


def knn(train_x, held_x, advantage, training_indices, ledger, name):
    record = ledger.start(name, neighbors=32, updates_completed=0)
    # Stable sort preserves the original (global context index) order for exact ties.
    squared = np.maximum((held_x ** 2).sum(axis=1)[:, None]
                         + (train_x ** 2).sum(axis=1)[None] - 2 * held_x @ train_x.T, 0)
    nearest = np.argsort(squared, axis=1, kind="stable")[:, :32]
    sample = advantage[nearest]
    probability = np.stack(((sample < 0).mean(axis=1), (sample == 0).mean(axis=1),
                            (sample > 0).mean(axis=1)), axis=-1)
    result = sample.mean(axis=1)
    ledger.finish(record)
    return result, probability, training_indices[nearest]


def fit_fold(data, fold, output, ledger):
    diag, base = data["diagnostics"], data["baseline_index"]
    held = np.flatnonzero((diag["market"] - 1001) // 4 == fold)
    train = np.flatnonzero((diag["market"] - 1001) // 4 != fold)
    train_markets, held_markets = diag["market"][train], diag["market"][held]
    require(set(train_markets).isdisjoint(held_markets) and len(set(train_markets)) == 12,
            "Training/held market leakage")
    utility, advantage, conversion, delta_conv = targets(data)
    legal = legal_candidates(base)
    score_scale = max(float(np.sqrt(np.mean(utility[train] ** 2))), .01)
    conv_scale = max(float(np.sqrt(np.mean(conversion[train] ** 2))), .01)
    public = [data[k] for k in ("states", "steps", "lengths", "baseline_index")]
    x, metadata, guards = {}, {}, {}
    for family in (0, 1):
        raw = features(*public, family)
        train_x, held_x, normalizer = normalize(raw[train], raw[held], train_markets)
        x[family] = train_x, held_x
        prediction, fitted = auxiliary(train_x, held_x, conversion[train], delta_conv[train],
                                      data["terminal_violation"][train], advantage[train] > 0,
                                      base[train], train_markets, legal[train], conv_scale, ledger,
                                      f"fold{fold}/F{family}/auxiliary")
        prediction["delta_conv"][np.arange(len(held)), base[held]] = 0
        prediction["positive"][np.arange(len(held)), base[held]] = 0
        guards[family] = prediction
        metadata[f"F{family}"] = {"normalizer": normalizer, "auxiliary": fitted,
            "positive_prior": positive_prior(advantage[train] > 0, train_markets, legal[train])}
    row_weight = context_weights(train_markets)
    predictions = np.zeros((7, len(held), 11))
    for index, family in ((0, 0), (1, 1)):
        predictions[index] = ridge(*x[family], advantage[train] / score_scale,
                                    row_weight, ledger, f"fold{fold}/{CONFIGS[index]}") * score_scale
    predictions[2], class_probability, class_metadata = three_class(
        *x[1], advantage[train], train_markets, legal[train], ledger, f"fold{fold}/{CONFIGS[2]}")
    direct, direct_metadata = regression_mlp(f"fold{fold}/{CONFIGS[3]}", *x[1],
                                            advantage[train] / score_scale, row_weight, ledger)
    absolute, absolute_metadata = regression_mlp(f"fold{fold}/{CONFIGS[4]}", *x[1],
                                                utility[train] / score_scale, row_weight, ledger)
    require(direct_metadata == absolute_metadata, "Direct/absolute capacity/init/batch mismatch")
    predictions[3] = direct * score_scale
    predictions[4] = (absolute - absolute[np.arange(len(held)), base[held], None]) * score_scale
    predictions[5], knn_probability, neighbors = knn(*x[1], advantage[train], train, ledger,
                                                   f"fold{fold}/{CONFIGS[5]}")
    permutation = np.random.default_rng(SEED).permutation(len(train))
    predictions[6] = ridge(*x[1], advantage[train][permutation] / score_scale, row_weight,
                            ledger, f"fold{fold}/{CONFIGS[6]}") * score_scale
    predictions[:, np.arange(len(held)), base[held]] = 0
    for probability in (class_probability, knn_probability):
        probability[np.arange(len(held)), base[held]] = [0, 1, 0]
        require(np.isfinite(probability).all() and np.allclose(probability.sum(axis=-1), 1)
                and np.all(probability >= 0), "Invalid three-class probabilities")
    selected = np.stack([choose(predictions[i], guards[f]["baseline_conv"], guards[f]["delta_conv"],
                                guards[f]["violation"], base[held]) for i, f in enumerate(FAMILIES)])
    require(np.isfinite(predictions).all(), "Nonfinite ranker forecast")
    payload = {
        "configuration_names": np.array(CONFIGS), "feature_families": np.array(FAMILIES),
        "context_index": held, "fold_id": np.full(len(held), fold), "training_context_index": train,
        "diagnostics": diag[held], "baseline_index": base[held], "legal_candidates": legal[held],
        "delta_prediction": predictions, "selected": selected,
        "guard_baseline_conv": np.stack([guards[f]["baseline_conv"] for f in (0, 1)]),
        "guard_delta_conv": np.stack([guards[f]["delta_conv"] for f in (0, 1)]),
        "guard_violation": np.stack([guards[f]["violation"] for f in (0, 1)]),
        "guard_positive": np.stack([guards[f]["positive"] for f in (0, 1)]),
        "prior_positive": np.array([metadata[f"F{f}"]["positive_prior"] for f in (0, 1)]),
        "class3_probability": class_probability, "knn_probability": knn_probability,
        "knn_training_context_index": neighbors,
        "actual_score": data["terminal_score"][held],
        "actual_conversions": data["terminal_conversions"][held],
        "actual_violation": data["terminal_violation"][held],
        "physical_scale": data["cpa"][held] / data["budget"][held],
        "actual_cost": (data["spent"][:, None] + data["endpoints"][:, :, 0] * data["budget"][:, None])[held],
    }
    path = output / f"fold{fold}-predictions.npz"
    np.savez_compressed(path, **payload)
    write_json(output / f"fold{fold}-fit.json", {
        "fold": fold, "training_markets": sorted(np.unique(train_markets).tolist()),
        "held_markets": sorted(np.unique(held_markets).tolist()),
        "score_scale": score_scale, "conversion_scale": conv_scale,
        "feature_families": metadata, "three_class": class_metadata,
        "direct": direct_metadata, "absolute": absolute_metadata,
        "shuffled_target_source_context_index": train[permutation].tolist(),
        "all_training_objects_exclude_held_markets": True,
    })
    ledger.check(force=True)
    return payload


def ratio(actual, baseline):
    return float(actual / baseline) if baseline != 0 else None


def scene_metrics(diag, baseline, selected, actual, markets):
    by_scene, market_rows = {}, []
    for si, scene in enumerate(SCENES):
        per_market = []
        for market in markets:
            mask = (diag["market"] == market) & (diag["scene_index"] == si)
            require(int(mask.sum()) == 18, "Missing market-scene evaluation contexts")
            rows = np.flatnonzero(mask)
            record = {"market": int(market), "scene": scene,
                      "switch_fraction": float(np.mean(selected[mask] != baseline[mask]))}
            for name, outcomes in actual.items():
                chosen = float(outcomes[rows, selected[rows]].mean())
                reference = float(outcomes[rows, baseline[rows]].mean())
                record[name] = {"selected": chosen, "feedback": reference,
                                "difference": chosen - reference, "ratio": ratio(chosen, reference)}
            per_market.append(record)
        cell = {"markets": len(per_market), "switch_fraction": float(np.mean([r["switch_fraction"] for r in per_market]))}
        for name in actual:
            chosen = float(np.mean([r[name]["selected"] for r in per_market]))
            reference = float(np.mean([r[name]["feedback"] for r in per_market]))
            cell[name] = {"selected": chosen, "feedback": reference,
                          "difference": chosen - reference, "ratio": ratio(chosen, reference)}
        by_scene[scene] = cell
        market_rows.extend(per_market)
    ratios = [by_scene[scene]["score"]["ratio"] for scene in PRIMARY]
    headline = None if any(x is None for x in ratios) else float(np.mean(ratios))
    return {"scenes": by_scene, "headline_score_ratio": headline, "markets": market_rows}


def weighted_average_precision(probability, positive, markets, legal):
    if np.any(legal.sum(axis=1) == 0):
        return None
    weights = candidate_weights(markets, legal)[legal]
    labels, scores = positive[legal], probability[legal]
    order = np.argsort(-scores, kind="stable")
    labels, scores, weights = labels[order], scores[order], weights[order]
    mass = float(np.sum(labels * weights))
    if mass == 0:
        return None
    # Evaluate whole equal-score groups, so tied probabilities cannot exploit row order.
    ends = np.r_[np.flatnonzero(np.diff(scores) != 0), len(scores) - 1]
    tp, total = np.cumsum(labels * weights)[ends], np.cumsum(weights)[ends]
    return float(np.sum((tp - np.r_[0, tp[:-1]]) * (tp / total)) / mass)


def summarize(folds):
    diag = np.concatenate([p["diagnostics"] for p in folds])
    base = np.concatenate([p["baseline_index"] for p in folds])
    legal = np.concatenate([p["legal_candidates"] for p in folds])
    actual = {name: np.concatenate([p[f"actual_{key}"] for p in folds])
              for name, key in (("score", "score"), ("conversions", "conversions"), ("violation", "violation"))}
    rows = np.arange(len(base))
    delta_score = actual["score"] - actual["score"][rows, base, None]
    positive = delta_score > 0
    result = {}
    for index, name in enumerate(CONFIGS):
        selected = np.concatenate([p["selected"][index] for p in folds])
        probability = np.concatenate([p["guard_positive"][FAMILIES[index]] for p in folds])
        prior = np.concatenate([np.full((len(p["baseline_index"]), 11), p["prior_positive"][FAMILIES[index]]) for p in folds])
        brier = weighted_brier(probability, positive, diag["market"], legal)
        prior_brier = weighted_brier(prior, positive, diag["market"], legal)
        improvement = (None if brier["value"] is None or prior_brier["value"] in (None, 0)
                       else 1 - brier["value"] / prior_brier["value"])
        metrics = scene_metrics(diag, base, selected, actual, MARKETS)
        fold_metrics = {}
        for fold, payload in enumerate(folds):
            fold_actual = {n: payload[f"actual_{key}"] for n, key in
                           (("score", "score"), ("conversions", "conversions"), ("violation", "violation"))}
            fold_metrics[str(fold)] = scene_metrics(payload["diagnostics"], payload["baseline_index"],
                                                    payload["selected"][index], fold_actual,
                                                    MARKETS[fold * 4:(fold + 1) * 4])
        positive_folds = sum(m["headline_score_ratio"] is not None and m["headline_score_ratio"] > 1 for m in fold_metrics.values())
        switches = int(np.count_nonzero(selected != base))
        scene_gate = {}
        for scene in PRIMARY:
            cell = metrics["scenes"][scene]
            scene_gate[scene] = {
                "score": cell["score"]["ratio"] is not None and cell["score"]["ratio"] >= .98,
                "conversions": cell["conversions"]["ratio"] is not None and cell["conversions"]["ratio"] >= .97,
                "violation": cell["violation"]["difference"] <= 0,
            }
        flags = {"positive_brier": improvement is not None and improvement >= .05,
                 "headline": metrics["headline_score_ratio"] is not None and metrics["headline_score_ratio"] >= 1.005,
                 "fold_direction": positive_folds >= 3, "nonfeedback_choice": switches > 0,
                 "primary_scene_guards": all(all(g.values()) for g in scene_gate.values())}
        result[name] = metrics | {
            "folds": fold_metrics, "switches": switches, "switch_fraction": switches / len(base),
            "positive_folds": positive_folds, "calibration": {
                "positive_brier": brier, "prior_brier": prior_brier,
                "relative_improvement": improvement,
                "positive_average_precision": weighted_average_precision(probability, positive, diag["market"], legal),
                "attribution": f"shared F{FAMILIES[index]} positive head"},
            "g1_flags": flags, "primary_scene_flags": scene_gate,
            "eligible": index < 6, "paired_target": index in DIRECT,
            "g1_conjunction": all(flags.values()),
            "g1_pass": index < 6 and all(flags.values()),
        }
    class_diagnostics = {}
    weights = candidate_weights(diag["market"], legal)
    for label, mask in (("negative", delta_score < 0), ("zero", delta_score == 0), ("positive", delta_score > 0)):
        mass = float(np.sum(weights * mask))
        class_diagnostics[label] = {"weighted_mass": mass,
            "mean_absolute_score_advantage": None if mass == 0 else float(np.sum(weights * mask * np.abs(delta_score)) / mass)}
    ranker_calibration = {}
    for key in ("class3_probability", "knn_probability"):
        probability = np.concatenate([p[key] for p in folds])
        labels = np.stack((delta_score < 0, delta_score == 0, delta_score > 0), axis=-1)
        ranker_calibration[key] = {
            "positive_brier": weighted_brier(probability[:, :, 2], positive, diag["market"], legal),
            "class_brier": [float(np.sum(weights * (probability[:, :, c] - labels[:, :, c]) ** 2)) for c in range(3)],
        }
    direct, absolute = result[CONFIGS[3]], result[CONFIGS[4]]
    a, b = direct["headline_score_ratio"], absolute["headline_score_ratio"]
    return {
        "status": "complete", "claim_scope": protocol()["scope"], "configurations": result,
        "package_signal_pass": any(result[name]["g1_pass"] for name in CONFIGS[:6]),
        "direct_config_pass": any(result[CONFIGS[i]]["g1_pass"] for i in DIRECT),
        "absolute_only_pass": result[CONFIGS[4]]["g1_pass"] and not any(result[CONFIGS[i]]["g1_pass"] for i in DIRECT),
        "direct_vs_absolute": {"headline_ratio_difference": None if a is None or b is None else a - b,
            "scene_score_difference": {s: direct["scenes"][s]["score"]["selected"] - absolute["scenes"][s]["score"]["selected"] for s in SCENES},
            "direct_pass": direct["g1_pass"], "absolute_pass": absolute["g1_pass"]},
        "class_diagnostics": class_diagnostics, "ranker_calibration_diagnostics": ranker_calibration,
        "feedback": scene_metrics(diag, base, base, actual, MARKETS),
        "independent_confirmation": False, "full_policy_rollout": False,
        "next_action": "distinct future preregistration only if gate passes; never automatically open new data",
    }


def seal(output):
    files = {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*"))
             if path.is_file() and path.name != "seal.json"}
    record = {"files": files, "content_identity": fingerprint(files), "sealed_utc": utc()}
    write_json(output / "seal.json", record)
    require(all(digest(output / name) == sha for name, sha in files.items()), "Seal reread failed")
    require(read_json(output / "seal.json")["content_identity"] == fingerprint(files), "Seal identity failed")
    return record["content_identity"]


def run(output, expected_freeze):
    output = Path(output).resolve()
    require(not output.exists(), "Output already exists; a new run must use a fresh directory")
    require(not (output == ROOT or ROOT.is_relative_to(output)), "Output cannot own a repository ancestor")
    data, frozen, checks = preflight()
    require(fingerprint(frozen) == expected_freeze, "Preflight freeze identity changed; review/freeze before run")
    require(shutil.disk_usage(output.parent).free >= MIN_FREE, "Less than10GiB free before run")
    with Path(__file__).open("rb") as source_lock:
        fcntl.flock(source_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(parents=True, exist_ok=False)
        ledger = Resources(output)
        ledger.started_utc = utc()
        protocol_path = output / "protocol.json"
        write_json(protocol_path, frozen["protocol"])
        write_json(output / "manifest.json", {"freeze_identity": expected_freeze, "frozen": frozen,
                   "created_utc": utc(), "frozen_before_first_update": True})
        write_json(output / "preflight.json", checks)
        write_json(output / "summary.json", {"status": "running", "package_signal_pass": None,
                   "direct_config_pass": None, "scientific_outcome_evaluable": False})
        ledger.save()
        folds, summary, exit_code = [], None, 0
        previous_handlers = {}

        def interrupt(signum, _frame):
            raise StudyStop(f"Interrupted by signal{signum}")

        for signum in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[signum] = signal.signal(signum, interrupt)
        try:
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            torch.use_deterministic_algorithms(True)
            for fold in range(4):
                # Lightweight source guard at fold boundaries; raw tape hashes are checked only at freeze/end.
                require(all(digest(ROOT / name) == sha for name, sha in frozen["sources"].items()),
                        "Source mutated after freeze")
                folds.append(fit_fold(data, fold, output, ledger))
            ledger.check(force=True)
            summary = summarize(folds)
            summary["scientific_outcome_evaluable"] = True
        except StudyStop as error:
            exit_code = 2
            summary = {"status": "incomplete", "reason": str(error), "completed_folds": len(folds),
                       "package_signal_pass": None, "direct_config_pass": None,
                       "scientific_outcome_evaluable": False}
        except Exception as error:
            exit_code = 3
            summary = {"status": "invalid", "reason": f"{type(error).__name__}: {error}",
                       "completed_folds": len(folds), "package_signal_pass": None,
                       "direct_config_pass": None, "scientific_outcome_evaluable": False}
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            # Even unsuccessful runs retain before/after identities and a seal.
            source_after = hashes_after_run(frozen["sources"])
            inputs_after = hashes_after_run(frozen["inputs"])
            try:
                protocol_unchanged = fingerprint(read_json(protocol_path)) == fingerprint(frozen["protocol"])
            except (OSError, ValueError):
                protocol_unchanged = False
            identities_unchanged = (source_after == frozen["sources"] and inputs_after == frozen["inputs"]
                                    and protocol_unchanged)
            if not identities_unchanged:
                exit_code = 3
                summary = {"status": "invalid", "reason": "Source/protocol/input mutation after freeze",
                           "package_signal_pass": None, "direct_config_pass": None,
                           "scientific_outcome_evaluable": False}
            if summary["status"] == "complete":
                try:
                    ledger.check(force=True)
                except StudyStop as error:
                    exit_code = 2
                    summary = {"status": "incomplete", "reason": str(error),
                               "completed_folds": len(folds), "package_signal_pass": None,
                               "direct_config_pass": None, "scientific_outcome_evaluable": False}
            for record in ledger.records:
                if record["status"] == "running":
                    record.update(status=summary["status"], end=ledger.snapshot())
            ledger.save()
            summary["freeze_identity"] = expected_freeze
            write_json(output / "summary.json", summary)
            write_json(output / "execution.json", {
                "status": summary["status"], "completed_utc": utc(), "resources": ledger.snapshot(),
                "source_hashes_before": frozen["sources"], "source_hashes_after": source_after,
                "input_hashes_after": inputs_after, "identities_unchanged": identities_unchanged,
                "protected_ten_sources_unchanged": all(source_after[k] == v for k, v in frozen["protected_sources"].items()),
                "power_configuration_changed": False, "process_wake_assertion": "owned externally by launcher, if used",
            })
            identity = seal(output)
            print(json.dumps({"output": str(output), "status": summary["status"],
                              "seal_identity": identity, "package_signal_pass": summary["package_signal_pass"]}), flush=True)
        return exit_code


def self_check():
    """Small deterministic synthetic checks; no cache reads or model fitting."""
    checks = []
    states, steps = np.zeros((2, 24, 13), dtype=np.float32), np.zeros((2, 24), dtype=np.int64)
    steps[1, :9] = np.arange(9)
    length, base = np.array([1, 9]), np.array([4, 4])
    f = features(states, steps, length, base, 1)
    require(f[0, -1] == 0 and f[1, -1] == 8, "tick0/no-win settlement semantics failed")
    altered, altered_steps = states.copy(), steps.copy()
    altered[0, 1:] = 999
    altered_steps[0, 1:] = 47
    altered[1, 9:] = 999
    require(np.array_equal(f, features(altered, altered_steps, length, base, 1)), "Padding/future affected features")
    states[1, 8, 12] = .5
    require(features(states, steps, length, base, 1)[1, -1] == 0, "Visible settlement did not affect no-win count")
    checks += ["right padding/future invariance", "tick0 has no settlement", "no-win cap8 and positive-win reset"]
    n = 8
    base = np.full(n, 4)
    advantage, delta_conv, risk = np.zeros((n, 11)), np.zeros((n, 11)), np.full((n, 11), .2)
    baseline_conv = np.ones(n)
    advantage[:, 5] = 1
    advantage[1, 5] = -1
    delta_conv[2, 5] = -.04
    risk[3, 5] = .3
    baseline_conv[4] = 0
    delta_conv[4, 5] = -.0001
    advantage[5, 6] = 1
    advantage[6, 10] = 100  # out of radius, never selectable
    advantage[6, 5] = 0
    advantage[7, 5] = 0
    selected = choose(advantage, baseline_conv, delta_conv, risk, base)
    require(np.array_equal(selected, [5, 4, 4, 4, 4, 5, 4, 4]), "Prediction guard/tie/fallback failed")
    checks += ["positive/negative/conversion/risk/zero-baseline guards", "fixed-radius and exact tie ordering", "full fallback"]
    synthetic = {"cpa": np.ones(2), "budget": np.ones(2), "baseline_index": np.array([0, 1]),
                 "terminal_score": np.array([[2, 4, 1], [0, 3, 3]], float),
                 "terminal_conversions": np.array([[2, 4, 1], [0, 3, 3]])}
    utility, advantage, _, conv_delta = targets(synthetic)
    require(np.array_equal(advantage, [[0, 2, -1], [-3, 0, 0]])
            and np.array_equal(advantage, conv_delta), "Paired sign/zero reconstruction failed")
    absolute = utility - utility[np.arange(2), synthetic["baseline_index"], None]
    require(np.array_equal(advantage, absolute), "Absolute subtraction identity failed")
    require(ratio(1, 0) is None and ratio(0, 0) is None, "Zero denominator must be undefined")
    checks += ["exact positive/negative/zero paired labels", "absolute/direct identity", "undefined zero denominator"]
    markets = np.array([1, 1, 2])
    legal = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], bool)
    y = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]], float)
    require(np.isclose(positive_prior(y, markets, legal), .25 * 1 + .25 * .5 + .5 / 3), "Training prior weights failed")
    require(np.isclose(weighted_brier(np.zeros_like(y), y, markets, legal)["value"], .25 + .125 + .5 / 3), "Brier hierarchy failed")
    legal[0] = False
    require(weighted_brier(y, y, markets, legal)["value"] is None, "Empty contexts silently dropped")
    checks += ["candidate/context/market equal weighting", "empty candidate set represented"]
    require(len(feature_names(0)) == 24 and len(feature_names(1)) == 45, "Feature allowlist changed")
    torch.manual_seed(SEED)
    direct = mlp(45, 11)
    torch.manual_seed(SEED)
    absolute = mlp(45, 11)
    require(state_digest(direct) == state_digest(absolute), "Matched initialization failed")
    require(CONFIGS[-1].endswith("control") and len(CONFIGS[:6]) == 6, "Configuration/control contract failed")
    checks += ["matched MLP capacity/initialization", "six eligible configurations plus ineligible control"]
    return {"status": "passed", "checks": checks, "check_count": len(checks),
            "cache_read": False, "training_fitted": False, "output_created": False,
            "protocol_sha256": fingerprint(protocol())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight", help="read-only train/input/causality checks; print freeze identity")
    commands.add_parser("self-check", help="synthetic controller/feature/aggregation checks; no data or fits")
    run_parser = commands.add_parser("run", help="all four folds and all six fixed configurations, after freeze")
    run_parser.add_argument("--output", type=Path, default=OUTPUT)
    run_parser.add_argument("--expect-freeze", required=True, help="reviewed preflight freeze_identity")
    args = parser.parse_args()
    if args.command == "self-check":
        print(json.dumps(self_check(), indent=2, sort_keys=True))
    elif args.command == "preflight":
        _, frozen, checks = preflight()
        print(json.dumps({"checks": checks, "frozen": frozen}, indent=2, sort_keys=True))
    else:
        return run(args.output, args.expect_freeze)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
