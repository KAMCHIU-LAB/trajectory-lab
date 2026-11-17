"""Exact public-kernel response tapes, hidden from the evaluated policies."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import os
import random
import time
from pathlib import Path

import numpy as np

from trajectorylab.utils import fingerprint, read_json, write_json

ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "third_party" / "auctionnet"
REVISION = "c20de631a40e24669d1d6a8e2acefd8f492f172d"
CPA = np.array(
    [
        100,
        70,
        90,
        110,
        60,
        130,
        120,
        80,
        70,
        130,
        100,
        110,
        120,
        90,
        60,
        80,
        130,
        80,
        110,
        100,
        90,
        120,
        60,
        70,
        120,
        60,
        90,
        70,
        100,
        110,
        130,
        80,
        120,
        90,
        70,
        80,
        100,
        110,
        60,
        130,
        90,
        100,
        110,
        80,
        60,
        70,
        130,
        120,
    ]
)
BUDGET = np.array(
    [
        2900,
        4350,
        3000,
        2400,
        4800,
        2000,
        2050,
        3500,
        4600,
        2000,
        2800,
        2350,
        2050,
        2900,
        4750,
        3450,
        2000,
        3500,
        2200,
        2700,
        3100,
        2100,
        4850,
        4100,
        2000,
        4800,
        3050,
        4250,
        2850,
        2250,
        2000,
        3900,
        2000,
        3250,
        4450,
        3550,
        2700,
        2100,
        4650,
        2000,
        3400,
        2650,
        2300,
        4100,
        4800,
        4450,
        2000,
        2050,
    ]
)


def _load(name: str, filename: str):
    numpy_state, python_state = np.random.get_state(), random.getstate()
    handlers = list(logging.getLogger().handlers)
    level = logging.getLogger().level
    options = np.get_printoptions()
    try:
        spec = importlib.util.spec_from_file_location(name, VENDOR / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
        logging.getLogger().handlers[:] = handlers
        logging.getLogger().setLevel(level)
        np.set_printoptions(**options)


def source_identity() -> dict:
    files = [Path(__file__), VENDOR / "NeurIPSPvGen.py", VENDOR / "BiddingEnv.py"]
    return {
        "upstream_revision": REVISION,
        "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


def effective_actions(study: dict) -> np.ndarray:
    return np.array(sorted({round(a * b, 8) for a in study["actions"] for b in study["biases"]}))


def kernel_equivalence() -> dict:
    """Runtime audit of the exact caching optimization, independent of training."""
    Env = _load("auctionnet_env_audit", "BiddingEnv.py").BiddingEnv
    rng = np.random.default_rng(90210)
    values = rng.uniform(0.0001, 0.002, (128, 48))
    sigmas = values * 0.1
    env = Env(reserve_pv_price=0.0001)
    env.reset(917)
    cached = _cached_env(Env, 917)
    cached.prepare(values, sigmas)
    checked = 0
    for factor in (0.0, 0.7, 2.1):
        bids = values * np.linspace(12, 140, 48)
        bids[:, 0] *= factor
        expected = env.simulate_ad_bidding(values, sigmas, bids)
        actual = cached.simulate_ad_bidding(values, sigmas, bids)
        if any(not np.array_equal(a, b) for a, b in zip(actual, expected, strict=True)):
            raise RuntimeError("Cached auction kernel differs from public kernel")
        checked += 1
    return {"exact_array_equality": True, "bid_cases": checked, "outputs_per_case": 7}


def _cached_env(Env, market_id: int):
    class CachedEnv(Env):
        def prepare(self, values, sigmas):
            self.cached_values = super()._generate_values_matrix(values, sigmas)
            self.cached_conversions = super()._calculate_conversion_action(
                self.cached_values, np.ones_like(values)
            )

        def _generate_values_matrix(self, values, sigmas):
            return self.cached_values

        def _calculate_conversion_action(self, values, exposure):
            return self.cached_conversions * exposure

    env = CachedEnv(reserve_pv_price=0.0001)
    env.reset(market_id)
    return env


def build_market(path: Path, study: dict, market_id: int) -> dict:
    identity = fingerprint(
        {
            "study": {k: study[k] for k in ("focal_agents", "pv_count", "actions", "biases")},
            "market": market_id,
            "source": source_identity(),
        }
    )
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists():
        meta = read_json(meta_path)
        if meta["identity"] != identity:
            raise RuntimeError("Market identity mismatch: use a new study directory")
        return meta
    started = time.perf_counter()
    Generator = _load("auctionnet_pv", "NeurIPSPvGen.py").NeurIPSPvGen
    Env = _load("auctionnet_env", "BiddingEnv.py").BiddingEnv
    pv = Generator(episode=market_id, pv_num=study["pv_count"])
    env = _cached_env(Env, market_id)
    focals = np.array(study["focal_agents"], dtype=int)
    grid = effective_actions(study)
    shape = (len(focals), 48, len(grid))
    costs, conversions, exposures = (np.zeros(shape) for _ in range(3))
    starts, lengths = np.zeros(shape, dtype=np.int64), np.zeros(shape, dtype=np.int64)
    prefix_cost, prefix_conv = [], []
    offset = 0
    pmeans, smeans = np.zeros((len(focals), 48)), np.zeros((len(focals), 48))
    counts = np.array([len(v) for v in pv.pv_values])
    rng = np.random.default_rng(market_id + 730001)
    opponent_alpha = rng.uniform(15, 145, 48)
    phase = rng.uniform(0, 2 * np.pi, 48)
    for tick, (values, sigmas) in enumerate(zip(pv.pv_values, pv.pValueSigmas, strict=True)):
        env.prepare(values, sigmas)
        bids = values * (opponent_alpha * (1 + 0.15 * np.sin(2 * np.pi * tick / 48 + phase)))
        for fi, focal in enumerate(focals):
            pmeans[fi, tick] = values[:, focal].mean()
            smeans[fi, tick] = sigmas[:, focal].mean()
            original = bids[:, focal].copy()
            for ai, multiplier in enumerate(grid):
                bids[:, focal] = values[:, focal] * CPA[focal] * multiplier
                out = env.simulate_ad_bidding(values, sigmas, bids)
                paid = out[2][focal] * out[3][focal]
                mask = paid > 0
                pc = np.concatenate(([0.0], np.cumsum(paid[mask], dtype=np.float64)))
                cv = np.concatenate(([0], np.cumsum(out[4][focal][mask], dtype=np.int32)))
                starts[fi, tick, ai], lengths[fi, tick, ai] = offset, len(pc)
                offset += len(pc)
                prefix_cost.append(pc)
                prefix_conv.append(cv.astype(np.int32))
                costs[fi, tick, ai], conversions[fi, tick, ai] = pc[-1], cv[-1]
                exposures[fi, tick, ai] = mask.sum()
            bids[:, focal] = original
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(
        tmp,
        grid=grid,
        focals=focals,
        costs=costs,
        conversions=conversions,
        exposures=exposures,
        starts=starts,
        lengths=lengths,
        prefix_cost=np.concatenate(prefix_cost),
        prefix_conv=np.concatenate(prefix_conv),
        pmeans=pmeans,
        smeans=smeans,
        counts=counts,
        budgets=BUDGET[focals] * study["pv_count"] / 500000,
        cpas=CPA[focals],
    )
    os.replace(tmp, path)
    meta = {
        "identity": identity,
        "market_id": market_id,
        "pv_count": int(counts.sum()),
        "requested_pv_count": study["pv_count"],
        "focal_agents": focals.tolist(),
        "effective_actions": grid.tolist(),
        "seconds": time.perf_counter() - started,
        "bytes": path.stat().st_size,
        "source": source_identity(),
    }
    write_json(meta_path, meta)
    return meta


class Tape:
    def __init__(self, path: Path):
        with np.load(path, allow_pickle=False) as data:
            self.arrays = {k: data[k] for k in data.files}
        self.grid_index = {round(float(x), 8): i for i, x in enumerate(self.arrays["grid"])}

    def index(self, action: float) -> int:
        try:
            return self.grid_index[round(float(action), 8)]
        except KeyError as error:
            raise ValueError("Action absent from exact response tape") from error

    def settle(self, fi: int, tick: int, effective_action: float, remaining: float) -> tuple:
        ai = self.index(effective_action)
        start = self.arrays["starts"][fi, tick, ai]
        end = start + self.arrays["lengths"][fi, tick, ai]
        pc = self.arrays["prefix_cost"][start:end]
        cv = self.arrays["prefix_conv"][start:end]
        n = max(0, int(np.searchsorted(pc, max(0, remaining), side="right")) - 1)
        return float(pc[n]), int(cv[n]), n, n < len(pc) - 1


def scenario_bias(name: str, tick: int) -> float:
    mapping = {
        "calibrated": 1.0,
        "over_25": 1.25,
        "over_50": 1.5,
        "under_25": 0.75,
        "mid_over_50": 1.0 if tick < 24 else 1.5,
    }
    if name not in mapping:
        raise ValueError("Unknown stress scenario: " + name)
    return mapping[name]


def score(cost: float, conversions: float, cpa: float) -> float:
    if conversions <= 0:
        return 0.0
    ratio = cpa * conversions / max(cost, 1e-12)
    return float(conversions * min(1.0, ratio * ratio))


def observation(
    tape: Tape, fi: int, tick: int, cost: float, conv: int, previous: tuple, bias: float
) -> np.ndarray:
    a = tape.arrays
    budget, cpa = float(a["budgets"][fi]), float(a["cpas"][fi])
    last_cost, last_conv, last_action, last_wins = previous
    count = max(1, int(a["counts"][max(0, tick - 1)]))
    return np.array(
        [
            tick / 48,
            (48 - tick) / 48,
            max(0, 1 - cost / budget),
            cost / budget,
            conv * cpa / budget,
            (cost - cpa * conv) / budget,
            last_cost / budget,
            last_conv * cpa / budget,
            last_action / 3,
            a["pmeans"][fi, tick] * bias / 0.0005,
            a["smeans"][fi, tick] / max(a["pmeans"][fi, tick], 1e-12),
            a["counts"][tick] / 10000.0,
            last_wins / count,
        ],
        dtype=np.float32,
    )


def feedback_action(
    state: np.ndarray, actions: np.ndarray, gain: float, prior: float, budget: float, cpa: float
) -> float:
    elapsed = float(state[0])
    spend = float(state[3]) * budget
    conv = float(state[4]) * budget / cpa
    pacing = ((elapsed + 0.05) * budget / (spend + 0.05 * budget)) ** gain
    quality = ((conv + prior) * cpa / (spend + prior * cpa)) ** gain
    desired = np.clip(pacing * quality, 0.05, 3.0)
    return float(actions[np.abs(actions - desired).argmin()])
