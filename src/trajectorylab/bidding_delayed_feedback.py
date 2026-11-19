"""Public AuctionNet event tape and environment-owned delayed label queue.

Only PublicCohort and arrived IDs belong in a policy view. Settlement and the
queue contain final environment truth and must never be passed to a policy.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from trajectorylab.bidding.market import (
    BUDGET,
    CPA,
    Tape,
    _cached_env,
    _load,
    effective_actions,
    source_identity,
)
from trajectorylab.utils import fingerprint, read_json, write_json


def event_source_identity():
    return {
        **source_identity(),
        "event_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


@dataclass(frozen=True)
class Settlement:
    cost: float
    conversions: int
    wins: int
    budget_blocked: bool
    event_ids: tuple[int, ...]
    public_p: tuple[float, ...]
    positive_ids: tuple[int, ...]


class EventTape(Tape):
    def settle_events(
        self, fi: int, tick: int, effective_action: float, remaining: float
    ) -> Settlement:
        cost, conv, n, blocked = self.settle(fi, tick, effective_action, remaining)
        ai = self.index(effective_action)
        start = int(self.arrays["starts"][fi, tick, ai])
        ids = self.arrays["prefix_event_ids"][start + 1 : start + 1 + n]
        cv = self.arrays["prefix_conv"][start : start + n + 1]
        positive = ids[np.diff(cv) > 0]
        return Settlement(
            cost,
            conv,
            n,
            bool(blocked),
            tuple(map(int, ids)),
            tuple(map(float, self.arrays["public_probability"][fi, ids])),
            tuple(map(int, positive)),
        )


@dataclass(frozen=True)
class PublicCohort:
    tick: int
    event_ids: tuple[int, ...]
    public_p: tuple[float, ...]


def event_delay(market: int, focal: int, event_id: int, delay_seed: int, regime: str) -> int:
    """Policy/call-order independent delay; every supported delay is at most W=48."""
    if regime not in ("immediate", "short", "long"):
        raise ValueError("Unknown delay regime: " + regime)
    if regime == "immediate":
        return 0
    key = f"{market}:{focal}:{event_id}:{delay_seed}".encode("ascii")
    value = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "little")
    # Use the top 53 bits so u is exactly representable and strictly below one.
    u = (value >> 11) / float(1 << 53)
    if regime == "long":
        if u < 0.25:
            return 0
        u = (u - 0.25) / 0.75
        p = 1 / 13
    else:
        p = 1 / 3
    return min(48, int(math.floor(math.log1p(-u) / math.log1p(-p))))


class DelayQueue:
    """One episode/focal queue; final zero labels mature only after 48 ticks.

    After action j, register(j, settlement); before decision t call advance(t-1).
    Thus D=0 becomes visible at decision j+1, with cohort age t-1-j. Calling
    advance(95) after the 48 opportunity ticks matures every accepted event.
    """

    def __init__(self, market: int, focal: int, delay_seed: int, regime: str):
        event_delay(market, focal, 0, delay_seed, regime)
        self.market, self.focal, self.delay_seed, self.regime = market, focal, delay_seed, regime
        self._due: dict[int, list[int]] = {}
        self._seen: set[int] = set()
        self._cohorts: list[PublicCohort] = []
        self._last_tick = -1

    def register(
        self, tick: int, settlement: Settlement, prediction_multiplier: float = 1.0
    ) -> PublicCohort:
        if tick < self._last_tick or tick < 0 or tick >= 48:
            raise ValueError("Register opportunities in chronological ticks 0..47 before advancing")
        if self._seen.intersection(settlement.event_ids):
            raise ValueError("An event may only be accepted once per focal episode")
        p = tuple(float(x * prediction_multiplier) for x in settlement.public_p)
        if any(not np.isfinite(x) or x < 0 or x > 1 for x in p):
            raise ValueError("Invalid public probability")
        cohort = PublicCohort(tick, settlement.event_ids, p)
        self._cohorts.append(cohort)
        self._seen.update(settlement.event_ids)
        for event in settlement.positive_ids:
            due = tick + event_delay(self.market, self.focal, event, self.delay_seed, self.regime)
            self._due.setdefault(due, []).append(event)
        return cohort

    def advance(self, tick: int) -> tuple[int, ...]:
        if tick < self._last_tick:
            raise ValueError("Cannot reverse as-of time")
        self._last_tick = tick
        due_ticks = sorted(k for k in self._due if k <= tick)
        return tuple(event for k in due_ticks for event in self._due.pop(k))

    def cohorts(self, as_of: int) -> tuple[PublicCohort, ...]:
        return tuple(c for c in self._cohorts if c.tick <= as_of)


def build_event_market(path: Path, study: dict, market_id: int) -> dict:
    identity = fingerprint(
        {
            "study": {k: study[k] for k in ("focal_agents", "pv_count", "actions", "biases")},
            "market": market_id,
            "source": event_source_identity(),
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
    prefix_cost, prefix_conv, prefix_ids = [], [], []
    public_probability = np.zeros((len(focals), int(sum(len(v) for v in pv.pv_values))))
    pv_offset = 0
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
            public_probability[fi, pv_offset : pv_offset + len(values)] = values[:, focal]
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
                prefix_ids.append(np.r_[-1, np.flatnonzero(mask) + pv_offset].astype(np.int64))
                prefix_cost.append(pc)
                prefix_conv.append(cv.astype(np.int32))
                costs[fi, tick, ai], conversions[fi, tick, ai] = pc[-1], cv[-1]
                exposures[fi, tick, ai] = mask.sum()
            bids[:, focal] = original
        pv_offset += len(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.npz")
    np.savez_compressed(
        tmp,
        prefix_event_ids=np.concatenate(prefix_ids),
        public_probability=public_probability,
        market_id=np.array(market_id),
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
        "source": event_source_identity(),
    }
    write_json(meta_path, meta)
    return meta
