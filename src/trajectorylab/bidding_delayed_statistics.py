"""Paired market-level summaries; campaign rows are never independent replicates."""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def _means(rows: list[dict], method: str, group: str) -> dict[int, np.ndarray]:
    selected = defaultdict(list)
    for row in rows:
        if row["method"] != method:
            continue
        if group != "all" and row["bias_group"] != group:
            continue
        selected[int(row["market"])].append(
            [row["score"], row["conversions"], row["violation"], row["budget_utilization"]]
        )
    return {m: np.mean(v, axis=0) for m, v in sorted(selected.items())}


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def paired_comparison(
    rows: list[dict],
    method: str,
    baseline: str,
    *,
    draws: int = 10000,
    seed: int = 20261119,
    minimum_markets: int = 24,
) -> dict:
    """Ratio of across-market means, preserving equal within-market scene weights.

    Bootstrap intervals are approximate, conditional on the finite declared
    training/arrival seeds. Seven protection endpoints use Bonferroni-adjusted
    one-sided intervals. They do not establish production constraint guarantees.
    """
    groups = {}
    market_ids = None
    indices = None
    rng = np.random.default_rng(seed)
    for group in ("all", "nominal", "under"):
        left, right = _means(rows, method, group), _means(rows, baseline, group)
        if not left or set(left) != set(right):
            raise ValueError(f"Missing paired market coverage: {method}/{baseline}/{group}")
        markets = sorted(left)
        if market_ids is None:
            market_ids = markets
            indices = rng.integers(0, len(markets), size=(draws, len(markets)))
        if markets != market_ids:
            raise ValueError("Protection groups must use the same paired markets")
        x, y = np.array([left[m] for m in markets]), np.array([right[m] for m in markets])
        xb, yb = x[indices].mean(axis=1), y[indices].mean(axis=1)
        ratio = _ratio(x.mean(axis=0), y.mean(axis=0))
        rb = _ratio(xb, yb)
        vd = xb[:, 2] - yb[:, 2]
        if np.isnan(ratio[:2]).any() or np.isnan(rb[:, :2]).any():
            groups[group] = {"status": "incomparable_zero_baseline", "markets": len(markets)}
            continue
        alpha = 0.05 / 7
        groups[group] = {
            "status": "estimated",
            "markets": len(markets),
            "score_ratio": float(ratio[0]),
            "conversion_ratio": float(ratio[1]),
            "score_ratio_ci95": np.quantile(rb[:, 0], [0.025, 0.975]).tolist(),
            "conversion_ratio_ci95": np.quantile(rb[:, 1], [0.025, 0.975]).tolist(),
            "score_ratio_protection_lower": float(np.quantile(rb[:, 0], alpha)),
            "conversion_ratio_protection_lower": float(np.quantile(rb[:, 1], alpha)),
            "violation_delta": float(x[:, 2].mean() - y[:, 2].mean()),
            "violation_delta_ci95": np.quantile(vd, [0.025, 0.975]).tolist(),
            "violation_delta_protection_upper": float(np.quantile(vd, 1 - alpha)),
            "method_means": dict(zip(("score", "conversions", "violation", "budget_utilization"), x.mean(axis=0).tolist(), strict=True)),
            "baseline_means": dict(zip(("score", "conversions", "violation", "budget_utilization"), y.mean(axis=0).tolist(), strict=True)),
        }
    estimated = all(v["status"] == "estimated" for v in groups.values())
    protection = estimated and all(
        groups[g]["score_ratio_protection_lower"] >= 0.97
        and groups[g]["conversion_ratio_protection_lower"] >= 0.97
        for g in ("nominal", "under")
    ) and all(groups[g]["violation_delta_protection_upper"] <= 0.03 for g in groups)
    uplift = estimated and groups["all"]["score_ratio"] >= 1.03 and groups["all"]["score_ratio_ci95"][0] > 1
    return {
        "method": method,
        "baseline": baseline,
        "independent_unit": "market",
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "protection_endpoints": 7,
        "groups": groups,
        "minimum_markets": minimum_markets,
        "sufficient_independent_markets": len(market_ids) >= minimum_markets,
        "business_target_pass": bool(uplift and protection and len(market_ids) >= minimum_markets),
        "protection_pass": bool(protection),
        "uplift_target_pass": bool(uplift),
    }


def validate_rows(rows: list[dict]) -> dict:
    """Reject silent changes in campaign coverage or invalid accounting."""
    keys = set()
    coverage = defaultdict(set)
    for row in rows:
        key = tuple(row[k] for k in ("market", "focal", "bias_group", "delay", "arrival_seed"))
        pair = (row["method"], *key)
        if pair in keys:
            raise ValueError("Duplicate campaign row")
        keys.add(pair)
        coverage[row["method"]].add(key)
        for name in ("score", "cost", "conversions", "budget_utilization"):
            if not np.isfinite(row[name]) or row[name] < 0:
                raise ValueError(f"Invalid accounting: {name}")
        if row["cost"] > row["budget"] + 1e-6:
            raise ValueError("Settlement exceeded hard budget")
        if row["violation"] not in (0, 1):
            raise ValueError("Violation is not an indicator")
    if not coverage or any(v != next(iter(coverage.values())) for v in coverage.values()):
        raise ValueError("Methods have different campaign coverage")
    return {"rows": len(rows), "methods": sorted(coverage), "paired_campaigns": len(next(iter(coverage.values())))}
