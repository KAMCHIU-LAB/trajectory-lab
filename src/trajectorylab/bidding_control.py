"""Checkpoint-matched controller diagnostics with validation-only selection.

This module is outside the frozen first-study package. Dense response tapes
remain evaluator-private. Empirical response rates use training markets only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.learning import Feedback, Learned, rollout
from trajectorylab.bidding.market import ROOT, Tape, build_market
from trajectorylab.utils import fingerprint, read_json, write_json


class CheckpointForecast:
    def __init__(self, path, actions, context):
        self.policy = Learned(path, actions, context, "cpu")
        if self.policy.kind != "response":
            raise ValueError("Paced control requires a response checkpoint")

    @torch.no_grad()
    def __call__(self, states, budget, cpa):
        p = self.policy
        end = len(states)
        start = max(0, end - p.context)
        x = torch.as_tensor((np.asarray(states[start:]) - p.mean) / p.std)[None]
        steps = torch.arange(start, end)[None]
        mask = torch.ones((1, end - start), dtype=torch.bool)
        curves = p.model(x, steps, mask)[0, -1, :2].numpy()
        return curves * np.array([budget, budget / cpa])[:, None]


class EmpiricalForecast:
    """Pooled training response per opportunity, in budget-normalized units."""

    def __init__(self, rates):
        self.rates = np.asarray(rates)

    def __call__(self, states, budget, cpa):
        opportunities = float(states[-1][11]) * 10000.0
        return self.rates * opportunities * np.array([budget, budget / cpa])[:, None]


class PacedResponse:
    """One-tick primal decisions with observed-outcome budget and CPA prices."""

    def __init__(self, forecast, actions, rates, calibration="none", prior=3.0, half_life=8.0):
        self.forecast, self.actions = forecast, np.asarray(actions)
        self.eta_budget, self.eta_quality = rates
        self.calibration, self.prior = calibration, prior
        self.rho = 2 ** (-1 / half_life) if calibration == "discounted" else 1.0
        self.reset()

    def reset(self):
        self.budget_price = self.quality_price = 0.0
        self.actual = np.zeros(2)
        self.expected = np.zeros(2)

    def __call__(self, states, used, rewards, budget, cpa):
        curves = self.forecast(states, budget, cpa)
        if not np.isfinite(curves).all() or (curves < 0).any():
            raise RuntimeError("Invalid response forecast")
        corrected = curves.copy()
        if self.calibration != "none":
            pseudo = np.array([self.prior * cpa, self.prior])
            ratios = np.clip((self.actual + pseudo) / (self.expected + pseudo), 0.25, 4)
            corrected *= ratios[:, None]
        normalized_cost = corrected[0] / budget
        normalized_quality = corrected[1] * cpa / budget
        utility = (1 + self.quality_price) * normalized_quality
        utility -= (self.budget_price + self.quality_price) * normalized_cost
        remaining = max(0.0, 1 - float(states[-1][3]))
        ai = int(np.argmax(utility)) if remaining > 1e-9 else 0
        return float(self.actions[ai]), {
            "predicted_tick_cost": float(curves[0, ai]),
            "predicted_tick_conversions": float(curves[1, ai]),
            "forecast_feasible": bool(curves[0, ai] <= remaining * budget),
            "pacing_allowance": remaining / (49 - len(states)),
            "normalizing_budget": budget,
            "target_cpa": cpa,
        }

    def observe(self, cost, conv, diagnostic):
        budget, cpa = diagnostic["normalizing_budget"], diagnostic["target_cpa"]
        c, q = cost / budget, conv * cpa / budget
        self.budget_price = max(
            0.0, self.budget_price + self.eta_budget * (c - diagnostic["pacing_allowance"])
        )
        self.quality_price = max(0.0, self.quality_price + self.eta_quality * (c - q))
        # Resource prices use actual settlement, including truncated ticks.
        # Forecast ratios use uncensored feedback only; forgetting is per tick.
        self.actual *= self.rho
        self.expected *= self.rho
        if not diagnostic.get("budget_blocked", False):
            self.actual += (cost, conv)
            self.expected += (
                diagnostic["predicted_tick_cost"],
                diagnostic["predicted_tick_conversions"],
            )


def empirical_rates(paths, actions):
    totals, opportunities = np.zeros((2, len(actions))), 0
    for path in paths:
        # No settlement prefixes or held-out responses are needed to fit this baseline.
        with np.load(path, allow_pickle=False) as arrays:
            grid = arrays["grid"]
            idx = [int(np.flatnonzero(np.isclose(grid, a, rtol=0, atol=1e-8))[0]) for a in actions]
            costs, conv = arrays["costs"][:, :, idx], arrays["conversions"][:, :, idx]
            budget, cpa = arrays["budgets"], arrays["cpas"]
            totals[0] += (costs / budget[:, None, None]).sum((0, 1))
            totals[1] += (conv * cpa[:, None, None] / budget[:, None, None]).sum((0, 1))
            opportunities += int(arrays["counts"].sum()) * len(budget)
    return totals / opportunities


def metrics(rows, method, scenario=None):
    group = [
        r for r in rows if r["method"] == method and (scenario is None or r["scenario"] == scenario)
    ]
    if not group:
        raise ValueError(f"No evidence for {method}")
    return {
        key: float(np.mean([r[key] for r in group]))
        for key in (
            "score",
            "conversions",
            "violation",
            "cost",
            "budget_utilization",
            "zero_conversions",
        )
    }


def evaluate(paths, scenarios, actions, policies, directory, status):
    directory.mkdir(parents=True, exist_ok=True)
    signature = fingerprint(sorted(policies))
    all_rows = []
    for path in paths:
        result_path = directory / f"{path.stem}.json"
        focals = read_json(path.with_suffix(".json"))["focal_agents"]
        expected_keys = {
            (int(path.stem), f, s, m, seed)
            for f in focals
            for s in scenarios
            for m, seed in policies
        }
        if result_path.exists():
            saved = read_json(result_path)
            if saved["policies"] != signature:
                raise RuntimeError("Cannot resume a different candidate set")
            rows = saved["rows"]
        else:
            tape = Tape(path)
            rows = []
            for fi, focal in enumerate(tape.arrays["focals"]):
                for scenario in scenarios:
                    for (method, seed), policy in policies.items():
                        if hasattr(policy, "reset"):
                            policy.reset()
                        row = rollout(tape, fi, actions, scenario, policy)
                        rows.append(
                            {
                                "market": int(path.stem),
                                "focal": int(focal),
                                "scenario": scenario,
                                "method": method,
                                "seed": seed,
                                **row,
                            }
                        )
            if len(rows) != len(expected_keys):
                raise RuntimeError("Incomplete control evaluation")
            write_json(result_path, {"policies": signature, "rows": rows})
        keys = {(r["market"], r["focal"], r["scenario"], r["method"], r["seed"]) for r in rows}
        if (
            keys != expected_keys
            or len(keys) != len(rows)
            or any(
                not np.isfinite(v) for r in rows for v in r.values() if isinstance(v, (int, float))
            )
        ):
            raise RuntimeError("Nonfinite or duplicated control evidence")
        all_rows.extend(rows)
        status("evaluation", market=int(path.stem), directory=directory.name, rows=len(all_rows))
    return all_rows


def paired(rows, method, reference, scenario, samples):
    grouped = defaultdict(list)
    for r in rows:
        if r["scenario"] == scenario and r["method"] in (method, reference):
            grouped[r["method"], r["market"], r["focal"]].append(r)
    markets = sorted({key[1] for key in grouped})
    values = []
    for market in markets:
        focal_ids = sorted({key[2] for key in grouped if key[1] == market})
        campaign = []
        for focal in focal_ids:
            a, b = grouped[method, market, focal], grouped[reference, market, focal]
            if not a or not b:
                raise RuntimeError("Unpaired confirmation evidence")
            ma = {k: np.mean([r[k] for r in a]) for k in ("score", "conversions", "violation")}
            mb = {k: np.mean([r[k] for r in b]) for k in ma}
            campaign.append(
                [
                    ma["score"] - mb["score"],
                    ma["conversions"],
                    mb["conversions"],
                    ma["violation"] - mb["violation"],
                ]
            )
        values.append(np.mean(campaign, axis=0))
    values = np.asarray(values)
    rng = np.random.default_rng(29117)
    draws = values[rng.integers(0, len(values), (samples, len(values)))].mean(axis=1)
    retention_defined = bool(np.all(draws[:, 2] > 0))
    if retention_defined:
        draws[:, 1] /= draws[:, 2]
    means = values.mean(axis=0)
    result = {
        "markets": len(markets),
        "method": method,
        "reference": reference,
        "scenario": scenario,
    }
    for name, column, estimate in (("score_delta", 0, means[0]), ("violation_delta", 3, means[3])):
        result[name] = {
            "mean": float(estimate),
            "market_bootstrap_95": np.quantile(draws[:, column], [0.025, 0.975]).tolist(),
        }
    result["conversion_ratio"] = {
        "mean": float(means[1] / means[2]) if means[2] > 0 else None,
        "market_bootstrap_95": np.quantile(draws[:, 1], [0.025, 0.975]).tolist()
        if retention_defined
        else None,
        "defined_for_all_resamples": retention_defined,
    }
    result["uncertainty_scope"] = (
        "Market bootstrap after averaging advertisers and training seeds; conditional on three trained checkpoints. Not a full training-seed interval."
    )
    seed_rows = defaultdict(list)
    for r in rows:
        if r["method"] == method and r["scenario"] == scenario:
            refs = grouped[reference, r["market"], r["focal"]]
            matching = [b for b in refs if b["seed"] == r["seed"]]
            refs = matching or refs
            seed_rows[r["seed"]].append(r["score"] - np.mean([b["score"] for b in refs]))
    result["seed_score_deltas"] = {str(k): float(np.mean(v)) for k, v in seed_rows.items()}
    return result


def write_report(directory, rows, selection, confirmation=None):
    methods = sorted({r["method"] for r in rows})
    scenarios = sorted({r["scenario"] for r in rows})
    summaries = [
        {"scenario": s, "method": m, **metrics(rows, m, s)} for s in scenarios for m in methods
    ]
    write_json(
        directory / "summary.json",
        {"groups": summaries, "selection": selection, "confirmation": confirmation},
    )
    write_json(directory / "results.json", rows)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (directory / "results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    columns = ["scenario", "method", "score", "conversions", "violation", "budget_utilization"]
    table = "<tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>"
    table += "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c]))}</td>"
            for c in columns
        )
        + "</tr>"
        for r in summaries
    )
    body = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>生成式出价：控制器诊断</title>
<style>body{{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;padding:0 20px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>生成式出价：控制器诊断与独立验证</h1>
<p>公开 AuctionNet 自定义固定对手模拟；不是官方榜单、线上收益或通用安全保证。calibrated 仅表示未额外注入偏差。</p>
<p>固定模型检查点，比较常动作尾部筛选、一时段资源价格控制、累计及八时段半衰期校正。
经验响应曲线只用训练市场拟合，和学习模型使用相同控制器；固定出价及反馈基线保留。
资源价格最大化线性的预期转化收益，并非精确优化非线性 CPA 惩罚分数。硬预算来自结算器。</p>
<h2>预先锁定的选择与判断</h2><pre>{html.escape(json.dumps(selection, ensure_ascii=False, indent=2))}</pre>
<h2>逐场景结果</h2><table>{table}</table>
<h2>独立确认</h2><pre>{html.escape(json.dumps(confirmation, ensure_ascii=False, indent=2))}</pre>
<p>同一市场中的广告主和训练种子不是更多独立市场。区间按市场重采样，不能替代训练随机性或分布外验证。
未达到验证晋级规则时，不生成独立确认市场；确认结果不用于重调本轮参数。
<a href="results.csv">原始记录</a> · <a href="summary.json">统计结果</a> · <a href="manifest.json">来源身份</a></p></html>"""
    (directory / "report.html").write_text(body)


def run(config_path):
    session_start = time.perf_counter()
    opts = tomllib.loads(config_path.read_text())
    main = ROOT / opts["main_run"]
    manifest = read_json(main / "manifest.json")
    if read_json(main / "status.json")["phase"] != "complete":
        raise RuntimeError("Finish the frozen first study before controller selection")
    config, study = manifest["config"], manifest["config"]["study"]
    if fingerprint(provenance(config)) != manifest["identity"]:
        raise RuntimeError("Imported first-study sources or runtime differ from its manifest")
    old_ids = set(
        sum([study[k] for k in ("train_markets", "validation_markets", "test_markets")], [])
    )
    confirm_ids = opts["confirmation_markets"]
    if len(set(confirm_ids)) != len(confirm_ids) or old_ids.intersection(confirm_ids):
        raise ValueError("Confirmation markets must be new and unique")
    torch.set_num_threads(config["training"]["threads"])
    directory = ROOT / opts["output"]
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl

    lock = (directory / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    selection_path = main / "selection.json"
    main_selection = read_json(selection_path)
    selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    checkpoint_hashes = {}
    for seed in study["training_seeds"]:
        path = main / "models" / f"response-{seed}" / "best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved["identity"] != manifest["identity"]:
            raise RuntimeError("Response checkpoint is not from the declared first study")
        checkpoint_hashes[str(seed)] = hashlib.sha256(path.read_bytes()).hexdigest()
    identity = fingerprint(
        {
            "config": opts,
            "main": manifest["identity"],
            "implementation": source_hash,
            "checkpoints": checkpoint_hashes,
            "main_selection_sha256": selection_hash,
        }
    )
    identity_path = directory / "manifest.json"
    if identity_path.exists() and read_json(identity_path)["identity"] != identity:
        raise RuntimeError("Control output belongs to different sources/configuration")
    write_json(
        identity_path,
        {
            "identity": identity,
            "config": opts,
            "main_identity": manifest["identity"],
            "implementation_sha256": source_hash,
            "checkpoint_sha256": checkpoint_hashes,
            "main_selection_sha256": selection_hash,
            "main_selection": main_selection,
            "evidence_layout": {
                "validation_markets": study["validation_markets"],
                "confirmation_markets": confirm_ids,
                "focal_agents": study["focal_agents"],
                "training_seeds": study["training_seeds"],
                "scenarios": config["evaluation"]["scenarios"],
            },
            "experiment_kind": "validation-selected controller refinement with untouched confirmation markets",
        },
    )

    def status(phase, **extra):
        write_json(
            directory / "status.json",
            {
                "phase": phase,
                "updated_unix": time.time(),
                "elapsed_this_session_seconds": time.perf_counter() - session_start,
                **extra,
            },
        )

    actions = np.asarray(study["actions"])
    prior = opts["calibration_prior"]

    def guard_sources():
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != source_hash:
            raise RuntimeError("Control implementation changed during execution")
        if hashlib.sha256(selection_path.read_bytes()).hexdigest() != selection_hash:
            raise RuntimeError("First-study selection changed during execution")
        if fingerprint(provenance(config)) != manifest["identity"]:
            raise RuntimeError("Imported first-study dependencies changed during execution")
        for seed, expected in checkpoint_hashes.items():
            path = main / "models" / f"response-{seed}" / "best.pt"
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Response checkpoint changed during execution")

    forecasts = {
        s: CheckpointForecast(
            main / "models" / f"response-{s}" / "best.pt", actions, config["training"]["context"]
        )
        for s in study["training_seeds"]
    }
    train_paths = [main / "markets" / f"{m}.npz" for m in study["train_markets"]]
    empirical = EmpiricalForecast(empirical_rates(train_paths, actions))
    write_json(
        directory / "empirical-rates.json",
        {
            "rates": empirical.rates.tolist(),
            "fit_split": "training only",
            "units": "cost/budget and conversions*target_cpa/budget per opportunity",
        },
    )
    rates = opts["dual_rates"]

    def make_paced(forecast, rate, calibration="none"):
        return PacedResponse(
            forecast, actions, rate, calibration, prior, opts["discount_half_life"]
        )

    policies = {}
    for i, rate in enumerate(rates):
        # Empirical forecasts receive every calibration option, rather than
        # reserving online correction for the neural model.
        for calibration in ("none", "cumulative", "discounted", "budget-only"):
            empirical_rate = (rate[0], 0.0) if calibration == "budget-only" else rate
            mode = "none" if calibration == "budget-only" else calibration
            policies[f"empirical-paced-{i}-{calibration}", -1] = make_paced(
                empirical, empirical_rate, mode
            )
        for seed, forecast in forecasts.items():
            policies[f"response-paced-{i}", seed] = make_paced(forecast, rate)
    policies["feedback", -1] = Feedback(actions, **main_selection["feedback"])
    for name, action in (("fixed-1", 1.0), ("fixed-tuned", main_selection["fixed"]["action"])):
        policies[name, -1] = lambda s, u, r, b, c, k=action: (k, {})
    for seed in study["training_seeds"]:
        for corrected in (False, True):
            name = "response+correction" if corrected else "response"
            policies[name, seed] = Learned(
                main / "models" / f"response-{seed}" / "best.pt",
                actions,
                config["training"]["context"],
                "cpu",
                corrected,
                main_selection["correction"]["response"]["prior"],
            )
    val_paths = [main / "markets" / f"{m}.npz" for m in study["validation_markets"]]
    scenarios = config["evaluation"]["scenarios"]
    rows = evaluate(val_paths, scenarios, actions, policies, directory / "validation-rates", status)
    best_rate = max(range(len(rates)), key=lambda i: metrics(rows, f"response-paced-{i}")["score"])
    variants = {}
    for seed, forecast in forecasts.items():
        variants["response-paced-cumulative", seed] = make_paced(
            forecast, rates[best_rate], "cumulative"
        )
        variants["response-paced-discounted", seed] = make_paced(
            forecast, rates[best_rate], "discounted"
        )
        variants["response-paced-budget-only", seed] = make_paced(
            forecast, (rates[best_rate][0], 0.0)
        )
    extra = evaluate(
        val_paths, scenarios, actions, variants, directory / "validation-calibration", status
    )
    rows.extend(extra)
    candidates = [
        f"response-paced-{best_rate}",
        "response-paced-cumulative",
        "response-paced-discounted",
        "response-paced-budget-only",
    ]
    candidate = max(candidates, key=lambda m: metrics(rows, m)["score"])
    simple = ["feedback", "fixed-1", "fixed-tuned"] + sorted(
        {method for method, _ in policies if method.startswith("empirical-paced-")}
    )
    reference = max(simple, key=lambda m: metrics(rows, m)["score"])
    primary_scenario = opts["primary_scenario"]
    primary_reference = max(simple, key=lambda m: metrics(rows, m, primary_scenario)["score"])
    a, b, old = metrics(rows, candidate), metrics(rows, reference), metrics(rows, "response")
    pa, pb = (
        metrics(rows, candidate, primary_scenario),
        metrics(rows, primary_reference, primary_scenario),
    )
    primary_old = metrics(rows, "response", primary_scenario)
    gate = opts["promotion"]

    def passes_gate(candidate_metrics, baseline_metrics, original):
        return bool(
            baseline_metrics["score"] > 0
            and baseline_metrics["conversions"] > 0
            and candidate_metrics["score"]
            >= gate["minimum_score_ratio"] * max(baseline_metrics["score"], original["score"])
            and candidate_metrics["conversions"]
            >= gate["minimum_conversion_ratio"] * baseline_metrics["conversions"]
            and candidate_metrics["violation"]
            <= baseline_metrics["violation"] + gate["maximum_violation_increase"]
        )

    promoted = passes_gate(a, b, old) and passes_gate(pa, pb, primary_old)
    selected_calibration = {
        f"response-paced-{best_rate}": "none",
        "response-paced-cumulative": "cumulative",
        "response-paced-discounted": "discounted",
        "response-paced-budget-only": "budget-only",
    }[candidate]
    matched_empirical = f"empirical-paced-{best_rate}-{selected_calibration}"
    selection = {
        "split": "validation only; equal weight per scenario and advertiser, averaged training seeds",
        "best_rate_index": best_rate,
        "best_rate": rates[best_rate],
        "candidate": candidate,
        "reference": reference,
        "primary_reference": primary_reference,
        "primary_scenario": primary_scenario,
        "matched_empirical": matched_empirical,
        "selected_calibration": selected_calibration,
        "candidate_metrics": a,
        "reference_metrics": b,
        "original_response_metrics": old,
        "primary_candidate_metrics": pa,
        "primary_reference_metrics": pb,
        "primary_original_response_metrics": primary_old,
        "gate_scope": "Both pooled scenarios and the primary scenario must pass",
        "promotion_rule": gate,
        "promoted": promoted,
        "interpretation": "Selection estimates are optimistic; independent confirmation is required.",
    }
    all_policies = policies | variants
    keep = {
        candidate,
        reference,
        primary_reference,
        matched_empirical,
        f"response-paced-{best_rate}",
        "response-paced-cumulative",
        "response-paced-discounted",
        "response-paced-budget-only",
        "response",
        "response+correction",
        "feedback",
        "fixed-1",
        "fixed-tuned",
    }
    confirmation_policies = {key: value for key, value in all_policies.items() if key[0] in keep}
    current_manifest = read_json(identity_path)
    current_manifest["evidence_layout"].update(
        validation_policies=sorted(all_policies),
        confirmation_policies=sorted(confirmation_policies),
    )
    write_json(identity_path, current_manifest)
    guard_sources()
    write_json(directory / "selection.json", selection)
    val_dir = directory / "validation-report"
    val_dir.mkdir(exist_ok=True)
    write_json(val_dir / "manifest.json", read_json(identity_path))
    write_report(val_dir, rows, selection)
    print(f"validation promoted={promoted} candidate={candidate} reference={reference}", flush=True)
    if not promoted:
        status("complete-not-promoted", reason="No candidate met the predeclared validation gate")
        return directory
    confirm_paths = []
    for market in confirm_ids:
        status("confirmation-data", market=market)
        path = directory / "markets" / f"{market}.npz"
        build_market(path, study, market)
        confirm_paths.append(path)
    confirmation_rows = evaluate(
        confirm_paths,
        scenarios,
        actions,
        confirmation_policies,
        directory / "confirmation-results",
        status,
    )
    effect = paired(
        confirmation_rows,
        candidate,
        primary_reference,
        opts["primary_scenario"],
        opts["confirmation"]["bootstrap_samples"],
    )
    limits = opts["confirmation"]
    effect["accepted"] = bool(
        effect["score_delta"]["market_bootstrap_95"][0] > limits["minimum_score_delta"]
        and effect["conversion_ratio"]["defined_for_all_resamples"]
        and effect["conversion_ratio"]["market_bootstrap_95"][0]
        >= limits["minimum_conversion_ratio"]
        and effect["violation_delta"]["market_bootstrap_95"][1]
        <= limits["maximum_violation_increase"]
        and sum(v > 0 for v in effect["seed_score_deltas"].values())
        >= limits["minimum_positive_seeds"]
    )
    effect["decision_rule"] = limits
    effect["model_contribution_diagnostic"] = paired(
        confirmation_rows,
        candidate,
        matched_empirical,
        primary_scenario,
        limits["bootstrap_samples"],
    )
    effect["model_contribution_diagnostic"]["interpretation"] = (
        "Context/history-conditioned learned forecast versus pooled empirical forecast; "
        "does not isolate history or neural architecture value."
    )
    effect["exploratory_ablations"] = [
        paired(
            confirmation_rows, candidate, baseline, primary_scenario, limits["bootstrap_samples"]
        )
        for baseline in (
            "response",
            "response+correction",
            f"response-paced-{best_rate}",
            "response-paced-cumulative",
            "response-paced-discounted",
            "response-paced-budget-only",
        )
        if baseline != candidate
    ]
    guard_sources()
    write_report(directory, confirmation_rows, selection, effect)
    status("complete", accepted=effect["accepted"], rows=len(confirmation_rows))
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config), flush=True)
