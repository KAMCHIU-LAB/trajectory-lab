"""Bounded feedback-use extension; first-study confirmation is never an input.

Calibration posteriors assume matched stable mature TRAIN delay history. Their
probabilities and information proxy are working diagnostics, not CPA guarantees.
No future cost/conversion joint response model is available here.
"""

from __future__ import annotations

import argparse
import hashlib
import time
from itertools import product
from pathlib import Path

import numpy as np

from trajectorylab import bidding_delayed_experiment as base
from trajectorylab.bidding.market import score
from trajectorylab.bidding_delayed_feedback import DelayQueue, EventTape, build_event_market
from trajectorylab.bidding_delayed_statistics import validate_rows
from trajectorylab.bidding_pending_response import PendingEstimator
from trajectorylab.utils import fingerprint, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


def variants():
    """Ordered simplicity tie-break; exactly twelve registered configurations."""
    result = [
        dict(name="fixed-naive", family="fixed", pending=0.0),
        dict(name="fixed-posterior", family="fixed", pending=1.0),
    ]
    result += [dict(name=f"partial-{x:g}", family="partial", pending=x) for x in (0.25, 0.5)]
    result += [
        dict(name=f"gate-{x:g}", family="gate", pending=1.0, threshold=x) for x in (0.8, 0.95)
    ]
    result += [
        dict(name=f"centered-eta{eta:g}-pending{p:g}", family="centered", eta=eta, pending=p)
        for eta in (0.5, 1.0)
        for p in (0.0, 1.0)
    ]
    result += [
        dict(name=f"absolute-eta{eta:g}", family="absolute", eta=eta, pending=1.0)
        for eta in (0.5, 1.0)
    ]
    return result


def posterior_state(view, estimator):
    """One distribution calculation per view; all policy statistics reuse it.

    The private frozen-core tuple is an explicit adapter contract covered by
    tests and included in source fingerprints. It does not expose latent truth.
    """
    rows, weights, pending, _ = estimator._distribution(view.cohorts)
    prior = np.exp(estimator.log_prior - estimator.log_prior.max())
    prior /= prior.sum()
    logs = np.log(estimator.scales)
    return dict(
        pending=float(weights @ pending),
        log_scale=float(weights @ logs),
        prior_log_scale=float(prior @ logs),
        over_probability=float(weights[estimator.scales < 0.9].sum()),
        information=float(sum(n * p * f for _, n, p, _, f in rows)),
    )


def policy_action(view, actions, variant, state, settings):
    """Multiply continuous action BEFORE one projection onto the declared grid."""
    lam, factor = variant["pending"], 1.0
    family = variant["family"]
    if (
        family == "gate"
        and state["information"] >= settings["information_threshold"]
        and state["over_probability"] >= variant["threshold"]
    ):
        lam = 0.25
    if family in {"centered", "absolute"}:
        logfactor = state["log_scale"] - state["prior_log_scale"]
        if family == "absolute":
            # Zero information -> neutral; abundant information -> absolute k^eta.
            j = state["information"]
            logfactor = j / (j + settings["information_threshold"]) * state["log_scale"]
        factor = float(
            np.clip(
                np.exp(variant["eta"] * logfactor),
                settings["factor_lower"],
                settings["factor_upper"],
            )
        )
    v = view.arrived + lam * state["pending"]
    gain, prior = settings["gain"], settings["prior"]
    if factor == 1.0:
        return base.feedback_from_view(view, actions, v, gain, prior)
    elapsed = float(np.float32(view.tick / 48))
    spent = float(np.float32(view.spent / view.budget)) * view.budget
    conv = float(np.float32(v * view.cpa / view.budget)) * view.budget / view.cpa
    pacing = (((elapsed + 0.05) * view.budget) / (spent + 0.05 * view.budget)) ** gain
    quality = (((conv + prior) * view.cpa) / (spent + prior * view.cpa)) ** gain
    desired = np.clip(pacing * quality * factor, 0.05, 3.0)
    return float(actions[np.abs(actions - desired).argmin()])


def episode(tape, fi, multiplier, regime, seed, variant, cdf, config):
    market, focal = int(tape.arrays["market_id"]), int(tape.arrays["focals"][fi])
    budget, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
    queue, ledger = (
        DelayQueue(market, focal, seed, regime),
        base.PublicLedger(config["evaluation"]["probability_bin_edges"]),
    )
    estimator = PendingEstimator(cdf, prior_log_sd=config["history"]["calibration_prior_log_sd"])
    spent, truth, wins, errors = 0.0, 0, 0, []
    for tick in range(48):
        ledger.receive(queue.advance(tick - 1))
        view = base.PolicyView(tick, spent, ledger.arrived, budget, cpa, ledger.snapshot(tick))
        state = posterior_state(view, estimator)
        action = policy_action(
            view, np.asarray(config["study"]["actions"]), variant, state, config["strategy"]
        )
        errors.append(abs(state["pending"] - (truth - view.arrived)))
        settled = tape.settle_events(fi, tick, action * multiplier, budget - spent)
        ledger.accept(queue.register(tick, settled, prediction_multiplier=multiplier))
        spent += settled.cost
        truth += settled.conversions
        wins += settled.wins
    ledger.receive(queue.advance(48 + config["history"]["maturity_window"]))
    if ledger.arrived != truth or spent > budget + 1e-6:
        raise RuntimeError("Settlement/maturity accounting failed")
    return dict(
        market=market,
        focal=focal,
        bias_group={0.75: "under", 1.0: "nominal", 1.5: "over"}[multiplier],
        delay=regime,
        arrival_seed=seed,
        method=variant["name"],
        cost=spent,
        conversions=truth,
        score=score(spent, truth, cpa),
        exposures=wins,
        budget=budget,
        target_cpa=cpa,
        cpa=spent / truth if truth else None,
        violation=int(spent > cpa * truth + 1e-8),
        budget_utilization=spent / budget,
        posterior_pending_mae=float(np.mean(errors)),
        deployable=True,
    )


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_record(config_path):
    names = [
        "src/trajectorylab/bidding_delayed_strategy.py",
        "scripts/run_bidding_delayed_strategy.py",
    ]
    return dict(
        frozen_core=base.source_record(),
        extension={n: file_hash(ROOT / n) for n in names},
        config_sha256=file_hash(config_path),
        registration_sha256=file_hash(ROOT / "docs/bidding-delayed-strategy-protocol.md"),
    )


def freeze(path, payload):
    if path.exists() and read_json(path) != payload:
        raise RuntimeError("Frozen artifact differs; use a new output identity")
    write_json(path, payload)


def rows_for(rows, method, group=None, delay=None):
    return [
        r
        for r in rows
        if r["method"] == method
        and (group is None or r["bias_group"] == group)
        and (delay is None or r["delay"] == delay)
    ]


def means(rows):
    return {
        k: float(np.mean([r[k] for r in rows]))
        for k in ("score", "conversions", "violation", "budget_utilization")
    }


def retention(candidate, reference):
    c, b = means(candidate), means(reference)
    return dict(
        score_ratio=c["score"] / b["score"] if b["score"] else None,
        conversion_ratio=c["conversions"] / b["conversions"] if b["conversions"] else None,
        violation_delta=c["violation"] - b["violation"],
    )


def development_selection(rows, config):
    references = ["fixed-naive", "fixed-posterior"]
    stronger = max(references, key=lambda m: means(rows_for(rows, m))["score"])
    table = []
    for variant in variants():
        name = variant["name"]
        macro = retention(rows_for(rows, name), rows_for(rows, stronger))
        bias = {
            g: retention(rows_for(rows, name, g), rows_for(rows, "fixed-naive", g))
            for g in ("under", "nominal", "over")
        }
        over = {
            d: retention(rows_for(rows, name, "over", d), rows_for(rows, "fixed-naive", "over", d))
            for d in config["evaluation"]["regimes"]
        }
        per_market = {
            str(m): means([r for r in rows_for(rows, name) if r["market"] == m])
            for m in config["study"]["validation_markets"]
        }
        wins = sum(
            v["score"]
            > means([r for r in rows_for(rows, stronger) if r["market"] == int(m)])["score"]
            for m, v in per_market.items()
        )
        gates = dict(
            macro_uplift=macro["score_ratio"] is not None and macro["score_ratio"] >= 1.03,
            nominal_under_retention=all(
                bias[g]["score_ratio"] is not None
                and bias[g]["conversion_ratio"] is not None
                and bias[g]["score_ratio"] >= 0.97
                and bias[g]["conversion_ratio"] >= 0.97
                for g in ("nominal", "under")
            ),
            bias_violation=all(v["violation_delta"] <= 0.03 for v in bias.values()),
            over_cell_retention=all(
                v["score_ratio"] is not None
                and v["conversion_ratio"] is not None
                and v["score_ratio"] >= 0.99
                and v["conversion_ratio"] >= 0.97
                and v["violation_delta"] <= 0.03
                for v in over.values()
            ),
            market_wins=wins >= config["strategy"]["minimum_dev_wins"],
        )
        table.append(
            dict(
                variant=variant,
                means=means(rows_for(rows, name)),
                macro_vs_stronger=macro,
                vs_fixed_references={
                    r: retention(rows_for(rows, name), rows_for(rows, r)) for r in references
                },
                bias_vs_naive=bias,
                over_cells_vs_naive=over,
                market_means=per_market,
                market_wins=wins,
                gates=gates,
                passed=all(gates.values()),
            )
        )
    passing = [r for r in table if r["passed"]]
    selected = None
    if passing:
        best = max(r["means"]["score"] for r in passing)
        eligible = [r for r in passing if r["means"]["score"] >= 0.995 * best]
        rank = {"fixed": 0, "partial": 1, "gate": 2, "centered": 3, "absolute": 3}
        order = {v["name"]: i for i, v in enumerate(variants())}
        selected = min(
            eligible, key=lambda r: (rank[r["variant"]["family"]], order[r["variant"]["name"]])
        )["variant"]
    return dict(
        gate=selected is not None,
        selected=selected,
        stronger_fixed_reference=stronger,
        table=table,
        stop_reason=None if selected else "All registered configurations failed; no further search",
        protection_endpoints=10,
        confirmation_scope="One selected policy; primary score vs frozen stronger fixed reference; protection vs fixed naive",
    )


def confirmation_statistics(rows, selection, config):
    """Market bootstrap; ten simultaneous one-sided protection endpoints."""
    selected, baseline = selection["selected"]["name"], selection["stronger_fixed_reference"]
    markets = sorted({r["market"] for r in rows})
    rng = np.random.default_rng(config["evaluation"]["bootstrap_seed"])
    index = rng.integers(0, len(markets), (config["evaluation"]["bootstrap_draws"], len(markets)))

    def arrays(method, group=None):
        return np.array(
            [
                [
                    means([r for r in rows_for(rows, method, group) if r["market"] == m])[k]
                    for k in ("score", "conversions", "violation")
                ]
                for m in markets
            ]
        )

    x, b = arrays(selected), arrays(baseline)
    if b[:, 0].mean() == 0 or np.any(b[index, 0].mean(axis=1) == 0):
        return dict(status="incomparable_zero_baseline", passed=False)
    ratios = x[index, 0].mean(axis=1) / b[index, 0].mean(axis=1)
    primary = dict(
        score_ratio=float(x[:, 0].mean() / b[:, 0].mean()),
        ci95=np.quantile(ratios, [0.025, 0.975]).tolist(),
    )
    alpha = 0.05 / 10
    protection = {}
    valid = True
    for g in (None, "under", "nominal", "over"):
        xx, nn = arrays(selected, g), arrays("fixed-naive", g)
        xb, nb = xx[index].mean(axis=1), nn[index].mean(axis=1)
        detail = dict(
            violation_delta=float(xx[:, 2].mean() - nn[:, 2].mean()),
            violation_upper=float(np.quantile(xb[:, 2] - nb[:, 2], 1 - alpha)),
        )
        valid &= detail["violation_upper"] <= 0.03
        if g is not None:
            for col, key in [(0, "score"), (1, "conversion")]:
                if nn[:, col].mean() == 0 or np.any(nb[:, col] == 0):
                    detail[key + "_retention_lower"] = None
                    valid = False
                else:
                    detail[key + "_retention_lower"] = float(
                        np.quantile(xb[:, col] / nb[:, col], alpha)
                    )
                    valid &= detail[key + "_retention_lower"] >= 0.97
        protection[g or "all"] = detail
    sufficient = len(markets) >= config["strategy"]["minimum_confirmation_markets"]
    return dict(
        status="estimated",
        independent_unit="market",
        independent_markets=len(markets),
        primary_baseline=baseline,
        protection_baseline="fixed-naive",
        primary=primary,
        protection=protection,
        protection_endpoints=10,
        one_sided_alpha=alpha,
        protection_pass=bool(valid),
        sufficient_markets=sufficient,
        passed=bool(
            valid and sufficient and primary["score_ratio"] >= 1.03 and primary["ci95"][0] > 1
        ),
    )


def validate_grid(rows, config, markets, methods):
    validate_rows(rows)
    expected = set(
        product(
            markets,
            config["study"]["focal_agents"],
            ("under", "nominal", "over"),
            config["evaluation"]["regimes"],
            config["evaluation"]["arrival_seeds"],
        )
    )
    if {r["method"] for r in rows} != {v["name"] for v in methods}:
        raise RuntimeError("Method coverage differs from registered grid")
    for variant in methods:
        actual = {
            tuple(r[k] for k in ("market", "focal", "bias_group", "delay", "arrival_seed"))
            for r in rows_for(rows, variant["name"])
        }
        if actual != expected:
            raise RuntimeError("Campaign coverage differs from registered grid")


def run(config_path, input_dir, output, stage):
    config_path, input_dir, output = Path(config_path), Path(input_dir), Path(output)
    config = base.load_config(config_path)
    # Hard boundary: neither selected scenarios nor first-study confirmation may
    # be silently substituted through a modified configuration.
    if config["study"]["validation_markets"] != list(range(2001, 2007)) or config["study"][
        "confirmation_markets"
    ] != list(range(7101, 7125)):
        raise RuntimeError("Only registered development and fresh confirmation IDs allowed")
    if output.resolve() == input_dir.resolve() or input_dir.resolve() in output.resolve().parents:
        raise RuntimeError("Extension output must be separate from frozen input")
    history_manifest = read_json(input_dir / "history-manifest.json")
    history_hash = file_hash(input_dir / "history.json")
    if (
        history_manifest["identity"] != fingerprint(history_manifest["core"])
        or history_manifest["history_sha256"] != history_hash
        or history_manifest["core"]["source"] != base.source_record()
    ):
        raise RuntimeError("Frozen training history/source identity changed")
    provenance = dict(
        config=config,
        source=source_record(config_path),
        history_sha256=history_hash,
        training_history_identity=history_manifest["identity"],
    )
    selection = None
    if stage == "confirm":
        selection = read_json(output / "selection.json")
        body = {k: v for k, v in selection.items() if k != "identity"}
        if (
            selection["identity"] != fingerprint(body)
            or selection["provenance"] != provenance
            or not selection["gate"]
        ):
            raise RuntimeError("Confirmation requires unchanged frozen passing selection")
        methods = [selection["selected"], *variants()[:2]]
        methods = list({v["name"]: v for v in methods}.values())
        split = "confirmation"
        cache_root = output
        for market in config["study"]["confirmation_markets"]:
            build_event_market(output / "markets" / f"{market}.npz", config["study"], market)
    else:
        if (output / "confirmation/manifest.json").exists():
            raise RuntimeError("Development is sealed after confirmation opens")
        methods, split, cache_root = variants(), "validation", input_dir
    markets = config["study"][f"{split}_markets"]
    core = dict(
        **provenance,
        split=split,
        methods=methods,
        markets=base.cache_provenance(config, cache_root, split),
        selection_identity=selection["identity"] if selection else None,
    )
    identity = fingerprint(core)
    destination = output / split
    freeze(destination / "manifest.json", dict(identity=identity, core=core))
    history = base.read_history(input_dir)
    started = time.perf_counter()
    rows = []
    for market in markets:
        checkpoint = destination / "market-results" / f"{market}.json"
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if saved["identity"] != identity or saved["rows_sha256"] != fingerprint(saved["rows"]):
                raise RuntimeError("Market checkpoint integrity changed")
            local = saved["rows"]
            validate_grid(local, config, [market], methods)
        else:
            tape = EventTape(cache_root / "markets" / f"{market}.npz")
            local = []
            for variant in methods:
                begin = time.perf_counter()
                for multiplier, regime, seed, fi in product(
                    config["study"]["biases"],
                    config["evaluation"]["regimes"],
                    config["evaluation"]["arrival_seeds"],
                    range(len(tape.arrays["focals"])),
                ):
                    local.append(
                        episode(
                            tape, fi, multiplier, regime, seed, variant, history[regime], config
                        )
                    )
                print(
                    f"evaluated split={split} market={market} method={variant['name']} seconds={time.perf_counter() - begin:.2f}",
                    flush=True,
                )
            validate_grid(local, config, [market], methods)
            freeze(checkpoint, dict(identity=identity, rows_sha256=fingerprint(local), rows=local))
        rows.extend(local)
        write_json(
            destination / "status.json",
            dict(
                identity=identity,
                phase="running",
                completed_markets=len({r["market"] for r in rows}),
                rows=len(rows),
            ),
        )
    validate_grid(rows, config, markets, methods)
    freeze(destination / "results.json", rows)
    if stage == "develop":
        summary = development_selection(rows, config)
        selected = dict(
            **summary,
            provenance=provenance,
            development_identity=identity,
            development_results_sha256=fingerprint(rows),
        )
        selected["identity"] = fingerprint(selected)
        freeze(output / "selection.json", selected)
    else:
        summary = confirmation_statistics(rows, selection, config)
        summary["method_means"] = {v["name"]: means(rows_for(rows, v["name"])) for v in methods}
        summary["cells"] = [
            dict(method=v["name"], bias_group=g, delay=d, **means(rows_for(rows, v["name"], g, d)))
            for v, g, d in product(
                methods, ("under", "nominal", "over"), config["evaluation"]["regimes"]
            )
        ]
        summary["vs_full_posterior"] = retention(
            rows_for(rows, selection["selected"]["name"]), rows_for(rows, "fixed-posterior")
        )
    freeze(destination / "summary.json", dict(identity=identity, **summary))
    write_json(
        destination / "status.json",
        dict(
            identity=identity,
            phase="complete",
            rows=len(rows),
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["develop", "confirm"], required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/bidding-delayed-strategy.toml")
    )
    parser.add_argument("--input", type=Path, default=Path("artifacts/bidding/delayed-feedback"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bidding/delayed-strategy"))
    args = parser.parse_args()
    run(args.config, args.input, args.output, args.stage)


if __name__ == "__main__":
    main()
