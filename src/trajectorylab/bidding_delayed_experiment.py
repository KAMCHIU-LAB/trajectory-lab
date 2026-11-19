"""Source-only delayed-feedback study with explicit environment/policy boundaries."""

from __future__ import annotations

import argparse
import hashlib
import resource
import time
import tomllib
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

from trajectorylab.bidding.market import feedback_action, score
from trajectorylab.bidding_delayed_feedback import (
    DelayQueue,
    EventTape,
    build_event_market,
    event_delay,
    event_source_identity,
)
from trajectorylab.bidding_delayed_statistics import paired_comparison, validate_rows
from trajectorylab.bidding_pending_response import Cohort, EmpiricalDelayCDF, PendingEstimator
from trajectorylab.utils import fingerprint, read_json, write_json

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PolicyView:
    tick: int
    spent: float
    arrived: int
    budget: float
    cpa: float
    cohorts: tuple[Cohort, ...]


class PublicLedger:
    """Only accepted public p and arrived event IDs; no latent positive IDs."""

    def __init__(self, edges):
        self.edges = np.asarray(edges)
        self.groups = {}
        self.accepted = {}
        self.arrived = 0

    def accept(self, public_cohort):
        probabilities = np.asarray(public_cohort.public_p)
        bins = np.searchsorted(self.edges, probabilities)
        for b in range(len(self.edges) + 1):
            mask = bins == b
            if mask.any():
                self.groups[(public_cohort.tick, b)] = [
                    int(mask.sum()),
                    float(probabilities[mask].sum()),
                    0,
                ]
        self.accepted.update(
            (event, (public_cohort.tick, int(b)))
            for event, b in zip(public_cohort.event_ids, bins, strict=True)
        )

    def receive(self, arrived_ids):
        for event in arrived_ids:
            key = self.accepted.pop(event)
            self.groups[key][2] += 1
            self.arrived += 1

    def snapshot(self, decision_tick):
        return tuple(
            Cohort(decision_tick - 1 - tick, psum / n, arrived, n)
            for (tick, _), (n, psum, arrived) in self.groups.items()
        )


def feedback_from_view(view: PolicyView, actions, total_estimate: float, gain: float, prior: float):
    """No hidden scenario, final labels, or unselected action response inputs."""
    state = np.zeros(13, dtype=np.float32)
    state[0] = view.tick / 48
    state[3] = view.spent / view.budget
    state[4] = total_estimate * view.cpa / view.budget
    return feedback_action(state, actions, gain, prior, view.budget, view.cpa)


def variants(config):
    e = config["evaluation"]
    return [
        dict(name=f"{method}-g{gain:g}-p{prior:g}", estimator=method, gain=gain, prior=prior)
        for method in e["methods"]
        for gain, prior in zip(e["gains"], e["priors"], strict=True)
    ]


def episode(
    tape,
    fi,
    prediction_multiplier,
    regime,
    arrival_seed,
    variant,
    cdf,
    config,
    *,
    collect_delays=False,
):
    """Truth is owned here; only PolicyView is used by ordinary decision code."""
    market = int(tape.arrays["market_id"])
    focal = int(tape.arrays["focals"][fi])
    budget, cpa = float(tape.arrays["budgets"][fi]), float(tape.arrays["cpas"][fi])
    actions = np.asarray(config["study"]["actions"])
    queue = DelayQueue(market, focal, arrival_seed, regime)
    ledger = PublicLedger(config["evaluation"]["probability_bin_edges"])
    kind = variant["estimator"]
    estimator = (
        None
        if kind == "oracle_past"
        else PendingEstimator(cdf, kind, prior_log_sd=config["history"]["calibration_prior_log_sd"])
    )
    spent, truth, wins = 0.0, 0, 0
    chosen, delay_samples, pending_errors = [], [], []
    for tick in range(48):
        ledger.receive(queue.advance(tick - 1))
        view = PolicyView(tick, spent, ledger.arrived, budget, cpa, ledger.snapshot(tick))
        if kind == "oracle_past":
            # Explicit nondeployable diagnostic: past final labels, never future opportunities.
            estimated = float(truth)
        else:
            estimated = estimator.estimate(view.cohorts).total_mean
        pending_errors.append(abs((estimated - view.arrived) - (truth - view.arrived)))
        action = feedback_from_view(view, actions, estimated, variant["gain"], variant["prior"])
        settled = tape.settle_events(fi, tick, action * prediction_multiplier, budget - spent)
        public = queue.register(tick, settled, prediction_multiplier=prediction_multiplier)
        ledger.accept(public)
        spent += settled.cost
        truth += settled.conversions
        wins += settled.wins
        chosen.append(float(action))
        if collect_delays:
            delay_samples.extend(
                event_delay(market, focal, event, arrival_seed, regime)
                for event in settled.positive_ids
            )
    ledger.receive(queue.advance(48 + config["history"]["maturity_window"]))
    if ledger.arrived != truth:
        raise RuntimeError("Final maturity failed or labels counted twice")
    if spent > budget + 1e-6:
        raise RuntimeError("Settlement exceeded hard budget")
    group = {0.75: "under", 1.0: "nominal", 1.5: "over"}[prediction_multiplier]
    row = dict(
        market=market,
        focal=focal,
        bias_group=group,
        delay=regime,
        arrival_seed=arrival_seed,
        method=variant["name"],
        estimator=kind,
        gain=variant["gain"],
        prior=variant["prior"],
        cost=spent,
        conversions=truth,
        exposures=wins,
        budget=budget,
        target_cpa=cpa,
        cpa=spent / truth if truth else None,
        violation=int(spent > cpa * truth + 1e-8),
        score=score(spent, truth, cpa),
        budget_utilization=spent / budget,
        pending_mae=float(np.mean(pending_errors)),
        actions=chosen,
        final_arrived=ledger.arrived,
        zero_conversions=int(truth == 0),
        deployable=kind != "oracle_past",
    )
    return row, delay_samples


def load_config(path):
    return tomllib.loads(Path(path).read_text())


def source_record():
    names = [
        "bidding_delayed_experiment.py",
        "bidding_delayed_feedback.py",
        "bidding_pending_response.py",
        "bidding_delayed_statistics.py",
    ]
    source = {
        f"src/trajectorylab/{name}": hashlib.sha256(
            (ROOT / "src/trajectorylab" / name).read_bytes()
        ).hexdigest()
        for name in names
    }
    for name in ("market.py", "learning.py"):
        source[f"src/trajectorylab/bidding/{name}"] = hashlib.sha256(
            (ROOT / "src/trajectorylab/bidding" / name).read_bytes()
        ).hexdigest()
    return source


def prepare(config, output, split):
    markets = config["study"][f"{split}_markets"]
    records = []
    for market in markets:
        meta = build_event_market(output / "markets" / f"{market}.npz", config["study"], market)
        records.append(meta)
        print(
            f"market-ready id={market} seconds={meta['seconds']:.2f} bytes={meta['bytes']}",
            flush=True,
        )
    return records


def fit_history(config, output):
    """Mature actual-winner train logs; never use confirmation positives for F."""
    if (output / "selection.json").exists():
        raise RuntimeError("Cannot refit delay history after selection is frozen")
    provenance = dict(
        markets=cache_provenance(config, output, "train"),
        source=source_record(),
        config_identity=fingerprint(config),
        behavior="arrived-only feedback, gain=.3, prior=8, nominal prediction; selected winners only",
        context_contract="separate mature training logs per stable deployment context; no test observations fitted",
    )
    result = {}
    reference = dict(name="history-feedback", estimator="naive", gain=0.3, prior=8.0)
    dummy = EmpiricalDelayCDF((1.0,) * 49, 0)
    for regime in config["evaluation"]["regimes"]:
        samples = []
        for market in config["study"]["train_markets"]:
            tape = EventTape(output / "markets" / f"{market}.npz")
            for fi in range(len(tape.arrays["focals"])):
                _, delays = episode(
                    tape,
                    fi,
                    1.0,
                    regime,
                    config["history"]["arrival_seed"],
                    reference,
                    dummy,
                    config,
                    collect_delays=True,
                )
                samples.extend(delays)
        cdf = EmpiricalDelayCDF.fit(
            samples,
            config["history"]["maturity_window"],
            split="train",
            all_mature=True,
            pseudocount=config["history"]["total_delay_pseudocount"],
        )
        result[regime] = dict(
            probabilities=list(cdf.probabilities),
            training_sample_count=cdf.training_sample_count,
            training_markets=config["study"]["train_markets"],
            all_mature=True,
        )
        print(f"history-fit regime={regime} mature_positive_samples={len(samples)}", flush=True)
    write_json(output / "history.json", result)
    write_json(
        output / "history-manifest.json",
        dict(
            identity=fingerprint(provenance),
            core=provenance,
            history_sha256=hashlib.sha256((output / "history.json").read_bytes()).hexdigest(),
        ),
    )
    return result


def read_history(output):
    return {
        k: EmpiricalDelayCDF(tuple(v["probabilities"]), v["training_sample_count"])
        for k, v in read_json(output / "history.json").items()
    }


def check_frozen_selection(config, output):
    selection = read_json(output / "selection.json")
    if selection["source"] != source_record() or selection["config_identity"] != fingerprint(
        config
    ):
        raise RuntimeError("Source/config changed after development selection")
    if (
        selection["history_sha256"]
        != hashlib.sha256((output / "history.json").read_bytes()).hexdigest()
    ):
        raise RuntimeError("Mature training history changed after selection")
    return selection


def cache_provenance(config, output, split):
    """Reject stale/swapped sources and include actual tape bytes in the manifest."""
    records = {}
    study = {k: config["study"][k] for k in ("focal_agents", "pv_count", "actions", "biases")}
    source = event_source_identity()
    for market in config["study"][f"{split}_markets"]:
        path = output / "markets" / f"{market}.npz"
        meta = read_json(path.with_suffix(".json"))
        expected = fingerprint(dict(study=study, market=market, source=source))
        if meta["identity"] != expected or meta["source"] != source:
            raise RuntimeError("Market cache source identity changed")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        records[str(market)] = dict(
            identity=meta["identity"], sha256=digest.hexdigest(), source=source
        )
    return records


def validate_grid(rows, config, split, methods):
    coverage = validate_rows(rows)
    expected = set(
        product(
            config["study"][f"{split}_markets"],
            config["study"]["focal_agents"],
            ("under", "nominal", "over"),
            config["evaluation"]["regimes"],
            config["evaluation"]["arrival_seeds"],
        )
    )
    for method in methods:
        actual = {
            tuple(r[k] for k in ("market", "focal", "bias_group", "delay", "arrival_seed"))
            for r in rows
            if r["method"] == method["name"]
        }
        if actual != expected:
            raise RuntimeError("Evaluation is missing declared campaign cells")
    if set(coverage["methods"]) != {m["name"] for m in methods}:
        raise RuntimeError("Unexpected evaluation method")
    return coverage


def evaluate(config, output, split, selected=None):
    if split == "confirmation":
        frozen = check_frozen_selection(config, output)
        if selected != frozen["frozen_variants"]:
            raise RuntimeError("Confirmation methods differ from frozen selection")
    methods = variants(config) if selected is None else selected
    core = dict(
        config=config,
        source=source_record(),
        markets=cache_provenance(config, output, split),
        history_sha256=hashlib.sha256((output / "history.json").read_bytes()).hexdigest(),
        methods=methods,
        split=split,
    )
    identity = fingerprint(core)
    destination = output / split
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path)["identity"] != identity:
        raise RuntimeError("Frozen evaluation changed; use a new output identity")
    write_json(
        manifest_path,
        dict(
            identity=identity,
            core=core,
            claim="custom fixed-opponent simulation; mature labels; historical delay regime matched; finite declared arrival seeds",
        ),
    )
    history = read_history(output)
    rows = []
    started = time.perf_counter()
    for market in config["study"][f"{split}_markets"]:
        path = destination / "market-results" / f"{market}.json"
        if path.exists():
            saved = read_json(path)
            if saved["identity"] != identity:
                raise RuntimeError("Market checkpoint identity changed")
            rows.extend(saved["rows"])
            continue
        tape = EventTape(output / "markets" / f"{market}.npz")
        local = []
        for variant in methods:
            tick_start = time.perf_counter()
            for bias in config["study"]["biases"]:
                for regime in config["evaluation"]["regimes"]:
                    for arrival_seed in config["evaluation"]["arrival_seeds"]:
                        for fi in range(len(tape.arrays["focals"])):
                            row, _ = episode(
                                tape,
                                fi,
                                bias,
                                regime,
                                arrival_seed,
                                variant,
                                history[regime],
                                config,
                            )
                            local.append(row)
            print(
                f"evaluated split={split} market={market} method={variant['name']} seconds={time.perf_counter() - tick_start:.2f}",
                flush=True,
            )
        validate_rows(local)
        write_json(path, dict(identity=identity, rows=local))
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
    coverage = validate_grid(rows, config, split, methods)
    write_json(destination / "results.json", rows)
    write_json(
        destination / "status.json",
        dict(
            identity=identity,
            phase="complete",
            **coverage,
            elapsed_seconds=time.perf_counter() - started,
            peak_rss_native=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        ),
    )
    return rows


def select_development(config, output):
    if (output / "confirmation" / "manifest.json").exists():
        raise RuntimeError("Cannot reselect after confirmation has been opened")
    rows = read_json(output / "validation" / "results.json")
    options = variants(config)
    validate_grid(rows, config, "validation", options)
    means = {
        v["name"]: np.mean([r["score"] for r in rows if r["method"] == v["name"]]) for v in options
    }
    winners = [
        max((v for v in options if v["estimator"] == k), key=lambda v: means[v["name"]])
        for k in config["evaluation"]["methods"]
    ]
    point = max(
        (v for v in winners if v["estimator"] != "oracle_past"), key=lambda v: means[v["name"]]
    )
    oracle = next(v for v in winners if v["estimator"] == "oracle_past")
    a = {
        (r["market"], r["focal"], r["bias_group"], r["delay"], r["arrival_seed"]): r
        for r in rows
        if r["method"] == point["name"]
    }
    b = {
        (r["market"], r["focal"], r["bias_group"], r["delay"], r["arrival_seed"]): r
        for r in rows
        if r["method"] == oracle["name"]
    }
    action_change = float(
        np.mean(
            [x != y for key in a for x, y in zip(a[key]["actions"], b[key]["actions"], strict=True)]
        )
    )
    gain = (
        float(means[oracle["name"]] / means[point["name"]] - 1)
        if means[point["name"]] > 0
        else None
    )
    selection = dict(
        validation_identity=read_json(output / "validation" / "manifest.json")["identity"],
        frozen_variants=winners,
        strongest_point=point,
        diagnostic_oracle=oracle,
        mean_scores={v["name"]: float(means[v["name"]]) for v in winners},
        oracle_diagnostic_gain=gain,
        oracle_action_change_fraction=action_change,
        generative_investment_gate=bool(
            gain is not None
            and gain >= config["limits"]["oracle_diagnostic_gain"]
            and action_change >= config["limits"]["oracle_diagnostic_action_change"]
        ),
        diagnostic_scope="past-outcome feedback heuristic; not a global optimum or proof that no better policy exists",
        selection_scope="best macro development score within each estimator; confirmation protection gates remain binding",
        source=source_record(),
        config_identity=fingerprint(config),
        history_sha256=hashlib.sha256((output / "history.json").read_bytes()).hexdigest(),
        history_contract="Each deployment context supplies its own frozen mature TRAIN history. Context-specific history is matched and stable; the policy gets neither test delay parameters nor context/regime labels. This does not infer unknown delay regimes or handle drift.",
        mature_only_scope="Mature-cohort calibration plus pending imputation. H=W=48 provides no fully mature live cohort, so calibration remains the prior MAP; this is not a controller that discards all immature conversions.",
    )
    selection["identity"] = fingerprint(selection)
    write_json(output / "selection.json", selection)
    return selection


def summarise(config, output, split):
    rows = read_json(output / split / "results.json")
    selection = check_frozen_selection(config, output)
    baseline = selection["strongest_point"]["name"]
    methods = sorted({r["method"] for r in rows})
    naive = next(v["name"] for v in selection["frozen_variants"] if v["estimator"] == "naive")
    comparison_args = dict(
        draws=config["evaluation"]["bootstrap_draws"],
        seed=config["evaluation"]["bootstrap_seed"],
        minimum_markets=config["limits"]["minimum_confirmation_markets"],
    )
    comparisons = [
        paired_comparison(rows, method, baseline, **comparison_args)
        for method in methods
        if method != baseline
    ]
    primary = paired_comparison(rows, baseline, naive, **comparison_args)
    cells = []
    for method in methods:
        for group in ("nominal", "under", "over"):
            for regime in config["evaluation"]["regimes"]:
                s = [
                    r
                    for r in rows
                    if r["method"] == method and r["bias_group"] == group and r["delay"] == regime
                ]
                cells.append(
                    dict(
                        method=method,
                        bias_group=group,
                        delay=regime,
                        campaigns=len(s),
                        score=float(np.mean([r["score"] for r in s])),
                        conversions=float(np.mean([r["conversions"] for r in s])),
                        violation=float(np.mean([r["violation"] for r in s])),
                        budget_utilization=float(np.mean([r["budget_utilization"] for r in s])),
                        pending_mae=float(np.mean([r["pending_mae"] for r in s])),
                    )
                )
    result = dict(
        split=split,
        coverage=validate_rows(rows),
        strongest_point=baseline,
        primary_estimand="development-selected strongest deployable point vs development-tuned arrived-only feedback; no generative novelty claim",
        primary_comparison=primary,
        other_comparisons_scope="secondary/descriptive; not additional unadjusted primary hypothesis claims",
        selection=selection,
        comparisons=comparisons,
        cells=cells,
        final_metrics="matured labels at H+W; no additional opportunities after H",
        independent_market_count=len({r["market"] for r in rows}),
    )
    write_json(output / split / "summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "configs/bidding-delayed-feedback.toml")
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/bidding/delayed-feedback")
    p.add_argument(
        "--stage", choices=["prepare", "history", "evaluate", "select", "summary"], required=True
    )
    p.add_argument("--split", choices=["train", "validation", "confirmation"], default="validation")
    args = p.parse_args()
    config = load_config(args.config)
    if args.stage == "prepare":
        if args.split == "confirmation" and not (args.output / "selection.json").exists():
            raise RuntimeError("Freeze development selection before opening confirmation markets")
        if args.split == "confirmation":
            check_frozen_selection(config, args.output)
        prepare(config, args.output, args.split)
    elif args.stage == "history":
        fit_history(config, args.output)
    elif args.stage == "evaluate":
        selected = (
            read_json(args.output / "selection.json")["frozen_variants"]
            if args.split == "confirmation"
            else None
        )
        evaluate(config, args.output, args.split, selected)
    elif args.stage == "select":
        select_development(config, args.output)
    else:
        summarise(config, args.output, args.split)


if __name__ == "__main__":
    main()
