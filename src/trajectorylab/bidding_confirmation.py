"""Independent market confirmation of frozen original response versus feedback."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import html
import io
import json
import os
import tempfile
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from trajectorylab import bidding_control
from trajectorylab.bidding.experiment import evaluate, provenance
from trajectorylab.bidding.learning import Feedback, Learned
from trajectorylab.bidding.market import ROOT, build_market, effective_actions
from trajectorylab.bidding_control import metrics, paired
from trajectorylab.utils import fingerprint, read_json, write_json

PROTOCOL = ROOT / "docs/bidding-confirmation-protocol.md"
SEEDS = [11, 23, 37]
SCENARIOS = ["calibrated", "over_25", "over_50", "under_25", "mid_over_50"]
SLOTS = [("response", s) for s in SEEDS] + [("feedback", -1), ("fixed-1", -1), ("fixed-tuned", -1)]
METRICS = ("score", "conversions", "violation", "cost", "budget_utilization", "zero_conversions")
PRIMARY = {
    "method": "response",
    "reference": "feedback",
    "scenario": "over_50",
    "bootstrap_samples": 10000,
    "bootstrap_rng": 29117,
    "score_ci_lower_strictly_above": 0.0,
    "conversion_ratio_ci_lower_at_least": 0.95,
    "violation_delta_ci_upper_at_most": 0.05,
    "positive_seed_means_required": 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Configuration paths must be repository-relative")
    return ROOT / path


def _prepare(config_path: Path) -> tuple[dict, dict, dict]:
    opts = tomllib.loads(config_path.read_text())
    main, control = _relative(opts["main_run"]), _relative(opts["control_run"])
    _relative(opts["output"])
    main_manifest, control_manifest = [read_json(p / "manifest.json") for p in (main, control)]
    main_selection, control_selection = [read_json(p / "selection.json") for p in (main, control)]
    main_status, control_status = [read_json(p / "status.json") for p in (main, control)]
    config = main_manifest["config"]
    study = config["study"]
    if main_status["phase"] != "complete" or control_status["phase"] != "complete-not-promoted":
        raise RuntimeError(
            "The frozen main/control studies must have their declared terminal states"
        )
    if fingerprint(provenance(config)) != main_manifest["identity"]:
        raise RuntimeError("Current main provenance differs from the frozen study")
    if main_manifest["identity"] != opts["frozen"]["main_identity"]:
        raise RuntimeError("Unexpected main source identity")
    if control_manifest["main_identity"] != main_manifest["identity"]:
        raise RuntimeError("Control and main records do not share a source identity")
    if (
        control_selection["primary_reference"] != "feedback"
        or control_selection["primary_scenario"] != "over_50"
        or main_selection["feedback"] != {"gain": 0.3, "prior": 8.0}
        or main_selection["fixed"]["action"] != 1.25
    ):
        raise RuntimeError("Frozen strongest primary reference or main policy parameters changed")
    if (
        opts["markets"] != list(range(3301, 3325))
        or opts["training_seeds"] != SEEDS
        or opts["focal_agents"] != [0, 8, 16, 24, 32, 40]
        or opts["pv_count"] != 500000
        or opts["scenarios"] != SCENARIOS
        or opts["primary"] != PRIMARY
        or opts["feedback_gain"] != 0.3
        or opts["feedback_prior"] != 8.0
        or opts["fixed_tuned"] != 1.25
        or opts["device"] != "cpu"
        or opts["torch_threads"] != 2
        or config["training"]["context"] != 24
        or any(
            opts[k] != study[k]
            for k in ("training_seeds", "focal_agents", "pv_count", "actions", "biases")
        )
        or config["evaluation"]["scenarios"] != SCENARIOS
    ):
        raise ValueError("Configuration differs from the fixed confirmation design")
    previous = sum([study[k] for k in ("train_markets", "validation_markets", "test_markets")], [])
    previous += control_manifest["evidence_layout"]["confirmation_markets"]
    if set(previous).intersection(opts["markets"]):
        raise ValueError("Confirmation markets must be entirely new")
    sources = {
        f"{label}_{kind}_sha256": _sha256(directory / f"{kind}.json")
        for label, directory in (("main", main), ("control", control))
        for kind in ("manifest", "selection", "status")
    }
    if any(opts["frozen"][k] != v for k, v in sources.items()):
        raise RuntimeError("A frozen main/control record changed")
    checkpoints = {str(s): _sha256(main / "models" / f"response-{s}" / "best.pt") for s in SEEDS}
    if checkpoints != opts["checkpoint_sha256"]:
        raise RuntimeError("A frozen response checkpoint changed")
    finished = [r for r in read_json(main / "training.json") if r["kind"] == "response"]
    if (
        len(finished) != 3
        or {r["seed"] for r in finished} != set(SEEDS)
        or any(r["steps"] != config["training"]["steps"] for r in finished)
    ):
        raise RuntimeError("All three response trainings must be finished")
    dependencies = {
        p.name: _sha256(p)
        for p in (
            Path(__file__),
            Path(bidding_control.__file__),
            ROOT / "src/trajectorylab/utils.py",
        )
    }
    if dependencies["bidding_control.py"] != control_manifest["implementation_sha256"]:
        raise RuntimeError("The frozen paired-bootstrap implementation changed")
    core = {
        "config": opts,
        "config_sha256": _sha256(config_path),
        "protocol_sha256": _sha256(PROTOCOL),
        "implementation": dependencies,
        "main_identity": main_manifest["identity"],
        "main_implementation": main_manifest["implementation"],
        "main_auction": main_manifest["auction"],
        "frozen_records": sources,
        "checkpoint_sha256": checkpoints,
        "training_sha256": _sha256(main / "training.json"),
        "control_selection": control_selection,
        "control_status": control_status,
        "design": "independent confirmation D, declared after main/control discovery",
        "uncertainty_scope": "24 new markets, conditional on three fixed response checkpoints",
        "policy_slots": [list(slot) for slot in SLOTS],
        "expected_rows": 4320,
    }
    return opts, config, core


def _seal_markets(directory: Path, opts: dict, study: dict, core: dict, identity: str) -> dict:
    paths = [directory / "markets" / f"{m}.npz" for m in opts["markets"]]
    # Establish completeness before reading any tape bytes or policy-result cache.
    if not all(p.is_file() and p.with_suffix(".json").is_file() for p in paths):
        raise RuntimeError("All 24 complete market caches are required before evaluation")
    expected_files = {f"{m}{suffix}" for m in opts["markets"] for suffix in (".npz", ".json")}
    actual = {p.name for p in (directory / "markets").iterdir() if p.suffix in (".npz", ".json")}
    if actual != expected_files:
        raise RuntimeError("Unexpected market cache coverage")
    auction = core["main_auction"]
    caches = {}
    for path in paths:
        market, meta = int(path.stem), read_json(path.with_suffix(".json"))
        expected = fingerprint(
            {
                "study": {k: study[k] for k in ("focal_agents", "pv_count", "actions", "biases")},
                "market": market,
                "source": auction,
            }
        )
        if (
            meta["identity"] != expected
            or meta["market_id"] != market
            or meta["source"] != auction
            or meta["focal_agents"] != opts["focal_agents"]
            or meta["requested_pv_count"] != opts["pv_count"]
            or meta["bytes"] != path.stat().st_size
            or meta["effective_actions"] != effective_actions(study).tolist()
        ):
            raise RuntimeError("Market metadata differs from the frozen construction")
        caches[str(market)] = {
            "identity": expected,
            "npz_sha256": _sha256(path),
            "metadata_sha256": _sha256(path.with_suffix(".json")),
        }
    seal = {"identity": identity, "markets": caches}
    seal_path = directory / "market-cache.json"
    if seal_path.exists() and read_json(seal_path) != seal:
        raise RuntimeError("A sealed market cache changed")
    write_json(seal_path, seal)
    return seal


def _rows(rows: list[dict], markets: list[int], opts: dict) -> None:
    names = ("market", "focal", "scenario", "method", "seed")
    expected = {
        (m, f, s, method, seed)
        for m in markets
        for f in opts["focal_agents"]
        for s in SCENARIOS
        for method, seed in SLOTS
    }
    keys = {tuple(r[k] for k in names) for r in rows}
    if keys != expected or len(keys) != len(rows):
        raise RuntimeError("Missing, duplicate or unexpected campaign evidence")
    for row in rows:
        if any(type(row[k]) is not int for k in ("market", "focal", "seed")):
            raise RuntimeError("Noninteger campaign identity")
        if any(k not in row or not isinstance(row[k], (int, float)) for k in METRICS):
            raise RuntimeError("Missing numerical campaign metric")
        for key, value in row.items():
            if key in ("scenario", "method"):
                continue
            if value is None and key == "cpa" and row["conversions"] == 0:
                continue
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise RuntimeError("Nonfinite or unsupported campaign evidence")


def _policies(opts: dict, config: dict, identity: str) -> dict:
    main, actions = _relative(opts["main_run"]), np.asarray(opts["actions"])
    architecture = {k: config["training"][k] for k in ("width", "layers", "heads")}
    policies = {}
    for seed in SEEDS:
        path = main / "models" / f"response-{seed}" / "best.pt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (
            saved["identity"] != identity
            or saved["kind"] != "response"
            or saved["seed"] != seed
            or saved["architecture"] != architecture
            or not 0 < saved["step"] <= config["training"]["steps"]
            or any(
                np.asarray(saved[k]).shape != (13,) or not np.isfinite(saved[k]).all()
                for k in ("mean", "std")
            )
            or np.any(saved["std"] <= 0)
            or any(not torch.isfinite(v).all() for v in saved["model"].values())
        ):
            raise RuntimeError("Response checkpoint metadata, normalization or weights are invalid")
        policies["response", seed] = Learned(
            path, actions, 24, "cpu", corrected=False, hard_cpa=False
        )
    policies["feedback", -1] = Feedback(actions, opts["feedback_gain"], opts["feedback_prior"])
    for method, action in (("fixed-1", 1.0), ("fixed-tuned", opts["fixed_tuned"])):
        policies[method, -1] = lambda states, used, rewards, budget, cpa, a=action: (a, {})
    return policies


def _summary(rows: list[dict], identity: str) -> dict:
    effects = {s: paired(rows, "response", "feedback", s, 10000) for s in SCENARIOS}
    primary = effects["over_50"]
    ratio, seeds = primary["conversion_ratio"], primary["seed_score_deltas"]
    defined = ratio["defined_for_all_resamples"] and ratio["mean"] is not None
    gates = {
        "score_ci_lower_positive": primary["score_delta"]["market_bootstrap_95"][0] > 0,
        "conversion_ratios_all_defined": defined,
        "conversion_ratio_ci_lower_at_least_0.95": ratio["market_bootstrap_95"][0] >= 0.95
        if defined
        else None,
        "violation_delta_ci_upper_at_most_0.05": primary["violation_delta"]["market_bootstrap_95"][
            1
        ]
        <= 0.05,
        "all_three_seed_score_means_positive": set(seeds) == {str(s) for s in SEEDS}
        and all(v > 0 for v in seeds.values()),
    }
    decision = (
        "inconclusive" if not defined else ("confirmed" if all(gates.values()) else "not-confirmed")
    )
    return {
        "identity": identity,
        "decision": decision,
        "primary_scenario": "over_50",
        "gates": gates,
        "primary_effect": primary,
        "scenario_effects": effects,
        "groups": [
            {"scenario": s, "method": m, **metrics(rows, m, s)}
            for s in SCENARIOS
            for m in ("response", "feedback", "fixed-1", "fixed-tuned")
        ],
        "response_seed_groups": [
            {
                "scenario": s,
                "seed": seed,
                **metrics([r for r in rows if r["seed"] == seed], "response", s),
            }
            for s in SCENARIOS
            for seed in SEEDS
        ],
        "sample_size": {
            "markets": 24,
            "focals_per_market": 6,
            "scenarios": 5,
            "policy_slots": 6,
            "rows": len(rows),
        },
        "limits": [
            "New confirmation declared after main/control discovery; not a reopened controller gate.",
            primary["uncertainty_scope"],
            "Only over_50 is primary; other scenarios and fixed baselines are descriptive secondary evidence.",
            "No training-randomness, deployment or causal architecture/history/bottleneck conclusion.",
        ],
    }


def _atomic_text(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _report(directory: Path, rows: list[dict], summary: dict) -> None:
    write_json(directory / "results.json", rows)
    write_json(directory / "summary.json", summary)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(directory / "results.csv", stream.getvalue())
    columns = ("scenario", "method", *METRICS)
    table = "<tr>" + "".join(f"<th>{c}</th>" for c in columns) + "</tr>"
    for group in summary["groups"]:
        table += (
            "<tr>" + "".join(f"<td>{html.escape(str(group[c]))}</td>" for c in columns) + "</tr>"
        )
    detail = html.escape(
        json.dumps(
            {
                k: summary[k]
                for k in ("gates", "scenario_effects", "response_seed_groups", "limits")
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    _atomic_text(
        directory / "report.html",
        f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>Independent confirmation D: original response versus feedback</title>
<style>body{{font:16px/1.6 system-ui;max-width:1150px;margin:32px auto;padding:0 20px}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:6px;border:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}</style>
<h1>Independent confirmation D</h1><p>Decision: <strong>{summary["decision"]}</strong>.</p>
<p>Primary: frozen original response − feedback in over_50, 24 new markets, three fixed response checkpoints.
Declared after main/control discovery. This does not reopen the failed paced/correction gate.</p>
<p>All five strata are reported separately. Secondary strata and fixed baselines do not replace the primary.
Intervals are conditional on three fixed models, not a training-seed confidence interval.</p>
<h2>Complete scenario strata</h2><table>{table}</table><h2>Decision and paired evidence</h2><pre>{detail}</pre>
<p><a href="results.csv">Campaign rows CSV</a> · <a href="results.json">Campaign rows JSON</a> ·
<a href="summary.json">Summary</a> · <a href="manifest.json">Manifest</a></p></html>""",
    )


def run(config_path: Path, stage: str = "all") -> Path:
    opts, config, core = _prepare(config_path)
    identity = fingerprint(core)
    directory = _relative(opts["output"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = directory / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            if manifest.get("identity") != identity or manifest.get("core") != core:
                raise RuntimeError("Output belongs to a different confirmation identity")
        else:
            if (directory / "markets").exists() and any((directory / "markets").iterdir()):
                raise RuntimeError(
                    "Create the locked confirmation manifest before generating markets"
                )
            manifest = {"identity": identity, "core": core}
            write_json(manifest_path, manifest)
        started = time.perf_counter()

        def status(phase, **extra):
            write_json(
                directory / "status.json",
                {
                    "identity": identity,
                    "phase": phase,
                    "updated_unix": time.time(),
                    "elapsed_this_session_seconds": time.perf_counter() - started,
                    **extra,
                },
            )

        def guard():
            if _prepare(config_path)[2] != core:
                raise RuntimeError("Confirmation sources or frozen inputs changed during execution")

        prior_status = (
            read_json(directory / "status.json") if (directory / "status.json").exists() else {}
        )
        if prior_status and prior_status.get("identity") != identity:
            raise RuntimeError("Existing status belongs to a different confirmation identity")
        try:
            if stage in ("all", "data") and prior_status.get("phase") != "complete":
                for market in opts["markets"]:
                    guard()
                    status("data", market=market)
                    build_market(directory / "markets" / f"{market}.npz", config["study"], market)
            guard()
            seal = _seal_markets(directory, opts, config["study"], core, identity)
            guard()
            if stage == "data":
                if prior_status.get("phase") != "complete":
                    status("data-complete", markets=24)
                return directory
            expected_outputs = {
                "results.json",
                "results.csv",
                "summary.json",
                "report.html",
                "market-cache.json",
            } | {f"market-results/{m}.json" for m in opts["markets"]}
            if prior_status.get("phase") == "complete":
                hashes = manifest.get("output_sha256", {})
                if set(hashes) != expected_outputs or any(
                    _sha256(directory / p) != h for p, h in hashes.items()
                ):
                    raise RuntimeError("Completed confirmation output hashes differ")
                _rows(read_json(directory / "results.json"), opts["markets"], opts)
                published = read_json(directory / "summary.json")
                if (
                    published["identity"] != identity
                    or published["decision"] != prior_status["decision"]
                    or prior_status["rows"] != 4320
                    or prior_status["markets"] != 24
                ):
                    raise RuntimeError("Completed confirmation status and summary differ")
                guard()
                return directory
            torch.set_num_threads(opts["torch_threads"])
            policies = _policies(opts, config, core["main_identity"])
            guard()
            rows = []
            for market in opts["markets"]:
                result_path = directory / "market-results" / f"{market}.json"
                if result_path.exists():
                    saved = read_json(result_path)
                    group = saved["rows"]
                    if (
                        saved["identity"] != identity
                        or saved["market_cache"] != seal["markets"][str(market)]
                        or saved["rows_sha256"] != fingerprint(group)
                    ):
                        raise RuntimeError("Market result cache identity or hash differs")
                else:
                    status("evaluation", market=market, completed_markets=len(rows) // 180)
                    group = evaluate(
                        [directory / "markets" / f"{market}.npz"],
                        config["study"],
                        SCENARIOS,
                        policies,
                    )
                    _rows(group, [market], opts)
                    guard()
                    write_json(
                        result_path,
                        {
                            "identity": identity,
                            "market_cache": seal["markets"][str(market)],
                            "rows_sha256": fingerprint(group),
                            "rows": group,
                        },
                    )
                _rows(group, [market], opts)
                rows.extend(group)
            _rows(rows, opts["markets"], opts)
            summary = _summary(rows, identity)
            guard()
            _seal_markets(directory, opts, config["study"], core, identity)
            _report(directory, rows, summary)
            guard()
            manifest["output_sha256"] = {
                p: _sha256(directory / p) for p in sorted(expected_outputs)
            }
            guard()
            write_json(manifest_path, manifest)
            status("complete", decision=summary["decision"], rows=len(rows), markets=24)
            return directory
        except KeyboardInterrupt:
            status("stopped", reason="Operational interruption; resume the same fixed identity")
            raise
        except Exception as error:
            status("failed", error_type=type(error).__name__)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bidding-confirmation.toml")
    parser.add_argument("--stage", choices=("data", "evaluate", "all"), default="all")
    args = parser.parse_args()
    run(args.config, args.stage)


if __name__ == "__main__":
    main()
