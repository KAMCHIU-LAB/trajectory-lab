"""Post-hoc validation replay auditing frozen policies' chosen-action forecasts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.learning import Learned, rollout
from trajectorylab.bidding.market import ROOT, Tape, scenario_bias
from trajectorylab.bidding_control import CheckpointForecast, PacedResponse, metrics
from trajectorylab.utils import fingerprint, read_json, write_json

SEEDS = (11, 23, 37)
TICKS = 48
THREADS = 2
RTOL = 1e-9
ATOL = 1e-8
GROUPS = ("current_cost", "current_conversion", "remaining_cost", "remaining_conversion")
SCALES = np.array([48.0, 48.0, 1.0, 1.0])
PROTOCOL = ROOT / "docs/bidding-onpolicy-protocol.md"
SETTINGS = {
    "design": "post-hoc after viewing main-v2 and control-v1 outcomes; descriptive only",
    "validation_markets": 6,
    "focal_advertisers": 6,
    "response_seeds": list(SEEDS),
    "scenarios": 5,
    "methods": 2,
    "expected_rollouts": 1080,
    "expected_ticks": 51840,
    "device": "cpu",
    "torch_threads": THREADS,
    "context": 24,
    "policy_parameters": "original control definitions, frozen selection; no fitting or retuning",
    "forecast_capture": "read-only forward hook copies actual final-token four curves and returns None",
    "prediction_units": "normalized cost/budget and conversions*CPA/budget",
    "prediction_values": "raw model curves; current groups cross-checked against policy diagnostics",
    "current_truth": "private unthrottled response at tick, nominal chosen action * current scenario bias",
    "remaining_truth": "private unthrottled sum tick..47 at constant current effective action, including current tick",
    "remaining_truth_excludes": "later policy actions, remaining-budget truncation, future scenario-bias changes",
    "private_label_feedback": "never passed to the policy or included in its diagnostic",
    "budget_blocked_ticks": "included in all forecast audits using private unthrottled labels",
    "target_scales": SCALES.tolist(),
    "log_error": "log1p(prediction*scale)-log1p(private label*scale)",
    "smooth_l1": "beta=1; 0.5*error^2 if abs(error)<1 else abs(error)-0.5",
    "group_weighting": "equal ticks and cost/conversion weights within current or remaining groups",
    "aggregate_ratio": "sum normalized predictions / sum normalized private labels, per curve",
    "undefined_ratio": "null when the true aggregate is zero",
    "early_spend": "actual cumulative budget utilization after each of first 12 settled ticks",
    "action_histogram": "11 nominal policy actions, not scenario-multiplied effective actions",
    "replay_exact_fields": ["conversions", "violation"],
    "replay_float_fields": ["score", "cost"],
    "replay_rtol": RTOL,
    "replay_atol": ATOL,
    "mismatch_rule": "fail entire diagnostic; do not change tolerances or omit rollouts",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite(name: str, values) -> None:
    if not np.isfinite(values).all():
        raise RuntimeError(f"Nonfinite {name}")


def _key(row: dict) -> tuple:
    return tuple(row[name] for name in ("market", "focal", "scenario", "method", "seed"))


class _Audit:
    """Keep private audit labels outside the delegate's observations and feedback."""

    def __init__(self, policy, model, tape: Tape, fi: int, actions: np.ndarray, scenario: str):
        policy.reset()
        self.policy, self.tape, self.fi = policy, tape, fi
        self.actions, self.scenario = actions, scenario
        self.budget = float(tape.arrays["budgets"][fi])
        self.cpa = float(tape.arrays["cpas"][fi])
        self.curves = None
        self.pending = None
        self.count, self.blocked, self.spent = 0, 0, 0.0
        self.histogram = np.zeros(len(actions), dtype=np.int64)
        self.early = []
        self.sums = {
            name: np.zeros(4) for name in ("predicted", "true", "signed", "squared", "smooth")
        }
        self.hook = model.register_forward_hook(self._capture)

    def _capture(self, _module, _inputs, output):
        if output.ndim != 4 or output.shape[0] != 1 or output.shape[2:] != (4, len(self.actions)):
            raise RuntimeError("Unexpected response forecast shape")
        self.curves = output[0, -1].detach().cpu().numpy().copy().astype(np.float64)
        # Returning None leaves the exact model output consumed by the policy intact.

    def __call__(self, states, used, rewards, budget, cpa):
        if self.pending is not None or len(states) != self.count + 1:
            raise RuntimeError("Audit call sequence differs from the frozen rollout")
        self.curves = None
        action, diagnostic = self.policy(states, used, rewards, budget, cpa)
        if self.curves is None or budget != self.budget or cpa != self.cpa:
            raise RuntimeError("Missing forecast capture or advertiser normalization mismatch")
        ai = int(np.abs(self.actions - action).argmin())
        if not np.isclose(self.actions[ai], action, rtol=0, atol=1e-8):
            raise RuntimeError("Audit received an invalid nominal action")
        predicted = self.curves[:, ai].copy()
        diagnostic_current = np.array(
            [
                diagnostic["predicted_tick_cost"] / budget,
                diagnostic["predicted_tick_conversions"] * cpa / budget,
            ]
        )
        if not np.allclose(predicted[:2], diagnostic_current, rtol=1e-12, atol=1e-12):
            raise RuntimeError("Captured current curves differ from original diagnostics")
        predicted[:2] = diagnostic_current
        _finite("chosen-action predictions", predicted)
        if np.any(predicted < 0):
            raise RuntimeError("Negative chosen-action prediction")
        self.pending = (ai, float(action), predicted)
        return action, diagnostic

    def observe(self, cost, conv, diagnostic):
        # Only the unchanged actual settlement and original diagnostic reach the policy.
        self.policy.observe(cost, conv, diagnostic)
        if self.pending is None:
            raise RuntimeError("Audit settlement has no matching action")
        ai, action, predicted = self.pending
        tick = self.count
        effective = self.tape.index(action * scenario_bias(self.scenario, tick))
        a = self.tape.arrays
        costs = a["costs"][self.fi, tick:, effective]
        conversions = a["conversions"][self.fi, tick:, effective]
        true = np.array(
            [
                costs[0] / self.budget,
                conversions[0] * self.cpa / self.budget,
                costs.sum() / self.budget,
                conversions.sum() * self.cpa / self.budget,
            ]
        )
        _finite("private unthrottled labels", true)
        if np.any(true < 0):
            raise RuntimeError("Negative private unthrottled response")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            error = np.log1p(predicted * SCALES) - np.log1p(true * SCALES)
            squared = error * error
            smooth = np.where(np.abs(error) < 1, 0.5 * squared, np.abs(error) - 0.5)
        for name, value in (
            ("predicted", predicted),
            ("true", true),
            ("signed", error),
            ("squared", squared),
            ("smooth", smooth),
        ):
            self.sums[name] += value
            _finite("audit sufficient statistics", self.sums[name])
        self.histogram[ai] += 1
        self.blocked += int(diagnostic["budget_blocked"])
        self.spent += cost
        if tick < 12:
            self.early.append(self.spent / self.budget)
        self.count += 1
        self.pending = None

    def close(self) -> None:
        self.hook.remove()

    def record(self) -> dict:
        if self.count != TICKS or self.pending is not None or len(self.early) != 12:
            raise RuntimeError("Incomplete audit tick coverage")
        return {
            "ticks": self.count,
            "blocked": self.blocked,
            "histogram": self.histogram,
            "early": np.asarray(self.early),
            **self.sums,
        }


def _aggregate(rows: list[dict], dimensions: tuple[str, ...], actions: np.ndarray) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[name] for name in dimensions)].append(row)
    result = []
    for key, group in sorted(groups.items()):
        ticks = sum(row["ticks"] for row in group)
        totals = {
            name: np.sum([row[name] for row in group], axis=0)
            for name in ("predicted", "true", "signed", "squared", "smooth")
        }
        errors = {}
        for name, indices in (
            ("current", slice(0, 2)),
            ("remaining_constant_effective_action", slice(2, 4)),
        ):
            errors[name] = {
                "chosen_action_log_mse": float(totals["squared"][indices].sum() / (ticks * 2)),
                "chosen_action_smooth_l1_beta1": float(
                    totals["smooth"][indices].sum() / (ticks * 2)
                ),
            }
        curves = {
            name: {
                "chosen_action_log_mse": float(totals["squared"][i] / ticks),
                "chosen_action_smooth_l1_beta1": float(totals["smooth"][i] / ticks),
                "mean_signed_log_error": float(totals["signed"][i] / ticks),
                "normalized_predicted_to_true_ratio": float(
                    totals["predicted"][i] / totals["true"][i]
                )
                if totals["true"][i] > 0
                else None,
            }
            for i, name in enumerate(GROUPS)
        }
        histogram = np.sum([row["histogram"] for row in group], axis=0)
        blocked = sum(row["blocked"] for row in group)
        entry = {
            **dict(zip(dimensions, key, strict=True)),
            "campaigns": len(group),
            "ticks": ticks,
            "chosen_action_errors": errors,
            "curves": curves,
            "nominal_action_histogram": {
                str(float(action)): int(count)
                for action, count in zip(actions, histogram, strict=True)
            },
            "nominal_action_frequency": {
                str(float(action)): float(count / ticks)
                for action, count in zip(actions, histogram, strict=True)
            },
            "first_12_tick_cumulative_budget_utilization_mean": np.mean(
                [row["early"] for row in group], axis=0
            ).tolist(),
            "budget_blocked_ticks": blocked,
            "budget_blocked_fraction": blocked / ticks,
            "outcome_means": {
                name: float(np.mean([row[name] for row in group]))
                for name in ("score", "cost", "conversions", "violation")
            },
        }
        result.append(entry)
    return result


def _inputs(
    main_run: Path, control_run: Path
) -> tuple[dict, dict, dict, dict, list[Path], dict, dict]:
    main = read_json(main_run / "manifest.json")
    config, study = main["config"], main["config"]["study"]
    if (
        fingerprint({key: value for key, value in main.items() if key != "identity"})
        != main["identity"]
        or fingerprint(provenance(config)) != main["identity"]
        or read_json(main_run / "status.json")["phase"] != "complete"
    ):
        raise RuntimeError("Main study is incomplete or differs from current provenance")
    if (
        len(study["validation_markets"]) != 6
        or len(set(study["validation_markets"])) != 6
        or len(study["focal_agents"]) != 6
        or len(set(study["focal_agents"])) != 6
        or tuple(study["training_seeds"]) != SEEDS
        or len(config["evaluation"]["scenarios"]) != 5
        or len(set(config["evaluation"]["scenarios"])) != 5
        or config["training"]["context"] != 24
        or len(study["actions"]) != 11
    ):
        raise RuntimeError(
            "Expected the frozen 6-market/6-focal/3-seed/5-scenario validation design"
        )
    control = read_json(control_run / "manifest.json")
    selection = read_json(control_run / "selection.json")
    control_status = read_json(control_run / "status.json")
    opts, layout = control["config"], control["evidence_layout"]
    declared_identity = fingerprint(
        {
            "config": opts,
            "main": control["main_identity"],
            "implementation": control["implementation_sha256"],
            "checkpoints": control["checkpoint_sha256"],
            "main_selection_sha256": control["main_selection_sha256"],
        }
    )
    if (
        control_status["phase"] not in {"complete", "complete-not-promoted"}
        or control["identity"] != declared_identity
        or control["main_identity"] != main["identity"]
        or _sha256(ROOT / "src/trajectorylab/bidding_control.py")
        != control["implementation_sha256"]
        or read_json(control_run / "validation-report/manifest.json") != control
    ):
        raise RuntimeError("Control validation evidence source or completion mismatch")
    snapshot = (
        json.dumps(control["main_selection"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if (
        hashlib.sha256(snapshot).hexdigest() != control["main_selection_sha256"]
        or _sha256(main_run / "selection.json") != control["main_selection_sha256"]
    ):
        raise RuntimeError("Frozen main selection provenance mismatch")
    for field, expected in (
        ("validation_markets", study["validation_markets"]),
        ("focal_agents", study["focal_agents"]),
        ("training_seeds", study["training_seeds"]),
        ("scenarios", config["evaluation"]["scenarios"]),
    ):
        if layout[field] != expected:
            raise RuntimeError("Control evidence layout differs from the main validation design")
    modes = {
        f"response-paced-{selection['best_rate_index']}": "none",
        "response-paced-cumulative": "cumulative",
        "response-paced-discounted": "discounted",
        "response-paced-budget-only": "budget-only",
    }
    rate_index = selection["best_rate_index"]
    if (
        not isinstance(rate_index, int)
        or not 0 <= rate_index < len(opts["dual_rates"])
        or selection["best_rate"] != opts["dual_rates"][rate_index]
        or modes.get(selection["candidate"]) != selection["selected_calibration"]
    ):
        raise RuntimeError("Selected candidate does not match the frozen control configuration")
    raw = read_json(control_run / "validation-report/results.json")
    reference = [row for row in raw if row["method"] in {"response", selection["candidate"]}]
    expected = {
        (market, focal, scenario, method, seed)
        for market in study["validation_markets"]
        for focal in study["focal_agents"]
        for scenario in config["evaluation"]["scenarios"]
        for method in ("response", selection["candidate"])
        for seed in SEEDS
    }
    if len(reference) != 1080 or {_key(row) for row in reference} != expected:
        raise RuntimeError(
            "Frozen comparison rows do not uniquely cover all expected validation jobs"
        )
    for row in reference:
        _finite(
            "reference replay outcomes",
            [row[name] for name in ("score", "cost", "conversions", "violation")],
        )
    if (
        metrics(raw, selection["candidate"]) != selection["candidate_metrics"]
        or metrics(raw, "response") != selection["original_response_metrics"]
    ):
        raise RuntimeError("Frozen validation rows contradict selected aggregate outcomes")
    paths = [main_run / "markets" / f"{market}.npz" for market in study["validation_markets"]]
    bound = {
        "implementation": Path(__file__),
        "protocol": PROTOCOL,
        "control_implementation": ROOT / "src/trajectorylab/bidding_control.py",
        "utils_implementation": ROOT / "src/trajectorylab/utils.py",
        "main_manifest": main_run / "manifest.json",
        "main_status": main_run / "status.json",
        "main_selection": main_run / "selection.json",
        "control_manifest": control_run / "manifest.json",
        "control_status": control_run / "status.json",
        "control_selection": control_run / "selection.json",
        "control_validation_manifest": control_run / "validation-report/manifest.json",
        "control_validation_results": control_run / "validation-report/results.json",
    }
    market_identity = {}
    for market, path in zip(study["validation_markets"], paths, strict=True):
        meta = read_json(path.with_suffix(".json"))
        expected_identity = fingerprint(
            {
                "study": {
                    key: study[key] for key in ("focal_agents", "pv_count", "actions", "biases")
                },
                "market": market,
                "source": main["auction"],
            }
        )
        if (
            meta.get("identity") != expected_identity
            or meta.get("focal_agents") != study["focal_agents"]
            or meta.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError("Validation cache identity or size mismatch")
        market_identity[str(market)] = {"identity": expected_identity, "bytes": meta["bytes"]}
        bound[f"validation_market_{market}_metadata"] = path.with_suffix(".json")
    for seed in SEEDS:
        path = main_run / "models" / f"response-{seed}" / "best.pt"
        if _sha256(path) != control["checkpoint_sha256"][str(seed)]:
            raise RuntimeError("Response checkpoint differs from the frozen control run")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if (
            saved["identity"] != main["identity"]
            or saved["kind"] != "response"
            or saved["seed"] != seed
            or saved["architecture"]
            != {key: config["training"][key] for key in ("width", "layers", "heads")}
        ):
            raise RuntimeError("Response checkpoint metadata mismatch")
        bound[f"response_{seed}_checkpoint"] = path
    core = {
        "schema_version": 1,
        "main_identity": main["identity"],
        "main_implementation": main["implementation"],
        "control_identity": control["identity"],
        "files_sha256": {name: _sha256(path) for name, path in bound.items()},
        "validation_markets": market_identity,
        "focal_agents": study["focal_agents"],
        "scenarios": config["evaluation"]["scenarios"],
        "actions": study["actions"],
        "selected_controller": {
            name: selection[name]
            for name in ("candidate", "best_rate_index", "best_rate", "selected_calibration")
        },
        "original_response": {
            "corrected": False,
            "hard_cpa": False,
            "prior": control["main_selection"]["correction"]["response"]["prior"],
        },
        "calibration_prior": opts["calibration_prior"],
        "discount_half_life": opts["discount_half_life"],
        "decision_curves": {
            "response": "remaining",
            selection["candidate"]: "current with frozen control calibration and resource prices",
        },
        "settings": SETTINGS,
        "runtime": {"numpy": np.__version__, "torch": torch.__version__},
    }
    return core, config, opts, selection, paths, {_key(row): row for row in reference}, bound


def run(main_run: Path, control_run: Path, output: Path) -> Path:
    core, config, opts, selection, paths, reference, bound = _inputs(main_run, control_run)
    identity = fingerprint(core)

    def guard() -> None:
        if fingerprint(provenance(config)) != core["main_identity"]:
            raise RuntimeError("Main provenance changed during the audit")
        if any(_sha256(path) != core["files_sha256"][name] for name, path in bound.items()):
            raise RuntimeError("Audit source, protocol, checkpoint, or frozen evidence changed")
        for path in paths:
            if path.stat().st_size != core["validation_markets"][path.stem]["bytes"]:
                raise RuntimeError("Validation cache size changed")

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this on-policy audit output") from error
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            previous = read_json(manifest_path)
            previous_core = {
                key: value
                for key, value in previous.items()
                if key not in {"identity", "summary_sha256"}
            }
            if previous.get("identity") != identity or fingerprint(previous_core) != identity:
                raise RuntimeError("Output belongs to a different on-policy audit identity")
            status_path = output / "status.json"
            if status_path.exists() and read_json(status_path).get("phase") == "complete":
                status = read_json(status_path)
                summary_path = output / "summary.json"
                if (
                    status.get("identity") != identity
                    or status.get("summary_sha256") != previous.get("summary_sha256")
                    or _sha256(summary_path) != previous.get("summary_sha256")
                    or read_json(summary_path).get("identity") != identity
                ):
                    raise RuntimeError("Completed on-policy audit hashes or identity mismatch")
                guard()
                return output
        elif any(path.name != ".run.lock" for path in output.iterdir()):
            raise RuntimeError("Nonempty output lacks an on-policy manifest; use a new directory")
        guard()
        write_json(manifest_path, core | {"identity": identity})
        started = time.perf_counter()

        def status(phase: str, **details) -> None:
            write_json(
                output / "status.json",
                {
                    "phase": phase,
                    "identity": identity,
                    "updated_unix": time.time(),
                    "elapsed_this_session_seconds": time.perf_counter() - started,
                    **details,
                },
            )

        try:
            status("loading-policies")
            torch.set_num_threads(THREADS)
            actions = np.asarray(config["study"]["actions"])
            policies = {}
            for seed in SEEDS:
                path = main_run / "models" / f"response-{seed}" / "best.pt"
                original = Learned(
                    path, actions, 24, "cpu", False, core["original_response"]["prior"]
                )
                forecast = CheckpointForecast(path, actions, 24)
                mode = selection["selected_calibration"]
                rate = selection["best_rate"]
                if mode == "budget-only":
                    rate, mode = (rate[0], 0.0), "none"
                paced = PacedResponse(
                    forecast,
                    actions,
                    rate,
                    mode,
                    opts["calibration_prior"],
                    opts["discount_half_life"],
                )
                policies["response", seed] = (original, original.model)
                policies[selection["candidate"], seed] = (paced, forecast.policy.model)
            rows, seen = [], set()
            max_difference = {name: 0.0 for name in ("score", "cost", "conversions", "violation")}
            for path in paths:
                tape = Tape(path)
                if tape.arrays["focals"].tolist() != core["focal_agents"]:
                    raise RuntimeError("Validation tape focal coverage mismatch")
                for fi, focal in enumerate(core["focal_agents"]):
                    for scenario in core["scenarios"]:
                        for (method, seed), (policy, model) in policies.items():
                            status(
                                "replaying",
                                market=int(path.stem),
                                focal=focal,
                                scenario=scenario,
                                method=method,
                                seed=seed,
                                completed_rollouts=len(rows),
                            )
                            audit = _Audit(policy, model, tape, fi, actions, scenario)
                            try:
                                outcome = rollout(tape, fi, actions, scenario, audit)
                                stats = audit.record()
                            finally:
                                audit.close()
                            key = (int(path.stem), focal, scenario, method, seed)
                            if key in seen or key not in reference:
                                raise RuntimeError("Unexpected or duplicate replay identity")
                            for name in max_difference:
                                actual, expected = outcome[name], reference[key][name]
                                exact = name in {"conversions", "violation"}
                                if not np.isfinite(actual) or (
                                    actual != expected
                                    if exact
                                    else not np.isclose(actual, expected, rtol=RTOL, atol=ATOL)
                                ):
                                    raise RuntimeError(
                                        f"Replay mismatch: {key}, metric={name}, actual={actual}, expected={expected}"
                                    )
                                max_difference[name] = max(
                                    max_difference[name], abs(float(actual) - float(expected))
                                )
                            if stats["blocked"] != outcome[
                                "budget_blocked_ticks"
                            ] or not np.isclose(audit.spent, outcome["cost"], rtol=0, atol=1e-12):
                                raise RuntimeError(
                                    "Audit settlement counters differ from the frozen rollout"
                                )
                            seen.add(key)
                            rows.append(
                                {
                                    "market": key[0],
                                    "focal": focal,
                                    "scenario": scenario,
                                    "method": method,
                                    "seed": seed,
                                    **{name: outcome[name] for name in max_difference},
                                    **stats,
                                }
                            )
                            del audit
                del tape
            if (
                seen != set(reference)
                or len(rows) != SETTINGS["expected_rollouts"]
                or sum(row["ticks"] for row in rows) != SETTINGS["expected_ticks"]
            ):
                raise RuntimeError("Incomplete frozen validation replay coverage")
            summary = {
                "identity": identity,
                "main_identity": core["main_identity"],
                "control_identity": core["control_identity"],
                "replay_agreement": {
                    "all_rollouts_match": True,
                    "checked_rollouts": len(rows),
                    "max_absolute_difference": max_difference,
                    "rtol": RTOL,
                    "atol": ATOL,
                },
                "sample_counts": {
                    "rollouts": len(rows),
                    "ticks": sum(row["ticks"] for row in rows),
                    "independent_validation_markets": len(paths),
                },
                "decision_curves": core["decision_curves"],
                "aggregates": {
                    name: _aggregate(rows, dimensions, actions)
                    for name, dimensions in {
                        "by_method": ("method",),
                        "by_scenario_method": ("scenario", "method"),
                        "by_seed_method": ("seed", "method"),
                        "by_market_method": ("market", "method"),
                        "by_scenario_method_seed_market": ("scenario", "method", "seed", "market"),
                    }.items()
                },
                "limitations": [
                    "Post-hoc after observing main and control outcomes; not independent confirmation or a pre-result protocol.",
                    "Policies visit different states and choose different actions; on-policy errors do not isolate predictor quality or a causal bottleneck.",
                    "All ticks, including budget-blocked ticks, use evaluator-private unthrottled labels that never reach either policy.",
                    "Remaining truth holds the current effective action fixed through tick 47, even in midshift; it is not realized subsequent policy reward or a future-bias-path forecast.",
                    "Audit curves are raw model outputs; paced calibration and resource prices remain internal to the unchanged selected policy.",
                    "Current state contains cumulative and previous-tick feedback; this is not a history-free comparison.",
                    "Only six independent validation markets; ticks, focal campaigns and training seeds are correlated within markets.",
                    "No fitting, retuning, test or confirmation market access, or policy superiority claim is part of this audit.",
                ],
            }
            guard()
            summary_path = output / "summary.json"
            write_json(summary_path, summary)
            summary_hash = _sha256(summary_path)
            write_json(manifest_path, core | {"identity": identity, "summary_sha256": summary_hash})
            guard()
            status("complete", summary_sha256=summary_hash)
            return output
        except BaseException as error:
            status(
                "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                error_type=type(error).__name__,
            )
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-run", type=Path, default=ROOT / "artifacts/bidding/main-v2")
    parser.add_argument("--control-run", type=Path, default=ROOT / "artifacts/bidding/control-v1")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/bidding/onpolicy-v1")
    args = parser.parse_args()
    run(args.main_run, args.control_run, args.output)


if __name__ == "__main__":
    main()
