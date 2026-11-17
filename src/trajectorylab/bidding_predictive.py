"""Matched, validation-only forecast diagnostics; no fitting or policy evaluation.

Response checkpoints and ridge regularization were selected using validation data.
These descriptive metrics do not establish policy superiority or a bottleneck cause.
"""

from __future__ import annotations

import argparse
import fcntl
import time
from pathlib import Path

import numpy as np
import torch

from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.learning import ResponseModel, trajectories
from trajectorylab.bidding_context import GROUPS, SCALES, RidgeForecast, _main_inputs, _sha256
from trajectorylab.bidding_control import empirical_rates
from trajectorylab.utils import fingerprint, read_json, write_json

SEEDS = (11, 23, 37)
CONTEXT = 24
BATCH_SIZE = 64
THREADS = 2
TICKS = 48
STATE_DIM = 13
ACTION_COUNT = 11
EXPECTED_TRAJECTORIES = 360
EXPECTED_ROWS = 17280
SETTINGS = {
    "seeds": list(SEEDS),
    "response_checkpoint": "best.pt from each separately completed response training run",
    "neural_ensemble": False,
    "device": "cpu",
    "torch_threads": THREADS,
    "batch_size_max": BATCH_SIZE,
    "context_max": CONTEXT,
    "state_dim": STATE_DIM,
    "action_count": ACTION_COUNT,
    "expected_validation_trajectories": EXPECTED_TRAJECTORIES,
    "expected_validation_rows": EXPECTED_ROWS,
    "row_order": "main validation market order, learning.trajectories behavior order, absolute tick",
    "neural_window": "current and preceding states, at most 24; right padded; select last valid token",
    "neural_steps": "absolute zero-based tick indices, with causal masks and model.eval()",
    "neural_normalization": "checkpoint mean/std, float32 arithmetic, no added clipping",
    "neural_projection": "unchanged ResponseModel monotone softplus curves; no additional clipping",
    "ridge_prediction": "unchanged RidgeForecast.predict_curves projection",
    "empirical_fit": "bidding_control.empirical_rates on declared main training markets only",
    "empirical_prediction": "normalized training rates * current state[11] * 10000",
    "groups": list(GROUPS),
    "target_scales": SCALES.tolist(),
    "error_space": "log1p(normalized prediction * scale) - log1p(normalized label * scale)",
    "metric_dtype": "float64 after predictions; neural inference and normalization float32",
    "log_mse": "mean squared error, equal weight per current row/group/action",
    "smooth_l1": "beta=1: 0.5*error^2 if abs(error)<1 else abs(error)-0.5; equal row/group/action weights",
    "training_loss_convention": "same log scales, beta=1 and group/action averaging as learning._loss",
    "comparable_current_groups": list(GROUPS[:2]),
    "remaining_groups": "neural and ridge only; empirical has no remaining-horizon prediction",
    "aggregate_ratio": "sum normalized predictions / sum normalized labels per group over rows/actions",
    "undefined_ratio": "null when the true aggregate is zero",
    "selection": "none in this diagnostic; all methods evaluated separately on identical validation rows",
}


def _finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"Nonfinite {name}")


def _ridge_evidence(
    ridge_run: Path, main_identity: str, actions: np.ndarray
) -> tuple[dict, RidgeForecast]:
    manifest = read_json(ridge_run / "manifest.json")
    status = read_json(ridge_run / "status.json")
    model = RidgeForecast(ridge_run / "model.npz")
    summary_path = ridge_run / "summary.json"
    summary = read_json(summary_path)
    if (
        status.get("phase") != "complete"
        or manifest.get("main_identity") != main_identity
        or model.identity != manifest.get("identity")
        or status.get("identity") != model.identity
        or summary.get("identity") != model.identity
        or status.get("model_sha256") != manifest.get("model_sha256")
        or summary.get("model_sha256") != manifest.get("model_sha256")
        or status.get("summary_sha256") != manifest.get("summary_sha256")
        or _sha256(summary_path) != manifest.get("summary_sha256")
        or not np.array_equal(model.actions, actions)
    ):
        raise RuntimeError("Ridge evidence is incomplete or belongs to a different main study")
    return {
        "identity": model.identity,
        "selected_lambda": manifest["selected_lambda"],
        "files_sha256": {
            name: _sha256(ridge_run / name)
            for name in ("manifest.json", "model.npz", "summary.json", "status.json")
        },
    }, model


def _checkpoint(path: Path, config: dict, identity: str, seed: int, *, final: bool) -> dict:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    opts = config["training"]
    architecture = {key: opts[key] for key in ("width", "layers", "heads")}
    step = saved.get("step")
    if (
        saved.get("identity") != identity
        or saved.get("kind") != "response"
        or saved.get("seed") != seed
        or saved.get("architecture") != architecture
        or not isinstance(step, int)
        or isinstance(step, bool)
        or not 1 <= step <= opts["steps"]
        or (final and step != opts["steps"])
    ):
        raise RuntimeError(
            "Response checkpoint provenance, architecture, or training step mismatch"
        )
    for key in ("mean", "std"):
        value = np.asarray(saved[key])
        if value.shape != (STATE_DIM,) or value.dtype != np.float32:
            raise RuntimeError("Unexpected checkpoint normalization shape or dtype")
        _finite(f"checkpoint {key}", value)
    if np.any(saved["std"] < 0.05):
        raise RuntimeError("Invalid checkpoint normalization scale")
    weights = saved["model"]
    if not weights or any(not torch.isfinite(value).all().item() for value in weights.values()):
        raise RuntimeError("Nonfinite or empty response weights")
    loss = saved.get("best_loss" if final else "validation_loss")
    if not isinstance(loss, (int, float)) or not np.isfinite(loss) or loss < 0:
        raise RuntimeError("Invalid checkpoint validation loss")
    return saved


def _response(
    main_run: Path, config: dict, identity: str, seed: int, record: dict
) -> tuple[ResponseModel, dict]:
    directory = main_run / "models" / f"response-{seed}"
    best = _checkpoint(directory / "best.pt", config, identity, seed, final=False)
    final = _checkpoint(directory / "checkpoint.pt", config, identity, seed, final=True)
    if (
        any(not np.array_equal(best[key], final[key]) for key in ("mean", "std"))
        or best["validation_loss"] != final["best_loss"]
        or best["validation_loss"] != record["best_validation_loss"]
        or best["parameters"] != record["parameters"]
    ):
        raise RuntimeError("Best, final, and completed training records disagree")
    model = ResponseModel(STATE_DIM, ACTION_COUNT, **best["architecture"]).cpu()
    model.load_state_dict(final["model"], strict=True)
    model.load_state_dict(best["model"], strict=True)
    if sum(value.numel() for value in model.parameters()) != best["parameters"]:
        raise RuntimeError("Response parameter count mismatch")
    model.eval()
    return model, best


def _training_records(main_run: Path, config: dict) -> dict:
    records = read_json(main_run / "training.json")
    if not isinstance(records, list):
        raise RuntimeError("Expected main training records")
    response = [record for record in records if record.get("kind") == "response"]
    if len(response) != len(SEEDS) or {record.get("seed") for record in response} != set(SEEDS):
        raise RuntimeError("All three declared response seeds must have finished training")
    for record in response:
        if record.get("steps") != config["training"]["steps"]:
            raise RuntimeError("Response training is incomplete")
    return {record["seed"]: record for record in response}


def _validate_caches(paths: dict, focals: list[int]) -> None:
    for group in paths.values():
        for path in group:
            with np.load(path, allow_pickle=False) as arrays:
                if arrays["focals"].tolist() != focals:
                    raise RuntimeError("Cached focal advertisers differ from the main declaration")
                counts = arrays["counts"]
                if counts.shape != (TICKS,) or np.any(counts < 0) or counts.sum() <= 0:
                    raise RuntimeError("Invalid cached opportunity counts")
                for key in ("budgets", "cpas"):
                    values = arrays[key]
                    _finite(key, values)
                    if values.shape != (len(focals),) or np.any(values <= 0):
                        raise RuntimeError("Invalid cached advertiser normalization")
                _finite("opportunity counts", counts)


def _validation_rows(
    paths: list[Path], actions: np.ndarray, focals: list[int]
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    runs, markets = [], []
    expected_per_market = len(focals) * 10
    for path in paths:
        group = trajectories([path], actions)
        if len(group) != expected_per_market:
            raise RuntimeError("Validation behavior trajectory coverage mismatch")
        for run in group:
            states, labels = run["states"], run["labels"]
            if states.shape != (TICKS, STATE_DIM) or labels.shape != (TICKS, 4, ACTION_COUNT):
                raise RuntimeError("Validation row shape mismatch")
            _finite("validation states", states)
            _finite("validation labels", labels)
            if (
                np.any(labels < 0)
                or np.any(labels[:, :, 0] != 0)
                or not np.allclose(states[:, 0], np.arange(TICKS) / TICKS, rtol=0, atol=1e-7)
            ):
                raise RuntimeError("Validation labels or chronological state ordering mismatch")
        runs.extend(group)
        markets.extend([int(path.stem)] * (len(group) * TICKS))
    if len(runs) != EXPECTED_TRAJECTORIES or len(markets) != EXPECTED_ROWS:
        raise RuntimeError("Expected exactly 360 validation trajectories and 17280 rows")
    states = np.concatenate([run["states"] for run in runs])
    labels = np.concatenate([run["labels"] for run in runs]).astype(np.float64)
    return runs, states, labels, np.asarray(markets)


@torch.inference_mode()
def _neural_predictions(
    model: ResponseModel, saved: dict, runs: list[dict], progress
) -> np.ndarray:
    total = len(runs) * TICKS
    predictions = np.empty((total, 4, ACTION_COUNT), dtype=np.float64)
    for offset in range(0, total, BATCH_SIZE):
        size = min(BATCH_SIZE, total - offset)
        states = np.zeros((size, CONTEXT, STATE_DIM), dtype=np.float32)
        steps = np.zeros((size, CONTEXT), dtype=np.int64)
        mask = np.zeros((size, CONTEXT), dtype=bool)
        lengths = np.empty(size, dtype=np.int64)
        for row in range(size):
            trajectory, tick = divmod(offset + row, TICKS)
            end = tick + 1
            start = max(0, end - CONTEXT)
            length = end - start
            states[row, :length] = (runs[trajectory]["states"][start:end] - saved["mean"]) / saved[
                "std"
            ]
            steps[row, :length] = np.arange(start, end)
            mask[row, :length] = True
            lengths[row] = length
        _finite("normalized neural inputs", states)
        curves = model(torch.from_numpy(states), torch.from_numpy(steps), torch.from_numpy(mask))
        predictions[offset : offset + size] = curves[
            torch.arange(size), torch.from_numpy(lengths - 1)
        ].numpy()
        if (offset // BATCH_SIZE) % 16 == 0 or offset + size == total:
            progress(offset + size)
    return predictions


def _metrics(
    predicted: np.ndarray, labels: np.ndarray, groups: tuple[str, ...], scales: np.ndarray
) -> dict:
    if predicted.shape != labels.shape or predicted.shape[1:] != (len(groups), ACTION_COUNT):
        raise RuntimeError("Forecast metric row/action coverage mismatch")
    _finite("predictions", predicted)
    if np.any(predicted < 0) or np.any(predicted[:, :, 0] != 0):
        raise RuntimeError("Forecast curves must be nonnegative and zero at action zero")
    if np.any(np.diff(predicted, axis=2) < -1e-12):
        raise RuntimeError("Forecast curves must be monotone over actions")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        errors = np.log1p(predicted * scales[None, :, None]) - np.log1p(
            labels * scales[None, :, None]
        )
        mse = errors * errors
        absolute = np.abs(errors)
        smooth = np.where(absolute < 1.0, 0.5 * mse, absolute - 0.5)
        predicted_totals = predicted.sum(axis=(0, 2))
        true_totals = labels.sum(axis=(0, 2))
        result = {
            "log_mse": float(mse.mean()),
            "smooth_l1_beta1": float(smooth.mean()),
            "groups": {
                name: {
                    "log_mse": float(mse[:, i].mean()),
                    "smooth_l1_beta1": float(smooth[:, i].mean()),
                    "aggregate_predicted_to_true_ratio": float(predicted_totals[i] / true_totals[i])
                    if true_totals[i] > 0
                    else None,
                }
                for i, name in enumerate(groups)
            },
        }
    _finite("forecast errors", np.r_[mse.ravel(), smooth.ravel(), predicted_totals, true_totals])
    for group in result["groups"].values():
        _finite("group metrics", np.array([value for value in group.values() if value is not None]))
    return result


def _method_metrics(predicted: np.ndarray, labels: np.ndarray, markets: np.ndarray) -> dict:
    def group(indices) -> dict:
        result = {
            "current": _metrics(predicted[indices, :2], labels[indices, :2], GROUPS[:2], SCALES[:2])
        }
        if predicted.shape[1] == 4:
            result["remaining"] = _metrics(
                predicted[indices, 2:], labels[indices, 2:], GROUPS[2:], SCALES[2:]
            )
            result["all_four"] = _metrics(predicted[indices], labels[indices], GROUPS, SCALES)
        return result

    return {
        "pooled": group(slice(None)),
        "per_market": {str(market): group(markets == market) for market in np.unique(markets)},
    }


def run(main_run: Path, ridge_run: Path, output: Path) -> Path:
    main_core, paths, config, actions = _main_inputs(main_run)
    study = config["study"]
    if tuple(study["training_seeds"]) != SEEDS or config["training"]["context"] != CONTEXT:
        raise RuntimeError("This diagnostic requires frozen seeds 11/23/37 and context 24")
    if len(paths["validation"]) * len(study["focal_agents"]) * 10 != EXPECTED_TRAJECTORIES:
        raise RuntimeError("Declared validation behavior coverage differs from this diagnostic")
    records = _training_records(main_run, config)
    ridge_evidence, ridge = _ridge_evidence(ridge_run, main_core["main_identity"], actions)
    package = Path(__file__).parent
    dependencies = {
        name: _sha256(package / name)
        for name in ("bidding_context.py", "bidding_control.py", "utils.py")
    }
    checkpoints = {
        str(seed): {
            name: _sha256(main_run / "models" / f"response-{seed}" / name)
            for name in ("best.pt", "checkpoint.pt")
        }
        for seed in SEEDS
    }
    core = {
        "schema_version": 1,
        "purpose": "matched validation-only predictive diagnostic; no fitting or policy evaluation",
        "source_sha256": _sha256(Path(__file__)),
        "main_identity": main_core["main_identity"],
        "main_manifest_sha256": _sha256(main_run / "manifest.json"),
        "main_implementation": main_core["main_implementation"],
        "expected_response_training_steps": config["training"]["steps"],
        "response_architecture": {
            key: config["training"][key] for key in ("width", "layers", "heads")
        },
        "dependency_hashes": dependencies,
        "training_records_sha256": _sha256(main_run / "training.json"),
        "response_checkpoints_sha256": checkpoints,
        "ridge": ridge_evidence,
        "data": main_core["data"],
        "focal_agents": study["focal_agents"],
        "actions": actions.tolist(),
        "settings": SETTINGS,
        "runtime": {"numpy": np.__version__, "torch": torch.__version__},
    }
    identity = fingerprint(core)

    def guard() -> None:
        if (
            _sha256(Path(__file__)) != core["source_sha256"]
            or fingerprint(provenance(config)) != core["main_identity"]
        ):
            raise RuntimeError("Predictive or main source identity changed")
        if any(_sha256(package / name) != digest for name, digest in dependencies.items()):
            raise RuntimeError("Predictive dependency identity changed")
        if _sha256(main_run / "manifest.json") != core["main_manifest_sha256"]:
            raise RuntimeError("Main manifest changed")
        if _sha256(main_run / "training.json") != core["training_records_sha256"]:
            raise RuntimeError("Main training evidence changed")
        for seed, files in checkpoints.items():
            if any(
                _sha256(main_run / "models" / f"response-{seed}" / name) != digest
                for name, digest in files.items()
            ):
                raise RuntimeError("Response checkpoint changed")
        if any(
            _sha256(ridge_run / name) != digest
            for name, digest in ridge_evidence["files_sha256"].items()
        ):
            raise RuntimeError("Ridge evidence changed")

    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this predictive diagnostic output") from error
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            manifest = read_json(manifest_path)
            previous_core = {
                key: value
                for key, value in manifest.items()
                if key not in {"identity", "summary_sha256"}
            }
            if fingerprint(previous_core) != identity or manifest.get("identity") != identity:
                raise RuntimeError("Output belongs to a different predictive diagnostic identity")
            status_path = output / "status.json"
            if status_path.exists() and read_json(status_path).get("phase") == "complete":
                status = read_json(status_path)
                summary_path = output / "summary.json"
                if (
                    status.get("identity") != identity
                    or status.get("summary_sha256") != manifest.get("summary_sha256")
                    or _sha256(summary_path) != manifest.get("summary_sha256")
                    or read_json(summary_path).get("identity") != identity
                ):
                    raise RuntimeError("Completed predictive evidence identity/hash mismatch")
                guard()
                return output
        elif any(path.name != ".run.lock" for path in output.iterdir()):
            raise RuntimeError("Nonempty output lacks a predictive manifest; use a new directory")
        guard()
        write_json(manifest_path, core | {"identity": identity})
        started = time.perf_counter()

        def status(phase: str, **details) -> None:
            write_json(
                output / "status.json",
                {
                    "phase": phase,
                    "identity": identity,
                    "elapsed_this_session_seconds": time.perf_counter() - started,
                    "updated_unix": time.time(),
                    **details,
                },
            )

        try:
            torch.set_num_threads(THREADS)
            status("validating-inputs")
            _validate_caches(paths, study["focal_agents"])
            models = {}
            for seed in SEEDS:
                model, saved = _response(
                    main_run, config, core["main_identity"], seed, records[seed]
                )
                models[seed] = (model, saved)
            status("building-validation-trajectories")
            runs, states, labels, markets = _validation_rows(
                paths["validation"], actions, study["focal_agents"]
            )
            rates = np.asarray(empirical_rates(paths["train"], actions), dtype=np.float64)
            _finite("training empirical rates", rates)
            if rates.shape != (2, ACTION_COUNT) or np.any(rates < 0) or np.any(states[:, 11] < 0):
                raise RuntimeError("Invalid empirical rates or observed opportunity counts")
            methods = []
            status("predicting", method="empirical")
            predicted = rates[None] * (states[:, 11].astype(np.float64) * 10000.0)[:, None, None]
            methods.append(
                {"method": "empirical", "seed": None, **_method_metrics(predicted, labels, markets)}
            )
            status("predicting", method="ridge")
            methods.append(
                {
                    "method": "ridge",
                    "seed": None,
                    **_method_metrics(ridge.predict_curves(states), labels, markets),
                }
            )
            for seed in SEEDS:
                model, saved = models.pop(seed)
                predicted = _neural_predictions(
                    model,
                    saved,
                    runs,
                    lambda count, seed=seed: status(
                        "predicting", method="response", seed=seed, rows_complete=count
                    ),
                )
                methods.append(
                    {
                        "method": "response",
                        "seed": seed,
                        "checkpoint_step": saved["step"],
                        **_method_metrics(predicted, labels, markets),
                    }
                )
                del model, saved, predicted
            summary = {
                "identity": identity,
                "main_identity": core["main_identity"],
                "ridge_identity": ridge.identity,
                "sample_counts": {
                    "rows": len(states),
                    "behavior_trajectories": len(runs),
                    "independent_validation_markets": len(paths["validation"]),
                    "training_markets_for_empirical": len(paths["train"]),
                    "market_focal_pairs": len(paths["validation"]) * len(study["focal_agents"]),
                    "per_market_rows": {
                        str(market): int(np.count_nonzero(markets == market))
                        for market in np.unique(markets)
                    },
                },
                "empirical_rates_normalized_per_opportunity": rates.tolist(),
                "methods": methods,
                "limitations": [
                    "Current state includes cumulative and previous-tick feedback; this is not a history-free comparison.",
                    "Neural checkpoints and ridge lambda were selected using validation data; these are descriptive selected-validation estimates.",
                    "One prediction uses each row's maximum causal context; sampled-window token weighting in training logs differs, so reported losses do not reproduce checkpoint validation losses.",
                    "Behavior trajectories repeat labels; rows within trajectories, focal campaigns, and markets are correlated.",
                    "Neural seeds are reported separately and are not an ensemble or independent validation markets.",
                    "Current curves are comparable across all methods; empirical has no remaining-horizon curves.",
                    "Forecast errors do not establish policy superiority or attribute a performance bottleneck to one cause.",
                    "No test or confirmation market, main selection, or main policy outcome is read.",
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
    parser.add_argument("--main-run", required=True, type=Path)
    parser.add_argument("--ridge-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.main_run, args.ridge_run, args.output)


if __name__ == "__main__":
    main()
