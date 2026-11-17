"""Validation-only ridge response diagnostic using current causal state covariates.

The current state includes cumulative and previous-tick feedback. This comparator
therefore does not isolate history or architecture from the trained response model.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from trajectorylab.bidding.experiment import provenance
from trajectorylab.bidding.learning import trajectories
from trajectorylab.utils import fingerprint, read_json, write_json

STATE_DIM = 13
ACTION_COUNT = 11
FEATURE_DIM = 105
GROUPS = ("current_cost", "current_conversion", "remaining_cost", "remaining_conversion")
SCALES = np.array([48.0, 48.0, 1.0, 1.0])
LAMBDAS = (1e-6, 1e-4, 1e-2, 1.0)
STD_FLOOR = 0.05
STATE_CLIP = 8.0
LOG_CLIP = 20.0
SETTINGS = {
    "state_dim": STATE_DIM,
    "action_count": ACTION_COUNT,
    "feature_dim": FEATURE_DIM,
    "features": "intercept, z[0:13], z[i]*z[j] in lexicographic i<=j order",
    "normalization": "training-row population mean/std; std floored before clipping",
    "std_floor": STD_FLOOR,
    "standardized_state_clip": [-STATE_CLIP, STATE_CLIP],
    "groups": list(GROUPS),
    "label_units": "cost/budget and conversions*cpa/budget; current then remaining",
    "target_scales": SCALES.tolist(),
    "target": "log1p(normalized_labels * target_scales)",
    "numerical_dtype": "float64 for normalization, features, fitting, projection, and metrics",
    "ridge_lambdas": list(LAMBDAS),
    "ridge_objective": "XTX/N + lambda*diag(0,1,...,1); intercept unpenalized",
    "ridge_solver": "numpy.linalg.solve with all training rows and 44 simultaneous targets",
    "prediction_log_clip": [0.0, LOG_CLIP],
    "prediction_log_clip_reason": "fixed numerical protection; independent of validation labels",
    "projection": "clip log, expm1/scale, action zero=0, cumulative maximum over actions",
    "selection": "minimum mean squared log-space error after actual projection; validation only",
    "loss_weighting": "equal weight per row, group, and action; all 4 groups and all 11 actions",
    "tie_break": "first lambda in declared order, with exact floating-point ties",
    "behavior_trajectories_per_focal_market": 10,
    "ticks_per_trajectory": 48,
    "trajectory_source": "unchanged learning.trajectories on main train/validation caches",
    "group_ratio": "sum projected normalized predictions / sum normalized labels over rows/actions",
    "undefined_ratio": "null when the true aggregate is zero",
}
MODEL_FIELDS = {
    "identity",
    "source_sha256",
    "main_identity",
    "mean",
    "std",
    "coefficients",
    "actions",
    "target_scales",
    "selected_lambda",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite(name: str, value: np.ndarray) -> None:
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Nonfinite {name}")


def _features(states: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] != STATE_DIM or not len(states):
        raise ValueError("States must have nonempty shape (rows, 13)")
    _finite("states", states)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        z = np.clip((states - mean) / std, -STATE_CLIP, STATE_CLIP)
        i, j = np.triu_indices(STATE_DIM)
        result = np.concatenate((np.ones((len(z), 1)), z, z[:, i] * z[:, j]), axis=1)
    _finite("features", result)
    return result


def _project(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        logits = features @ coefficients
        _finite("unprojected log predictions", logits)
        logits = np.clip(logits.reshape(-1, len(GROUPS), ACTION_COUNT), 0.0, LOG_CLIP)
        curves = np.expm1(logits) / SCALES[None, :, None]
        curves[:, :, 0] = 0.0
        curves = np.maximum.accumulate(curves, axis=2)
    _finite("projected predictions", curves)
    return curves


def _metrics(predicted: np.ndarray, labels: np.ndarray) -> dict:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        difference = np.log1p(predicted * SCALES[None, :, None]) - np.log1p(
            labels * SCALES[None, :, None]
        )
        mse = float(np.mean(difference * difference))
        predicted_totals = predicted.sum(axis=(0, 2))
        true_totals = labels.sum(axis=(0, 2))
    _finite("metric aggregates", np.r_[mse, predicted_totals, true_totals])
    ratios = {
        group: float(predicted_totals[i] / true_totals[i]) if true_totals[i] > 0 else None
        for i, group in enumerate(GROUPS)
    }
    _finite("defined aggregate ratios", np.array([v for v in ratios.values() if v is not None]))
    return {"log_mse": mse, "aggregate_predicted_to_true_ratio": ratios}


def _manifest_core(manifest: dict) -> dict:
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"identity", "model_sha256", "summary_sha256", "selected_lambda"}
    }


def _check_manifest(manifest: dict) -> None:
    if fingerprint(_manifest_core(manifest)) != manifest.get("identity"):
        raise RuntimeError("Ridge manifest identity mismatch")
    if manifest.get("settings") != SETTINGS or manifest.get("schema_version") != 1:
        raise RuntimeError("Unsupported ridge manifest settings")
    if manifest.get("source_sha256") != _sha256(Path(__file__)):
        raise RuntimeError("Ridge source differs from the fitted model source")
    if manifest.get("utils_sha256") != _sha256(Path(__file__).with_name("utils.py")):
        raise RuntimeError("Utility source differs from the fitted model dependency")
    package = Path(__file__).parent
    files = list((package / "bidding").glob("*.py")) + [package / "models.py"]
    implementation = {path.name: _sha256(path) for path in files}
    if manifest.get("main_implementation") != implementation:
        raise RuntimeError("Main implementation differs from the fitted model dependencies")


class RidgeForecast:
    """Load a verified model; use only the last current state for deployment.

    ``predict_curves`` returns (rows, 4, 11) curves in normalized label units.
    ``__call__`` returns (2, 11) current cost/conversion curves in raw units,
    matching the forecast interface consumed by ``PacedResponse``.
    """

    def __init__(self, model_path: Path | str):
        model_path = Path(model_path)
        manifest = read_json(model_path.parent / "manifest.json")
        _check_manifest(manifest)
        if _sha256(model_path) != manifest.get("model_sha256"):
            raise RuntimeError("Ridge model file hash mismatch")
        with np.load(model_path, allow_pickle=False) as saved:
            if set(saved.files) != MODEL_FIELDS:
                raise RuntimeError("Unexpected ridge model fields")
            for key in ("identity", "source_sha256", "main_identity"):
                if saved[key].shape != () or saved[key].item() != manifest[key]:
                    raise RuntimeError(f"Ridge model {key} mismatch")
            self.mean = np.array(saved["mean"], dtype=np.float64)
            self.std = np.array(saved["std"], dtype=np.float64)
            self.coefficients = np.array(saved["coefficients"], dtype=np.float64)
            self.actions = np.array(saved["actions"], dtype=np.float64)
            scales = np.array(saved["target_scales"], dtype=np.float64)
            selected = saved["selected_lambda"]
            if selected.shape != () or selected.item() != manifest.get("selected_lambda"):
                raise RuntimeError("Ridge selected lambda mismatch")
        if self.mean.shape != (STATE_DIM,) or self.std.shape != (STATE_DIM,):
            raise RuntimeError("Invalid ridge normalization shape")
        if self.coefficients.shape != (FEATURE_DIM, len(GROUPS) * ACTION_COUNT):
            raise RuntimeError("Invalid ridge coefficient shape")
        _validate_actions(self.actions)
        if self.actions.tolist() != manifest["actions"] or not np.array_equal(scales, SCALES):
            raise RuntimeError("Ridge action grid or target scale mismatch")
        for name in ("mean", "std", "coefficients", "actions"):
            _finite(name, getattr(self, name))
        if np.any(self.std < STD_FLOOR) or manifest["selected_lambda"] not in LAMBDAS:
            raise RuntimeError("Invalid ridge normalization or regularization")
        self.identity = manifest["identity"]

    def predict_curves(self, states: np.ndarray) -> np.ndarray:
        return _project(_features(states, self.mean, self.std), self.coefficients)

    def __call__(self, states, budget: float, cpa: float) -> np.ndarray:
        if not np.isfinite(budget) or not np.isfinite(cpa) or budget <= 0 or cpa <= 0:
            raise ValueError("Budget and CPA must be finite and positive")
        current = np.asarray(states, dtype=np.float64)
        if current.ndim != 2 or current.shape[1] != STATE_DIM or not len(current):
            raise ValueError("Expected nonempty causal state sequence with 13 columns")
        curves = self.predict_curves(current[-1:])[0, :2]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            curves = curves * np.array([budget, budget / cpa])[:, None]
        _finite("raw current predictions", curves)
        return curves


def _validate_actions(actions: np.ndarray) -> None:
    if actions.shape != (ACTION_COUNT,):
        raise ValueError("This diagnostic requires exactly 11 actions")
    _finite("actions", actions)
    if actions[0] != 0 or actions[-1] != 3 or np.any(np.diff(actions) <= 0):
        raise ValueError("Actions must increase strictly from zero to three")


def _main_inputs(main_run: Path) -> tuple[dict, dict, dict, np.ndarray]:
    main = read_json(main_run / "manifest.json")
    config = main["config"]
    identity = main.get("identity")
    stored = {key: value for key, value in main.items() if key != "identity"}
    if fingerprint(stored) != identity or fingerprint(provenance(config)) != identity:
        raise RuntimeError("Main manifest does not match current source/configuration provenance")
    study = config["study"]
    actions = np.asarray(study["actions"], dtype=np.float64)
    _validate_actions(actions)
    focals = study["focal_agents"]
    if not focals or len(set(focals)) != len(focals):
        raise ValueError("Main focal agents must be nonempty and unique")
    ids = {split: study[f"{split}_markets"] for split in ("train", "validation")}
    joined = ids["train"] + ids["validation"]
    if not all(ids.values()) or len(set(joined)) != len(joined):
        raise ValueError("Train/validation market IDs must be nonempty, unique, and disjoint")
    paths, evidence = {}, {}
    for split, market_ids in ids.items():
        paths[split] = []
        market_identities = {}
        for market in market_ids:
            if not isinstance(market, int) or isinstance(market, bool) or market < 0:
                raise ValueError("Market IDs must be nonnegative integers")
            path = main_run / "markets" / f"{market}.npz"
            meta_path = path.with_suffix(".json")
            if not path.is_file() or not meta_path.is_file():
                raise RuntimeError(f"Missing {split} market cache or metadata")
            meta = read_json(meta_path)
            expected = fingerprint(
                {
                    "study": {
                        key: study[key] for key in ("focal_agents", "pv_count", "actions", "biases")
                    },
                    "market": market,
                    "source": main["auction"],
                }
            )
            if (
                meta.get("identity") != expected
                or meta.get("market_id") != market
                or meta.get("focal_agents") != focals
                or meta.get("requested_pv_count") != study["pv_count"]
                or meta.get("source") != main["auction"]
                or meta.get("bytes") != path.stat().st_size
            ):
                raise RuntimeError(f"Invalid {split} market cache metadata")
            market_identities[str(market)] = expected
            paths[split].append(path)
        evidence[split] = {"market_ids": market_ids, "market_identities": market_identities}
    core = {
        "schema_version": 1,
        "purpose": "current-state quadratic ridge validation diagnostic",
        "source_sha256": _sha256(Path(__file__)),
        "main_identity": identity,
        "main_implementation": main["implementation"],
        "actions": actions.tolist(),
        "focal_agents": focals,
        "data": evidence,
        "settings": SETTINGS,
        "numpy_version": np.__version__,
        "utils_sha256": _sha256(Path(__file__).with_name("utils.py")),
    }
    return core, paths, config, actions


def _rows(
    paths: list[Path], actions: np.ndarray, focal_count: int
) -> tuple[np.ndarray, np.ndarray]:
    runs = trajectories(paths, actions)
    expected = len(paths) * focal_count * SETTINGS["behavior_trajectories_per_focal_market"]
    if len(runs) != expected:
        raise RuntimeError("Unexpected behavior trajectory coverage")
    for run in runs:
        if np.shape(run["states"]) != (48, STATE_DIM) or np.shape(run["labels"]) != (
            48,
            len(GROUPS),
            ACTION_COUNT,
        ):
            raise RuntimeError("Unexpected trajectory state/label shape")
    states = np.concatenate([run["states"] for run in runs]).astype(np.float64)
    labels = np.concatenate([run["labels"] for run in runs]).astype(np.float64)
    _finite("trajectory states", states)
    _finite("trajectory labels", labels)
    if np.any(labels < 0) or np.any(labels[:, :, 0] != 0):
        raise RuntimeError("Labels must be nonnegative with zero response at action zero")
    return states, labels


def _save_model(path: Path, arrays: dict) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".ridge-", suffix=".npz")
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def run(main_run: Path, output: Path) -> Path:
    core, paths, config, actions = _main_inputs(main_run)
    identity = fingerprint(core)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another process owns this ridge output") from error
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            existing = read_json(manifest_path)
            _check_manifest(existing)
            if existing["identity"] != identity:
                raise RuntimeError("Output belongs to a different ridge diagnostic identity")
            status_path = output / "status.json"
            if status_path.exists() and read_json(status_path).get("phase") == "complete":
                model = RidgeForecast(output / "model.npz")
                status = read_json(status_path)
                summary_path = output / "summary.json"
                if (
                    model.identity != identity
                    or status.get("identity") != identity
                    or status.get("model_sha256") != existing.get("model_sha256")
                    or status.get("summary_sha256") != existing.get("summary_sha256")
                    or _sha256(summary_path) != existing.get("summary_sha256")
                    or read_json(summary_path).get("identity") != identity
                ):
                    raise RuntimeError("Completed ridge evidence identity/hash mismatch")
                return output
        elif any(path.name != ".run.lock" for path in output.iterdir()):
            raise RuntimeError("Nonempty output lacks a ridge manifest; use a new directory")
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
            data = {}
            for split in ("train", "validation"):
                status("building-trajectories", split=split)
                states, labels = _rows(paths[split], actions, len(core["focal_agents"]))
                data[split] = {"states": states, "labels": labels}
            train = data["train"]
            mean = train["states"].mean(axis=0)
            std = np.maximum(train["states"].std(axis=0), STD_FLOOR)
            _finite("training mean/std", np.r_[mean, std])
            for rows in data.values():
                rows["features"] = _features(rows.pop("states"), mean, std)
            x = train["features"]
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                y = np.log1p(train["labels"] * SCALES[None, :, None]).reshape(len(x), -1)
                gram = x.T @ x / len(x)
                cross = x.T @ y / len(x)
            _finite("ridge normal equations", np.r_[gram.ravel(), cross.ravel()])
            penalty = np.eye(FEATURE_DIM)
            penalty[0, 0] = 0.0
            candidates = []
            best = None
            for strength in LAMBDAS:
                status("fitting", ridge_lambda=strength)
                coefficients = np.linalg.solve(gram + strength * penalty, cross)
                _finite("ridge coefficients", coefficients)
                metrics = {
                    split: _metrics(_project(rows["features"], coefficients), rows["labels"])
                    for split, rows in data.items()
                }
                candidates.append({"ridge_lambda": strength, **metrics})
                if best is None or metrics["validation"]["log_mse"] < best["validation_log_mse"]:
                    best = {
                        "ridge_lambda": strength,
                        "validation_log_mse": metrics["validation"]["log_mse"],
                        "coefficients": coefficients,
                    }
            if (
                _sha256(Path(__file__)) != core["source_sha256"]
                or fingerprint(provenance(config)) != core["main_identity"]
            ):
                raise RuntimeError("Diagnostic or main source identity changed during fitting")
            if _sha256(Path(__file__).with_name("utils.py")) != core["utils_sha256"]:
                raise RuntimeError("Utility source changed during fitting")
            status("saving", selected_lambda=best["ridge_lambda"])
            model_path = output / "model.npz"
            _save_model(
                model_path,
                {
                    "identity": np.array(identity),
                    "source_sha256": np.array(core["source_sha256"]),
                    "main_identity": np.array(core["main_identity"]),
                    "mean": mean,
                    "std": std,
                    "coefficients": best["coefficients"],
                    "actions": actions,
                    "target_scales": SCALES,
                    "selected_lambda": np.array(best["ridge_lambda"]),
                },
            )
            model_hash = _sha256(model_path)
            summary = {
                "identity": identity,
                "main_identity": core["main_identity"],
                "model_sha256": model_hash,
                "selected_lambda": best["ridge_lambda"],
                "candidates": candidates,
                "sample_counts": {
                    split: {
                        "rows": len(rows["features"]),
                        "independent_markets": len(paths[split]),
                        "behavior_trajectories": len(rows["features"]) // 48,
                        "market_focal_pairs": len(paths[split]) * len(core["focal_agents"]),
                    }
                    for split, rows in data.items()
                },
                "limitations": [
                    "Current state contains cumulative and previous-tick feedback; it is not history-free.",
                    "Behavior trajectories repeat action-curve labels; rows within markets and trajectories are correlated.",
                    "Validation selects lambda and reports its error on the same validation data; the selected estimate is optimistic.",
                    "No test or confirmation market is read and no held-out policy improvement is established.",
                    "This comparator cannot establish a single-factor causal conclusion about history or model architecture.",
                    "Aggregate curve ratios are descriptive normalized-label diagnostics, not rollout spend or conversion guarantees.",
                ],
            }
            summary_path = output / "summary.json"
            write_json(summary_path, summary)
            summary_hash = _sha256(summary_path)
            write_json(
                manifest_path,
                core
                | {
                    "identity": identity,
                    "model_sha256": model_hash,
                    "summary_sha256": summary_hash,
                    "selected_lambda": best["ridge_lambda"],
                },
            )
            RidgeForecast(model_path)
            status("complete", model_sha256=model_hash, summary_sha256=summary_hash)
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
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.main_run, args.output)


if __name__ == "__main__":
    main()
