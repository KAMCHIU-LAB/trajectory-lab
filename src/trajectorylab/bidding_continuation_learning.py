"""Cheap conditional-value learnability checks for exact feedback continuations.

Only causal observations and the public feedback baseline action form features.
The simulator's counterfactual labels supervise the heads and diagnostic scores;
they never enter the deployed feature encoder. This is simulator research, and
the point predictions and development diagnostics provide no safety guarantee.
"""

from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trajectorylab.bidding.learning import ResponseModel
from trajectorylab.utils import write_json

FAMILIES = ("utility", "joint_mean")
ACTION_COUNT = 11
STATE_DIMENSION = 13
ENCODER_WIDTH = 128
CONDITION_DIMENSION = ENCODER_WIDTH + STATE_DIMENSION + ACTION_COUNT
TARGET_STD_FLOOR = 0.01
ARCHITECTURE = {"input": CONDITION_DIMENSION, "width": 128, "hidden_layers": 2}


class ContinuationHead(nn.Module):
    def __init__(self, family: str):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError("Expected continuation family=utility|joint_mean")
        self.network = nn.Sequential(
            nn.Linear(CONDITION_DIMENSION, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU(),
            nn.Linear(128, 33 if family == "utility" else 24),
        )

    def forward(self, condition):
        return self.network(condition)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actions(values) -> np.ndarray:
    actions = np.asarray(values, dtype=np.float64)
    if (
        actions.shape != (ACTION_COUNT,)
        or not np.isfinite(actions).all()
        or actions[0] != 0
        or np.any(np.diff(actions) <= 0)
    ):
        raise ValueError("Continuation models require eleven increasing actions from zero")
    return actions


def _observations(states, steps, lengths, baseline_index, context: int):
    states = np.asarray(states, dtype=np.float32)
    steps = np.asarray(steps)
    lengths = np.asarray(lengths)
    baseline = np.asarray(baseline_index)
    count = len(states)
    if (
        not count
        or states.shape != (count, context, STATE_DIMENSION)
        or not np.isfinite(states).all()
        or steps.shape != (count, context)
        or lengths.shape != (count,)
        or baseline.shape != (count,)
        or not np.issubdtype(steps.dtype, np.integer)
        or not np.issubdtype(lengths.dtype, np.integer)
        or not np.issubdtype(baseline.dtype, np.integer)
        or np.any(lengths < 1)
        or np.any(lengths > context)
        or np.any(baseline < 0)
        or np.any(baseline >= ACTION_COUNT)
    ):
        raise ValueError("Expected nonempty right-padded causal observation contexts")
    mask = np.arange(context)[None] < lengths[:, None]
    if np.any(steps[mask] < 0) or np.any(steps[mask] >= 48):
        raise ValueError("Valid observation steps must be actual tick IDs in [0,47]")
    for row, length in zip(steps, lengths, strict=True):
        if np.any(np.diff(row[:length]) != 1):
            raise ValueError("Valid causal observation tick IDs must increase consecutively")
    current = states[np.arange(count), lengths - 1]
    return (
        states,
        steps.astype(np.int64),
        lengths.astype(np.int64),
        baseline.astype(np.int64),
        mask,
        current,
    )


def _validate_data(data: dict, context: int) -> None:
    observations = _observations(
        data["states"], data["steps"], data["lengths"], data["baseline_index"], context
    )
    count = len(observations[0])
    for field, shape in (
        ("endpoints", (count, ACTION_COUNT, 2)),
        ("terminal_score", (count, ACTION_COUNT)),
        ("terminal_conversions", (count, ACTION_COUNT)),
        ("terminal_violation", (count, ACTION_COUNT)),
        ("budget", (count,)),
        ("cpa", (count,)),
    ):
        if np.shape(data[field]) != shape or not np.isfinite(data[field]).all():
            raise ValueError("Invalid continuation supervision: " + field)
    if (
        np.any(data["budget"] <= 0)
        or np.any(data["cpa"] <= 0)
        or np.any(data["endpoints"] < 0)
        or np.any(data["terminal_score"] < 0)
        or np.any(data["terminal_conversions"] < 0)
        or not np.isin(data["terminal_violation"], [0, 1]).all()
    ):
        raise ValueError("Continuation labels must use positive scales and nonnegative outcomes")


def _encoder(frozen: dict) -> ResponseModel:
    if (
        frozen.get("family") != "response"
        or frozen.get("kind") != "response"
        or frozen["architecture"]["width"] != ENCODER_WIDTH
    ):
        raise RuntimeError("A frozen augmented response encoder with width 128 is required")
    mean, std = np.asarray(frozen["mean"]), np.asarray(frozen["std"])
    if (
        mean.shape != (STATE_DIMENSION,)
        or std.shape != (STATE_DIMENSION,)
        or not np.isfinite(np.r_[mean, std]).all()
        or np.any(std <= 0)
    ):
        raise RuntimeError("Invalid frozen observation normalizer")
    model = ResponseModel(STATE_DIMENSION, ACTION_COUNT, **frozen["architecture"])
    model.load_state_dict(frozen["model"])
    model.requires_grad_(False)
    model.eval()
    _finite_parameters(model)
    return model


@torch.no_grad()
def _features(
    encoder: ResponseModel, mean, std, states, steps, lengths, baseline_index, context: int
) -> np.ndarray:
    states, steps, lengths, baseline, mask, _current = _observations(
        states, steps, lengths, baseline_index, context
    )
    features = []
    for start in range(0, len(states), 64):
        stop = min(start + 64, len(states))
        x = torch.as_tensor((states[start:stop] - mean) / std, dtype=torch.float32)
        time_ids = torch.as_tensor(steps[start:stop])
        valid = torch.as_tensor(mask[start:stop])
        # Padded timestep IDs are irrelevant, but keep their embeddings in range.
        time_ids = time_ids.masked_fill(~valid, 0)
        hidden = encoder.embedding(x) + encoder.time(time_ids)
        for block in encoder.blocks:
            hidden = block(hidden, valid)
        hidden = encoder.norm(hidden)
        rows = torch.arange(stop - start)
        last = torch.as_tensor(lengths[start:stop] - 1)
        one_hot = F.one_hot(torch.as_tensor(baseline[start:stop]), ACTION_COUNT).float()
        condition = torch.cat((hidden[rows, last], x[rows, last], one_hot), dim=-1)
        features.append(condition.numpy())
    result = np.concatenate(features)
    if result.shape != (len(states), CONDITION_DIMENSION) or not np.isfinite(result).all():
        raise RuntimeError("Invalid frozen continuation features")
    return result


def _targets(family: str, data: dict) -> np.ndarray:
    baseline = np.asarray(data["baseline_index"], dtype=np.int64)
    rows = np.arange(len(baseline))
    if family == "utility":
        scale = np.asarray(data["cpa"]) / np.asarray(data["budget"])
        values = np.stack(
            (
                data["terminal_score"] * scale[:, None],
                data["terminal_conversions"] * scale[:, None],
                data["terminal_violation"],
            ),
            axis=-1,
        )
        return values.reshape(-1, 33).astype(np.float32)
    endpoints = np.asarray(data["endpoints"], dtype=np.float64)
    base = endpoints[rows, baseline]
    delta = endpoints - base[:, None]
    delta[rows, baseline] = 0.0
    return np.concatenate((base, delta.reshape(-1, 22)), axis=-1).astype(np.float32)


def _loss(head: ContinuationHead, family: str, features, targets):
    raw = head(features)
    if family == "joint_mean":
        return F.mse_loss(raw, targets)
    output = raw.reshape(-1, ACTION_COUNT, 3)
    target = targets.reshape(-1, ACTION_COUNT, 3)
    return F.mse_loss(
        F.softplus(output[..., :2]), target[..., :2]
    ) + F.binary_cross_entropy_with_logits(output[..., 2], target[..., 2])


def _decode(
    family: str,
    raw: np.ndarray,
    center,
    std,
    current: np.ndarray,
    baseline_index: np.ndarray,
    remaining_budget=None,
) -> dict:
    count = len(raw)
    rows = np.arange(count)
    baseline = np.asarray(baseline_index, dtype=np.int64)
    if family == "utility":
        output = torch.as_tensor(raw).reshape(count, ACTION_COUNT, 3)
        positive = F.softplus(output[..., :2]).numpy()
        result = {
            "utility": positive[..., 0],
            "conv": positive[..., 1],
            "violation": torch.sigmoid(output[..., 2]).numpy(),
        }
    else:
        decoded = np.asarray(raw, dtype=np.float64) * std + center
        base = decoded[:, :2]
        delta = decoded[:, 2:].reshape(count, ACTION_COUNT, 2)
        delta[rows, baseline] = 0.0
        endpoints = base[:, None] + delta
        remaining = current[:, 2] if remaining_budget is None else np.asarray(remaining_budget)
        if (
            np.shape(remaining) != (count,)
            or not np.isfinite(remaining).all()
            or np.any(remaining < 0)
        ):
            raise ValueError("remaining_budget must contain normalized nonnegative budgets")
        support_clipped = np.stack(
            (
                (endpoints[..., 0] < 0) | (endpoints[..., 0] > remaining[:, None]),
                endpoints[..., 1] < 0,
            ),
            axis=-1,
        )
        endpoints[..., 0] = np.minimum(np.maximum(endpoints[..., 0], 0), remaining[:, None])
        endpoints[..., 1] = np.maximum(endpoints[..., 1], 0)
        # Baseline is represented once, and every candidate is its paired delta.
        baseline_endpoint = np.stack(
            (np.minimum(np.maximum(base[:, 0], 0), remaining), np.maximum(base[:, 1], 0)), axis=-1
        )
        endpoints[rows, baseline] = baseline_endpoint
        cost = current[:, 3, None] + endpoints[..., 0]
        conv = current[:, 4, None] + endpoints[..., 1]
        utility = conv * np.minimum(1.0, np.square(conv / np.maximum(cost, 1e-12)))
        result = {
            "utility": utility,
            "conv": conv,
            "violation": (cost > conv).astype(np.float64),
            "endpoints": endpoints,
            "support_clipped": support_clipped,
        }
    result["delta_score"] = result["utility"] - result["utility"][rows, baseline, None]
    result["delta_score"][rows, baseline] = 0.0
    if not all(np.isfinite(value).all() for value in result.values()):
        raise RuntimeError("Nonfinite continuation predictions")
    return result


def _diagnostic(data: dict, prediction: dict) -> dict:
    baseline = np.asarray(data["baseline_index"], dtype=np.int64)
    rows = np.arange(len(baseline))
    scales = np.asarray(data["cpa"]) / np.asarray(data["budget"])
    scores = np.asarray(data["terminal_score"])
    normalized = scores * scales[:, None]
    delta = normalized - normalized[rows, baseline, None]
    nonbaseline = np.ones(delta.shape, dtype=bool)
    nonbaseline[rows, baseline] = False
    errors = prediction["delta_score"] - delta
    chosen = np.argmax(prediction["delta_score"], axis=-1)
    top = normalized.max(axis=-1)
    baseline_score = scores[rows, baseline]
    chosen_score = scores[rows, chosen]
    conv = np.asarray(data["terminal_conversions"])
    violation = np.asarray(data["terminal_violation"])
    base_conv = conv[rows, baseline]
    selected_conv = conv[rows, chosen]
    value = {
        "contexts": len(baseline),
        "delta_score_rmse_normalized_nonbaseline": float(
            np.sqrt(np.mean(errors[nonbaseline] ** 2))
        ),
        "top_choice_tie_aware_accuracy": float(np.mean(normalized[rows, chosen] >= top - 1e-10)),
        "baseline_delta_max_absolute": float(
            np.max(np.abs(prediction["delta_score"][rows, baseline]))
        ),
        "oracle_delta_score_mean_normalized": float(np.mean(top - normalized[rows, baseline])),
        "greedy_selected_delta_score_mean_normalized": float(np.mean(delta[rows, chosen])),
        "unconstrained_greedy_one_shot_diagnostic": {
            "score_ratio": float(chosen_score.mean() / baseline_score.mean())
            if baseline_score.mean()
            else None,
            "conversion_ratio": float(selected_conv.mean() / base_conv.mean())
            if base_conv.mean()
            else None,
            "violation_frequency_difference": float(
                np.mean(violation[rows, chosen] - violation[rows, baseline])
            ),
        },
    }
    if "support_clipped" in prediction:
        clipped = np.asarray(prediction["support_clipped"], dtype=bool)
        value["physical_support_clipped_fraction"] = float(np.mean(clipped))
        value["physical_support_clipped_cost_fraction"] = float(np.mean(clipped[..., 0]))
        value["physical_support_clipped_conv_fraction"] = float(np.mean(clipped[..., 1]))
        value["physical_support_clipped_any_endpoint_fraction"] = float(
            np.mean(np.any(clipped, axis=-1))
        )
    if "diagnostics" in data and "scene_index" in (data["diagnostics"].dtype.names or ()):
        value["by_scene_index"] = {}
        for scene in np.unique(data["diagnostics"]["scene_index"]):
            mask = data["diagnostics"]["scene_index"] == scene
            local = {
                key: np.asarray(item)[mask]
                for key, item in data.items()
                if isinstance(item, np.ndarray)
                and len(item) == len(baseline)
                and key != "diagnostics"
            }
            local_prediction = {key: item[mask] for key, item in prediction.items()}
            value["by_scene_index"][str(int(scene))] = _diagnostic(local, local_prediction)
    return value


def _finite_parameters(model: nn.Module) -> None:
    parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    if not torch.isfinite(parameters).all():
        raise RuntimeError("Nonfinite continuation model parameters")


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fit(
    family: str,
    seed: int,
    config: dict,
    train: dict,
    validation: dict,
    encoder_path: Path,
    directory: Path,
    identity: str,
) -> dict:
    """Fit the cheap CPU pilot; identity must bind data, source and encoder SHA256s."""
    if family not in FAMILIES or not identity:
        raise ValueError("A registered continuation family and nonempty identity are required")
    opts = {
        "steps": 2000,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "threads": 2,
        "context": 24,
        "device": "cpu",
        "log_every": 100,
        "checkpoint_every": 250,
    }
    opts.update(config.get("training", {}))
    if opts["device"] != "cpu":
        raise ValueError("The continuation learnability pilot is CPU only")
    if (
        not 1 <= opts["threads"] <= 2
        or not 1 <= opts["context"] <= 48
        or any(
            opts[key] <= 0
            for key in ("steps", "batch_size", "learning_rate", "log_every", "checkpoint_every")
        )
    ):
        raise ValueError("Invalid bounded CPU continuation training options")
    actions = _actions(config["study"]["actions"])
    _validate_data(train, opts["context"])
    _validate_data(validation, opts["context"])
    torch.set_num_threads(opts["threads"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder_path, directory = Path(encoder_path), Path(directory)
    encoder_sha = _sha256(encoder_path)
    frozen = torch.load(encoder_path, map_location="cpu", weights_only=False)
    if (
        _sha256(encoder_path) != encoder_sha
        or frozen.get("actions") != actions.tolist()
        or frozen.get("context") != opts["context"]
    ):
        raise RuntimeError("Frozen continuation encoder identity/grid/context mismatch")
    encoder = _encoder(frozen)
    feature_started = time.perf_counter()
    features = {}
    for name, data in (("train", train), ("validation", validation)):
        features[name] = _features(
            encoder,
            frozen["mean"],
            frozen["std"],
            data["states"],
            data["steps"],
            data["lengths"],
            data["baseline_index"],
            opts["context"],
        )
    feature_seconds = time.perf_counter() - feature_started
    targets = {
        name: _targets(family, data)
        for name, data in (("train", train), ("validation", validation))
    }
    if family == "joint_mean":
        center = targets["train"].astype(np.float64).mean(axis=0).astype(np.float32)
        std = np.maximum(targets["train"].astype(np.float64).std(axis=0), TARGET_STD_FLOOR).astype(
            np.float32
        )
    else:
        center, std = np.zeros(33, dtype=np.float32), np.ones(33, dtype=np.float32)
    tensors = {
        name: (torch.as_tensor(features[name]), torch.as_tensor((targets[name] - center) / std))
        for name in ("train", "validation")
    }
    head = ContinuationHead(family)
    optimizer = torch.optim.AdamW(head.parameters(), lr=opts["learning_rate"], weight_decay=1e-4)
    metadata = {
        "schema_version": 1,
        "identity": identity,
        "family": family,
        "seed": seed,
        "actions": actions.tolist(),
        "context": opts["context"],
        "device": "cpu",
        "architecture": ARCHITECTURE,
        "encoder_sha256": encoder_sha,
        "encoder_identity": frozen["identity"],
        "encoder_architecture": frozen["architecture"],
        "encoder_mean": frozen["mean"],
        "encoder_std": frozen["std"],
        "target_center": center,
        "target_std": std,
        "target_std_floor": TARGET_STD_FLOOR,
        "feature_fields": ["states", "steps", "lengths", "baseline_index"],
        "training_options": {
            key: opts[key]
            for key in ("batch_size", "learning_rate", "threads", "context", "device")
        },
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "frozen_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "validation_selection": "minimum full-validation training objective",
    }
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, best = directory / "checkpoint.pt", directory / "best.pt"
    step, best_loss, total_seconds, history = 0, float("inf"), 0.0, []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in (
            "schema_version",
            "identity",
            "family",
            "seed",
            "actions",
            "context",
            "device",
            "architecture",
            "encoder_sha256",
            "encoder_identity",
            "training_options",
        ):
            if saved.get(key) != metadata[key]:
                raise RuntimeError("Continuation checkpoint metadata mismatch: " + key)
        for key in ("target_center", "target_std", "encoder_mean", "encoder_std"):
            if not np.array_equal(saved[key], metadata[key]):
                raise RuntimeError("Continuation checkpoint normalization mismatch: " + key)
        if saved["step"] > opts["steps"]:
            raise RuntimeError("Requested step budget is below the existing checkpoint")
        head.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        step, best_loss, total_seconds = saved["step"], saved["best_loss"], saved["seconds"]
        history = saved["history"]
    elif best.exists() or (directory / "history.json").exists():
        raise RuntimeError("Continuation output exists without a resumable checkpoint")
    _finite_parameters(head)
    started = time.perf_counter()

    def payload():
        return metadata | {"model": head.state_dict(), "encoder_model": encoder.state_dict()}

    while step < opts["steps"]:
        head.train()
        indices = torch.as_tensor(rng.integers(len(features["train"]), size=opts["batch_size"]))
        objective = _loss(head, family, tensors["train"][0][indices], tensors["train"][1][indices])
        if not torch.isfinite(objective):
            raise RuntimeError("Nonfinite continuation training loss")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("Nonfinite continuation gradient")
        optimizer.step()
        step += 1
        if (
            step % opts["log_every"] == 0
            or step % opts["checkpoint_every"] == 0
            or step == opts["steps"]
        ):
            _finite_parameters(head)
        if step % opts["log_every"] == 0 or step == opts["steps"]:
            head.eval()
            with torch.no_grad():
                validation_loss = float(_loss(head, family, *tensors["validation"]))
            if not np.isfinite(validation_loss):
                raise RuntimeError("Nonfinite continuation validation loss")
            history.append(
                {
                    "step": step,
                    "training_loss": float(objective.detach()),
                    "validation_loss": validation_loss,
                    "seconds": total_seconds + time.perf_counter() - started,
                }
            )
            print(
                f"continuation family={family} seed={seed} step={step} "
                f"validation_loss={validation_loss:.6f}",
                flush=True,
            )
            write_json(directory / "history.json", history)
            if validation_loss < best_loss:
                best_loss = validation_loss
                _save(best, payload() | {"step": step, "validation_loss": validation_loss})
        if step % opts["checkpoint_every"] == 0 or step == opts["steps"]:
            _save(
                checkpoint,
                payload()
                | {
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_loss": best_loss,
                    "history": history,
                    "seconds": total_seconds + time.perf_counter() - started,
                    "torch_rng": torch.get_rng_state(),
                    "numpy_rng": rng.bit_generator.state,
                },
            )
    if not best.exists() or not np.isfinite(best_loss):
        raise RuntimeError("Continuation fitting did not produce a finite best model")
    if _sha256(encoder_path) != encoder_sha:
        raise RuntimeError("Frozen encoder file changed during fitting")
    best_saved = torch.load(best, map_location="cpu", weights_only=False)
    head.load_state_dict(best_saved["model"])
    head.eval()
    with torch.no_grad():
        raw = head(tensors["validation"][0]).numpy()
    valid_lengths = np.asarray(validation["lengths"], dtype=np.int64)
    current = np.asarray(validation["states"])[np.arange(len(valid_lengths)), valid_lengths - 1]
    prediction = _decode(family, raw, center, std, current, validation["baseline_index"])
    result = {
        "family": family,
        "seed": seed,
        "steps": step,
        "best_step": best_saved["step"],
        "best_validation_loss": best_loss,
        "parameters": metadata["parameters"],
        "frozen_parameters": metadata["frozen_parameters"],
        "encoder_sha256": encoder_sha,
        "feature_precompute_seconds": feature_seconds,
        "seconds": total_seconds + time.perf_counter() - started,
        "validation": _diagnostic(validation, prediction),
    }
    write_json(directory / "fit-summary.json", result)
    return result


class ContinuationModel:
    """Load a self-contained predictor; all outputs use normalized budget units."""

    def __init__(self, path: Path, actions: np.ndarray, device: str = "cpu"):
        if device != "cpu":
            raise ValueError("The continuation learnability pilot is CPU only")
        path = Path(path)
        checkpoint_sha = _sha256(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if _sha256(path) != checkpoint_sha:
            raise RuntimeError("Continuation checkpoint changed while loading")
        actions = _actions(actions)
        if (
            saved.get("schema_version") != 1
            or saved.get("family") not in FAMILIES
            or saved.get("architecture") != ARCHITECTURE
            or saved.get("actions") != actions.tolist()
        ):
            raise RuntimeError("Unsupported continuation model or action grid")
        self.identity, self.training_identity = checkpoint_sha, saved["identity"]
        self.family, self.seed, self.context = saved["family"], saved["seed"], saved["context"]
        self.actions, self.device = actions, device
        self.mean, self.std = saved["encoder_mean"], saved["encoder_std"]
        self.target_center, self.target_std = saved["target_center"], saved["target_std"]
        target_dimension = 33 if self.family == "utility" else 24
        if (
            np.shape(self.target_center) != (target_dimension,)
            or np.shape(self.target_std) != (target_dimension,)
            or not np.isfinite(np.r_[self.target_center, self.target_std]).all()
            or np.any(self.target_std <= 0)
        ):
            raise RuntimeError("Invalid continuation target normalizer")
        frozen = {
            "family": "response",
            "kind": "response",
            "architecture": saved["encoder_architecture"],
            "mean": self.mean,
            "std": self.std,
            "model": saved["encoder_model"],
        }
        self.encoder = _encoder(frozen)
        self.head = ContinuationHead(self.family)
        self.head.load_state_dict(saved["model"])
        self.head.requires_grad_(False)
        self.head.eval()
        _finite_parameters(self.head)

    @torch.no_grad()
    def predict(self, states, steps, lengths, baseline_index, remaining_budget=None) -> dict:
        """Return utility, conv, violation and delta_score, each shaped (N,11).

        Utility is E[terminal score]*cpa/budget and conv is E[terminal conv]*cpa/
        budget. Joint means also return remaining cost/budget and conv*cpa/budget
        in endpoints[N,11,2]. Its violation output is a deterministic plug-in
        indicator; the direct utility head returns a learned probability.
        ``remaining_budget`` is normalized by budget and defaults to state[2].
        """
        features = _features(
            self.encoder, self.mean, self.std, states, steps, lengths, baseline_index, self.context
        )
        raw = self.head(torch.as_tensor(features)).numpy()
        lengths = np.asarray(lengths, dtype=np.int64)
        current = np.asarray(states)[np.arange(len(lengths)), lengths - 1]
        return _decode(
            self.family,
            raw,
            self.target_center,
            self.target_std,
            current,
            np.asarray(baseline_index),
            remaining_budget,
        )


def diagnostic(model: ContinuationModel, data: dict) -> dict:
    """Development-only ranking and exact-label one-shot greedy diagnostics."""
    _validate_data(data, model.context)
    prediction = model.predict(
        data["states"], data["steps"], data["lengths"], data["baseline_index"]
    )
    return _diagnostic(data, prediction)
