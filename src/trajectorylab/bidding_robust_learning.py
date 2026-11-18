"""Stress-trained response forecasts using the frozen soft-CPA controller."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trajectorylab.bidding.learning import Learned, ResponseModel, _batch
from trajectorylab.utils import write_json

FAMILIES = ("response", "mlp")
LOSSES = ("log", "mean")
STATE_DIMENSION = 13


class PointwiseResponseModel(nn.Module):
    """Current-state MLP; its 13 inputs include cumulative/previous feedback."""

    def __init__(self, dimension: int, action_count: int, width: int = 448, layers: int = 3):
        super().__init__()
        if dimension != STATE_DIMENSION or width != 448 or layers != 3:
            raise ValueError("The registered MLP uses 13 inputs and three 448-unit hidden layers")
        self.action_count = action_count
        blocks = []
        for index in range(layers):
            blocks.extend([nn.Linear(dimension if index == 0 else width, width), nn.SiLU()])
        self.hidden = nn.Sequential(*blocks)
        self.decoder = nn.Linear(width, 4 * (action_count - 1))
        with torch.no_grad():
            self.decoder.bias.reshape(4, action_count - 1)[:2].fill_(-5.0)
            self.decoder.bias.reshape(4, action_count - 1)[2:].fill_(-2.0)

    def forward(self, states, steps, mask):
        raw = self.decoder(self.hidden(states)).reshape(*states.shape[:2], 4, self.action_count - 1)
        return F.pad(F.softplus(raw).cumsum(dim=-1), (1, 0), value=0.0)


def _model(family: str, action_count: int, architecture: dict) -> nn.Module:
    if family == "response":
        return ResponseModel(STATE_DIMENSION, action_count, **architecture)
    if family == "mlp":
        return PointwiseResponseModel(STATE_DIMENSION, action_count, **architecture)
    raise ValueError("Unknown response model family: " + family)


def _loss(model: nn.Module, loss: str, batch) -> torch.Tensor:
    states, _actions, _returns, labels, steps, mask = batch
    prediction = model(states, steps, mask)[..., 1:]
    labels = labels[..., 1:]
    if not torch.isfinite(prediction).all() or (prediction < 0).any():
        raise RuntimeError("Response predictions must be finite and nonnegative")
    scales = states.new_tensor([48.0, 48.0, 1.0, 1.0])[None, None, :, None]
    predicted, target = prediction * scales, labels * scales
    # Action zero is structurally zero. Exclude it before taking any logarithm;
    # for log loss this is the frozen loss times A/(A-1), with identical optimum.
    if loss == "log":
        errors = F.smooth_l1_loss(torch.log1p(predicted), torch.log1p(target), reduction="none")
    elif loss == "mean":
        errors = predicted - target * torch.log(torch.clamp(predicted, min=1e-8))
    else:
        raise ValueError("Unknown response loss: " + loss)
    return (errors.mean((-1, -2)) * mask).sum() / mask.sum()


def _validate_data(data: list[dict], action_count: int) -> None:
    if not data:
        raise ValueError("Training and validation data must both be nonempty")
    for run in data:
        shapes = {
            "states": (48, STATE_DIMENSION),
            "actions": (48,),
            "returns": (48,),
            "labels": (48, 4, action_count),
        }
        for field, shape in shapes.items():
            if np.shape(run[field]) != shape or not np.isfinite(run[field]).all():
                raise ValueError("Invalid trajectory shape or nonfinite field: " + field)
        labels = run["labels"]
        if np.any(labels < 0) or np.any(labels[:, :, 0] != 0):
            raise ValueError("Targets must be nonnegative and zero at action zero")
        if not np.isfinite(run["score"]):
            raise ValueError("Trajectory scores must be finite")


def _finite_parameters(model: nn.Module) -> None:
    if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise RuntimeError("Nonfinite response model parameters")


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fit(
    family: str,
    loss: str,
    seed: int,
    config: dict,
    train: list[dict],
    validation: list[dict],
    directory: Path,
    identity: str,
) -> dict:
    """Fit one registered family/loss with identity-checked complete resume."""
    if family not in FAMILIES or loss not in LOSSES:
        raise ValueError("Expected family=response|mlp and loss=log|mean")
    opts = config["training"]
    device = opts["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS is unavailable; declare a different device explicitly")
    if not identity:
        raise ValueError("A nonempty training identity is required")
    if any(
        opts[name] <= 0
        for name in (
            "steps",
            "batch_size",
            "context",
            "log_every",
            "checkpoint_every",
            "threads",
            "learning_rate",
        )
    ):
        raise ValueError("Training budgets and intervals must be positive")
    if opts["context"] > 48:
        raise ValueError("Context cannot exceed the 48-tick horizon")
    actions = np.asarray(config["study"]["actions"], dtype=np.float64)
    if (
        actions.ndim != 1
        or len(actions) < 2
        or not np.isfinite(actions).all()
        or actions[0] != 0
        or np.any(np.diff(actions) <= 0)
    ):
        raise ValueError("Actions must be finite, increase strictly, and start at zero")
    _validate_data(train, len(actions))
    _validate_data(validation, len(actions))
    torch.set_num_threads(opts["threads"])
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    rng = np.random.default_rng(seed)
    states = np.concatenate([run["states"] for run in train])
    mean, std = states.mean(axis=0), np.maximum(states.std(axis=0), 0.05)
    target = float(np.quantile([run["score"] for run in train], 0.9))
    return_scale = max(1.0, target)
    if not np.isfinite(np.r_[mean, std, target, return_scale]).all():
        raise RuntimeError("Nonfinite training normalization")
    architecture = (
        {name: opts[name] for name in ("width", "layers", "heads")}
        if family == "response"
        else {"width": 448, "layers": 3}
    )
    model = _model(family, len(actions), architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=opts["learning_rate"], weight_decay=1e-4)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, best = directory / "checkpoint.pt", directory / "best.pt"
    metadata = {
        "identity": identity,
        "kind": "response",
        "family": family,
        "loss": loss,
        "seed": seed,
        "mean": mean,
        "std": std,
        "return_scale": return_scale,
        "target_return": target,
        "architecture": architecture,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "actions": actions.tolist(),
        "context": opts["context"],
        "device": device,
        "current_state_contains_history_summary": True,
        "zero_action_excluded_from_loss": True,
    }
    step, best_loss, total_seconds, history = 0, float("inf"), 0.0, []
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for key in (
            "identity",
            "family",
            "loss",
            "seed",
            "architecture",
            "actions",
            "context",
            "device",
        ):
            if saved.get(key) != metadata[key]:
                raise RuntimeError("Training checkpoint metadata mismatch: " + key)
        if (
            not np.array_equal(saved["mean"], mean)
            or not np.array_equal(saved["std"], std)
            or saved["target_return"] != target
            or saved["return_scale"] != return_scale
        ):
            raise RuntimeError("Training checkpoint data normalization mismatch")
        if saved["step"] > opts["steps"]:
            raise RuntimeError("Requested step budget is smaller than the existing checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        step, best_loss, total_seconds = saved["step"], saved["best_loss"], saved["seconds"]
        history = saved["history"]
    elif best.exists() or (directory / "history.json").exists():
        raise RuntimeError("Training output exists without a resumable checkpoint")
    _finite_parameters(model)
    started = time.perf_counter()
    while step < opts["steps"]:
        model.train()
        batch = _batch(
            train, rng, opts["batch_size"], opts["context"], mean, std, return_scale, device
        )
        objective = _loss(model, loss, batch)
        if not torch.isfinite(objective):
            raise RuntimeError("Nonfinite training loss")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient")
        optimizer.step()
        step += 1
        if (
            step % opts["log_every"] == 0
            or step % opts["checkpoint_every"] == 0
            or step == opts["steps"]
        ):
            _finite_parameters(model)
        if step % opts["log_every"] == 0 or step == opts["steps"]:
            model.eval()
            validation_rng = np.random.default_rng(800001)
            with torch.no_grad():
                losses = [
                    float(
                        _loss(
                            model,
                            loss,
                            _batch(
                                validation,
                                validation_rng,
                                opts["batch_size"],
                                opts["context"],
                                mean,
                                std,
                                return_scale,
                                device,
                            ),
                        )
                    )
                    for _ in range(8)
                ]
            if not np.isfinite(losses).all():
                raise RuntimeError("Nonfinite validation loss")
            val = float(np.mean(losses))
            history.append(
                {
                    "step": step,
                    "training_loss": float(objective.detach()),
                    "validation_loss": val,
                    "seconds": total_seconds + time.perf_counter() - started,
                }
            )
            print(
                f"train family={family} loss={loss} seed={seed} step={step} "
                f"validation_loss={val:.6f}",
                flush=True,
            )
            write_json(directory / "history.json", history)
            if val < best_loss:
                best_loss = val
                _save(
                    best,
                    metadata | {"model": model.state_dict(), "step": step, "validation_loss": val},
                )
        if step % opts["checkpoint_every"] == 0 or step == opts["steps"]:
            _save(
                checkpoint,
                metadata
                | {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_loss": best_loss,
                    "history": history,
                    "seconds": total_seconds + time.perf_counter() - started,
                    "torch_rng": torch.get_rng_state(),
                    "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
                    "numpy_rng": rng.bit_generator.state,
                },
            )
    if not best.exists() or not np.isfinite(best_loss):
        raise RuntimeError("Training did not produce a finite best checkpoint")
    return {
        "kind": "response",
        "family": family,
        "loss": loss,
        "seed": seed,
        "best_validation_loss": best_loss,
        "steps": step,
        "parameters": metadata["parameters"],
        "seconds": total_seconds + time.perf_counter() - started,
        "zero_action_excluded_from_loss": True,
    }


class RobustLearned(Learned):
    """Load either forecast family and reuse the unchanged response controller."""

    def __init__(self, path: Path, actions: np.ndarray, context: int, device: str):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("family") not in FAMILIES or saved.get("loss") not in LOSSES:
            raise RuntimeError("Not a registered robust response checkpoint")
        if saved.get("actions") != np.asarray(actions, dtype=np.float64).tolist():
            raise RuntimeError("Inference action grid differs from the training grid")
        if context != saved.get("context"):
            raise RuntimeError("Inference context differs from the training context")
        self.kind = "response"
        self.family, self.loss = saved["family"], saved["loss"]
        self.actions, self.context, self.device = np.asarray(actions), context, device
        self.mean, self.std = saved["mean"], saved["std"]
        if (
            np.shape(self.mean) != (STATE_DIMENSION,)
            or np.shape(self.std) != (STATE_DIMENSION,)
            or not np.isfinite(np.r_[self.mean, self.std]).all()
            or np.any(self.std <= 0)
        ):
            raise RuntimeError("Invalid response checkpoint normalization")
        self.target, self.scale = saved["target_return"], saved["return_scale"]
        self.model = _model(self.family, len(actions), saved["architecture"]).to(device)
        self.model.load_state_dict(saved["model"])
        _finite_parameters(self.model)
        self.model.eval()
        self.corrected, self.prior, self.hard_cpa = False, 3.0, False
        self.reset()
