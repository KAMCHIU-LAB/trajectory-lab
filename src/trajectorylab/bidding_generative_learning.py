"""Conditional joint response bundles over the frozen response encoder.

The forty generated coordinates are positive-action curve increments, jointly
sampled across cost, conversions, current response and remaining response. The
13 observation columns include cumulative and previous-tick feedback. Neither
scenario identifiers nor response labels belong in the sampler interface.
"""

from __future__ import annotations

import hashlib
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trajectorylab.bidding.learning import ResponseModel, _batch
from trajectorylab.utils import write_json

FAMILIES = ("flow", "gmm")
STATE_DIMENSION = 13
ENCODER_WIDTH = 128
CONDITION_DIMENSION = 141
ACTION_COUNT = 11
TARGET_DIMENSION = 40
SCALES = np.array([48.0, 48.0, 1.0, 1.0], dtype=np.float32)
TARGET_STD_FLOOR = 0.1
TRANSFORMED_CAP = 20.0
ARCHITECTURE = {
    "condition_dimension": CONDITION_DIMENSION,
    "target_dimension": TARGET_DIMENSION,
    "width": 256,
    "layers": 2,
    "time_features": 16,
    "mixture_components": 8,
    "time_frequency_base": 2.0,
    "time_frequency_multiplier": 2.0 * math.pi,
}


class FlowField(nn.Module):
    """Conditional flow field with sixteen fixed sine/cosine time features."""

    def __init__(self):
        super().__init__()
        self.register_buffer("frequencies", 2.0 * math.pi * 2.0 ** torch.arange(8))
        self.network = nn.Sequential(
            nn.Linear(CONDITION_DIMENSION + TARGET_DIMENSION + 16, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, TARGET_DIMENSION),
        )

    def forward(self, condition, latent, flow_time):
        phase = flow_time * self.frequencies
        time_features = torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
        return self.network(torch.cat((condition, latent, time_features), dim=-1))


class JointGMM(nn.Module):
    """Eight diagonal Gaussians with one shared component per complete bundle."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(CONDITION_DIMENSION, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 8 * (1 + 2 * TARGET_DIMENSION)),
        )

    def forward(self, condition):
        output = self.network(condition).reshape(-1, 8, 1 + 2 * TARGET_DIMENSION)
        return output[..., 0], output[..., 1:41], output[..., 41:].clamp(-5.0, 2.0)


def _head(family: str) -> nn.Module:
    if family == "flow":
        return FlowField()
    if family == "gmm":
        return JointGMM()
    raise ValueError("Expected generative family=flow|gmm")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actions(actions) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float64)
    if (
        actions.shape != (ACTION_COUNT,)
        or not np.isfinite(actions).all()
        or actions[0] != 0
        or np.any(np.diff(actions) <= 0)
    ):
        raise ValueError("The registered bundle model requires eleven increasing actions from zero")
    return actions


def _transformed(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    if (
        labels.shape != (48, 4, ACTION_COUNT)
        or not np.isfinite(labels).all()
        or np.any(labels < 0)
        or np.any(labels[..., 0] != 0)
    ):
        raise ValueError("Expected finite nonnegative 48x4x11 labels with a zero first action")
    increments = np.diff(labels, axis=-1)
    if np.any(increments < 0):
        raise ValueError("Response curve targets must be monotone over nominal actions")
    return np.log1p(increments * SCALES[None, :, None]).reshape(48, TARGET_DIMENSION)


def _validate_runs(data: list[dict]) -> None:
    if not data:
        raise ValueError("Training and validation trajectories must be nonempty")
    for run in data:
        for field, shape in (
            ("states", (48, STATE_DIMENSION)),
            ("actions", (48,)),
            ("returns", (48,)),
        ):
            if np.shape(run[field]) != shape or not np.isfinite(run[field]).all():
                raise ValueError("Invalid trajectory field: " + field)
        _transformed(run["labels"])


def _target_normalizer(train: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    # Accumulate in float64 without materializing all repeated behavior labels.
    total = np.zeros(TARGET_DIMENSION, dtype=np.float64)
    squared = np.zeros(TARGET_DIMENSION, dtype=np.float64)
    count = 0
    for run in train:
        values = _transformed(run["labels"])
        total += values.sum(axis=0)
        squared += np.square(values).sum(axis=0)
        count += len(values)
    center = total / count
    std = np.maximum(np.sqrt(np.maximum(squared / count - center * center, 0)), TARGET_STD_FLOOR)
    if not np.isfinite(np.r_[center, std]).all():
        raise RuntimeError("Nonfinite target normalization")
    return center.astype(np.float32), std.astype(np.float32)


def _encoder(metadata: dict, actions: np.ndarray, context: int, device: str) -> ResponseModel:
    if metadata.get("family") != "response" or metadata.get("kind") != "response":
        raise RuntimeError("A frozen augmented response checkpoint is required")
    if metadata.get("actions") != actions.tolist() or metadata.get("context") != context:
        raise RuntimeError("Frozen encoder action grid or context mismatch")
    if metadata["architecture"]["width"] != ENCODER_WIDTH:
        raise RuntimeError("The registered condition requires a 128-dimensional response encoder")
    mean, std = np.asarray(metadata["mean"]), np.asarray(metadata["std"])
    if (
        mean.shape != (STATE_DIMENSION,)
        or std.shape != (STATE_DIMENSION,)
        or not np.isfinite(np.r_[mean, std]).all()
        or np.any(std <= 0)
    ):
        raise RuntimeError("Invalid frozen observation normalization")
    model = ResponseModel(STATE_DIMENSION, ACTION_COUNT, **metadata["architecture"]).to(device)
    model.load_state_dict(metadata["model"])
    model.requires_grad_(False)
    model.eval()
    _finite_parameters(model)
    return model


@torch.no_grad()
def _condition(encoder: ResponseModel, states, steps, mask):
    hidden = encoder.embedding(states) + encoder.time(steps)
    for block in encoder.blocks:
        hidden = block(hidden, mask)
    return torch.cat((encoder.norm(hidden), states), dim=-1)


def _loss(
    head: nn.Module,
    family: str,
    encoder: ResponseModel,
    batch,
    center: torch.Tensor,
    std: torch.Tensor,
    rng: np.random.Generator,
) -> torch.Tensor:
    states, _actions_used, _returns, labels, steps, mask = batch
    condition = _condition(encoder, states, steps, mask)[mask]
    scales = labels.new_tensor(SCALES)[None, None, :, None]
    target = torch.log1p(torch.diff(labels, dim=-1) * scales).flatten(-2)[mask]
    target = (target - center) / std
    if family == "flow":
        normal = torch.as_tensor(
            rng.standard_normal(target.shape).astype(np.float32), device=states.device
        )
        flow_time = torch.as_tensor(
            rng.random((len(target), 1), dtype=np.float32), device=states.device
        )
        interpolated = (1 - flow_time) * normal + flow_time * target
        return F.mse_loss(head(condition, interpolated, flow_time), target - normal)
    if family == "gmm":
        logits, means, log_std = head(condition)
        residual = (target[:, None] - means) * torch.exp(-log_std)
        log_density = -0.5 * (residual.square() + 2 * log_std + math.log(2 * math.pi)).sum(-1)
        log_weights = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return -torch.logsumexp(log_weights + log_density, dim=-1).mean()
    raise ValueError("Expected generative family=flow|gmm")


def _finite_parameters(model: nn.Module) -> None:
    # Only called at initialization, log/checkpoint boundaries and final save.
    parameters = torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])
    if not torch.isfinite(parameters).all():
        raise RuntimeError("Nonfinite generative model parameters")


def _save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".partial.pt")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def fit(
    family: str,
    seed: int,
    config: dict,
    train: list[dict],
    validation: list[dict],
    encoder_path: Path,
    directory: Path,
    identity: str,
) -> dict:
    """Fit a joint generator; caller identity must bind data/metadata file SHA256s.

    The full frozen response model is embedded in every saved model, so inference
    does not depend on the source robust checkpoint remaining at its old path.
    """
    if family not in FAMILIES or not identity:
        raise ValueError("A registered family and data-hash-bound identity are required")
    opts = config["training"]
    device = opts["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("Requested MPS is unavailable; declare another device explicitly")
    if any(
        opts[key] <= 0
        for key in (
            "steps",
            "batch_size",
            "context",
            "learning_rate",
            "log_every",
            "checkpoint_every",
            "threads",
        )
    ):
        raise ValueError("Training budgets and intervals must be positive")
    if opts["context"] > 48:
        raise ValueError("Context exceeds the 48-tick horizon")
    actions = _actions(config["study"]["actions"])
    _validate_runs(train)
    _validate_runs(validation)
    target_center, target_std = _target_normalizer(train)
    encoder_path, directory = Path(encoder_path), Path(directory)
    encoder_sha = _sha256(encoder_path)
    frozen = torch.load(encoder_path, map_location="cpu", weights_only=False)
    if _sha256(encoder_path) != encoder_sha:
        raise RuntimeError("Frozen encoder changed while loading")
    torch.set_num_threads(opts["threads"])
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    rng = np.random.default_rng(seed)
    encoder = _encoder(frozen, actions, opts["context"], device)
    head = _head(family).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=opts["learning_rate"], weight_decay=1e-4)
    mean, observation_std = frozen["mean"], frozen["std"]
    return_scale = float(frozen["return_scale"])
    if not np.isfinite(return_scale) or return_scale <= 0:
        raise RuntimeError("Invalid frozen return scale")
    center_tensor = torch.as_tensor(target_center, device=device)
    std_tensor = torch.as_tensor(target_std, device=device)
    metadata = {
        "schema_version": 1,
        "identity": identity,
        "family": family,
        "seed": seed,
        "actions": actions.tolist(),
        "context": opts["context"],
        "device": device,
        "architecture": ARCHITECTURE,
        "encoder_sha256": encoder_sha,
        "encoder_identity": frozen["identity"],
        "encoder_architecture": frozen["architecture"],
        "encoder_mean": mean,
        "encoder_std": observation_std,
        "encoder_return_scale": return_scale,
        "encoder_target_return": frozen["target_return"],
        "target_center": target_center,
        "target_std": target_std,
        "target_std_floor": TARGET_STD_FLOOR,
        "target_scales": SCALES.tolist(),
        "transformed_cap": TRANSFORMED_CAP,
        "parameters": sum(parameter.numel() for parameter in head.parameters()),
        "frozen_parameters": sum(parameter.numel() for parameter in encoder.parameters()),
        "optimizer_options": {"learning_rate": opts["learning_rate"], "weight_decay": 1e-4},
        "batch_size": opts["batch_size"],
        "threads": opts["threads"],
        "current_state_contains_history_summary": True,
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
            "encoder_architecture",
            "optimizer_options",
            "batch_size",
            "threads",
        ):
            if saved.get(key) != metadata[key]:
                raise RuntimeError("Generative checkpoint metadata mismatch: " + key)
        for key in ("target_center", "target_std", "encoder_mean", "encoder_std"):
            if not np.array_equal(saved[key], metadata[key]):
                raise RuntimeError("Generative checkpoint normalization mismatch: " + key)
        if saved["step"] > opts["steps"]:
            raise RuntimeError("Requested step budget is below the existing checkpoint")
        head.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        rng.bit_generator.state = saved["numpy_rng"]
        step, best_loss, total_seconds = saved["step"], saved["best_loss"], saved["seconds"]
        history = saved["history"]
    elif best.exists() or (directory / "history.json").exists():
        raise RuntimeError("Generative output exists without a resumable checkpoint")
    _finite_parameters(head)
    started = time.perf_counter()

    def payload() -> dict:
        return metadata | {"model": head.state_dict(), "encoder_model": encoder.state_dict()}

    while step < opts["steps"]:
        head.train()
        batch = _batch(
            train,
            rng,
            opts["batch_size"],
            opts["context"],
            mean,
            observation_std,
            return_scale,
            device,
        )
        objective = _loss(head, family, encoder, batch, center_tensor, std_tensor, rng)
        if not torch.isfinite(objective):
            raise RuntimeError("Nonfinite generative training loss")
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite generative gradient")
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
            validation_rng = np.random.default_rng(800003)
            with torch.no_grad():
                losses = [
                    float(
                        _loss(
                            head,
                            family,
                            encoder,
                            _batch(
                                validation,
                                validation_rng,
                                opts["batch_size"],
                                opts["context"],
                                mean,
                                observation_std,
                                return_scale,
                                device,
                            ),
                            center_tensor,
                            std_tensor,
                            validation_rng,
                        )
                    )
                    for _ in range(8)
                ]
            if not np.isfinite(losses).all():
                raise RuntimeError("Nonfinite generative validation loss")
            value = float(np.mean(losses))
            history.append(
                {
                    "step": step,
                    "training_loss": float(objective.detach()),
                    "validation_loss": value,
                    "seconds": total_seconds + time.perf_counter() - started,
                }
            )
            print(
                f"train family={family} seed={seed} step={step} validation_loss={value:.6f}",
                flush=True,
            )
            write_json(directory / "history.json", history)
            if value < best_loss:
                best_loss = value
                _save(best, payload() | {"step": step, "validation_loss": value})
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
                    "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
                    "numpy_rng": rng.bit_generator.state,
                },
            )
    if not best.exists() or not np.isfinite(best_loss):
        raise RuntimeError("Generative fitting did not produce a finite best model")
    if _sha256(encoder_path) != encoder_sha:
        raise RuntimeError("Frozen encoder file changed during fitting")
    return {
        "family": family,
        "seed": seed,
        "steps": step,
        "best_validation_loss": best_loss,
        "parameters": metadata["parameters"],
        "frozen_parameters": metadata["frozen_parameters"],
        "encoder_sha256": encoder_sha,
        "seconds": total_seconds + time.perf_counter() - started,
    }


class BundleSampler:
    """Generate complete response bundles from caller-supplied common randomness."""

    def __init__(self, path: Path, actions: np.ndarray, context: int, device: str):
        path = Path(path)
        checkpoint_sha = _sha256(path)
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if _sha256(path) != checkpoint_sha:
            raise RuntimeError("Generator checkpoint changed while loading")
        actions = _actions(actions)
        if saved.get("schema_version") != 1 or saved.get("family") not in FAMILIES:
            raise RuntimeError("Not a registered conditional response generator")
        if saved.get("architecture") != ARCHITECTURE:
            raise RuntimeError("Generative architecture differs from the registered model")
        if saved.get("actions") != actions.tolist() or saved.get("context") != context:
            raise RuntimeError("Inference action grid or context differs from fitted data")
        if (
            saved.get("target_scales") != SCALES.tolist()
            or saved.get("transformed_cap") != TRANSFORMED_CAP
        ):
            raise RuntimeError("Generative target decoding settings differ")
        self.family, self.seed = saved["family"], saved["seed"]
        self.actions, self.context, self.device = actions, context, device
        self.identity, self.training_identity = checkpoint_sha, saved["identity"]
        self.encoder_sha256 = saved["encoder_sha256"]
        self.mean, self.std = saved["encoder_mean"], saved["encoder_std"]
        self.target_center = np.asarray(saved["target_center"], dtype=np.float32)
        self.target_std = np.asarray(saved["target_std"], dtype=np.float32)
        if (
            self.target_center.shape != (TARGET_DIMENSION,)
            or self.target_std.shape != (TARGET_DIMENSION,)
            or not np.isfinite(np.r_[self.target_center, self.target_std]).all()
            or np.any(self.target_std < TARGET_STD_FLOOR)
        ):
            raise RuntimeError("Invalid generative target normalization")
        frozen = {
            "family": "response",
            "kind": "response",
            "actions": saved["actions"],
            "context": context,
            "architecture": saved["encoder_architecture"],
            "mean": self.mean,
            "std": self.std,
            "model": saved["encoder_model"],
        }
        self.encoder = _encoder(frozen, actions, context, device)
        self.head = _head(self.family).to(device)
        self.head.load_state_dict(saved["model"])
        self.head.requires_grad_(False)
        self.head.eval()
        _finite_parameters(self.head)
        self.last_diagnostics = {}
        self._coordinates = self._upper_hits = self._lower_hits = self._bundles = 0

    def _states(self, states):
        states = np.asarray(states, dtype=np.float32)
        if (
            states.ndim != 2
            or states.shape[1] != STATE_DIMENSION
            or not 1 <= len(states) <= 48
            or not np.isfinite(states).all()
        ):
            raise ValueError("Expected one to 48 causal observations with thirteen finite columns")
        end = len(states)
        start = max(0, end - self.context)
        normalized = (states[start:] - self.mean) / self.std
        x = torch.as_tensor(normalized, dtype=torch.float32, device=self.device)[None]
        steps = torch.arange(start, end, device=self.device)[None]
        mask = torch.ones((1, end - start), dtype=torch.bool, device=self.device)
        return x, steps, mask

    @torch.no_grad()
    def point(self, states) -> np.ndarray:
        """The same frozen deterministic response used by the projection control."""
        x, steps, mask = self._states(states)
        curves = self.encoder(x, steps, mask)[0, -1].cpu().numpy()
        if not np.isfinite(curves).all():
            raise RuntimeError("Nonfinite frozen point response")
        return curves

    @torch.no_grad()
    def curves(
        self, states, normal: np.ndarray, uniforms: np.ndarray, integration_steps: int = 8
    ) -> np.ndarray:
        """Return (samples, 4, 11) monotone curves in the original label units."""
        normal = np.asarray(normal, dtype=np.float32)
        uniforms = np.asarray(uniforms, dtype=np.float64)
        if (
            normal.ndim != 2
            or normal.shape[1] != TARGET_DIMENSION
            or not len(normal)
            or uniforms.shape != (len(normal),)
            or not np.isfinite(normal).all()
            or not np.isfinite(uniforms).all()
            or np.any(uniforms < 0)
            or np.any(uniforms >= 1)
        ):
            raise ValueError("Provide finite normal[S,40] and categorical uniforms[S] in [0,1)")
        if not isinstance(integration_steps, (int, np.integer)) or integration_steps <= 0:
            raise ValueError("Euler integration requires a positive integer step count")
        x, steps, mask = self._states(states)
        condition = _condition(self.encoder, x, steps, mask)[:, -1]
        noise = torch.as_tensor(normal, device=self.device)
        if self.family == "flow":
            latent = noise.clone()
            repeated_condition = condition.expand(len(normal), -1)
            for step in range(integration_steps):
                flow_time = latent.new_full((len(normal), 1), step / integration_steps)
                latent = (
                    latent + self.head(repeated_condition, latent, flow_time) / integration_steps
                )
        else:
            logits, means, log_std = self.head(condition)
            cumulative = torch.softmax(logits[0], dim=-1).cumsum(dim=-1)
            uniform = torch.as_tensor(uniforms, dtype=torch.float32, device=self.device)
            component = (uniform[:, None] >= cumulative[None]).sum(dim=-1).clamp(max=7)
            latent = means[0, component] + noise * torch.exp(log_std[0, component])
        transformed = latent.cpu().numpy() * self.target_std + self.target_center
        if not np.isfinite(transformed).all():
            raise RuntimeError("Nonfinite generated transformed increments")
        upper = int(np.count_nonzero(transformed >= TRANSFORMED_CAP))
        lower = int(np.count_nonzero(transformed < 0))
        self._coordinates += transformed.size
        self._upper_hits += upper
        self._lower_hits += lower
        self._bundles += len(normal)
        self.last_diagnostics = {
            "samples": len(normal),
            "transformed_coordinates": transformed.size,
            "transformed_upper_cap_hits": upper,
            "transformed_lower_clamp_hits": lower,
            "transformed_upper_cap_fraction": upper / transformed.size,
            "transformed_lower_clamp_fraction": lower / transformed.size,
        }
        increments = np.expm1(np.clip(transformed, 0, TRANSFORMED_CAP)).reshape(-1, 4, 10)
        increments = increments / SCALES[None, :, None]
        curves = np.pad(increments.cumsum(axis=-1), ((0, 0), (0, 0), (1, 0)))
        if not np.isfinite(curves).all():
            raise RuntimeError("Nonfinite generated response curves")
        return curves

    def diagnostics(self) -> dict:
        """Cumulative numerical clipping counts; the cap is not calibration."""
        return {
            "generated_bundles": self._bundles,
            "transformed_coordinates": self._coordinates,
            "transformed_upper_cap": TRANSFORMED_CAP,
            "transformed_upper_cap_hits": self._upper_hits,
            "transformed_lower_clamp_hits": self._lower_hits,
            "transformed_upper_cap_fraction": self._upper_hits / max(1, self._coordinates),
            "transformed_lower_clamp_fraction": self._lower_hits / max(1, self._coordinates),
        }
