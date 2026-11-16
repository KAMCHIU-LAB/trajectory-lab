"""BC, Implicit Q-Learning, and a compact Decision Transformer baseline.

IQL: Kostrikov et al., https://arxiv.org/abs/2110.06169
DT: Chen et al., https://arxiv.org/abs/2106.01345
Implementation is original to this repository; methods are not claimed as novel.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trajectorylab.data import Episode, Normalizer, returns_to_go, transitions
from trajectorylab.models import DecisionTransformer, GaussianPolicy, TwinQ, mlp


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return (weight * diff.square()).mean()


def bellman_target(
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    next_value: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Time-limit truncation does not remove the next-state bootstrap."""
    return rewards + gamma * (1.0 - terminated.float()) * next_value


class Replay:
    def __init__(self, episodes: list[Episode], normalizer: Normalizer, device: str):
        arrays = transitions(episodes)
        for k in ("observations", "next_observations"):
            arrays[k] = normalizer(arrays[k])
        self.tensors = {
            k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in arrays.items()
        }
        self.size = len(arrays["actions"])
        self.device = device

    def sample(self, rng: np.random.Generator, count: int) -> dict:
        ids = torch.as_tensor(rng.integers(self.size, size=count), device=self.device)
        return {k: v[ids] for k, v in self.tensors.items()}


class Sequences:
    def __init__(
        self,
        episodes: list[Episode],
        normalizer: Normalizer,
        context: int,
        return_scale: float,
        device: str,
    ):
        self.offsets = np.cumsum([0] + [len(e) for e in episodes])
        self.size = int(self.offsets[-1])
        self.context, self.device = context, device
        self.arrays = {
            "observations": normalizer(np.concatenate([e.observations[:-1] for e in episodes])),
            "actions": np.concatenate([e.actions for e in episodes]),
            "returns": np.concatenate([returns_to_go(e.rewards) / return_scale for e in episodes]),
            "timesteps": np.concatenate([np.arange(len(e)) for e in episodes]),
        }

    def sample(self, rng: np.random.Generator, count: int) -> dict:
        starts = rng.integers(self.size, size=count)
        episodes = np.searchsorted(self.offsets[1:], starts, side="right")
        ends = self.offsets[episodes + 1]
        indices = starts[:, None] + np.arange(self.context)[None, :]
        mask = indices < ends[:, None]
        safe = np.minimum(indices, ends[:, None] - 1)
        result = {}
        for k, values in self.arrays.items():
            padded = values[safe].copy()
            padded[~mask] = 0
            dtype = torch.long if k == "timesteps" else torch.float32
            result[k] = torch.as_tensor(padded, dtype=dtype, device=self.device)
        result["mask"] = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        return result


class Learner:
    def __init__(self, algorithm: str, observation_dim: int, action_dim: int, config: dict):
        if algorithm not in ("bc", "iql", "dt"):
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.algorithm, self.config = algorithm, config
        self.device = config["device"]
        width, lr = config["width"], config["learning_rate"]
        self.modules: dict[str, nn.Module] = {}
        if algorithm == "dt":
            self.modules["actor"] = DecisionTransformer(
                observation_dim,
                action_dim,
                width,
                config["dt_layers"],
                config["dt_heads"],
                config["max_timestep"],
            ).to(self.device)
        else:
            self.modules["actor"] = GaussianPolicy(observation_dim, action_dim, width).to(
                self.device
            )
        if algorithm == "iql":
            self.modules["q"] = TwinQ(observation_dim, action_dim, width).to(self.device)
            self.modules["target_q"] = copy.deepcopy(self.modules["q"]).requires_grad_(False)
            self.modules["value"] = mlp(observation_dim, 1, width).to(self.device)
        self.optimizers = {
            k: torch.optim.Adam(module.parameters(), lr=lr)
            for k, module in self.modules.items()
            if k != "target_q"
        }

    @property
    def actor(self) -> nn.Module:
        return self.modules["actor"]

    def _optimize(self, name: str, loss: torch.Tensor) -> None:
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {name} loss")
        optimizer = self.optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.modules[name].parameters(), self.config["grad_clip"], error_if_nonfinite=True
        )
        optimizer.step()

    def update(self, batch: dict) -> dict[str, float]:
        self.actor.train()
        if self.algorithm == "dt":
            predictions = self.actor(
                batch["observations"],
                batch["actions"],
                batch["returns"],
                batch["timesteps"],
                batch["mask"],
            )
            loss = (predictions - batch["actions"]).square().mean(-1)
            loss = loss[batch["mask"]].mean()
            self._optimize("actor", loss)
            return {"actor_loss": float(loss.detach())}
        obs, actions = batch["observations"], batch["actions"]
        if self.algorithm == "bc":
            loss = -self.actor.log_prob(obs, actions).mean()
            self._optimize("actor", loss)
            return {"actor_loss": float(loss.detach())}
        q, target_q, value = (self.modules[k] for k in ("q", "target_q", "value"))
        with torch.no_grad():
            reference_q = torch.minimum(*target_q(obs, actions))
        predicted_v = value(obs).squeeze(-1)
        value_loss = expectile_loss(reference_q - predicted_v, self.config["expectile"])
        self._optimize("value", value_loss)
        with torch.no_grad():
            target = bellman_target(
                batch["rewards"] * self.config["reward_scale"],
                batch["terminated"],
                value(batch["next_observations"]).squeeze(-1),
                self.config["gamma"],
            )
        q1, q2 = q(obs, actions)
        q_loss = 0.5 * (F.mse_loss(q1, target) + F.mse_loss(q2, target))
        self._optimize("q", q_loss)
        with torch.no_grad():
            advantage = reference_q - value(obs).squeeze(-1)
            log_weight = (self.config["advantage_beta"] * advantage).clamp(max=np.log(100.0))
            weight = log_weight.exp()
        actor_loss = -(weight * self.actor.log_prob(obs, actions)).mean()
        self._optimize("actor", actor_loss)
        with torch.no_grad():
            for p, target_p in zip(q.parameters(), target_q.parameters(), strict=True):
                target_p.lerp_(p, self.config["target_tau"])
        ess_fraction = weight.sum().square() / (
            len(weight) * weight.square().sum().clamp_min(1e-12)
        )
        return {
            "actor_loss": float(actor_loss.detach()),
            "q_loss": float(q_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "advantage_mean": float(advantage.mean()),
            "weight_ess_fraction": float(ess_fraction),
            "weight_clip_fraction": float((weight >= 99.99).float().mean()),
        }

    def state(self) -> dict:
        return {
            "modules": {k: m.state_dict() for k, m in self.modules.items()},
            "optimizers": {k: o.state_dict() for k, o in self.optimizers.items()},
        }

    def load_state(self, state: dict) -> None:
        for k, values in state["modules"].items():
            self.modules[k].load_state_dict(values)
        for k, values in state["optimizers"].items():
            self.optimizers[k].load_state_dict(values)
