"""Fixed-seed policy evaluation and descriptive data-support diagnostics."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch
from scipy.spatial import cKDTree

from trajectorylab.data import Normalizer, flatten_observation, pendulum_controller


def make_environment(metadata: dict):
    if metadata["source"] == "minari":
        import minari

        return minari.load_dataset(metadata["minari_id"]).recover_environment(eval_env=True)
    return gym.make(metadata["env_id"])


class Policy:
    def __init__(self, learner, normalizer: Normalizer, config: dict, target_return: float):
        self.learner, self.normalizer, self.config = learner, normalizer, config
        self.target_return = target_return
        self.reset()

    def reset(self) -> None:
        self.observations, self.actions, self.returns = [], [], []
        self.current_return = self.target_return

    @torch.no_grad()
    def act(self, observation: np.ndarray) -> np.ndarray:
        actor = self.learner.actor
        actor.eval()
        if self.learner.algorithm != "dt":
            state = torch.as_tensor(self.normalizer(observation), device=self.config["device"])
            return actor(state[None])[0].cpu().numpy()
        self.observations.append(self.normalizer(observation))
        action_dim = actor.action_head.out_features
        self.actions.append(np.zeros(action_dim, dtype=np.float32))
        self.returns.append(self.current_return / self.config["return_scale"])
        length = len(self.observations)
        start = max(0, length - self.config["context"])
        times = torch.arange(start, length, device=self.config["device"])[None]
        if length > self.config["max_timestep"]:
            raise ValueError("Evaluation horizon exceeds DT timestep vocabulary")
        prediction = (
            actor(
                torch.as_tensor(
                    np.asarray(self.observations[start:]), device=self.config["device"]
                )[None],
                torch.as_tensor(np.asarray(self.actions[start:]), device=self.config["device"])[
                    None
                ],
                torch.as_tensor(
                    [self.returns[start:]], dtype=torch.float32, device=self.config["device"]
                ),
                times,
                torch.ones_like(times, dtype=torch.bool),
            )[0, -1]
            .cpu()
            .numpy()
        )
        self.actions[-1] = prediction.copy()
        return prediction

    def observe_reward(self, reward: float) -> None:
        self.current_return -= reward


def evaluate(
    metadata: dict,
    seeds: list[int],
    policy: Policy | None = None,
    reference: str = "random",
) -> dict:
    env = make_environment(metadata)
    low, high = np.asarray(metadata["action_low"]), np.asarray(metadata["action_high"])
    episode_returns, lengths, successes, states, normalized_actions = [], [], [], [], []
    for seed in seeds:
        observation, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed)
        if policy:
            policy.reset()
        total, success, success_available, steps_taken = 0.0, False, False, 0
        for _ in range(metadata["max_episode_steps"]):
            steps_taken += 1
            flat = flatten_observation(observation)
            if policy:
                action = policy.act(flat)
            elif reference == "controller":
                if metadata["env_id"] != "Pendulum-v1":
                    raise ValueError("The reference controller is only defined for Pendulum-v1")
                action = pendulum_controller(flat) / 2.0
            else:
                action = rng.uniform(-1, 1, size=low.shape)
            if not np.isfinite(action).all():
                raise FloatingPointError("Policy produced non-finite actions")
            action = np.clip(action, -1, 1).astype(np.float32)
            native_action = (low + (action + 1.0) * (high - low) / 2.0).astype(np.float32)
            states.append(flat)
            normalized_actions.append(action)
            observation, reward, terminal, timeout, info = env.step(native_action)
            total += float(reward)
            if policy:
                policy.observe_reward(float(reward))
            for key in ("success", "is_success"):
                if key in info:
                    success_available = True
                    success = success or bool(info[key])
            if terminal or timeout:
                break
        episode_returns.append(total)
        lengths.append(steps_taken)
        if success_available:
            successes.append(success)
    env.close()
    return {
        "seeds": seeds,
        "returns": episode_returns,
        "episode_lengths": lengths,
        "return_mean": float(np.mean(episode_returns)),
        "return_std_across_episodes": float(np.std(episode_returns, ddof=1))
        if len(seeds) > 1
        else 0.0,
        "success_rate": float(np.mean(successes)) if successes else None,
        "action_saturation_fraction": float(np.mean(np.abs(normalized_actions) >= 0.99)),
        "states": np.asarray(states, dtype=np.float32),
    }


class SupportDiagnostic:
    """Nearest-neighbor distance, not a calibrated OOD detector or safety score."""

    def __init__(
        self,
        train_states: np.ndarray,
        validation_states: np.ndarray,
        normalizer: Normalizer,
        seed: int,
        max_reference: int = 5000,
    ):
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(train_states), min(max_reference, len(train_states)), replace=False)
        self.normalizer = normalizer
        self.tree = cKDTree(normalizer(train_states[ids]))
        self.reference_size = len(ids)
        distances = self.tree.query(normalizer(validation_states), workers=1)[0]
        self.threshold = max(float(np.quantile(distances, 0.95)), 1e-8)

    def summarize(self, states: np.ndarray) -> dict:
        distances = self.tree.query(self.normalizer(states), workers=1)[0]
        return {
            "reference_states": self.reference_size,
            "validation_distance_p95": self.threshold,
            "distance_mean": float(np.mean(distances)),
            "distance_p95": float(np.quantile(distances, 0.95)),
            "above_validation_p95_fraction": float(np.mean(distances > self.threshold)),
            "interpretation": "Representation-dependent descriptive nearest-neighbor distance; not an OOD guarantee",
        }
