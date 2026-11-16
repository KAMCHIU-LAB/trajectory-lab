"""Episode-preserving datasets, public-data adapters, and controlled data views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from trajectorylab.utils import array_digest, fingerprint, read_json, write_json

SCHEMA_VERSION = 1
VIEWS = ("uniform", "top_return", "return_stratified")


@dataclass
class Episode:
    id: str
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray

    def __len__(self) -> int:
        return len(self.actions)

    @property
    def total_return(self) -> float:
        return float(self.rewards.sum(dtype=np.float64))

    @property
    def digest(self) -> str:
        return array_digest(
            [self.observations, self.actions, self.rewards, self.terminated, self.truncated]
        )

    def validate(self) -> None:
        n = len(self)
        if n == 0 or self.observations.ndim != 2 or self.actions.ndim != 2:
            raise ValueError(f"{self.id}: empty or non-vector episode")
        if len(self.observations) != n + 1:
            raise ValueError(f"{self.id}: observations must contain T+1 states")
        for x in (self.rewards, self.terminated, self.truncated):
            if x.shape != (n,):
                raise ValueError(f"{self.id}: transition fields must have shape (T,)")
        for x in (self.observations, self.actions, self.rewards):
            if not np.isfinite(x).all():
                raise ValueError(f"{self.id}: non-finite data")
        if np.abs(self.actions).max() > 1.0001:
            raise ValueError(f"{self.id}: actions must be normalized to [-1, 1]")
        if self.terminated.dtype != np.bool_ or self.truncated.dtype != np.bool_:
            raise ValueError(f"{self.id}: termination fields must be boolean")
        if self.terminated[:-1].any() or self.truncated[:-1].any():
            raise ValueError(f"{self.id}: episode contains an internal boundary")


@dataclass
class Dataset:
    episodes: list[Episode]
    metadata: dict

    def validate(self) -> None:
        if len(self.episodes) < 3:
            raise ValueError("At least three episodes are required")
        if len({e.id for e in self.episodes}) != len(self.episodes):
            raise ValueError("Episode IDs must be unique")
        shapes = set()
        for e in self.episodes:
            e.validate()
            shapes.add((e.observations.shape[1], e.actions.shape[1]))
        if len(shapes) != 1:
            raise ValueError("All episodes must share observation and action dimensions")

    @property
    def digest(self) -> str:
        return fingerprint(
            {"metadata": self.metadata, "episodes": [(e.id, e.digest) for e in self.episodes]}
        )

    def save(self, directory: Path) -> None:
        self.validate()
        directory.mkdir(parents=True, exist_ok=True)
        offsets = np.cumsum([0] + [len(e) for e in self.episodes])
        # A separate final state per episode preserves exact boundaries without pickle.
        arrays = {
            "offsets": offsets,
            "observations": np.concatenate([e.observations[:-1] for e in self.episodes]),
            "final_observations": np.stack([e.observations[-1] for e in self.episodes]),
            "actions": np.concatenate([e.actions for e in self.episodes]),
            "rewards": np.concatenate([e.rewards for e in self.episodes]),
            "terminated": np.concatenate([e.terminated for e in self.episodes]),
            "truncated": np.concatenate([e.truncated for e in self.episodes]),
        }
        temporary = directory / "episodes.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(directory / "episodes.npz")
        write_json(
            directory / "dataset.json",
            {
                "schema_version": SCHEMA_VERSION,
                "metadata": self.metadata,
                "episode_ids": [e.id for e in self.episodes],
                "digest": self.digest,
                "transitions": int(offsets[-1]),
            },
        )

    @classmethod
    def load(cls, directory: Path) -> Dataset:
        manifest = read_json(directory / "dataset.json")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported dataset schema")
        episodes = []
        with np.load(directory / "episodes.npz", allow_pickle=False) as arrays:
            for i, eid in enumerate(manifest["episode_ids"]):
                lo, hi = arrays["offsets"][i : i + 2]
                observations = np.concatenate(
                    [arrays["observations"][lo:hi], arrays["final_observations"][i : i + 1]]
                )
                episodes.append(
                    Episode(
                        eid,
                        observations,
                        *[
                            arrays[k][lo:hi].copy()
                            for k in ("actions", "rewards", "terminated", "truncated")
                        ],
                    )
                )
        result = cls(episodes, manifest["metadata"])
        result.validate()
        if result.digest != manifest["digest"]:
            raise ValueError("Dataset fingerprint mismatch: corrupted or changed data")
        return result


def flatten_observation(observation: np.ndarray | dict) -> np.ndarray:
    """Stable key ordering, identical at import and evaluation time."""
    if isinstance(observation, dict):
        return np.concatenate([flatten_observation(observation[k]) for k in sorted(observation)])
    return np.asarray(observation, dtype=np.float32).reshape(-1)


def pendulum_controller(observation: np.ndarray) -> np.ndarray:
    """Public classical-control energy shaping, followed by local stabilization.

    Uses only the standard Pendulum observation. This is a data generator, not a
    learned expert or a claim of an optimal controller.
    """
    cosine, sine, omega = observation
    theta = np.arctan2(sine, cosine)
    if abs(theta) < 0.35 and abs(omega) < 2.0:
        torque = -12.0 * theta - 4.0 * omega
    else:
        energy = omega**2 / 6.0 + 5.0 * (cosine - 1.0)
        torque = -1.5 * energy * omega
        if abs(omega) < 0.05:
            torque = 2.0
    return np.asarray([np.clip(torque, -2, 2)], dtype=np.float32)


def generate_pendulum(count: int, seed: int) -> Dataset:
    if count < 12:
        raise ValueError("Use at least 12 source episodes")
    env = gym.make("Pendulum-v1")
    rng = np.random.default_rng(seed)
    episodes = []
    for i in range(count):
        obs, _ = env.reset(seed=seed * 1000 + i)
        mode = int(rng.choice(3, p=[0.35, 0.4, 0.25]))
        observations, actions, rewards, terminals, timeouts = [obs.copy()], [], [], [], []
        while True:
            if mode == 2:
                action = rng.uniform(-2, 2, 1).astype(np.float32)
            else:
                noise = 0.05 if mode == 0 else 0.9
                action = np.clip(pendulum_controller(obs) + rng.normal(0, noise, 1), -2, 2)
            obs, reward, terminal, timeout, _ = env.step(action)
            observations.append(obs.copy())
            actions.append(np.asarray(action / 2.0, dtype=np.float32))
            rewards.append(reward)
            terminals.append(terminal)
            timeouts.append(timeout)
            if terminal or timeout:
                break
        episodes.append(
            Episode(
                f"pendulum-{seed}-{i:05d}",
                np.asarray(observations, dtype=np.float32),
                np.asarray(actions, dtype=np.float32),
                np.asarray(rewards, dtype=np.float32),
                np.asarray(terminals, dtype=bool),
                np.asarray(timeouts, dtype=bool),
            )
        )
    env.close()
    return Dataset(
        episodes,
        {
            "source": "generated",
            "env_id": "Pendulum-v1",
            "source_seed": seed,
            "generator": "energy-shaping-mixture-v1",
            "source_episodes": count,
            "source_url": "https://gymnasium.farama.org/environments/classic_control/pendulum/",
            "action_low": [-2.0],
            "action_high": [2.0],
            "max_episode_steps": 200,
            "benchmark_status": "controlled study; not an official D4RL benchmark",
        },
    )


def import_minari(dataset_id: str, count: int, seed: int) -> Dataset:
    import minari

    dataset = minari.load_dataset(dataset_id, download=True)
    env = dataset.recover_environment(eval_env=True)
    low, high = env.action_space.low, env.action_space.high
    if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(high <= low):
        raise ValueError("Only finite continuous action spaces are supported")
    rng = np.random.default_rng(seed)
    indices = np.asarray(dataset.episode_indices)
    selected = np.sort(rng.choice(indices, min(count, len(indices)), replace=False))
    episodes = []
    for e in dataset.iterate_episodes(selected):
        observations = e.observations
        if isinstance(observations, dict):
            observations = np.concatenate(
                [
                    np.asarray(observations[k]).reshape(len(e.actions) + 1, -1)
                    for k in sorted(observations)
                ],
                axis=1,
            )
        actions = 2.0 * (np.asarray(e.actions) - low) / (high - low) - 1.0
        episodes.append(
            Episode(
                f"minari-{e.id}",
                np.asarray(observations, dtype=np.float32),
                np.asarray(actions, dtype=np.float32),
                np.asarray(e.rewards, dtype=np.float32),
                np.asarray(e.terminations, dtype=bool),
                np.asarray(e.truncations, dtype=bool),
            )
        )
    metadata = {
        "source": "minari",
        "minari_id": dataset_id,
        "source_seed": seed,
        "source_url": "https://minari.farama.org/",
        "available_episodes": len(indices),
        "selected_source_episodes": len(episodes),
        "selection": "seeded uniform episode sample",
        "action_low": low.tolist(),
        "action_high": high.tolist(),
        "env_id": env.spec.id,
        "max_episode_steps": env.spec.max_episode_steps,
        "license": "Upstream dataset terms apply; source data is not redistributed",
        "benchmark_status": "public benchmark subset; not a paper-score reproduction",
    }
    env.close()
    result = Dataset(episodes, metadata)
    result.validate()
    return result


def split_episodes(episodes: list[Episode], seed: int) -> dict[str, list[Episode]]:
    # Identical content is grouped, even when it has different source IDs.
    groups: dict[str, list[Episode]] = {}
    for episode in episodes:
        groups.setdefault(episode.digest, []).append(episode)
    if len(groups) < 10:
        raise ValueError("At least ten distinct trajectories are needed for a held-out split")
    keys = np.asarray(sorted(groups))
    np.random.default_rng(seed).shuffle(keys)
    n = len(keys)
    boundaries = (int(n * 0.8), int(n * 0.9))
    pieces = np.split(keys, boundaries)
    return {
        name: [e for key in piece for e in groups[key]]
        for name, piece in zip(("train", "validation", "heldout"), pieces, strict=True)
    }


def select_view(
    episodes: list[Episode],
    name: str,
    transition_budget: int,
    seed: int,
    tolerance: float = 0.05,
) -> list[Episode]:
    if name not in VIEWS:
        raise ValueError(f"Unknown view {name!r}; expected one of {VIEWS}")
    if transition_budget < 1 or not 0 <= tolerance < 1:
        raise ValueError("Invalid transition budget or tolerance")
    rng = np.random.default_rng(seed)
    if name == "uniform":
        order = rng.permutation(len(episodes)).tolist()
    elif name == "top_return":
        order = sorted(
            range(len(episodes)), key=lambda i: (-episodes[i].total_return, episodes[i].id)
        )
    else:
        ranked = sorted(range(len(episodes)), key=lambda i: episodes[i].total_return)
        bins = [rng.permutation(b).tolist() for b in np.array_split(ranked, 4)]
        order = []
        while any(bins):
            for b in bins:
                if b:
                    order.append(b.pop())
    result, used = [], 0
    for i in order:
        episode = episodes[i]
        if used + len(episode) <= transition_budget:
            result.append(episode)
            used += len(episode)
    if used < transition_budget * (1 - tolerance):
        raise ValueError(
            f"{name}: only {used}/{transition_budget} transitions fit complete episodes; "
            "increase source size or adjust the budget explicitly"
        )
    if not result:
        raise ValueError("No complete episode fits the budget")
    return result


def transitions(episodes: list[Episode]) -> dict[str, np.ndarray]:
    return {
        "observations": np.concatenate([e.observations[:-1] for e in episodes]),
        "next_observations": np.concatenate([e.observations[1:] for e in episodes]),
        "actions": np.concatenate([e.actions for e in episodes]),
        "rewards": np.concatenate([e.rewards for e in episodes]),
        "terminated": np.concatenate([e.terminated for e in episodes]),
        "truncated": np.concatenate([e.truncated for e in episodes]),
    }


@dataclass
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, observations: np.ndarray) -> Normalizer:
        return cls(observations.mean(0), np.maximum(observations.std(0), 1e-3))

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        return np.asarray((observations - self.mean) / self.std, dtype=np.float32)

    def state(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> Normalizer:
        return cls(
            np.asarray(state["mean"], dtype=np.float32), np.asarray(state["std"], dtype=np.float32)
        )


def returns_to_go(rewards: np.ndarray) -> np.ndarray:
    return np.cumsum(rewards[::-1], dtype=np.float64)[::-1].astype(np.float32)
