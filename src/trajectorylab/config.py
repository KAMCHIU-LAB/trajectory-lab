"""Explicit configuration, with early validation instead of silent defaults."""

from __future__ import annotations

import tomllib
from pathlib import Path

TRAIN_DEFAULTS = {
    "steps": 5000,
    "batch_size": 128,
    "width": 64,
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "expectile": 0.7,
    "advantage_beta": 3.0,
    "target_tau": 0.005,
    "reward_scale": 0.01,
    "grad_clip": 10.0,
    "context": 20,
    "dt_layers": 2,
    "dt_heads": 4,
    "max_timestep": 8192,
    "return_scale": 1000.0,
    "target_return_quantile": 0.9,
    "device": "cpu",
    "threads": 1,
    "log_every": 100,
    "eval_every": 2500,
    "checkpoint_every": 2500,
}
ALGORITHMS = ("bc", "iql", "dt")


def load_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text())
    for section in ("suite", "data", "evaluation"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    unknown = set(config.get("training", {})) - set(TRAIN_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown training options: {sorted(unknown)}")
    config["training"] = TRAIN_DEFAULTS | config.get("training", {})
    suite, data, train, evaluation = (
        config[k] for k in ("suite", "data", "training", "evaluation")
    )
    if not suite["seeds"] or len(set(suite["seeds"])) != len(suite["seeds"]):
        raise ValueError("Training seeds must be nonempty and unique")
    if not suite["algorithms"] or set(suite["algorithms"]) - set(ALGORITHMS):
        raise ValueError(f"Algorithms must be selected from {ALGORITHMS}")
    if len(set(suite["algorithms"])) != len(suite["algorithms"]):
        raise ValueError("Repeated algorithms are not independent runs")
    if not data["views"] or len(set(data["views"])) != len(data["views"]):
        raise ValueError("Data views must be nonempty and unique")
    if data["kind"] not in ("pendulum", "minari"):
        raise ValueError("Data kind must be pendulum or minari")
    validation, test = evaluation["validation_seeds"], evaluation["test_seeds"]
    if not validation or not test or set(validation) & set(test):
        raise ValueError("Validation and test evaluation seeds must be nonempty and disjoint")
    if len(set(validation)) != len(validation) or len(set(test)) != len(test):
        raise ValueError("Evaluation seeds must be unique")
    for field in (
        "steps",
        "batch_size",
        "width",
        "context",
        "dt_layers",
        "dt_heads",
        "max_timestep",
        "threads",
        "log_every",
        "eval_every",
        "checkpoint_every",
    ):
        if not isinstance(train[field], int) or train[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if not 0 < train["expectile"] < 1 or not 0 <= train["gamma"] <= 1:
        raise ValueError("Invalid expectile or discount")
    if not 0 < train["target_tau"] <= 1 or train["advantage_beta"] < 0:
        raise ValueError("Invalid target update rate or advantage temperature")
    for field in ("reward_scale", "return_scale", "learning_rate", "grad_clip"):
        if train[field] <= 0:
            raise ValueError(f"{field} must be positive")
    if train["width"] % train["dt_heads"] or not 0 <= train["target_return_quantile"] <= 1:
        raise ValueError("Invalid Transformer width or return quantile")
    if train["device"] not in ("cpu", "mps", "cuda"):
        raise ValueError("Device must be explicit: cpu, mps, or cuda")
    return config
