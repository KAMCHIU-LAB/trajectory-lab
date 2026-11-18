"""Offline stress augmentation with nominal-action response targets.

Scenario names and the response tape are used only to generate data. Policies
receive the existing 13 observation columns, which include historical feedback.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from trajectorylab.bidding.learning import Feedback, rollout
from trajectorylab.bidding.learning import trajectories as nominal_trajectories
from trajectorylab.bidding.market import Tape, scenario_bias

SCENARIOS = ("calibrated", "over_25", "over_50", "under_25", "mid_over_50")
TICKS = 48
BEHAVIORS_PER_FOCAL = 10


def _actions(actions: np.ndarray) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.isfinite(values).all()
        or values[0] != 0
        or np.any(np.diff(values) <= 0)
    ):
        raise ValueError("Actions must be finite, strictly increasing, and start at zero")
    return values


def _labels(tape: Tape, fi: int, actions: np.ndarray, scenario: str) -> np.ndarray:
    # The output axis stays nominal; each future tick uses its own bias.
    indices = np.array(
        [
            [tape.index(action * scenario_bias(scenario, tick)) for action in actions]
            for tick in range(TICKS)
        ],
        dtype=np.int64,
    )
    ticks = np.arange(TICKS)[:, None]
    raw_cost = tape.arrays["costs"][fi][ticks, indices]
    raw_conv = tape.arrays["conversions"][fi][ticks, indices]
    budget, cpa = tape.arrays["budgets"][fi], tape.arrays["cpas"][fi]
    if not np.isfinite([budget, cpa]).all() or budget <= 0 or cpa <= 0:
        raise ValueError("Response normalization requires positive finite budget and CPA")
    labels = np.stack(
        [
            raw_cost / budget,
            raw_conv * cpa / budget,
            raw_cost[::-1].cumsum(axis=0)[::-1] / budget,
            raw_conv[::-1].cumsum(axis=0)[::-1] * cpa / budget,
        ],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise RuntimeError("Response labels must be finite and nonnegative")
    if np.any(labels[:, :, 0] != 0) or np.any(np.diff(labels, axis=-1) < 0):
        raise RuntimeError("Response labels must be monotone and zero at action zero")
    # All ten behavior runs share one immutable label array for this pair.
    labels.setflags(write=False)
    return labels


def _runs(tape: Tape, path: Path, actions: np.ndarray, scenarios: list[str]) -> list[dict]:
    result = []
    for scenario in scenarios:
        for fi, focal in enumerate(tape.arrays["focals"]):
            labels = _labels(tape, fi, actions, scenario)
            policies = [Feedback(actions, gain, 3.0) for gain in (0.3, 0.7, 1.2)]
            for fixed in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
                policies.append(
                    lambda s, u, r, b, c, k=fixed: (
                        float(actions[np.abs(actions - k).argmin()]),
                        {},
                    )
                )
            for behavior, policy in enumerate(policies):
                run = rollout(tape, fi, actions, scenario, policy, retain=True)
                rewards = run.pop("rewards")
                run["returns"] = rewards[::-1].cumsum()[::-1].copy()
                run["labels"] = labels
                run["data_diagnostics"] = {
                    "market": path.stem,
                    "focal": int(focal),
                    "scenario": scenario,
                    "behavior": behavior,
                }
                result.append(run)
    return result


def trajectories(paths: list[Path], actions: np.ndarray, scenarios: list[str]) -> list[dict]:
    """Generate the frozen ten behavior policies in each requested scenario.

    Targets have shape (48, 4, actions), normalized exactly like the old data.
    ``data_diagnostics`` is bookkeeping and never belongs in model features.
    """
    actions = _actions(actions)
    if not paths or not scenarios or len(set(scenarios)) != len(scenarios):
        raise ValueError("Provide markets and nonempty, unique scenarios")
    for scenario in scenarios:
        scenario_bias(scenario, 0)
    result = []
    for path in paths:
        path = Path(path)
        result.extend(_runs(Tape(path), path, actions, scenarios))
    return result


def audit(path: Path, actions: np.ndarray) -> dict:
    """Independently check every focal/tick/action in one market and all scenes.

    Raises on a failed invariant; returns only JSON-compatible diagnostics.
    Remaining labels are checked by explicit per-future-tick tape lookups.
    """
    path, actions = Path(path), _actions(actions)
    tape = Tape(path)
    runs = _runs(tape, path, actions, list(SCENARIOS))
    focal_count = len(tape.arrays["focals"])
    expected_runs = focal_count * len(SCENARIOS) * BEHAVIORS_PER_FOCAL
    if len(runs) != expected_runs:
        raise RuntimeError("Augmented behavior trajectory coverage mismatch")
    errors = np.zeros(4, dtype=np.float64)
    checked = 0
    for si, scenario in enumerate(SCENARIOS):
        for fi in range(focal_count):
            first = (si * focal_count + fi) * BEHAVIORS_PER_FOCAL
            group = runs[first : first + BEHAVIORS_PER_FOCAL]
            labels = group[0]["labels"]
            if not all(run["labels"] is labels for run in group):
                raise RuntimeError("Behavior runs did not share one response target array")
            if labels.dtype != np.float32 or labels.shape != (TICKS, 4, len(actions)):
                raise RuntimeError("Response target dtype/shape mismatch")
            for run in group:
                if run["states"].shape != (TICKS, 13):
                    raise RuntimeError("Policy observation schema changed")
                for field in ("states", "actions", "returns", "labels"):
                    if not np.isfinite(run[field]).all():
                        raise RuntimeError("Nonfinite trajectory field: " + field)
            budget, cpa = tape.arrays["budgets"][fi], tape.arrays["cpas"][fi]
            for tick in range(TICKS):
                for ai, action in enumerate(actions):
                    index = tape.index(action * scenario_bias(scenario, tick))
                    future_cost = sum(
                        tape.arrays["costs"][
                            fi, future, tape.index(action * scenario_bias(scenario, future))
                        ]
                        for future in range(tick, TICKS)
                    )
                    future_conv = sum(
                        tape.arrays["conversions"][
                            fi, future, tape.index(action * scenario_bias(scenario, future))
                        ]
                        for future in range(tick, TICKS)
                    )
                    expected = np.array(
                        [
                            tape.arrays["costs"][fi, tick, index] / budget,
                            tape.arrays["conversions"][fi, tick, index] * cpa / budget,
                            future_cost / budget,
                            future_conv * cpa / budget,
                        ]
                    )
                    actual = labels[tick, :, ai]
                    errors = np.maximum(errors, np.abs(expected - actual))
                    if not np.allclose(actual, expected, rtol=2e-6, atol=2e-7):
                        raise RuntimeError("Nominal-action target differs from exact tick lookup")
                    checked += 1
            if not np.allclose(
                labels[:-1, 2:], labels[:-1, :2] + labels[1:, 2:], rtol=2e-6, atol=2e-7
            ):
                raise RuntimeError("Remaining targets do not equal the sum of future targets")
            if not np.array_equal(labels[-1, :2], labels[-1, 2:]):
                raise RuntimeError("Final remaining targets differ from current targets")
    old = nominal_trajectories([path], actions)
    calibrated = runs[: focal_count * BEHAVIORS_PER_FOCAL]
    for original, augmented in zip(old, calibrated, strict=True):
        for field, value in original.items():
            other = augmented[field]
            if isinstance(value, np.ndarray):
                equal = np.array_equal(value, other)
            else:
                equal = value == other
            if not equal:
                raise RuntimeError("Calibrated data differ from frozen trajectories: " + field)
    return {
        "passed": True,
        "market": path.stem,
        "scenarios": list(SCENARIOS),
        "focals": focal_count,
        "ticks": TICKS,
        "actions": len(actions),
        "behavior_trajectories": len(runs),
        "current_and_remaining_action_cells_checked": checked,
        "max_absolute_normalized_error_by_head": errors.tolist(),
        "calibrated_frozen_trajectory_fields_exact": True,
        "per_future_tick_bias_checked": True,
        "monotone_nonnegative_zero_action": True,
        "labels_shared_per_focal_scenario": True,
        "observation_columns": 13,
        "current_state_contains_history_summary": True,
    }
