"""Simple scale controls and an abstaining calibration gate for bidding.

The gate uses completed, accepted-exposure logs only. Its Gamma-Poisson
probabilities are working-model diagnostics, not validated confidence levels.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammainc


def scaled_feedback_action(state, actions, budget, cpa, scale=1.0) -> float:
    """Scale native feedback before its clip and projection onto the action grid."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    elapsed = float(state[0])
    spend = float(state[3]) * budget
    conversions = float(state[4]) * budget / cpa
    pacing = ((elapsed + 0.05) * budget / (spend + 0.05 * budget)) ** 0.3
    quality = ((conversions + 8.0) * cpa / (spend + 8.0 * cpa)) ** 0.3
    desired = float(np.clip(pacing * quality * scale, 0.05, 3.0))
    return float(actions[np.abs(actions - desired).argmin()])


def diagnose(
    past_conversions: int,
    past_prediction: float,
    reference_rate: float,
    minimum_expected: float = 8.0,
    probability: float = 0.95,
    margin: float = 0.1,
) -> dict:
    """Assess calibration relative to a frozen nominal selected-winner reference.

    A ratio below one means fewer conversions than the nominal reference would
    predict. It does not identify the simulator's injected bias: selection,
    population drift and nominal model error can produce the same observation.
    No current-tick outcome or scenario identifier is accepted by this API.
    """
    if (
        past_conversions < 0
        or int(past_conversions) != past_conversions
        or not math.isfinite(past_prediction)
        or past_prediction < 0
        or not math.isfinite(reference_rate)
        or reference_rate <= 0
        or not math.isfinite(minimum_expected)
        or minimum_expected < 0
        or not 0.5 < probability < 1
        or not 0 < margin < 1
    ):
        raise ValueError("Invalid calibration inputs")
    expected = reference_rate * past_prediction
    shape, rate = 1.0 + past_conversions, 1.0 + expected
    probability_over = float(gammainc(shape, rate * (1.0 - margin)))
    probability_under = float(1.0 - gammainc(shape, rate * (1.0 + margin)))
    label = "unknown"
    if expected >= minimum_expected:
        if probability_over >= probability:
            label = "over"
        elif probability_under >= probability:
            label = "under"
    return {
        "label": label,
        "relative_mean": shape / rate,
        "probability_over": probability_over,
        "probability_under": probability_under,
        "expected_information": expected,
    }


def branch(label: str) -> str:
    """Use the learned response only after an overprediction signal."""
    if label not in {"over", "under", "unknown"}:
        raise ValueError("Unknown calibration label")
    return "response" if label == "over" else "feedback"
