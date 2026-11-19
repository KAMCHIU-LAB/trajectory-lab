"""One expert choice after a shared public-feedback prefix.

Only the environment owns the tape, delayed positive-label queue and terminal
truth. Choosers receive copies of as-of public observations/actions/rewards in
the existing Learned call format. No online ``observe`` correction is applied.
"""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import dataclass

import numpy as np

from trajectorylab.bidding.learning import Feedback
from trajectorylab.bidding.market import scenario_bias, score
from trajectorylab.bidding_delayed_experiment import PolicyView, PublicLedger
from trajectorylab.bidding_delayed_feedback import DelayQueue, EventTape
from trajectorylab.utils import fingerprint

HORIZON = 48
HISTORY_GAIN = 0.3
HISTORY_PRIOR = 8.0
DEFAULT_ACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
OBSERVATION_NAMES = (
    "elapsed_fraction",
    "remaining_fraction",
    "remaining_budget_fraction",
    "spent_budget_fraction",
    "arrived_value_budget_fraction",
    "arrived_cpa_debt_budget_fraction",
    "previous_spend_budget_fraction",
    "newly_arrived_value_budget_fraction",
    "previous_action_scaled",
    "current_public_probability_scaled",
    "current_relative_uncertainty",
    "current_traffic_scaled",
    "previous_win_fraction",
)
AGE_BUCKETS = ((0, 4, "0_3"), (4, 8, "4_7"), (8, 16, "8_15"), (16, 48, "16_plus"))
FEATURE_NAMES = (
    *OBSERVATION_NAMES,
    "accepted_events_scaled",
    "accepted_public_value_budget_fraction",
    "smoothed_arrival_ratio",
    *(
        f"{name}_age_{label}"
        for _, _, label in AGE_BUCKETS
        for name in ("accepted_events_scaled", "public_value_budget_fraction", "arrived_value_budget_fraction")
    ),
)


@dataclass(frozen=True)
class PublicTraffic:
    """Current public prediction statistics; no unbiased p or bias identifier."""

    probability_mean: float
    relative_uncertainty: float
    count: int
    previous_count: int


def arrived_observation(view: PolicyView, traffic: PublicTraffic, previous: tuple) -> np.ndarray:
    """The old 13-column observation with as-of arrived conversion feedback.

    ``previous[1]`` is the newly arrived count, including arrivals from older
    cohorts. In immediate mode it equals the old last-tick conversion count.
    """
    last_cost, newly_arrived, last_action, last_wins = previous
    return np.array(
        [
            view.tick / HORIZON,
            (HORIZON - view.tick) / HORIZON,
            max(0, 1 - view.spent / view.budget),
            view.spent / view.budget,
            view.arrived * view.cpa / view.budget,
            (view.spent - view.cpa * view.arrived) / view.budget,
            last_cost / view.budget,
            newly_arrived * view.cpa / view.budget,
            last_action / 3,
            traffic.probability_mean / 0.0005,
            traffic.relative_uncertainty,
            traffic.count / 10000.0,
            last_wins / max(1, traffic.previous_count),
        ],
        dtype=np.float32,
    )


def public_features(view: PolicyView, traffic: PublicTraffic, previous: tuple) -> list[float]:
    """Selection features computed entirely from the public view and traffic."""
    features = arrived_observation(view, traffic, previous).astype(float).tolist()
    accepted = sum(c.event_count for c in view.cohorts)
    public_probability = sum(c.public_probability * c.event_count for c in view.cohorts)
    features.extend(
        (
            accepted / 10000.0,
            public_probability * view.cpa / view.budget,
            (view.arrived + HISTORY_PRIOR) / (public_probability + HISTORY_PRIOR),
        )
    )
    for low, high, _ in AGE_BUCKETS:
        cohorts = [c for c in view.cohorts if low <= c.age < high]
        features.extend(
            (
                sum(c.event_count for c in cohorts) / 10000.0,
                sum(c.public_probability * c.event_count for c in cohorts) * view.cpa / view.budget,
                sum(c.arrived_conversions for c in cohorts) * view.cpa / view.budget,
            )
        )
    if not np.isfinite(features).all():
        raise RuntimeError("Non-finite public selection features")
    return features


def _validate_expert(expert):
    if not callable(expert):
        raise TypeError("Expert must use the Learned-compatible callable interface")
    if getattr(expert, "corrected", False):
        raise ValueError("Expert selection requires an uncorrected frozen expert")


class ExpertSelectionEnv:
    """Cloneable episode state; this object is never passed to a chooser."""

    feature_names = FEATURE_NAMES

    def __init__(
        self,
        tape: EventTape,
        fi: int,
        bias: float | str,
        regime: str,
        arrival_seed: int,
        *,
        actions=None,
    ):
        self.actions = np.asarray(DEFAULT_ACTIONS if actions is None else actions, dtype=float).copy()
        if (
            self.actions.ndim != 1
            or not len(self.actions)
            or not np.isfinite(self.actions).all()
            or np.any(np.diff(self.actions) <= 0)
            or self.actions[0] != 0
            or self.actions[-1] > 3
        ):
            raise ValueError("Expected increasing finite nominal actions from zero up to three")
        self._tape, self._fi, self._bias = tape, fi, bias
        a = tape.arrays
        self.budget, self.cpa = float(a["budgets"][fi]), float(a["cpas"][fi])
        if not np.isfinite((self.budget, self.cpa)).all() or min(self.budget, self.cpa) <= 0:
            raise ValueError("Budget and CPA must be finite and positive")
        self._queue = DelayQueue(int(a["market_id"]), int(a["focals"][fi]), arrival_seed, regime)
        # One public probability bucket per acceptance tick is enough for these
        # aggregate features; no private label or delay parameter is used here.
        self._ledger = PublicLedger([])
        self.tick = 0
        self.spent = 0.0
        self._truth = self._wins = self._blocked = 0
        self.previous = (0.0, 0, 0.0, 0)
        self.states: list[np.ndarray] = []
        self.used: list[float] = []
        self.rewards: list[float] = []
        self._previous_public_score = 0.0
        self._ready = False
        self._finished = False

    def _multiplier(self, tick):
        multiplier = scenario_bias(self._bias, tick) if isinstance(self._bias, str) else float(self._bias)
        if not np.isfinite(multiplier) or multiplier <= 0:
            raise ValueError("Prediction multiplier must be finite and positive")
        return multiplier

    def _prepare(self):
        if self.tick >= HORIZON or self._finished:
            raise RuntimeError("No decision remains in this episode")
        if not self._ready:
            arrivals = self._queue.advance(self.tick - 1)
            self._ledger.receive(arrivals)
            self.previous = (self.previous[0], len(arrivals), self.previous[2], self.previous[3])
            current_score = score(self.spent, self._ledger.arrived, self.cpa)
            if self.tick:
                self.rewards.append(current_score - self._previous_public_score)
            self._previous_public_score = current_score
            self._ready = True

    def public_view(self) -> PolicyView:
        self._prepare()
        return PolicyView(
            self.tick, self.spent, self._ledger.arrived, self.budget, self.cpa,
            self._ledger.snapshot(self.tick),
        )

    def public_traffic(self) -> PublicTraffic:
        self._prepare()
        a, tick, fi = self._tape.arrays, self.tick, self._fi
        pmean = float(a["pmeans"][fi, tick])
        # Multiplicative prediction bias leaves the public relative uncertainty
        # unchanged. Compute the old ratio exactly, preserving float32 parity.
        return PublicTraffic(
            pmean * self._multiplier(tick),
            float(a["smeans"][fi, tick]) / max(pmean, 1e-12),
            int(a["counts"][tick]),
            int(a["counts"][max(0, tick - 1)]),
        )

    def observation(self) -> np.ndarray:
        return arrived_observation(self.public_view(), self.public_traffic(), self.previous)

    def publicfeatures(self) -> list[float]:
        return public_features(self.public_view(), self.public_traffic(), self.previous)

    def clone(self) -> ExpertSelectionEnv:
        """Share the immutable tape; isolate all mutable episode bookkeeping."""
        other = copy(self)
        for name in ("actions", "_queue", "_ledger", "previous", "states", "used", "rewards"):
            setattr(other, name, deepcopy(getattr(self, name)))
        return other

    def run(self, expert, *, stop_tick=HORIZON):
        _validate_expert(expert)
        if not isinstance(stop_tick, int) or not self.tick <= stop_tick <= HORIZON:
            raise ValueError("Stop tick must be an integer between current tick and 48")
        if self._finished:
            raise RuntimeError("Cannot continue a matured episode")
        while self.tick < stop_tick:
            state = self.observation()
            # A callable can retain or mutate these inputs without gaining the
            # tape/queue or corrupting the common-prefix checkpoint.
            action, _ = expert(
                [s.copy() for s in (*self.states, state)],
                self.used.copy(), self.rewards.copy(), self.budget, self.cpa,
            )
            matches = np.flatnonzero(np.isclose(self.actions, action, rtol=0, atol=1e-8))
            if not np.isfinite(action) or not len(matches):
                raise RuntimeError("Expert returned an invalid nominal action")
            action = float(self.actions[matches[0]])
            settled = self._tape.settle_events(
                self._fi, self.tick, action * self._multiplier(self.tick), self.budget - self.spent
            )
            if settled.cost < 0 or settled.cost > max(0, self.budget - self.spent) + 1e-6:
                raise RuntimeError("Settlement exceeded the hard budget")
            public = self._queue.register(
                self.tick, settled, prediction_multiplier=self._multiplier(self.tick)
            )
            self._ledger.accept(public)
            self.spent += settled.cost
            self._truth += settled.conversions
            self._wins += settled.wins
            self._blocked += int(settled.budget_blocked)
            self.states.append(state.copy())
            self.used.append(action)
            self.previous = (settled.cost, 0, action, settled.wins)
            self.tick += 1
            self._ready = False
        return self

    def prefix_record(self) -> dict:
        """Audit record containing only information visible at selection time."""
        view, traffic = self.public_view(), self.public_traffic()
        public_sum = sum(c.event_count * c.public_probability for c in view.cohorts)
        return {
            "tick": self.tick,
            "budget": self.budget,
            "target_cpa": self.cpa,
            "spent": self.spent,
            "arrived": view.arrived,
            "accepted_public_probability_sum": public_sum,
            "calibration_ratio": (view.arrived + HISTORY_PRIOR) / (public_sum + HISTORY_PRIOR),
            "states": [state.tolist() for state in self.states],
            "actions": self.used.copy(),
            "rewards": self.rewards.copy(),
            "observation": arrived_observation(view, traffic, self.previous).tolist(),
            "cohorts": [vars(c) for c in view.cohorts],
        }

    def finish(self) -> dict:
        """Mature all selected events and score the full 48-opportunity cycle."""
        if self.tick != HORIZON:
            raise RuntimeError("Terminal scoring requires all 48 opportunity ticks")
        if not self._finished:
            self._ledger.receive(self._queue.advance(2 * HORIZON - 1))
            if self._ledger.arrived != self._truth:
                raise RuntimeError("Final label maturity failed or labels were counted twice")
            if self.spent > self.budget + 1e-6:
                raise RuntimeError("Final settlement exceeded the hard budget")
            self.rewards.append(score(self.spent, self._ledger.arrived, self.cpa) - self._previous_public_score)
            self._finished = True
        return {
            "cost": self.spent,
            "conversions": self._truth,
            "score": score(self.spent, self._truth, self.cpa),
            "violation": int(self.spent > self.cpa * self._truth + 1e-8),
            "hard_budget_violation": int(self.spent > self.budget + 1e-6),
            "final_arrived": self._ledger.arrived,
            "exposures": self._wins,
            "budget": self.budget,
            "target_cpa": self.cpa,
            "cpa": self.spent / self._truth if self._truth else None,
            "cpa_debt": self.spent - self.cpa * self._truth,
            "zero_conversions": int(self._truth == 0),
            "budget_utilization": self.spent / self.budget,
            "budget_blocked_ticks": self._blocked,
            "actions": self.used.copy(),
        }


def paired_continuations(
    tape, fi, bias, regime, arrival_seed, prefix, response, *,
    gain=HISTORY_GAIN, prior=HISTORY_PRIOR, actions=None, feedback=None,
) -> dict:
    """Run frozen feedback once, clone, and evaluate two complete-cycle endings.

    Study registration chooses prefixes (8/16) and uses immediate feedback as
    primary; short/long are compatibility adapters. The public digest excludes
    private queue contents, market/focal IDs and future labels. Artifact/model
    provenance belongs to the caller, not this public-prefix digest.
    """
    if not isinstance(prefix, int) or not 0 <= prefix < HORIZON:
        raise ValueError("Prefix must be an integer from zero through 47")
    _validate_expert(response)
    env = ExpertSelectionEnv(tape, fi, bias, regime, arrival_seed, actions=actions)
    feedback = Feedback(env.actions, gain, prior) if feedback is None else feedback
    _validate_expert(feedback)
    env.run(Feedback(env.actions, HISTORY_GAIN, HISTORY_PRIOR), stop_tick=prefix)
    record = env.prefix_record()
    digest = fingerprint(record)
    features = env.publicfeatures()
    result = {
        "prefix": prefix,
        "prefix_digest": digest,
        "prefix_identity": fingerprint({"prefix": prefix, "digest": digest, "gain": HISTORY_GAIN, "prior": HISTORY_PRIOR}),
        "prefix_summary": {key: record[key] for key in ("tick", "spent", "arrived", "accepted_public_probability_sum", "calibration_ratio")},
        "features": features,
        "feature_names": list(FEATURE_NAMES),
    }
    for name, expert in (("feedback", feedback), ("response", response)):
        row = env.clone().run(expert).finish()
        row.update(
            expert=name,
            prefix=prefix,
            prefix_digest=digest,
            selection_identity=fingerprint({"prefix_digest": digest, "expert": name}),
        )
        result[name] = row
    return result


def whole_episode(tape, fi, bias, regime, arrival_seed, expert, *, actions=None) -> dict:
    """Fixed callable baseline using the same public adapter and terminal score."""
    return ExpertSelectionEnv(tape, fi, bias, regime, arrival_seed, actions=actions).run(expert).finish()
