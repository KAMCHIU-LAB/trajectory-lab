"""Targeted runner checks; synthetic positives exercise rare-event boundaries."""

from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest

import trajectorylab.bidding_delayed_experiment as runner
from trajectorylab.bidding.market import Tape, build_market, feedback_action, observation, score
from trajectorylab.bidding_delayed_feedback import (
    DelayQueue,
    EventTape,
    PublicCohort,
    Settlement,
    build_event_market,
)
from trajectorylab.bidding_pending_response import Cohort, EmpiricalDelayCDF


def configuration():
    return {
        "study": {"actions": [0.0, 0.5, 1.0, 2.0, 3.0]},
        "evaluation": {"probability_bin_edges": [0.1, 0.3]},
        "history": {"calibration_prior_log_sd": 0.5, "maturity_window": 48},
    }


def variant(method="naive"):
    return dict(name=method, estimator=method, gain=0.3, prior=8.0)


def test_public_ledger_groups_public_probability_and_counts_once():
    ledger = runner.PublicLedger([0.1, 0.3])
    ledger.accept(PublicCohort(4, (10, 11, 12, 13), (0.05, 0.1, 0.2, 0.8)))
    before = ledger.snapshot(5)
    assert [
        (c.age, c.event_count, c.public_probability, c.arrived_conversions) for c in before
    ] == [
        (0, 2, pytest.approx(0.075), 0),
        (0, 1, 0.2, 0),
        (0, 1, 0.8, 0),
    ]
    ledger.receive((10, 13))
    after = ledger.snapshot(6)
    assert [c.age for c in after] == [1, 1, 1]
    assert [c.arrived_conversions for c in after] == [1, 0, 1]
    assert ledger.arrived == 2
    with pytest.raises(KeyError):
        ledger.receive((10,))
    assert ledger.arrived == 2
    assert all(
        set(vars(c)) == {"age", "public_probability", "arrived_conversions", "event_count"}
        for c in after
    )


def test_ledger_queue_timing_and_final_flush():
    queue = DelayQueue(1001, 0, 5, "immediate")
    ledger = runner.PublicLedger([0.1])
    settlement = Settlement(1.0, 1, 2, False, (1, 2), (0.02, 0.2), (2,))
    assert queue.advance(-1) == ()
    ledger.accept(queue.register(0, settlement))
    assert ledger.arrived == 0
    ledger.receive(queue.advance(0))  # Before decision 1, D0 arrives at cohort age zero.
    assert ledger.arrived == 1
    assert [c.age for c in ledger.snapshot(1)] == [0, 0]
    ledger.receive(queue.advance(0))
    assert ledger.arrived == 1
    for regime in ("short", "long"):
        queue = DelayQueue(1001, 0, 5, regime)
        ledger = runner.PublicLedger([0.1])
        ledger.accept(queue.register(47, settlement))
        assert ledger.arrived == 0
        ledger.receive(queue.advance(96))
        assert ledger.arrived == settlement.conversions
        ledger.receive(queue.advance(96))
        assert ledger.arrived == settlement.conversions


def test_zero_delay_episode_matches_original_feedback(tmp_path):
    config = configuration()
    study = dict(config["study"], focal_agents=[0], pv_count=10000, biases=[1.0])
    build_market(tmp_path / "base.npz", study, 1001)
    build_event_market(tmp_path / "events.npz", study, 1001)
    tape, original = EventTape(tmp_path / "events.npz"), Tape(tmp_path / "base.npz")
    row, delays = runner.episode(
        tape,
        0,
        1.0,
        "immediate",
        7,
        variant(),
        EmpiricalDelayCDF((1.0,) * 49, 0),
        config,
        collect_delays=True,
    )
    budget, cpa = float(original.arrays["budgets"][0]), float(original.arrays["cpas"][0])
    spent, conversions, exposures = 0.0, 0, 0
    previous = (0.0, 0, 0.0, 0)
    actions = []
    for tick in range(48):
        state = observation(original, 0, tick, spent, conversions, previous, 1.0)
        action = feedback_action(state, np.array(config["study"]["actions"]), 0.3, 8.0, budget, cpa)
        cost, conv, wins, _ = original.settle(0, tick, action, budget - spent)
        spent += cost
        conversions += conv
        exposures += wins
        previous = (cost, conv, action, wins)
        actions.append(action)
    assert row["actions"] == actions
    assert (row["cost"], row["conversions"], row["exposures"]) == (spent, conversions, exposures)
    assert row["score"] == score(spent, conversions, cpa)
    assert row["violation"] == int(spent > cpa * conversions + 1e-8)
    assert row["final_arrived"] == conversions
    assert row["cost"] <= budget + 1e-8
    assert delays == [0] * conversions


class SyntheticTape:
    """Finite hard budget and a guaranteed positive in every accepted opportunity."""

    arrays = dict(
        market_id=np.array(1001),
        focals=np.array([0]),
        budgets=np.array([3.0]),
        cpas=np.array([1.0]),
    )

    def __init__(self):
        self.calls = []

    def settle_events(self, fi, tick, action, remaining):
        self.calls.append((tick, remaining))
        if remaining < 1:
            return Settlement(0.0, 0, 0, True, (), (), ())
        return Settlement(1.0, 1, 1, False, (tick,), (0.2,), (tick,))


@pytest.mark.parametrize("regime", ["immediate", "short", "long"])
def test_episode_matures_all_accepted_events_without_new_opportunities(regime):
    tape = SyntheticTape()
    row, _ = runner.episode(
        tape, 0, 1.0, regime, 8, variant(), EmpiricalDelayCDF((1.0,) * 49, 0), configuration()
    )
    assert (row["cost"], row["conversions"], row["exposures"], row["final_arrived"]) == (
        3.0,
        3,
        3,
        3,
    )
    assert len(tape.calls) == 48
    assert [tick for tick, _ in tape.calls] == list(range(48))
    assert min(remaining for _, remaining in tape.calls) == 0


def test_estimator_only_receives_public_cohorts_and_oracle_is_explicit(monkeypatch):
    calls = []

    class PublicOnlyEstimator:
        def __init__(self, cdf, kind, *, prior_log_sd):
            assert kind == "naive"

        def estimate(self, cohorts):
            assert isinstance(cohorts, tuple)
            assert all(isinstance(c, Cohort) for c in cohorts)
            assert {f.name for f in fields(Cohort)} == {
                "age",
                "public_probability",
                "arrived_conversions",
                "event_count",
            }
            assert all(c.age >= 0 and c.public_probability == 0.2 for c in cohorts)
            calls.append(cohorts)
            return SimpleNamespace(total_mean=sum(c.arrived_conversions for c in cohorts))

    monkeypatch.setattr(runner, "PendingEstimator", PublicOnlyEstimator)
    ordinary, _ = runner.episode(
        SyntheticTape(),
        0,
        1.0,
        "long",
        9,
        variant(),
        EmpiricalDelayCDF((1.0,) * 49, 0),
        configuration(),
    )
    assert len(calls) == 48
    assert calls[0] == ()
    assert ordinary["deployable"] is True
    assert (
        ordinary["pending_mae"] > 0
    )  # Hidden pending positives exist but were not passed into the estimator.
    calls.clear()
    oracle, _ = runner.episode(
        SyntheticTape(),
        0,
        1.0,
        "long",
        9,
        variant("oracle_past"),
        EmpiricalDelayCDF((1.0,) * 49, 0),
        configuration(),
    )
    assert calls == []
    assert oracle["deployable"] is False
    assert oracle["pending_mae"] == 0.0
