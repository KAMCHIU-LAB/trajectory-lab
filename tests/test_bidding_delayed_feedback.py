"""Physical replay and as-of feedback integrity on a small public market."""

import numpy as np
import pytest

from trajectorylab.bidding.market import Tape, build_market
from trajectorylab.bidding_delayed_feedback import (
    DelayQueue,
    EventTape,
    Settlement,
    build_event_market,
    event_delay,
)


def test_small_public_event_tape_exact_replay(tmp_path):
    study = dict(focal_agents=[0], pv_count=10000, actions=[0.0, 1.0, 3.0], biases=[1.0])
    build_market(tmp_path / "reference.npz", study, 1001)
    build_event_market(tmp_path / "events.npz", study, 1001)
    original, tape = Tape(tmp_path / "reference.npz"), EventTape(tmp_path / "events.npz")
    for name, array in original.arrays.items():
        np.testing.assert_array_equal(array, tape.arrays[name])
    seen_probability = {}
    for tick in range(48):
        for action in (0.0, 1.0, 3.0):
            for remaining in (0.0, 1.0, float("inf")):
                s = tape.settle_events(0, tick, action, remaining)
                assert (s.cost, s.conversions, s.wins, s.budget_blocked) == original.settle(
                    0, tick, action, remaining
                )
                assert len(s.event_ids) == s.wins == len(s.public_p)
                assert len(s.positive_ids) == s.conversions
                assert set(s.positive_ids) <= set(s.event_ids)
                for event, probability in zip(s.event_ids, s.public_p, strict=True):
                    assert seen_probability.setdefault(event, probability) == probability
                    assert (
                        sum(tape.arrays["counts"][:tick])
                        <= event
                        < sum(tape.arrays["counts"][: tick + 1])
                    )
    assert build_event_market(tmp_path / "events.npz", study, 1001)["market_id"] == 1001


def test_delay_is_stable_bounded_and_no_double_counting():
    s = Settlement(2.0, 2, 3, False, (1, 2, 3), (0.1, 0.2, 0.3), (1, 3))
    for regime in ("immediate", "short", "long"):
        q = DelayQueue(1001, 0, 9, regime)
        cohort = q.register(4, s, 0.75)
        assert cohort.public_p == (0.07500000000000001, 0.15000000000000002, 0.22499999999999998)
        assert not hasattr(cohort, "positive_ids")
        assert q.cohorts(3) == ()
        arrivals = []
        for tick in range(97):
            got = q.advance(tick)
            assert all(tick >= 4 + event_delay(1001, 0, e, 9, regime) for e in got)
            arrivals.extend(got)
        assert sorted(arrivals) == [1, 3]
        assert q.advance(97) == ()
        with pytest.raises(ValueError):
            q.advance(0)
        delays = [event_delay(1001, 0, e, 9, regime) for e in range(100)]
        assert all(0 <= d <= 48 for d in delays)
        assert delays == [event_delay(1001, 0, e, 9, regime) for e in range(100)]


def test_zero_budget_empty_and_duplicate_guard():
    q = DelayQueue(1, 0, 0, "long")
    empty = Settlement(0.0, 0, 0, True, (), (), ())
    q.register(0, empty)
    assert q.advance(96) == ()
    q = DelayQueue(1, 0, 0, "immediate")
    s = Settlement(1.0, 1, 1, False, (5,), (0.3,), (5,))
    q.register(0, s)
    assert q.advance(0) == (5,)
    with pytest.raises(ValueError):
        q.register(0, s)


def test_immediate_feedback_policy_replay(tmp_path):
    from trajectorylab.bidding.market import feedback_action, observation, score

    study = dict(focal_agents=[0], pv_count=10000, actions=[0.0, 1.0, 3.0], biases=[1.0])
    build_market(tmp_path / "reference.npz", study, 1001)
    build_event_market(tmp_path / "events.npz", study, 1001)
    reference, events = Tape(tmp_path / "reference.npz"), EventTape(tmp_path / "events.npz")
    budget, cpa = float(events.arrays["budgets"][0]), float(events.arrays["cpas"][0])
    actions = np.array(study["actions"])
    queue = DelayQueue(1001, 0, 0, "immediate")
    cost = reference_cost = 0.0
    conversions = reference_conversions = 0
    previous = reference_previous = (0.0, 0, 0.0, 0)
    for tick in range(48):
        arrived = queue.advance(tick - 1)
        conversions += len(arrived)
        state = observation(events, 0, tick, cost, conversions, previous, 1.0)
        expected_state = observation(
            reference, 0, tick, reference_cost, reference_conversions, reference_previous, 1.0
        )
        np.testing.assert_array_equal(state, expected_state)
        action = feedback_action(state, actions, 0.3, 8.0, budget, cpa)
        expected_action = feedback_action(expected_state, actions, 0.3, 8.0, budget, cpa)
        assert action == expected_action
        settlement = events.settle_events(0, tick, action, budget - cost)
        expected = reference.settle(0, tick, expected_action, budget - reference_cost)
        queue.register(tick, settlement)
        cost += settlement.cost
        reference_cost += expected[0]
        reference_conversions += expected[1]
        previous = (settlement.cost, settlement.conversions, action, settlement.wins)
        reference_previous = (expected[0], expected[1], expected_action, expected[2])
    conversions += len(queue.advance(95))
    assert (cost, conversions) == (reference_cost, reference_conversions)
    assert score(cost, conversions, cpa) == score(reference_cost, reference_conversions, cpa)
