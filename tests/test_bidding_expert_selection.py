"""Shared-prefix integrity and public-only expert inputs on synthetic events."""

from dataclasses import fields

import numpy as np
import pytest

from trajectorylab import bidding_expert_selection as selection
from trajectorylab.bidding.learning import Feedback, rollout
from trajectorylab.bidding.market import observation, score
from trajectorylab.bidding_delayed_feedback import EventTape


def synthetic_tape(tmp_path, *, budget=33.0, positive=True):
    actions = selection.DEFAULT_ACTIONS
    grid = np.array(sorted({round(a * b, 8) for a in actions for b in (0.75, 1.0, 1.5)}))
    shape = (1, 48, len(grid))
    starts = np.zeros(shape, dtype=np.int64)
    lengths = np.zeros(shape, dtype=np.int64)
    costs, conversions, event_ids = [], [], []
    offset = 0
    for tick in range(48):
        for ai, action in enumerate(grid):
            n = min(3, int(action + 0.75))
            pc = np.arange(n + 1, dtype=float) * (0.7 + (tick % 4) * 0.1)
            labels = np.array([int(positive and (tick + j) % 3 == 0) for j in range(n)])
            cv = np.r_[0, labels.cumsum()].astype(np.int32)
            starts[0, tick, ai], lengths[0, tick, ai] = offset, len(pc)
            costs.extend(pc)
            conversions.extend(cv)
            event_ids.extend([-1, *range(tick * 3, tick * 3 + n)])
            offset += len(pc)
    path = tmp_path / f"synthetic-{positive}-{budget}.npz"
    np.savez(
        path,
        grid=grid,
        focals=np.array([0]),
        market_id=np.array(1001),
        budgets=np.array([budget]),
        cpas=np.array([1.0]),
        counts=np.arange(48) + 7,
        pmeans=np.linspace(0.0001, 0.0008, 48)[None],
        smeans=np.linspace(0.00003, 0.00009, 48)[None],
        starts=starts,
        lengths=lengths,
        prefix_cost=np.array(costs),
        prefix_conv=np.array(conversions),
        prefix_event_ids=np.array(event_ids),
        public_probability=np.linspace(0.01, 0.1, 144)[None],
    )
    return EventTape(path)


def fixed(action):
    def choose(states, used, rewards, budget, cpa):
        return action, {}

    return choose


@pytest.mark.parametrize("bias", [0.75, 1.0, 1.5, "mid_over_50"])
def test_immediate_matches_legacy_every_observation_action_reward_and_terminal(tmp_path, bias):
    tape = synthetic_tape(tmp_path)
    actions = np.array(selection.DEFAULT_ACTIONS)
    scenario = bias if isinstance(bias, str) else {0.75: "under_25", 1.0: "calibrated", 1.5: "over_50"}[bias]
    legacy = rollout(tape, 0, actions, scenario, Feedback(actions, 0.3, 8), retain=True)
    env = selection.ExpertSelectionEnv(tape, 0, bias, "immediate", 4)
    result = env.run(Feedback(actions, 0.3, 8)).finish()
    np.testing.assert_array_equal(np.array(env.states), legacy["states"])
    np.testing.assert_array_equal(np.array(env.used), legacy["actions"])
    np.testing.assert_array_equal(np.array(env.rewards, dtype=np.float32), legacy["rewards"])
    for key in ("cost", "conversions", "exposures", "score", "violation", "budget_blocked_ticks"):
        assert result[key] == legacy[key]
    assert result["final_arrived"] == result["conversions"]
    assert result["hard_budget_violation"] == 0


@pytest.mark.parametrize("regime", ["immediate", "short", "long"])
def test_pair_runs_prefix_once_then_clones_and_scores_full_cycle(tmp_path, regime):
    tape = synthetic_tape(tmp_path)
    calls = []
    settle = tape.settle_events

    def track(*args):
        calls.append(args[1])
        return settle(*args)

    tape.settle_events = track
    paired = selection.paired_continuations(tape, 0, 1.0, regime, 9, 8, fixed(0.0))
    assert calls == [*range(8), *range(8, 48), *range(8, 48)]
    feedback, response = paired["feedback"], paired["response"]
    assert feedback["actions"][:8] == response["actions"][:8]
    assert response["actions"][8:] == [0.0] * 40
    assert feedback["prefix_digest"] == response["prefix_digest"] == paired["prefix_digest"]
    assert feedback["selection_identity"] != response["selection_identity"]
    assert response["conversions"] > 0  # Prefix outcomes remain in the terminal score.
    for row in (feedback, response):
        assert row["score"] == score(row["cost"], row["conversions"], row["target_cpa"])
        assert row["final_arrived"] == row["conversions"]
        assert row["cost"] <= row["budget"] + 1e-8
        assert row["hard_budget_violation"] == 0
        assert len(row["actions"]) == 48
    assert len(paired["features"]) == len(paired["feature_names"]) == 28
    assert paired["prefix_summary"]["accepted_public_probability_sum"] > 0


def test_identical_experts_produce_identical_endings_and_prefix_zero_baseline(tmp_path):
    tape = synthetic_tape(tmp_path)
    expert = fixed(1.0)
    paired = selection.paired_continuations(tape, 0, 1.5, "long", 12, 16, expert, feedback=expert)
    for key in ("cost", "conversions", "actions", "score", "final_arrived", "violation"):
        assert paired["feedback"][key] == paired["response"][key]
    zero = selection.paired_continuations(tape, 0, 1.0, "immediate", 0, 0, expert)
    baseline = selection.whole_episode(tape, 0, 1.0, "immediate", 0, expert)
    for key, value in baseline.items():
        assert zero["response"][key] == value


def test_clone_isolates_queue_ledger_budget_previous_and_histories(tmp_path):
    tape = synthetic_tape(tmp_path)
    env = selection.ExpertSelectionEnv(tape, 0, 1.0, "long", 17)
    env.run(fixed(1.0), stop_tick=8)
    record = env.prefix_record()
    left, right = env.clone(), env.clone()
    assert left._queue is not right._queue
    assert left._ledger is not right._ledger
    assert left.states[0] is not right.states[0]
    left.run(fixed(3.0), stop_tick=12)
    left.states[0][0] = 999
    assert right.prefix_record() == record == env.prefix_record()
    assert left.tick == 12 and right.tick == 8
    assert left.spent > right.spent
    assert left.previous != right.previous
    right.run(fixed(0.0)).finish()
    assert env.prefix_record() == record


def test_delayed_observation_uses_new_arrivals_not_hidden_previous_conversions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trajectorylab.bidding_delayed_feedback.event_delay",
        lambda market, focal, event, seed, regime: 2,
    )
    tape = synthetic_tape(tmp_path)
    env = selection.ExpertSelectionEnv(tape, 0, 1.5, "long", 1)
    env.run(fixed(1.0), stop_tick=1)
    state = env.observation()
    assert env._truth > 0
    assert state[4] == state[7] == 0
    assert env.previous[1] == 0
    env.run(fixed(1.0), stop_tick=3)
    state = env.observation()
    assert env.public_view().arrived == 1
    assert state[4] == np.float32(1 / env.budget)
    assert state[7] == np.float32(1 / env.budget)
    assert state[9] == observation(tape, 0, 3, env.spent, 1, env.previous, 1.5)[9]
    assert state[10] == observation(tape, 0, 3, env.spent, 1, env.previous, 1.5)[10]
    assert {f.name for f in fields(env.public_traffic())} == {
        "probability_mean", "relative_uncertainty", "count", "previous_count"
    }
    row = env.run(fixed(1.0)).finish()
    assert row["final_arrived"] == row["conversions"]


def test_private_pending_and_future_labels_do_not_change_prefix_features_or_public_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "trajectorylab.bidding_delayed_feedback.event_delay",
        lambda market, focal, event, seed, regime: 48,
    )
    with_truth = synthetic_tape(tmp_path, positive=True)
    without_truth = synthetic_tape(tmp_path, positive=False)
    left = selection.paired_continuations(with_truth, 0, 1.0, "long", 3, 8, fixed(1.0))
    right = selection.paired_continuations(without_truth, 0, 1.0, "long", 3, 8, fixed(1.0))
    assert left["features"] == right["features"]
    assert left["prefix_digest"] == right["prefix_digest"]
    assert left["prefix_identity"] == right["prefix_identity"]
    assert left["prefix_summary"] == right["prefix_summary"]
    assert left["response"]["conversions"] > right["response"]["conversions"] == 0
    assert not any(
        prohibited in name
        for name in left["feature_names"]
        for prohibited in ("market", "focal", "bias", "truth", "latent")
    )


def test_experts_get_copied_public_history_and_no_online_correction(tmp_path):
    tape = synthetic_tape(tmp_path)

    class PublicOnlyExpert:
        corrected = False

        def __call__(self, states, used, rewards, budget, cpa):
            tick = len(states) - 1
            assert all(isinstance(state, np.ndarray) and state.shape == (13,) for state in states)
            assert len(used) == len(rewards) == tick
            assert states[-1][0] == np.float32(tick / 48)
            states[0][:] = -999
            used.append(-999)
            rewards.append(-999)
            return 1.0, {}

        def observe(self, *args):
            pytest.fail("Frozen uncorrected experts must not receive settlement updates")

    env = selection.ExpertSelectionEnv(tape, 0, 1.0, "immediate", 0)
    row = env.run(PublicOnlyExpert()).finish()
    assert len(env.states) == len(env.used) == len(env.rewards) == 48
    assert env.states[0][0] == 0
    assert row["actions"] == [1.0] * 48
    before = env.finish()
    assert before == row and len(env.rewards) == 48  # Maturity is idempotent.


def test_invalid_policy_prefix_action_and_early_terminal_are_rejected(tmp_path):
    tape = synthetic_tape(tmp_path)

    class Corrected:
        corrected = True

        def __call__(self, *args):
            return 1.0, {}

    with pytest.raises(ValueError, match="uncorrected"):
        selection.paired_continuations(tape, 0, 1.0, "immediate", 0, 8, Corrected())
    with pytest.raises(ValueError, match="Prefix"):
        selection.paired_continuations(tape, 0, 1.0, "immediate", 0, 48, fixed(1.0))
    for action in (float("nan"), 9.0, -1.0):
        env = selection.ExpertSelectionEnv(tape, 0, 1.0, "immediate", 0)
        with pytest.raises(RuntimeError, match="invalid nominal action"):
            env.run(fixed(action))
        assert env.tick == 0 and env.spent == 0
    with pytest.raises(RuntimeError, match="all 48"):
        selection.ExpertSelectionEnv(tape, 0, 1.0, "immediate", 0).finish()
