import copy

import numpy as np
import pytest
import torch

from trajectorylab.algorithms import Learner, Replay, Sequences, bellman_target, expectile_loss
from trajectorylab.config import TRAIN_DEFAULTS
from trajectorylab.data import Normalizer
from trajectorylab.experiment import load_checkpoint, save_checkpoint
from trajectorylab.models import DecisionTransformer


def test_expectile_direction():
    diff = torch.tensor([-2.0, 1.0])
    assert torch.allclose(expectile_loss(diff, 0.8), torch.tensor((0.2 * 4 + 0.8) / 2))


def test_timeout_keeps_bootstrap_but_terminal_removes_it():
    rewards, terminals, values = (
        torch.tensor([1.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([10.0, 10.0]),
    )
    assert torch.allclose(
        bellman_target(rewards, terminals, values, 0.9), torch.tensor([1.0, 10.0])
    )


def test_sequences_do_not_cross_boundaries(episodes):
    for i, episode in enumerate(episodes):
        episode.observations[:] = i
    normalizer = Normalizer.fit(np.concatenate([e.observations[:-1] for e in episodes]))
    sampler = Sequences(episodes, normalizer, 12, 1.0, "cpu")
    batch = sampler.sample(np.random.default_rng(0), 100)
    for states, mask, times in zip(
        batch["observations"], batch["mask"], batch["timesteps"], strict=True
    ):
        assert torch.allclose(states[mask], states[mask][0].expand_as(states[mask]))
        assert times[mask][-1] <= 7
        assert torch.all(~mask[mask.sum() :])


def test_dt_cannot_see_current_action_or_future_inputs():
    model = DecisionTransformer(3, 1, width=16, layers=2, heads=4, max_timestep=32).eval()
    states, actions, returns = torch.randn(2, 5, 3), torch.randn(2, 5, 1), torch.randn(2, 5)
    times, mask = torch.arange(5).expand(2, -1), torch.ones(2, 5, dtype=torch.bool)
    before = model(states, actions, returns, times, mask)
    changed_actions, changed_states, changed_returns = (
        actions.clone(),
        states.clone(),
        returns.clone(),
    )
    changed_actions[:, 2:] += 100
    changed_states[:, 3:] -= 100
    changed_returns[:, 3:] += 100
    after = model(changed_states, changed_actions, changed_returns, times, mask)
    assert torch.allclose(before[:, :3], after[:, :3], atol=1e-6)


@pytest.mark.parametrize("algorithm", ["bc", "iql", "dt"])
def test_real_updates_checkpoint_and_exact_resume(tmp_path, episodes, algorithm):
    config = TRAIN_DEFAULTS | {"width": 16, "dt_layers": 1, "context": 4, "batch_size": 8}
    normalizer = Normalizer.fit(np.concatenate([e.observations[:-1] for e in episodes]))
    sampler = (
        Sequences(episodes, normalizer, 4, 1000, "cpu")
        if algorithm == "dt"
        else Replay(episodes, normalizer, "cpu")
    )
    learner = Learner(algorithm, 3, 1, config)
    initial = [p.detach().clone() for p in learner.actor.parameters()]
    batch = sampler.sample(np.random.default_rng(0), 8)
    metrics = learner.update(batch)
    assert all(np.isfinite(v) for v in metrics.values())
    assert any(
        not torch.equal(a, b) for a, b in zip(initial, learner.actor.parameters(), strict=True)
    )
    path = tmp_path / "model.pt"
    save_checkpoint(path, {"run_id": "identity", "learner": learner.state()})
    saved = load_checkpoint(path, "identity")
    learner.update(batch)
    expected = copy.deepcopy(learner.actor.state_dict())
    restored = Learner(algorithm, 3, 1, config)
    restored.load_state(saved["learner"])
    restored.update(batch)
    assert all(
        torch.equal(value, restored.actor.state_dict()[key]) for key, value in expected.items()
    )
    with pytest.raises(ValueError, match="fingerprint"):
        load_checkpoint(path, "other")


def test_zero_weighting_iql_actor_matches_bc(episodes):
    config = TRAIN_DEFAULTS | {"width": 16, "advantage_beta": 0.0}
    normalizer = Normalizer.fit(np.concatenate([e.observations[:-1] for e in episodes]))
    batch = Replay(episodes, normalizer, "cpu").sample(np.random.default_rng(0), 8)
    torch.manual_seed(4)
    bc = Learner("bc", 3, 1, config)
    torch.manual_seed(4)
    iql = Learner("iql", 3, 1, config)
    bc.update(batch)
    iql.update(batch)
    assert all(
        torch.equal(a, b)
        for a, b in zip(bc.actor.parameters(), iql.actor.parameters(), strict=True)
    )
    assert all(p.grad is None for p in iql.modules["target_q"].parameters())
