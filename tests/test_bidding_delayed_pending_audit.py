"""Offline diagnostic scoring, determinism, common paths, and public inputs."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from trajectorylab import bidding_delayed_experiment as experiment
from trajectorylab.bidding_delayed_feedback import Settlement
from trajectorylab.bidding_pending_response import Cohort, EmpiricalDelayCDF

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('pending_audit', ROOT/'scripts/audit_bidding_delayed_pending.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


@pytest.mark.parametrize(('samples', 'truth', 'expected'), [([0], 0, 0.), ([0], 2, 2.), ([0, 2], 1, .5), ([0, 2], 0, .5), ([1, 1, 1], 1, 0.)])
def test_integer_crps_hand_examples(samples, truth, expected):
    assert audit.integer_crps(samples, truth) == expected
    x = np.array(samples)
    direct = np.abs(x-truth).mean()-.5*np.abs(x[:, None]-x[None, :]).mean()
    assert audit.integer_crps(samples, truth) == pytest.approx(direct)


@pytest.mark.parametrize('samples', [[], [-1], [1.2], [float('nan')], [[1]]])
def test_integer_crps_rejects_invalid_samples(samples):
    with pytest.raises(ValueError):
        audit.integer_crps(samples, 0)


def test_seed_stability_and_identity():
    first = audit.audit_seed(1, 1001, 8, 'under', 'long', 16)
    assert first == audit.audit_seed(1, 1001, 8, 'under', 'long', 16)
    assert len({first, audit.audit_seed(2, 1001, 8, 'under', 'long', 16), audit.audit_seed(1, 1001, 8, 'under', 'long', 24)}) == 3


class TinyTape:
    arrays = dict(market_id=np.array(1001), focals=np.array([0]), budgets=np.array([3.]), cpas=np.array([1.]))

    def settle_events(self, fi, tick, action, remaining):
        if remaining < 1:
            return Settlement(0., 0, 0, True, (), (), ())
        return Settlement(1., 1, 1, False, (tick,), (.2,), (tick,))


def fixture():
    config = dict(study=dict(actions=[0., .5, 1., 2., 3.]),
                  evaluation=dict(probability_bin_edges=[.1, .3]),
                  history=dict(calibration_prior_log_sd=.5, maturity_window=48))
    variant = dict(name='delay_corrected-test', estimator='delay_corrected', gain=.3, prior=8.)
    cdf = EmpiricalDelayCDF(tuple(np.linspace(.1, 1., 49)), 10)
    return config, variant, cdf


def test_identical_reference_trajectory_deterministic_posteriors_and_public_inputs(monkeypatch):
    config, variant, cdf = fixture()
    original = audit.PendingEstimator
    received = []

    class CheckedEstimator(original):
        def estimate(self, cohorts):
            assert isinstance(cohorts, tuple)
            assert all(isinstance(c, Cohort) and set(vars(c)) == {'age', 'public_probability', 'arrived_conversions', 'event_count'} for c in cohorts)
            assert all(c.age >= 0 and c.public_probability == .2 for c in cohorts)
            received.append(cohorts)
            return super().estimate(cohorts)

    monkeypatch.setattr(audit, 'PendingEstimator', CheckedEstimator)
    rows, actual = audit.audit_trajectory(TinyTape(), 0, 1., 'long', variant, cdf, config)
    repeated, repeated_actual = audit.audit_trajectory(TinyTape(), 0, 1., 'long', variant, cdf, config)
    assert rows == repeated
    assert actual == repeated_actual
    reference, _ = experiment.episode(TinyTape(), 0, 1., 'long', 11, variant, cdf, config)
    audit.assert_reference(actual, reference)
    assert len(rows) == len(audit.TICKS)*len(audit.METHODS) == 24
    assert received[0] == ()
    assert actual['cost'] == actual['conversions'] == actual['final_arrived'] == 3
    for tick in audit.TICKS:
        group = [r for r in rows if r['tick'] == tick]
        assert len({r['truth_u'] for r in group}) == 1
        assert len({r['method'] for r in group}) == 4
        for row in group:
            assert row['error'] == row['mean_u']-row['truth_u']
            if row['method'] == 'posterior':
                assert row['crps'] >= 0
                assert row['covered'] == int(row['interval_lower'] <= row['truth_u'] <= row['interval_upper'])
            else:
                assert row['crps'] is None
    bad = dict(reference, cost=reference['cost']+.01)
    with pytest.raises(RuntimeError, match='cost'):
        audit.assert_reference(actual, bad)
    summary = audit.aggregate(rows)
    assert len(summary) == 24
    assert all(r['decisions'] == 1 for r in summary)
