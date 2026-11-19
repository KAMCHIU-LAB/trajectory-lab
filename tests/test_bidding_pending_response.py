import numpy as np
import pytest

from trajectorylab.bidding_pending_response import Cohort, EmpiricalDelayCDF, PendingEstimator


def cdf():
    return EmpiricalDelayCDF.fit([0, 2], 2)


def test_hand_pending_formula_and_integer_samples():
    model = PendingEstimator(cdf(), scales=[1.])
    cohort = Cohort(0, .2, 1, 10)
    # 9 unreturned events, each .2*.5/(1-.2*.5) = 1/9.
    result = model.estimate([cohort])
    assert result.arrived == 1
    assert result.pending_mean == pytest.approx(1.)
    assert result.total_mean == pytest.approx(2.)
    samples = model.sample([cohort], 20000, 13)
    assert samples.dtype.kind == 'i'
    assert samples.min() >= 0 and samples.max() <= 9
    assert samples.mean() == pytest.approx(1., abs=.04)


def test_f_zero_one_zero_probability_and_no_arrivals():
    f = EmpiricalDelayCDF.fit([2, 2], 2)
    model = PendingEstimator(f, scales=[1.])
    assert model.estimate([Cohort(0, [.2, .2], 0)]).pending_mean == pytest.approx(.4)
    assert model.estimate([Cohort(2, [.2, .2], 1)]).pending_mean == 0
    assert model.estimate([Cohort(0, [0., 0.], 0)]).total_mean == 0
    assert model.estimate([]).total_mean == 0
    immediate = PendingEstimator(EmpiricalDelayCDF.fit([0], 2), scales=[1.])
    assert immediate.estimate([Cohort(0, [.2], 0)]).pending_mean == 0


def test_train_only_and_age_boundary():
    with pytest.raises(ValueError):
        EmpiricalDelayCDF.fit([0], 2, split='test')
    with pytest.raises(ValueError):
        EmpiricalDelayCDF.fit([0], 2, all_mature=False)
    with pytest.raises(ValueError):
        EmpiricalDelayCDF.fit([], 2)
    f = cdf()
    assert f(-1) == 0 and f(0) == .5 and f(1) == .5 and f(2) == f(100) == 1
    with pytest.raises(ValueError):
        PendingEstimator(f).estimate([Cohort(-1, [.2], 0)])


def test_mature_only_ignores_immature_calibration_evidence():
    model = PendingEstimator(cdf(), method='mature_only')
    mature = Cohort(2, .2, 2, 10)
    a = model.estimate([mature, Cohort(0, .2, 0, 10)])
    b = model.estimate([mature, Cohort(0, .2, 5, 10)])
    assert a.calibration_mean == b.calibration_mean
    assert a.pending_mean > b.pending_mean


def test_baselines_prior_positive_and_shared_pending_draws():
    cohorts = [Cohort(0, .2, 0, 10)]
    model = PendingEstimator(cdf())
    assert model.estimate(cohorts).pending_mean > 0
    assert PendingEstimator(cdf(), 'delay_corrected').estimate(cohorts).pending_mean > 0
    naive = PendingEstimator(cdf(), 'naive')
    assert naive.estimate(cohorts).pending_mean == 0
    assert np.all(naive.sample(cohorts, 10, 1) == 0)
    draws = model.sample_shared(cohorts, 100, 11, 12)
    assert np.all(draws == draws[:, :1])
    assert not draws.flags.writeable
    assert np.array_equal(model.sample(cohorts, 100, 12), draws[:, 0])


def test_invalid_public_inputs_and_impossible_observation():
    for cohort in [Cohort(0, [1.2], 0), Cohort(0, [.2], 2), Cohort(0, [.2], 0, 2)]:
        with pytest.raises(ValueError):
            PendingEstimator(cdf()).estimate([cohort])
    with pytest.raises(ValueError):
        PendingEstimator(cdf()).estimate([Cohort(0, [0.], 1)])
    with pytest.raises(ValueError):
        PendingEstimator(cdf(), scales=[0])


def test_optional_total_pseudocount_and_naive_zero_cdf():
    raw = EmpiricalDelayCDF.fit([2, 2], 2)
    smoothed = EmpiricalDelayCDF.fit([2, 2], 2, pseudocount=.5)
    assert smoothed(0) == pytest.approx((.5/3)/2.5)
    assert smoothed(1) == pytest.approx((1./3)/2.5)
    assert smoothed(2) == 1.
    cohort = Cohort(0, .2, 1, 10)
    naive = PendingEstimator(raw, 'naive')
    assert naive.estimate([cohort]).total_mean == 1
    assert np.all(naive.sample([cohort], 10, 1) == 0)
    assert PendingEstimator(smoothed).estimate([cohort]).pending_mean > 0
    with pytest.raises(ValueError):
        EmpiricalDelayCDF.fit([2], 2, pseudocount=-1)


def test_vectorized_matches_sequential_reference():
    cohorts = [Cohort(i % 3, .01 + (i % 11)*.02, i % 4, 30+i) for i in range(200)]
    for method in ['mature_only', 'delay_corrected', 'posterior']:
        model = PendingEstimator(cdf(), method)
        logw = model.log_prior.copy()
        pending = np.zeros_like(logw)
        for c in cohorts:
            n, p, o = c.summary()
            f = cdf()(c.age)
            q = np.minimum(model.scales*p, 1.)
            if method != 'mature_only' or c.age >= cdf().maturity_window:
                logw += o*np.log(q*f) + (n-o)*np.log1p(-q*f)
            pending += (n-o)*q*(1-f)/(1-q*f)
        weights = np.exp(logw-logw.max())
        weights /= weights.sum()
        if method != 'posterior':
            maximum = weights.argmax()
            weights[:] = 0
            weights[maximum] = 1
        result = model.estimate(cohorts)
        assert result.pending_mean == pytest.approx(weights @ pending, rel=1e-12)
        assert result.calibration_mean == pytest.approx(weights @ model.scales, rel=1e-12)
