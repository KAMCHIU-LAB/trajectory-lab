import pytest

from trajectorylab.bidding_delayed_statistics import paired_comparison, validate_rows


def records(multiplier=1.1):
    rows = []
    for market in range(4):
        for group in ("nominal", "under", "over"):
            for method, factor in (("base", 1), ("candidate", multiplier)):
                rows.append(
                    dict(
                        market=market,
                        focal=0,
                        bias_group=group,
                        delay="short",
                        arrival_seed=11,
                        method=method,
                        score=(market + 1) * factor,
                        conversions=(market + 1) * factor,
                        cost=2,
                        budget=10,
                        budget_utilization=0.2,
                        violation=0,
                    )
                )
    return rows


def test_pairing_constant_ratio_and_protection():
    rows = records()
    assert validate_rows(rows)["paired_campaigns"] == 12
    result = paired_comparison(rows, "candidate", "base", draws=1000, minimum_markets=4)
    assert result["business_target_pass"]
    assert result["groups"]["all"]["score_ratio_ci95"] == pytest.approx([1.1, 1.1])


def test_small_development_set_cannot_pass_confirmation():
    result = paired_comparison(records(), "candidate", "base", draws=1000)
    assert result["uplift_target_pass"]
    assert not result["sufficient_independent_markets"]
    assert not result["business_target_pass"]


def test_missing_pair_and_duplicates_rejected():
    rows = records()
    with pytest.raises(ValueError, match="coverage"):
        validate_rows(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows(rows + rows[:1])


def test_zero_denominator_is_not_epsilon_uplift():
    rows = records()
    for row in rows:
        if row["method"] == "base":
            row["score"] = 0
    result = paired_comparison(rows, "candidate", "base", draws=100)
    assert not result["business_target_pass"]
    assert result["groups"]["all"]["status"] == "incomparable_zero_baseline"
