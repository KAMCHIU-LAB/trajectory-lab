"""Public inputs, split separation, market pairing and frozen experiment gates."""

from copy import deepcopy
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from trajectorylab import bidding_expert_experiment as runner
from trajectorylab.bidding_expert_selection import FEATURE_NAMES
from trajectorylab.bidding_pending_response import Cohort
from trajectorylab.utils import fingerprint, read_json, write_json


@pytest.fixture
def config():
    return {
        "study": {
            "train_markets": [1, 2, 3],
            "validation_markets": [11, 12, 13, 14, 15, 16],
            "confirmation_markets": list(range(21, 45)),
            "focal_agents": [0, 8],
            "pv_count": 100,
            "actions": [0.0, 0.5, 1.0, 1.25, 1.5, 2.0, 3.0],
            "biases": [0.75, 1.0, 1.5],
        },
        "selection": {
            "regime": "immediate",
            "arrival_seed": 11,
            "prefix": 16,
            "prefix_gain": 0.3,
            "prefix_prior": 8.0,
            "response_seed": 11,
            "ridge_alphas": [1.0, 10.0, 100.0],
            "rejection_thresholds": [0.0, 0.02],
        },
        "baselines": {
            "feedback_gains": [0.3],
            "feedback_priors": [8.0],
            "fixed_multipliers": [],
            "calibration_bases": [],
            "calibration_priors": [],
            "threshold_ratios": [],
        },
        "evaluation": {
            "bootstrap_seed": 2401,
            "bootstrap_draws": 128,
            "minimum_relative_gain": 0.03,
            "score_protection": 0.99,
            "conversion_protection": 0.98,
            "violation_delta_protection": 0.01,
            "alpha": 0.05,
            "power": 0.8,
        },
        "limits": {"confirmation_maximum_markets": 24, "torch_threads": 1, "peak_rss_gib": 10},
    }


def outcome(value, *, violation=0):
    return dict(
        score=float(value),
        conversions=float(value),
        cost=float(value) * 0.8,
        budget=500.0,
        target_cpa=1.0,
        violation=violation,
        hard_budget_violation=0,
        final_arrived=float(value),
    )


def records(config, split, *, response_gain=20.0, feature_offset=0.0):
    result = []
    markets = config["study"][f"{split}_markets"]
    for mi, market in enumerate(markets):
        gain = response_gain[mi] if isinstance(response_gain, list) else response_gain
        for focal, group in product(config["study"]["focal_agents"], runner.GROUPS.values()):
            outputs = {
                "prefix-feedback": outcome(100, violation=0.2),
                "prefix-response": outcome(100 + gain, violation=0.2 - (mi + 1) / 10000),
            }
            if split != "train":
                outputs.update(
                    {
                        "feedback-g0.3-p8": outcome(100, violation=0.2),
                        "full-response": outcome(100 + gain),
                        "mixture-50": outcome(100),
                        "hindsight-two-experts": outcome(100 + max(0, gain)),
                    }
                )
            result.append(
                dict(
                    market=market,
                    focal=focal,
                    bias_group=group,
                    split=split,
                    feature_names=list(FEATURE_NAMES),
                    features=[feature_offset + mi + focal / 8] + [0.0] * 27,
                    prefix_digest=f"public-{market}-{focal}-{group}",
                    prefix_summary={},
                    target=gain / 500,
                    outcomes=outputs,
                )
            )
    return result


def fake_freeze(monkeypatch, output):
    def freeze(config, destination):
        assert destination == output
        value = {"identity": "frozen-current"}
        write_json(destination / "experiment-freeze.json", value)
        return value

    monkeypatch.setattr(runner, "freeze", freeze)


def save_selection(config, output):
    value = dict(
        candidate={
            "name": "ridge-a1-t0",
            "threshold": 0.0,
            "model": {"mean": [0.0] * 28, "std": [1.0] * 28, "coef": [0.0] * 28, "intercept": 0.04},
        },
        reference="feedback-g0.3-p8",
        power={"required_markets": 24},
        identity="current",
        confirmation_markets=config["study"]["confirmation_markets"].copy(),
    )
    value["selection_identity"] = fingerprint(value)
    write_json(output / "selection-freeze.json", value)
    return value


def save_result(config, output, split):
    market = config["study"][f"{split}_markets"][0]
    value = dict(
        identity="current",
        market=market,
        split=split,
        records=[r for r in records(config, split) if r["market"] == market],
        tape_sha256="input",
        seconds=0.01,
        peak_rss_gib=0.01,
    )
    value["result_digest"] = fingerprint(value)
    write_json(output / split / f"{market}.json", value)
    return value


@pytest.mark.parametrize("base", ["fixed-1.25", "feedback-g0.3-p8"])
@pytest.mark.parametrize("prior", [3.0, 8.0])
def test_calibration_uses_only_arrived_labels_and_accepted_public_probability(
    monkeypatch, config, base, prior
):
    hidden_bias = object()
    environments = []

    class PublicOnlyEnv:
        def __init__(self, tape, fi, bias, regime, seed, *, actions):
            assert bias is hidden_bias and regime == "immediate"
            self.actions, self.tick, self.chosen = np.array(actions), 0, []
            environments.append(self)

        def __getattribute__(self, name):
            if name in {"bias", "_bias", "truth", "_truth", "tape", "_tape"}:
                pytest.fail(f"Calibration accessed private environment information: {name}")
            return object.__getattribute__(self, name)

        def public_view(self):
            return SimpleNamespace(
                cohorts=(Cohort(0, 0.25, self.tick // 5, self.tick),),
                arrived=self.tick // 5,
                budget=500,
                cpa=1,
            )

        def observation(self):
            return np.zeros(13, dtype=np.float32)

        def run(self, chooser, *, stop_tick):
            assert stop_tick == self.tick + 1
            action, diagnostic = chooser([], [], [], 500, 1)
            assert diagnostic == {}
            self.chosen.append(action)
            self.tick = stop_tick

        def finish(self):
            return {"actions": self.chosen.copy()}

    monkeypatch.setattr(runner, "ExpertSelectionEnv", PublicOnlyEnv)
    monkeypatch.setattr(runner, "feedback_action", lambda *_: 1.25)
    row = runner.calibrated_episode(object(), 0, hidden_bias, config, base, prior)
    grid = environments[0].actions
    expected = []
    for tick in range(48):
        ratio = np.clip((tick // 5 + prior) / (tick * 0.25 + prior), 0.5, 2.0)
        expected.append(float(grid[np.argmin(abs(grid - 1.25 * ratio))]))
    assert row["actions"] == expected
    assert len(row["actions"]) == 48


def test_evaluate_market_target_is_full_cycle_paired_advantage(monkeypatch, config):
    tape = SimpleNamespace(arrays={"market_id": np.array(1), "focals": np.array([0, 8])})
    calls = []

    def paired(*args, **kwargs):
        calls.append((args, kwargs))
        return {
            "feedback": {**outcome(100), "suffix_score": 9000},
            "response": {**outcome(120), "suffix_score": 0},
            "features": [0.0] * 28,
            "feature_names": list(FEATURE_NAMES),
            "prefix_digest": "public-prefix",
            "prefix_summary": {"calibration_ratio": 1},
        }

    monkeypatch.setattr(runner, "paired_continuations", paired)
    rows = runner.evaluate_market(tape, config, "train", object())
    assert len(rows) == len(calls) == 6
    assert {r["target"] for r in rows} == {20 / 500}
    assert all(set(r["outcomes"]) == {"prefix-feedback", "prefix-response"} for r in rows)
    assert {args[2] for args, _ in calls} == {0.75, 1.0, 1.5}
    assert all(args[5] == 16 for args, _ in calls)
    rows[0]["target"] = (0 - 9000) / 500
    with pytest.raises(RuntimeError, match="full-cycle paired advantage"):
        runner.validate_records(rows, config, [1], "train")


def test_training_normalizer_is_fitted_separately_from_development_and_each_held_market(
    monkeypatch, tmp_path, config
):
    train = records(config, "train")
    dev = records(config, "validation", feature_offset=10000)
    fake_freeze(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "read_records",
        lambda output, split, cfg: {"train": train, "validation": dev}[split],
    )
    monkeypatch.setattr(runner, "planned_markets", lambda *_: {"required_markets": 24})
    original = runner.fit_ridge
    fitted_markets = []

    def fit(rows, alpha):
        fitted_markets.append({r["market"] for r in rows})
        return original(rows, alpha)

    monkeypatch.setattr(runner, "fit_ridge", fit)
    summary = runner.develop(config, tmp_path)
    expected_mean = np.mean([r["features"] for r in train], axis=0)
    assert summary["status"] == "READY_FOR_CONFIRMATION"
    assert len(summary["candidate_results"]) == 6
    for candidate in summary["candidate_results"]:
        np.testing.assert_array_equal(candidate["model"]["mean"], expected_mean)
        assert candidate["model"]["std"][1:] == [1e-8] * 27
    assert len(fitted_markets) == 3 * (1 + len(config["study"]["train_markets"]))
    assert all(markets <= set(config["study"]["train_markets"]) for markets in fitted_markets)
    for held in config["study"]["train_markets"]:
        assert set(config["study"]["train_markets"]) - {held} in fitted_markets
    selected = read_json(tmp_path / "selection-freeze.json")
    assert selected["candidate"]["name"] == summary["selected"]
    assert selected["reference"] == "feedback-g0.3-p8"


def test_market_values_equal_weight_markets_instead_of_campaigns():
    rows = []
    for market, campaign_scores in ((1, [100.0] * 5), (2, [1000.0])):
        for score_value in campaign_scores:
            rows.append(
                {
                    "market": market,
                    "bias_group": "under",
                    "outcomes": {"policy": outcome(score_value)},
                }
            )
    values = runner.market_values(rows, "policy")
    np.testing.assert_array_equal(values, [100, 1000])
    assert values.mean() == 550
    assert values.mean() != np.mean([r["outcomes"]["policy"]["score"] for r in rows])


def test_bootstrap_resamples_paired_market_rows_with_shared_indices(config):
    rows = []
    for market, reference in enumerate((1.0, 3.0, 100.0, 7.0), start=1):
        for group in runner.GROUPS.values():
            for _ in range(market):
                rows.append(
                    dict(
                        market=market,
                        bias_group=group,
                        outcomes={
                            "reference": outcome(reference, violation=0.2),
                            "candidate": outcome(reference * 2, violation=0.3),
                        },
                    )
                )
    result = runner.compare(rows, "candidate", "reference", config, draws=1000)
    for stats in result.values():
        for metric in ("score", "conversions"):
            assert stats[metric]["effect"] == pytest.approx(1)
            assert stats[metric]["ci95"] == pytest.approx([1, 1])
        assert stats["violation"]["effect"] == pytest.approx(0.1)
        assert stats["violation"]["ci95"] == pytest.approx([0.1, 0.1])


def test_holm_nine_protections_have_pass_and_failure_cases(config):
    rows = records(config, "validation", response_gain=[19, 20, 21, 22, 23, 24])
    passed = runner.holm_protections(rows, "prefix-response", "prefix-feedback", config)
    assert len(passed) == 9
    assert all(check["pass"] for check in passed)
    failed_rows = deepcopy(rows)
    for row in failed_rows:
        if row["bias_group"] == "over":
            row["outcomes"]["prefix-response"]["conversions"] = 90
    failed = runner.holm_protections(failed_rows, "prefix-response", "prefix-feedback", config)
    failed_checks = [check for check in failed if not check["pass"]]
    assert [(check["group"], check["metric"]) for check in failed_checks] == [
        ("over", "conversions")
    ]


def test_holm_controls_family_of_nine_instead_of_nine_uncorrected_tests(monkeypatch, config):
    monkeypatch.setattr(runner, "student_t", SimpleNamespace(sf=lambda *_: 0.006))
    rows = records(config, "validation", response_gain=[1, 2, 3, 4, 5, 6])
    for row in rows:
        row["outcomes"]["prefix-response"]["violation"] = (row["market"] - 11) / 100
    checks = runner.holm_protections(rows, "prefix-response", "prefix-feedback", config)
    assert len(checks) == 9
    assert all(check["p_one_sided"] == 0.006 for check in checks)
    assert all(check["holm_adjusted_p"] == pytest.approx(0.054) for check in checks)
    assert not any(check["pass"] for check in checks)


def test_planning_rejects_nonpositive_margin_and_high_variance(config):
    weak = records(config, "validation", response_gain=-1)
    assert (
        runner.planned_markets(weak, "prefix-response", "prefix-feedback", config)[
            "required_markets"
        ]
        is None
    )
    noisy = records(config, "validation", response_gain=[-40, -30, -20, 20, 40, 90])
    plan = runner.planned_markets(noisy, "prefix-response", "prefix-feedback", config)
    assert plan["required_markets"] > config["limits"]["confirmation_maximum_markets"]
    assert len(plan["endpoints"]) == 10  # Primary plus nine protection endpoints.
    assert all(endpoint["variance_multiplier"] == 2 for endpoint in plan["endpoints"])


def test_insufficient_power_is_a_persisted_stop_without_confirmation_freeze(
    monkeypatch, tmp_path, config
):
    train = records(config, "train")
    dev = records(config, "validation", response_gain=[-40, -30, -20, 20, 40, 90])
    fake_freeze(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner,
        "read_records",
        lambda output, split, cfg: {"train": train, "validation": dev}[split],
    )
    summary = runner.develop(config, tmp_path)
    assert summary["status"] == "STOP_INSUFFICIENT_BUDGET"
    assert summary["power"]["required_markets"] > 24
    assert not (tmp_path / "selection-freeze.json").exists()
    assert read_json(tmp_path / "development.json")["status"] == "STOP_INSUFFICIENT_BUDGET"


def test_no_oracle_headroom_completes_negative_result_before_reading_training(
    monkeypatch, tmp_path, config
):
    dev = records(config, "validation", response_gain=-10)
    calls = []
    fake_freeze(monkeypatch, tmp_path)

    def read(output, split, cfg):
        calls.append(split)
        assert split == "validation"
        return dev

    monkeypatch.setattr(runner, "read_records", read)
    monkeypatch.setattr(
        runner, "fit_ridge", lambda *_: pytest.fail("Oracle stop must precede model fitting")
    )
    summary = runner.develop(config, tmp_path)
    assert calls == ["validation"]
    assert summary["status"] == "STOP_NO_HEADROOM"
    assert summary["candidate_results"] == []
    assert read_json(tmp_path / "development.json") == summary
    assert not (tmp_path / "selection-freeze.json").exists()


@pytest.mark.parametrize("drift", ["config", "source", "protocol"])
def test_freeze_mismatch_blocks_resume_before_weights_or_market_open(
    monkeypatch, tmp_path, config, drift
):
    root, output = tmp_path / "repo", tmp_path / "output"
    (root / "docs").mkdir(parents=True)
    protocol = root / "docs/bidding-expert-selection-protocol.md"
    protocol.write_text("initial protocol")
    state = {"source": "v1"}
    monkeypatch.setattr(runner, "ROOT", root)
    monkeypatch.setattr(
        runner, "source_identity", lambda cfg: {"config": deepcopy(cfg), "source": state["source"]}
    )
    initial = runner.freeze(config, output)
    assert runner.freeze(config, output) == initial
    if drift == "config":
        config["selection"]["prefix"] = 8
    elif drift == "source":
        state["source"] = "v2"
    else:
        protocol.write_text("changed protocol")
    monkeypatch.setattr(
        runner,
        "PublicResponse",
        lambda **_: pytest.fail("Frozen mismatch must block model opening"),
    )
    with pytest.raises(RuntimeError, match="identity changed"):
        runner.evaluate(config, output, "train")


def test_cached_market_identity_mismatch_is_rejected(monkeypatch, tmp_path, config):
    config["study"]["train_markets"] = [1]
    write_json(tmp_path / "train/1.json", {"identity": "stale"})
    monkeypatch.setattr(runner, "freeze", lambda *_: {"identity": "current"})
    monkeypatch.setattr(runner, "PublicResponse", lambda **_: object())
    monkeypatch.setattr(runner, "seed_everything", lambda *_: None)
    monkeypatch.setattr(runner, "attest_market", lambda *_: {"tape_sha256": "input"})
    with pytest.raises(RuntimeError, match="identity"):
        runner.evaluate(config, tmp_path, "train")


def test_read_records_rejects_stale_experiment_identity(monkeypatch, tmp_path, config):
    config["study"]["train_markets"] = [1]
    write_json(
        tmp_path / "train/1.json",
        {"identity": "stale", "market": 1, "records": records(config, "train")},
    )
    monkeypatch.setattr(runner, "freeze", lambda *_: {"identity": "current"})
    monkeypatch.setattr(runner, "attest_market", lambda *_: {"tape_sha256": "input"})
    with pytest.raises(RuntimeError, match="identity"):
        runner.read_records(tmp_path, "train", config)


@pytest.mark.parametrize(
    "change", ["identity", "market", "split", "row-market", "row-split", "input", "digest"]
)
def test_cached_result_binds_experiment_market_split_input_and_payload(
    monkeypatch, tmp_path, config, change
):
    config["study"]["train_markets"] = [1]
    value = save_result(config, tmp_path, "train")
    monkeypatch.setattr(runner, "freeze", lambda *_: {"identity": "current"})
    monkeypatch.setattr(runner, "attest_market", lambda *_: {"tape_sha256": "input"})
    assert runner.read_records(tmp_path, "train", config) == value["records"]
    if change == "identity":
        value["identity"] = "other-experiment"
    elif change == "market":
        value["market"] = 2
    elif change == "split":
        value["split"] = "confirmation"
    elif change == "row-market":
        value["records"][0]["market"] = 2
    elif change == "row-split":
        value["records"][0]["split"] = "confirmation"
    elif change == "input":
        value["tape_sha256"] = "other-input"
    else:
        value["records"][0]["features"][0] += 1
    if change != "digest":
        value["result_digest"] = fingerprint(
            {k: v for k, v in value.items() if k != "result_digest"}
        )
    write_json(tmp_path / "train/1.json", value)
    with pytest.raises(RuntimeError, match="provenance, digest, split, or input identity"):
        runner.read_records(tmp_path, "train", config)


def test_confirmation_requires_explicit_immutable_opening(monkeypatch, tmp_path, config):
    monkeypatch.setattr(runner, "freeze", lambda *_: {"identity": "current"})
    selection = save_selection(config, tmp_path)
    with pytest.raises(RuntimeError, match="Open confirmation"):
        runner.open_confirmation(config, tmp_path)
    assert not (tmp_path / "confirmation-open.json").exists()
    seal = runner.open_confirmation(config, tmp_path, create=True)
    assert seal == {
        "experiment_identity": "current",
        "selection_identity": selection["selection_identity"],
        "markets": config["study"]["confirmation_markets"],
    }
    assert runner.open_confirmation(config, tmp_path) == seal
    selection["candidate"]["threshold"] = 0.02
    selection["selection_identity"] = fingerprint(
        {k: v for k, v in selection.items() if k != "selection_identity"}
    )
    write_json(tmp_path / "selection-freeze.json", selection)
    with pytest.raises(RuntimeError, match="different immutable selection"):
        runner.open_confirmation(config, tmp_path, create=True)
    assert read_json(tmp_path / "confirmation-open.json") == seal


@pytest.mark.parametrize("change", ["candidate", "experiment", "markets"])
def test_confirmation_rejects_changed_selection_before_weights_or_inputs(
    monkeypatch, tmp_path, config, change
):
    monkeypatch.setattr(runner, "freeze", lambda *_: {"identity": "current"})
    selection = save_selection(config, tmp_path)
    if change == "candidate":
        selection["candidate"]["model"]["intercept"] = 999
    elif change == "experiment":
        selection["identity"] = "other-experiment"
    else:
        selection["confirmation_markets"].pop()
    if change != "candidate":
        selection["selection_identity"] = fingerprint(
            {k: v for k, v in selection.items() if k != "selection_identity"}
        )
    write_json(tmp_path / "selection-freeze.json", selection)
    monkeypatch.setattr(
        runner,
        "PublicResponse",
        lambda **_: pytest.fail("Selection drift must block model loading"),
    )
    monkeypatch.setattr(
        runner, "attest_market", lambda *_: pytest.fail("Selection drift must block market opening")
    )
    with pytest.raises(
        RuntimeError, match="Frozen selection identity, sources, or markets changed"
    ):
        runner.evaluate(config, tmp_path, "confirmation")


def test_confirmation_cache_must_belong_to_exact_opening_seal(monkeypatch, tmp_path, config):
    config["study"]["confirmation_markets"] = [21]
    monkeypatch.setattr(runner, "freeze", lambda *_: {"identity": "current"})
    monkeypatch.setattr(runner, "attest_market", lambda *_: {"tape_sha256": "input"})
    save_selection(config, tmp_path)
    seal = runner.open_confirmation(config, tmp_path, create=True)
    value = save_result(config, tmp_path, "confirmation")
    value["confirmation_seal"] = fingerprint(seal)
    value["result_digest"] = fingerprint({k: v for k, v in value.items() if k != "result_digest"})
    write_json(tmp_path / "confirmation/21.json", value)
    assert runner.read_records(tmp_path, "confirmation", config) == value["records"]
    value["confirmation_seal"] = "other-opening"
    value["result_digest"] = fingerprint({k: v for k, v in value.items() if k != "result_digest"})
    write_json(tmp_path / "confirmation/21.json", value)
    with pytest.raises(RuntimeError, match="frozen confirmation opening"):
        runner.read_records(tmp_path, "confirmation", config)


def test_confirmation_failure_persists_complete_negative_result(monkeypatch, tmp_path, config):
    monkeypatch.setattr(runner, "freeze", lambda *_: {"identity": "current"})
    save_selection(config, tmp_path)
    runner.open_confirmation(config, tmp_path, create=True)
    rows = records(config, "confirmation", response_gain=[1, 2] * 12)
    monkeypatch.setattr(runner, "read_records", lambda *_: rows)
    result = runner.confirm(config, tmp_path)
    assert result["status"] == "CONFIRM_FAILURE"
    assert len(result["protections"]) == 9
    assert read_json(tmp_path / "confirmation.json") == result
    assert runner.confirm(config, tmp_path) == result


@pytest.mark.parametrize("change", ["bytes", "sidecar", "study", "generator", "data-digest"])
def test_market_attestation_rejects_changed_input_or_generation_identity(
    monkeypatch, tmp_path, config, change
):
    generator = {"source": "generator-v1"}
    monkeypatch.setattr(runner, "event_source_identity", lambda: generator)
    data = dict(config=deepcopy(config), generator=deepcopy(generator), protocol_sha256="protocol")
    data["identity"] = fingerprint(data)
    write_json(tmp_path / "data-freeze.json", data)
    path = tmp_path / "markets/1.npz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"small local test fixture; never interpreted as a market")
    expected = fingerprint(
        {
            "study": {
                k: config["study"][k] for k in ("focal_agents", "pv_count", "actions", "biases")
            },
            "market": 1,
            "source": generator,
        }
    )
    write_json(path.with_suffix(".json"), {"market_id": 1, "identity": expected})
    first = runner.attest_market(config, tmp_path, 1)
    assert runner.attest_market(config, tmp_path, 1) == first
    if change == "bytes":
        path.write_bytes(b"different bytes")
    elif change == "sidecar":
        write_json(path.with_suffix(".json"), {"market_id": 1, "identity": "other-generator"})
    elif change == "study":
        config["study"]["pv_count"] += 1
    elif change == "generator":
        generator["source"] = "generator-v2"
    else:
        data["protocol_sha256"] = "changed-without-updating-digest"
        write_json(tmp_path / "data-freeze.json", data)
    with pytest.raises(RuntimeError, match="changed|identity mismatch|differs"):
        runner.attest_market(config, tmp_path, 1)


def test_all_records_missing_same_registered_baseline_are_rejected(config):
    rows = records(config, "validation")
    runner.validate_records(rows, config, config["study"]["validation_markets"], "validation")
    for row in rows:
        del row["outcomes"]["feedback-g0.3-p8"]
    with pytest.raises(RuntimeError, match="Method coverage mismatch"):
        runner.validate_records(rows, config, config["study"]["validation_markets"], "validation")


@pytest.mark.parametrize("epsilon", [0.0, 1e-14])
def test_zero_or_near_zero_variance_cannot_establish_protection_or_power(config, epsilon):
    rows = records(config, "validation", response_gain=[20 + i * epsilon for i in range(6)])
    for row in rows:
        row["outcomes"]["prefix-response"]["violation"] = 0.2 + (row["market"] - 11) * epsilon
    checks = runner.holm_protections(rows, "prefix-response", "prefix-feedback", config)
    assert all(check["p_one_sided"] == 1.0 for check in checks)
    assert not any(check["pass"] for check in checks)
    plan = runner.planned_markets(rows, "prefix-response", "prefix-feedback", config)
    assert plan["required_markets"] is None


@pytest.mark.parametrize("change", ["missing", "duplicate", "unregistered"])
def test_coverage_rejects_missing_duplicate_and_unregistered_campaigns(config, change):
    rows = records(config, "train")
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(deepcopy(rows[0]))
    else:
        rows[0]["focal"] = 999
    with pytest.raises(RuntimeError, match="Missing, duplicate, or unregistered"):
        runner.validate_records(rows, config, config["study"]["train_markets"], "train")


@pytest.mark.parametrize("change", ["method", "maturity", "budget", "feature", "target"])
def test_record_contract_rejects_broken_method_maturity_budget_and_target(config, change):
    rows = records(config, "train")
    row = rows[0]
    if change == "method":
        row["outcomes"]["extra"] = outcome(1)
    elif change == "maturity":
        row["outcomes"]["prefix-feedback"]["final_arrived"] += 1
    elif change == "budget":
        row["outcomes"]["prefix-response"]["hard_budget_violation"] = 1
    elif change == "feature":
        row["features"][0] = float("nan")
    else:
        row["target"] = -1
    with pytest.raises(RuntimeError):
        runner.validate_records(rows, config, config["study"]["train_markets"], "train")
