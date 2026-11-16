import copy
import json
from pathlib import Path

import numpy as np
import pytest

from trajectorylab.config import load_config
from trajectorylab.data import Normalizer
from trajectorylab.evaluation import SupportDiagnostic
from trajectorylab.report import aggregate_results, collect_results

ROOT = Path(__file__).parents[1]


def test_shipped_configs_validate():
    for path in (ROOT / "configs").glob("*.toml"):
        load_config(path)


def test_validation_and_test_seeds_cannot_overlap(tmp_path):
    original = (ROOT / "configs/smoke.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text(original.replace("test_seeds = [20000, 20001]", "test_seeds = [10000]"))
    with pytest.raises(ValueError, match="disjoint"):
        load_config(path)


def test_aggregation_unit_is_training_seed():
    runs = []
    for seed, value in enumerate([0.0, 10.0, 20.0]):
        runs.append(
            {
                "view": "uniform",
                "algorithm": "bc",
                "seed": seed,
                "wall_seconds": 1,
                "test": {
                    "return_mean": value,
                    "returns": [value] * 100,
                    "success_rate": None,
                    "support": {"distance_mean": 1},
                },
            }
        )
    result = aggregate_results(runs)[0]
    assert result["training_seeds"] == 3
    assert result["return_mean"] == 10
    assert result["return_std_across_training_seeds"] == 10
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_results(runs + [copy.deepcopy(runs[0])])


def test_incomplete_report_fails(tmp_path):
    with pytest.raises(ValueError, match="incomplete"):
        collect_results(tmp_path, {"jobs": [], "status": "failed", "expected_runs": 3})


def test_forged_summary_is_rejected(tmp_path):
    directory = tmp_path / "runs/a"
    directory.mkdir(parents=True)
    result = {
        "run_id": "a",
        "view": "uniform",
        "algorithm": "bc",
        "seed": 0,
        "updates": 1,
        "test": {"seeds": [2], "returns": [1.0], "return_mean": 9.0},
    }
    (directory / "result.json").write_text(json.dumps(result))
    manifest = {
        "status": "complete",
        "expected_runs": 1,
        "jobs": [
            {"id": "a", "view": "uniform", "algorithm": "bc", "seed": 0, "status": "complete"}
        ],
        "config": {"training": {"steps": 1}, "evaluation": {"test_seeds": [2]}},
    }
    with pytest.raises(ValueError, match="summary"):
        collect_results(tmp_path, manifest)


def test_support_distance_is_finite_and_sensitive_to_shift():
    rng = np.random.default_rng(0)
    train, validation = rng.normal(size=(100, 3)), rng.normal(size=(40, 3))
    diagnostic = SupportDiagnostic(train, validation, Normalizer.fit(train), 0)
    normal = diagnostic.summarize(validation)
    shifted = diagnostic.summarize(validation + 100)
    assert np.isfinite(normal["distance_mean"])
    assert shifted["distance_mean"] > normal["distance_mean"]
