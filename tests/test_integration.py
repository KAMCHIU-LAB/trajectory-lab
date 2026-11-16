"""Real simulator, optimizer, persistence and report checks without network access."""

from __future__ import annotations

import copy
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from trajectorylab import experiment
from trajectorylab.config import load_config
from trajectorylab.report import generate_report
from trajectorylab.utils import read_json


def tiny_config():
    config = load_config(Path(__file__).parents[1] / "configs/smoke.toml")
    config["data"].update(source_episodes=20, transition_budget=400)
    config["training"].update(
        steps=6,
        width=16,
        batch_size=8,
        context=4,
        log_every=1,
        eval_every=3,
        checkpoint_every=3,
    )
    config["evaluation"].update(validation_seeds=[10000], test_seeds=[20000])
    return config


def test_real_suite_report_checkpoint_and_completed_run_reuse(tmp_path, monkeypatch):
    config = tiny_config()
    directory = experiment.run_suite(config, tmp_path)
    manifest = read_json(directory / "suite.json")
    assert manifest["status"] == "complete"
    assert manifest["expected_runs"] == 3
    checkpoint_times = {}
    for job in manifest["jobs"]:
        checkpoint = directory / "runs" / job["id"] / "checkpoint.pt"
        checkpoint_times[job["id"]] = checkpoint.stat().st_mtime_ns
        result = read_json(checkpoint.with_name("result.json"))
        assert result["actor_parameter_delta_l2"] > 0
        replayed = experiment.evaluate_checkpoint(checkpoint, [20000])
        assert replayed["returns"] == result["test"]["returns"]

    report = generate_report(directory, tmp_path, tmp_path / "report")
    assert report.is_file()
    assert "3 / 3" in report.read_text()
    assert len(read_json(report.with_name("summary.json"))["results"]) == 3
    assert report.with_name("runs.csv").is_file()
    assert b"\r\n" not in report.with_name("runs.csv").read_bytes()
    assert len(list((report.parent / "figures").glob("*.png"))) == 4
    assert len(list((report.parent / "evidence").glob("*-example-trajectory.csv"))) == 3

    def unexpected_training(*args, **kwargs):
        raise AssertionError("Complete results must not be retrained")

    monkeypatch.setattr(experiment, "train_run", unexpected_training)
    assert experiment.run_suite(config, tmp_path) == directory
    for job in manifest["jobs"]:
        checkpoint = directory / "runs" / job["id"] / "checkpoint.pt"
        assert checkpoint.stat().st_mtime_ns == checkpoint_times[job["id"]]

    # Exercise the separate spawn-based CPU recovery driver on an absent result.
    job = manifest["jobs"][0]
    result_path = directory / "runs" / job["id"] / "result.json"
    expected_test = read_json(result_path)["test"]
    result_path.unlink()
    driver = Path(__file__).parents[1] / "scripts/complete_suite.py"
    subprocess.run(
        [sys.executable, str(driver), "--suite", str(directory), "--workers", "2"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert read_json(result_path)["test"] == expected_test
    assert read_json(directory / "suite.json")["execution_drivers"][-1]["workers"] == 2
    assert not (directory / "parallel-driver.lock").exists()


def test_interrupted_runner_resumes_to_identical_actor_and_final_returns(tmp_path, monkeypatch):
    config = tiny_config()
    config["suite"]["algorithms"] = ["iql"]
    reference = experiment.run_suite(copy.deepcopy(config), tmp_path / "reference")
    save = experiment.save_checkpoint
    interrupted = False

    def save_then_interrupt(path, state):
        nonlocal interrupted
        save(path, state)
        if state["step"] == 3 and not interrupted:
            interrupted = True
            raise RuntimeError("Simulated interruption after durable checkpoint")

    monkeypatch.setattr(experiment, "save_checkpoint", save_then_interrupt)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        experiment.run_suite(copy.deepcopy(config), tmp_path / "resumed")
    failed = next((tmp_path / "resumed/artifacts/suites").glob("*/suite.json"))
    assert read_json(failed)["status"] == "failed"
    with pytest.raises(ValueError, match="incomplete"):
        generate_report(failed.parent, tmp_path / "resumed", tmp_path / "bad-report")

    monkeypatch.setattr(experiment, "save_checkpoint", save)
    resumed = experiment.run_suite(config, tmp_path / "resumed")
    job_id = read_json(resumed / "suite.json")["jobs"][0]["id"]
    left, right = [
        experiment.load_checkpoint(directory / "runs" / job_id / "checkpoint.pt")
        for directory in (reference, resumed)
    ]
    for name, values in left["learner"]["modules"].items():
        assert all(
            torch.equal(value, right["learner"]["modules"][name][key])
            for key, value in values.items()
        )
    results = [read_json(p / "runs" / job_id / "result.json") for p in (reference, resumed)]
    assert results[0]["test"] == results[1]["test"]


def test_checkpoint_ablation_verifier_accepts_matched_actors_and_rejects_wrong_beta(tmp_path):
    config = tiny_config()
    config["suite"]["algorithms"] = ["bc"]
    reference = experiment.run_suite(copy.deepcopy(config), tmp_path)
    config["suite"]["algorithms"] = ["iql"]
    config["training"]["advantage_beta"] = 0.0
    ablation = experiment.run_suite(config, tmp_path)
    output = tmp_path / "check.json"
    driver = Path(__file__).parents[1] / "scripts/verify_ablation.py"
    command = [
        sys.executable,
        str(driver),
        "--reference",
        str(reference),
        "--ablation",
        str(ablation),
        "--output",
        str(output),
    ]
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, timeout=30)
    assert read_json(output)["status"] == "passed"
    manifest = read_json(ablation / "suite.json")
    manifest["config"]["training"]["advantage_beta"] = 3.0
    from trajectorylab.utils import write_json

    write_json(ablation / "suite.json", manifest)
    assert subprocess.run(command, cwd=tmp_path, capture_output=True, timeout=30).returncode != 0
