"""Verify the matched zero-temperature IQL/BC actor invariant on real checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajectorylab.experiment import load_checkpoint
from trajectorylab.report import collect_results
from trajectorylab.utils import read_json, write_json


def verify(reference: Path, ablation: Path) -> dict:
    left, right = [read_json(path / "suite.json") for path in (reference, ablation)]
    left_results, right_results = [
        collect_results(path, manifest) for path, manifest in ((reference, left), (ablation, right))
    ]
    if right["config"]["training"]["advantage_beta"] != 0:
        raise ValueError("The ablation must use advantage_beta = 0")
    keys = ("steps", "batch_size", "width", "learning_rate", "grad_clip", "threads", "device")
    for key in keys:
        if left["config"]["training"][key] != right["config"]["training"][key]:
            raise ValueError(f"Actor setting mismatch: {key}")
    if left["implementation_digest"] != right["implementation_digest"]:
        raise ValueError("Core implementation differs")
    if left["data_audit"]["source_digest"] != right["data_audit"]["source_digest"]:
        raise ValueError("Source data differs")
    comparisons = []
    for result in right_results:
        if result["algorithm"] != "iql":
            raise ValueError("The ablation suite must contain only IQL")
        matching = [
            r
            for r in left_results
            if r["algorithm"] == "bc" and (r["view"], r["seed"]) == (result["view"], result["seed"])
        ]
        if len(matching) != 1:
            raise ValueError("Each ablation needs exactly one matched BC run")
        baseline = matching[0]
        view = result["view"]
        if (
            left["data_audit"]["views"][view]["digest"]
            != right["data_audit"]["views"][view]["digest"]
        ):
            raise ValueError("Training trajectories differ")
        actors = [
            load_checkpoint(path / "runs" / item["run_id"] / "checkpoint.pt", item["run_id"])[
                "learner"
            ]["modules"]["actor"]
            for path, item in ((reference, baseline), (ablation, result))
        ]
        same_keys = actors[0].keys() == actors[1].keys()
        equal = same_keys and all(torch.equal(v, actors[1][key]) for key, v in actors[0].items())
        same_returns = (
            baseline["test"]["seeds"] == result["test"]["seeds"]
            and baseline["test"]["returns"] == result["test"]["returns"]
        )
        comparisons.append(
            {
                "view": view,
                "seed": result["seed"],
                "bc_run_id": baseline["run_id"],
                "iql_zero_beta_run_id": result["run_id"],
                "actor_parameters_bitwise_equal": equal,
                "same_test_seeds_and_exact_returns": same_returns,
            }
        )
    if not comparisons or not all(
        c["actor_parameters_bitwise_equal"] and c["same_test_seeds_and_exact_returns"]
        for c in comparisons
    ):
        raise ValueError("Zero-temperature IQL failed the BC actor invariant")
    return {
        "check": "zero-temperature IQL reproduces matched BC actor",
        "reference_suite": left["suite_id"],
        "ablation_suite": right["suite_id"],
        "status": "passed",
        "comparisons": comparisons,
        "interpretation": "An implementation invariant on these CPU checkpoints; not a novel algorithmic finding",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--ablation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = verify(args.reference, args.ablation)
    write_json(args.output, evidence)
    print(f"Passed {len(evidence['comparisons'])} matched actor checks: {args.output}")


if __name__ == "__main__":
    main()
