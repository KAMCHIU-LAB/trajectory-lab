#!/usr/bin/env python3
"""Source-only complete delayed-feedback replay, without a business success claim.

Run from an installed checkout, or with PYTHONPATH=src. Resume uses exact source,
configuration, mature history, market and selection identities; mismatches fail.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from trajectorylab import bidding_delayed_experiment as experiment
from trajectorylab.utils import fingerprint, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_record(path, record):
    """Create an immutable value, or verify an identical resumed invocation."""
    if path.exists():
        if read_json(path) != record:
            raise RuntimeError(f"Frozen smoke record changed: {path.name}")
    else:
        write_json(path, record)


def verify_history(config, output):
    manifest = read_json(output / "history-manifest.json")
    core = manifest["core"]
    if (
        manifest["identity"] != fingerprint(core)
        or core["source"] != experiment.source_record()
        or core["config_identity"] != fingerprint(config)
        or core["markets"] != experiment.cache_provenance(config, output, "train")
        or manifest["history_sha256"] != sha256(output / "history.json")
    ):
        raise RuntimeError("Frozen smoke training history changed")


def run(output, config_path=ROOT / "configs/bidding-delayed-smoke.toml"):
    started = time.perf_counter()
    output = Path(output)
    config = experiment.load_config(config_path)
    freeze_record(
        output / "smoke-manifest.json",
        dict(
            config=config,
            source=experiment.source_record(),
            script_sha256=sha256(__file__),
            scope="software replay only; one confirmation market provides no business evidence",
        ),
    )
    experiment.prepare(config, output, "train")
    experiment.prepare(config, output, "validation")
    if not (output / "history.json").exists():
        experiment.fit_history(config, output)
    verify_history(config, output)
    validation = experiment.evaluate(config, output, "validation")
    selection_path = output / "selection.json"
    if selection_path.exists():
        selection = experiment.check_frozen_selection(config, output)
    else:
        selection = experiment.select_development(config, output)
    lock = dict(selection_sha256=sha256(selection_path), selection_identity=selection["identity"])
    freeze_record(output / "smoke-selection-lock.json", lock)
    experiment.prepare(config, output, "confirmation")
    confirmation = experiment.evaluate(config, output, "confirmation", selection["frozen_variants"])
    summary = experiment.summarise(config, output, "confirmation")
    if sha256(selection_path) != lock["selection_sha256"]:
        raise RuntimeError("Selection changed during confirmation")
    comparisons = [summary["primary_comparison"], *summary["comparisons"]]
    if any(c["business_target_pass"] for c in comparisons):
        raise RuntimeError("Software smoke must not claim business success")
    evidence = dict(
        scope="software-reproducibility-only",
        business_target_pass=False,
        independent_confirmation_markets=summary["independent_market_count"],
        validation_rows=len(validation),
        confirmation_rows=len(confirmation),
        all_final_arrivals_mature=all(r["final_arrived"] == r["conversions"] for r in confirmation),
        all_hard_budgets_respected=all(r["cost"] <= r["budget"] + 1e-6 for r in confirmation),
        mature_training_positives={
            k: v["training_sample_count"] for k, v in read_json(output / "history.json").items()
        },
        selection_sha256=lock["selection_sha256"],
        elapsed_seconds=time.perf_counter() - started,
    )
    if not evidence["all_final_arrivals_mature"] or not evidence["all_hard_budgets_respected"]:
        raise RuntimeError("Smoke settlement invariants failed")
    write_json(output / "smoke-evidence.json", evidence)
    print(json.dumps(evidence, sort_keys=True, indent=2), flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/bidding/delayed-smoke")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bidding-delayed-smoke.toml")
    args = parser.parse_args()
    run(args.output, args.config)


if __name__ == "__main__":
    main()
