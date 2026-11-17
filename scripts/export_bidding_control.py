"""Validate and export an exact allowlist of completed controller evidence."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import io
import json
import math
import os
import re
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_FILES = ("manifest.json", "selection.json", "status.json", "empirical-rates.json")
REPORT_FILES = ("manifest.json", "results.csv", "results.json", "summary.json", "report.html")
PRIVACY_PATTERNS = (
    r"/Users/",
    r"/home/",
    r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
)


def digest(content):
    return hashlib.sha256(content).hexdigest()


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Nonfinite evidence cannot be exported")
    if isinstance(value, str) and (
        Path(value).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise RuntimeError("Absolute filesystem paths cannot be exported")
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def decode_json(content):
    value = json.loads(content)
    finite(value)
    return value


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".export.tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export(source: Path, destination: Path) -> None:
    source, destination = source.resolve(), destination.resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise RuntimeError("Export destination must be separate from the run directory")
    lock = (source / ".run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)

    def read(name):
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise RuntimeError("Export input must be a regular file inside the run")
        return path.read_bytes()

    status = decode_json(read("status.json"))
    phase = status["phase"]
    if phase not in {"complete", "complete-not-promoted"}:
        raise RuntimeError("Only completed controller evidence can be exported")
    names = list(BASE_FILES) + [f"validation-report/{name}" for name in REPORT_FILES]
    if phase == "complete":
        names += [name for name in REPORT_FILES if name != "manifest.json"]
    payload = {name: read(name) for name in names}
    if decode_json(payload["status.json"]) != status:
        raise RuntimeError("Run status changed while collecting evidence")
    manifest = decode_json(payload["manifest.json"])
    selection = decode_json(payload["selection.json"])
    config, layout = manifest["config"], manifest["evidence_layout"]
    identity_content = {
        "config": config,
        "main": manifest["main_identity"],
        "implementation": manifest["implementation_sha256"],
        "checkpoints": manifest["checkpoint_sha256"],
        "main_selection_sha256": manifest["main_selection_sha256"],
    }
    identity = digest(json.dumps(identity_content, sort_keys=True, allow_nan=False).encode())
    if identity != manifest["identity"]:
        raise RuntimeError("Controller identity does not match its declared inputs")
    if (
        digest((ROOT / "src/trajectorylab/bidding_control.py").read_bytes())
        != manifest["implementation_sha256"]
    ):
        raise RuntimeError("Controller implementation differs from the completed run")
    snapshot = (
        json.dumps(manifest["main_selection"], indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()
    if digest(snapshot) != manifest["main_selection_sha256"]:
        raise RuntimeError("Main selection snapshot does not match its declared hash")
    if set(manifest["checkpoint_sha256"]) != {str(seed) for seed in layout["training_seeds"]}:
        raise RuntimeError("Checkpoint identities do not cover declared training seeds")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in manifest["checkpoint_sha256"].values()
    ):
        raise RuntimeError("Invalid checkpoint hash")
    if layout["confirmation_markets"] != config["confirmation_markets"]:
        raise RuntimeError("Confirmation markets differ from the configuration")
    for field in (
        "validation_markets",
        "confirmation_markets",
        "focal_agents",
        "training_seeds",
        "scenarios",
    ):
        if not layout[field] or len(set(layout[field])) != len(layout[field]):
            raise RuntimeError("Evidence layout must contain nonempty unique dimensions")
    if set(layout["validation_markets"]) & set(layout["confirmation_markets"]):
        raise RuntimeError("Validation and confirmation evidence overlap")
    if (
        selection["primary_scenario"] != config["primary_scenario"]
        or config["primary_scenario"] not in layout["scenarios"]
    ):
        raise RuntimeError("Primary scenario differs from the declared evaluation")
    if bool(selection["promoted"]) != (phase == "complete"):
        raise RuntimeError("Completion phase contradicts the promotion decision")
    rates = decode_json(payload["empirical-rates.json"])
    if rates["fit_split"] != "training only" or len(rates["rates"]) != 2:
        raise RuntimeError("Empirical forecast provenance is inconsistent")
    if (
        not rates["rates"][0]
        or len(rates["rates"][0]) != len(rates["rates"][1])
        or any(v < 0 for row in rates["rates"] for v in row)
    ):
        raise RuntimeError("Invalid empirical forecast rates")

    # This import is read-only and occurs after verifying the frozen controller source.
    from trajectorylab.bidding_control import metrics, paired

    selected_methods = {
        selection[key]
        for key in ("candidate", "reference", "primary_reference", "matched_empirical")
    }
    best = selection["best_rate_index"]
    calibration = selection["selected_calibration"]
    candidate_modes = {
        f"response-paced-{best}": "none",
        "response-paced-cumulative": "cumulative",
        "response-paced-discounted": "discounted",
        "response-paced-budget-only": "budget-only",
    }
    if (
        not isinstance(best, int)
        or not 0 <= best < len(config["dual_rates"])
        or selection["best_rate"] != config["dual_rates"][best]
        or candidate_modes.get(selection["candidate"]) != calibration
        or selection["matched_empirical"] != f"empirical-paced-{best}-{calibration}"
        or selection["promotion_rule"] != config["promotion"]
    ):
        raise RuntimeError("Selected controller parameters differ from their declaration")
    counts = {}
    for prefix, split in (("validation-report/", "validation"), ("", "confirmation")):
        if split == "confirmation" and phase != "complete":
            continue
        rows = decode_json(payload[prefix + "results.json"])
        summary = decode_json(payload[prefix + "summary.json"])
        policies = [tuple(pair) for pair in layout[f"{split}_policies"]]
        if not policies or len(set(policies)) != len(policies):
            raise RuntimeError("Declared policies must be nonempty and unique")
        methods = {method for method, _ in policies}
        if not selected_methods <= methods or "response" not in methods:
            raise RuntimeError("Selected methods are absent from declared evidence")
        for method in methods:
            seeds = {seed for name, seed in policies if name == method}
            expected_seeds = (
                set(layout["training_seeds"]) if method.startswith("response") else {-1}
            )
            if seeds != expected_seeds:
                raise RuntimeError("A method is missing its declared training seeds")
        expected = {
            (market, focal, scenario, method, seed)
            for market in layout[f"{split}_markets"]
            for focal in layout["focal_agents"]
            for scenario in layout["scenarios"]
            for method, seed in policies
        }
        identities = [
            (r["market"], r["focal"], r["scenario"], r["method"], r["seed"]) for r in rows
        ]
        if set(identities) != expected or len(identities) != len(expected):
            raise RuntimeError("Rows do not exactly cover declared evaluation jobs")
        for row in rows:
            for field in (
                "score",
                "conversions",
                "violation",
                "cost",
                "budget_utilization",
                "zero_conversions",
            ):
                if isinstance(row[field], bool) or not isinstance(row[field], (int, float)):
                    raise RuntimeError("Row metric is not numeric")
        recomputed = [
            {"scenario": scenario, "method": method, **metrics(rows, method, scenario)}
            for scenario in sorted(layout["scenarios"])
            for method in sorted(methods)
        ]
        if summary["groups"] != recomputed or summary["selection"] != selection:
            raise RuntimeError("Summary is inconsistent with raw evidence or selection")
        confirmation = summary["confirmation"]
        if split == "validation":
            if (
                confirmation is not None
                or decode_json(payload[prefix + "manifest.json"]) != manifest
            ):
                raise RuntimeError("Validation report identity or phase is inconsistent")
            for field, method, scenario in (
                ("candidate_metrics", selection["candidate"], None),
                ("reference_metrics", selection["reference"], None),
                ("original_response_metrics", "response", None),
                ("primary_candidate_metrics", selection["candidate"], config["primary_scenario"]),
                (
                    "primary_reference_metrics",
                    selection["primary_reference"],
                    config["primary_scenario"],
                ),
                ("primary_original_response_metrics", "response", config["primary_scenario"]),
            ):
                if selection[field] != metrics(rows, method, scenario):
                    raise RuntimeError("Selection metrics differ from raw validation evidence")
        elif (
            confirmation["method"] != selection["candidate"]
            or confirmation["reference"] != selection["primary_reference"]
            or confirmation["scenario"] != config["primary_scenario"]
            or confirmation["markets"] != len(layout["confirmation_markets"])
            or confirmation["model_contribution_diagnostic"]["reference"]
            != selection["matched_empirical"]
        ):
            raise RuntimeError("Primary confirmation does not match selection")
        if split == "confirmation":
            limits = config["confirmation"]
            effect = paired(
                rows,
                selection["candidate"],
                selection["primary_reference"],
                config["primary_scenario"],
                limits["bootstrap_samples"],
            )
            for key, value in effect.items():
                if confirmation[key] != value:
                    raise RuntimeError("Confirmation intervals differ from raw evidence")
            accepted = bool(
                effect["score_delta"]["market_bootstrap_95"][0] > limits["minimum_score_delta"]
                and effect["conversion_ratio"]["defined_for_all_resamples"]
                and effect["conversion_ratio"]["market_bootstrap_95"][0]
                >= limits["minimum_conversion_ratio"]
                and effect["violation_delta"]["market_bootstrap_95"][1]
                <= limits["maximum_violation_increase"]
                and sum(v > 0 for v in effect["seed_score_deltas"].values())
                >= limits["minimum_positive_seeds"]
            )
            if confirmation["accepted"] != accepted or status["accepted"] != accepted:
                raise RuntimeError("Confirmation decision differs from the declared rule")
        reader = csv.DictReader(io.StringIO(payload[prefix + "results.csv"].decode(), newline=""))
        fields = list(dict.fromkeys(key for row in rows for key in row))
        csv_rows = list(reader)
        expected_csv = [
            {key: "" if row.get(key) is None else str(row[key]) for key in fields} for row in rows
        ]
        if reader.fieldnames != fields or csv_rows != expected_csv:
            raise RuntimeError("CSV differs from JSON raw evidence")
        counts[split] = len(rows)

    if phase == "complete" and status["rows"] != counts["confirmation"]:
        raise RuntimeError("Completed status row count differs from confirmation evidence")

    for name, content in payload.items():
        if name.endswith(".csv"):
            payload[name] = content.replace(b"\r\n", b"\n")
        text = payload[name].decode()
        if any(re.search(pattern, text) for pattern in PRIVACY_PATTERNS):
            raise RuntimeError(
                f"Potential private material in export file {name}; content withheld"
            )
        if name.endswith(".json"):
            decode_json(payload[name])
    previous = {}
    if destination.exists() and any(destination.iterdir()):
        previous_manifest = destination / "manifest.json"
        if (
            previous_manifest.is_symlink()
            or not previous_manifest.exists()
            or decode_json(previous_manifest.read_bytes())["identity"] != identity
        ):
            raise RuntimeError(
                "Destination contains unknown or different evidence; select a new directory"
            )
        previous_record = destination / "export.json"
        if previous_record.exists():
            if previous_record.is_symlink():
                raise RuntimeError("Export record must be a regular file")
            previous = decode_json(previous_record.read_bytes())
            if previous["run_identity"] != identity:
                raise RuntimeError("Destination export identity differs")
    # Check every collision before writing; unrelated destination files are preserved.
    for name, content in payload.items():
        target = destination / name
        if target.is_symlink() or not target.resolve().is_relative_to(destination):
            raise RuntimeError("Export destination contains a linked evidence path")
        if target.exists():
            existing = digest(target.read_bytes())
            if (
                existing != digest(content)
                and previous.get("files_sha256", {}).get(name) != existing
            ):
                raise RuntimeError(f"Destination file {name} contains unknown changes")
    record = {
        "exporter_sha256": digest(Path(__file__).read_bytes()),
        "csv_line_endings": "LF; raw run CSV retained separately",
        "run_identity": identity,
        "phase": phase,
        "rows": counts,
        "files_sha256": {name: digest(content) for name, content in payload.items()},
        "privacy_gate": "allowlisted text evidence; local-home paths, email addresses, common credential formats checked; not a proof against all privacy disclosures",
        "excluded": "traffic tapes, model checkpoints, runner state, machine logs",
    }
    for name, content in payload.items():
        atomic_write(destination / name, content)
    atomic_write(
        destination / "export.json",
        (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    print(
        f"exported validation_rows={counts['validation']} confirmation_rows={counts.get('confirmation', 0)} files={len(payload)}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.destination)
