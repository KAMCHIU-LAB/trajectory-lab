"""Export an explicit allowlist of completed bidding evidence, with a privacy gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from trajectorylab.bidding.report import aggregate


def export(source: Path, destination: Path) -> None:
    def load(name):
        return json.loads((source / name).read_text())

    if load("status.json")["phase"] != "complete":
        raise RuntimeError("Only completed runs can be exported")
    manifest, rows = load("manifest.json"), load("results.json")
    study = manifest["config"]["study"]
    policies = {(m, -1) for m in ("feedback", "fixed-1", "fixed-tuned")}
    policies.update(
        (m, s)
        for m in ("dt", "dt+correction", "response", "response+correction", "response-hard")
        for s in study["training_seeds"]
    )
    expected = {
        (m, f, c, k, s)
        for m in study["test_markets"]
        for f in study["focal_agents"]
        for c in manifest["config"]["evaluation"]["scenarios"]
        for k, s in policies
    }
    identities = {(r["market"], r["focal"], r["scenario"], r["method"], r["seed"]) for r in rows}
    if identities != expected or len(rows) != len(expected):
        raise RuntimeError("Evaluation does not match the declared experiment jobs")
    groups, comparisons = aggregate(rows)
    summary = load("summary.json")
    if summary["groups"] != groups or summary["paired_comparisons"] != comparisons:
        raise RuntimeError("Summary is inconsistent with raw evidence")
    names = (
        "report.html",
        "results.csv",
        "results.json",
        "summary.json",
        "selection.json",
        "manifest.json",
        "training.json",
        "dataset-summary.json",
        "kernel-audit.json",
        "status.json",
    )
    payload = {name: (source / name).read_bytes() for name in names}
    # CSV's standard writer emits CRLF; normalize exported text without changing values.
    payload["results.csv"] = payload["results.csv"].replace(b"\r\n", b"\n")
    for history in sorted((source / "models").glob("*/history.json")):
        payload[f"histories/{history.parent.name}.json"] = history.read_bytes()
    patterns = (
        r"/Users/",
        r"/home/",
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    )
    for name, content in payload.items():
        text = content.decode()
        if any(re.search(pattern, text) for pattern in patterns):
            raise RuntimeError(
                f"Potential private material in export file {name}; content withheld"
            )
    if destination.exists() and any(destination.iterdir()):
        previous = destination / "manifest.json"
        if (
            not previous.exists()
            or json.loads(previous.read_text())["identity"] != manifest["identity"]
        ):
            raise RuntimeError("Destination contains a different study; select a new directory")
    for name, content in payload.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    record = {
        "exporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "csv_line_endings": "LF; raw run CSV retained separately",
        "run_identity": manifest["identity"],
        "rows": len(rows),
        "files_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in payload.items()},
        "privacy_gate": "allowlisted text evidence; local-home paths, email addresses, common credential formats checked; not a proof against all privacy disclosures",
        "excluded": "traffic tapes, model checkpoints, runner state, machine logs",
    }
    (destination / "export.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"exported rows={len(rows)} files={len(payload)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    export(args.run, args.destination)
