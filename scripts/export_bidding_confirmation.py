"""Publish a completed confirmation after checking raw rows, gates and hashes."""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import json
import re
from pathlib import Path

from export_bidding_control import PRIVACY_PATTERNS, atomic_write, decode_json, digest

from trajectorylab.bidding_confirmation import _prepare, _rows, _summary
from trajectorylab.utils import fingerprint


def export(config_path: Path, destination: Path) -> None:
    opts, _, core = _prepare(config_path)
    root = Path(__file__).resolve().parents[1]
    source = (root / opts["output"]).resolve()
    destination = destination.resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise RuntimeError("Export must be separate from the private run")
    with (source / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)

        def read(name):
            path = source / name
            if path.is_symlink() or not path.resolve().is_relative_to(source):
                raise RuntimeError("Evidence must be regular files inside the run")
            return path.read_bytes()

        names = {
            "manifest.json",
            "status.json",
            "summary.json",
            "results.json",
            "results.csv",
            "report.html",
            "market-cache.json",
        }
        payload = {name: read(name) for name in names}
        manifest, status, summary, rows = [
            decode_json(payload[name])
            for name in ("manifest.json", "status.json", "summary.json", "results.json")
        ]
        identity = fingerprint(core)
        if (
            manifest.get("core") != core
            or manifest.get("identity") != identity
            or status.get("identity") != identity
            or status.get("phase") != "complete"
            or status.get("markets") != 24
            or status.get("rows") != 4320
            or status.get("decision") != summary.get("decision")
        ):
            raise RuntimeError("Incomplete confirmation or changed frozen inputs")
        expected_outputs = (names - {"manifest.json", "status.json"}) | {
            f"market-results/{m}.json" for m in opts["markets"]
        }
        hashes = manifest.get("output_sha256", {})
        if set(hashes) != expected_outputs or any(
            digest(read(name)) != expected for name, expected in hashes.items()
        ):
            raise RuntimeError("Completed evidence hashes or coverage differ")
        _rows(rows, opts["markets"], opts)
        if _summary(rows, identity) != summary:
            raise RuntimeError("Summary or primary gates differ from raw campaign evidence")
        table = csv.DictReader(io.StringIO(payload["results.csv"].decode()))
        fields = table.fieldnames
        if fields is None or len(fields) != len(set(fields)):
            raise RuntimeError("Missing or duplicate CSV headers")
        csv_rows = list(table)
        if len(csv_rows) != len(rows):
            raise RuntimeError("CSV/JSON campaign counts differ")
        if set(fields) != {key for row in rows for key in row}:
            raise RuntimeError("CSV headers differ from the union of JSON campaign fields")
        for encoded, row in zip(csv_rows, rows, strict=True):
            if set(encoded) != set(fields):
                raise RuntimeError("CSV/JSON campaign fields differ")
            for key in fields:
                value = row.get(key)
                actual = encoded[key]
                same = (
                    actual == ""
                    if value is None
                    else (actual == value if isinstance(value, str) else float(actual) == value)
                )
                if not same:
                    raise RuntimeError("CSV/JSON campaign values differ")
        for name, content in payload.items():
            if any(re.search(pattern, content.decode()) for pattern in PRIVACY_PATTERNS):
                raise RuntimeError(f"Potential private content in {name}; content withheld")
        if destination.exists() and any(destination.iterdir()):
            old_manifest = destination / "manifest.json"
            if (
                not old_manifest.is_file()
                or decode_json(old_manifest.read_bytes()).get("identity") != identity
                or {p.name for p in destination.iterdir()} - (names | {"export.json"})
            ):
                raise RuntimeError("Destination contains different or unreviewed evidence")
        for name, content in payload.items():
            atomic_write(destination / name, content)
        record = {
            "run_identity": identity,
            "decision": summary["decision"],
            "exporter_sha256": digest(Path(__file__).read_bytes()),
            "files_sha256": {name: digest(content) for name, content in payload.items()},
            "verification": "Frozen inputs and every completed output hash checked; paired statistics and gates recomputed from all 4,320 rows; CSV/JSON cells compared.",
            "scope": "24 new simulated markets; only over_50 is primary; conditional on three fixed checkpoints. Does not reopen the failed controller gate.",
            "privacy_gate": "Allowlisted text scanned for home paths, emails and common credential patterns; not proof against every possible disclosure.",
            "excluded": "Traffic, tapes, weights, per-market operational caches and machine logs.",
        }
        atomic_write(
            destination / "export.json",
            (json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
        )
        print(f"exported confirmation rows={len(rows)} decision={summary['decision']}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/bidding-confirmation.toml"))
    parser.add_argument("--destination", type=Path, default=Path("reports/bidding-confirmation"))
    args = parser.parse_args()
    export(args.config, args.destination)
