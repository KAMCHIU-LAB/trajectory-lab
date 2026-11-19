"""Check a published source manifest and optional CPU replay invariants.

These checks establish file identity and the declared software contract, not
research efficacy or the authenticity of an arbitrary external evidence file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "releases/v0.2.0.json"


def verify_sources(root: Path, record: dict) -> int:
    files = record["files_sha256"]
    if not isinstance(files, dict) or not files:
        raise ValueError("Release must record a nonempty source inventory")
    for name, expected in files.items():
        relative = Path(name)
        path = root / relative
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise ValueError(f"Unsafe release path: {name}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid release hash: {name}")
        if not path.is_file():
            raise ValueError(f"Missing release file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Release source changed: {name}")
    return len(files)


def verify_replay(evidence: dict, expected: dict) -> None:
    if not expected:
        raise ValueError("Release must declare replay invariants")
    for key, value in expected.items():
        actual = evidence.get(key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f"Replay invariant failed: {key}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--replay", type=Path)
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Check replay invariants without release source hashes",
    )
    args = parser.parse_args()
    if args.replay_only and args.replay is None:
        parser.error("--replay-only requires --replay")
    record = json.loads(args.manifest.read_text())
    count = 0 if args.replay_only else verify_sources(ROOT, record)
    if args.replay:
        verify_replay(json.loads(args.replay.read_text()), record["replay"]["expected"])
    print(
        json.dumps(
            {
                "release": record["release"],
                "verified_files": count,
                "source_checked": not args.replay_only,
                "replay_checked": args.replay is not None,
                "scope": "source identity and software invariants only",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
