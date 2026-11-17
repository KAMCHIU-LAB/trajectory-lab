"""Prepare local frozen-model replay inputs; never upload or include traffic."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

from export_bidding_control import PRIVACY_PATTERNS, atomic_write

from trajectorylab.bidding.market import ROOT
from trajectorylab.bidding_confirmation import _prepare
from trajectorylab.utils import fingerprint


def package(destination: Path) -> None:
    opts, _, core = _prepare(ROOT / "configs/bidding-confirmation.toml")
    destination = destination.resolve()
    if not destination.is_relative_to((ROOT / "artifacts").resolve()):
        raise RuntimeError("Replay inputs must remain in local artifacts until publication review")
    inputs = {}
    for label, kinds in (
        ("main", ("manifest", "selection", "status", "training")),
        ("control", ("manifest", "selection", "status")),
    ):
        directory = ROOT / opts[f"{label}_run"]
        for kind in kinds:
            inputs[f"{directory.name}/{kind}.json"] = (directory / f"{kind}.json").read_bytes()
    for seed in opts["training_seeds"]:
        path = ROOT / opts["main_run"] / "models" / f"response-{seed}" / "best.pt"
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != opts["checkpoint_sha256"][str(seed)]:
            raise RuntimeError("A frozen checkpoint changed")
        inputs[f"{Path(opts['main_run']).name}/models/response-{seed}/best.pt"] = content
    for name, content in inputs.items():
        if any(re.search(pattern.encode(), content) for pattern in PRIVACY_PATTERNS):
            raise RuntimeError(f"Potential private material in {name}; content withheld")
    record = {
        "confirmation_identity": fingerprint(core),
        "main_identity": core["main_identity"],
        "files_sha256": {
            name: hashlib.sha256(content).hexdigest() for name, content in sorted(inputs.items())
        },
        "runtime": {"python": "3.12.13", "numpy": "2.5.5", "torch": "2.15.0"},
        "scope": "Frozen inference inputs, not training-state or traffic data. Local publication candidate only.",
        "excluded": "Optimizer/RNG states, raw traffic, labels, trajectories, market caches and logs.",
        "replay_limit": "Requires matching frozen repository and runtime, then regeneration of all 24 markets. Cross-platform equivalence has not been established.",
    }
    inputs["replay-inputs.json"] = (json.dumps(record, indent=2, sort_keys=True) + "\n").encode()
    inputs["REPLAY.md"] = (
        b"# Frozen bidding replay inputs\n\n"
        b"This local candidate is not an uploaded release. It contains three response best.pt files "
        b"and seven study records, exclusively from the declared public simulation study.\n\n"
        b"Use a fresh clone at the matching source revision. Check replay-inputs.json hashes before "
        b"loading checkpoints; best.pt uses PyTorch serialization with NumPy metadata. Do not "
        b"load substituted or untrusted files. Extract under an empty artifacts/bidding directory. "
        b"Do not overwrite another run.\n\n"
        b"The confirmation protocol requires Python 3.12.13, NumPy 2.5.5 and PyTorch 2.15.0. "
        b"Keep the main configuration's historical mps string: confirmation inference itself is CPU. "
        b"Run python -m trajectorylab.bidding_confirmation to generate all 24 markets before "
        b"evaluation. Inputs do not prove the training process, guarantee cross-platform numerical "
        b"equivalence, or enlarge the conclusion beyond three fixed models.\n"
    )
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", filename="", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, content in sorted(inputs.items()):
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = len(content), 0o644, 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(content))
    content = buffer.getvalue()
    # Re-open every member and check exact bytes before keeping the local package.
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        if set(archive.getnames()) != set(inputs):
            raise RuntimeError("Archive member coverage differs")
        for member in archive.getmembers():
            if not member.isfile() or archive.extractfile(member).read() != inputs[member.name]:
                raise RuntimeError("Archive bytes differ from the allowlisted inputs")
    atomic_write(destination, content)
    atomic_write(
        destination.with_suffix(destination.suffix + ".sha256"),
        f"{hashlib.sha256(content).hexdigest()}  {destination.name}\n".encode(),
    )
    print(f"Local replay candidate: files={len(inputs)} bytes={len(content)}; not uploaded")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/bidding/publication/bidding-frozen-inputs.tar.gz",
    )
    package(parser.parse_args().output)
