"""Portable frozen response experts from the public AuctionNet simulation study.

The assets contain numerical state dictionaries and allowlisted JSON metadata,
without pickle. They are conversions of the original public-simulator-trained
checkpoints, with no commercial code or data. They reproduce frozen inference;
they do not establish a complete bitwise reproduction of the training process.
The original expert consumes immediate feedback. A delayed input adapter is a
separate expert identity and requires its own evaluation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from trajectorylab.bidding.learning import Learned, ResponseModel

DEFAULT_ASSET_DIRECTORY = Path(__file__).resolve().parents[2] / "assets/bidding-response-v1"
FORMAT = "trajectorylab-public-response-v1"
STUDY_IDENTITY = "16cf4929440a33b5b6ea67a713ac669d96f5087fea841e38c783db6993b9ff7a"
UPSTREAM_REVISION = "c20de631a40e24669d1d6a8e2acefd8f492f172d"
ARCHITECTURE = {"width": 128, "layers": 2, "heads": 4}
ACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
CHECKPOINT_SHA256 = {
    11: "d7c09784f3748e5e880ec0cfe08af3c018458427f47d0a23474ee1bf6bd5baf0",
    23: "407a780646b825a48b3700984a711b6c89649ca9849d124e8851867ef63ae9d0",
    37: "ae388409c27c21782bc4aa71d8bcc1db88f9e36b7689b385f1b3a27429c855d9",
}
PROVENANCE = {
    "upstream_repository": "https://github.com/alimama-tech/AuctionNet",
    "data": "Public-simulator-generated fixed-opponent markets; no commercial code or data.",
    "scope": "Original frozen response inference; not complete bitwise training reproduction.",
}
METADATA_FIELDS = {
    "format",
    "seed",
    "architecture",
    "actions",
    "context",
    "mean",
    "std",
    "target_return",
    "return_scale",
    "selected_step",
    "checkpoint_sha256",
    "weights_sha256",
    "study_identity",
    "upstream_revision",
    "provenance",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metadata(path: Path) -> dict:
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or set(metadata) != METADATA_FIELDS:
        raise ValueError("Unexpected public response metadata fields")
    seed = metadata["seed"]
    if (
        type(seed) is not int
        or seed not in CHECKPOINT_SHA256
        or metadata["format"] != FORMAT
        or metadata["architecture"] != ARCHITECTURE
        or metadata["actions"] != list(ACTIONS)
        or metadata["context"] != 24
        or metadata["checkpoint_sha256"] != CHECKPOINT_SHA256[seed]
        or metadata["study_identity"] != STUDY_IDENTITY
        or metadata["upstream_revision"] != UPSTREAM_REVISION
        or metadata["provenance"] != PROVENANCE
        or type(metadata["selected_step"]) is not int
        or not 0 < metadata["selected_step"] <= 15000
    ):
        raise ValueError("Public response metadata differs from the registered expert")
    mean, std = (np.asarray(metadata[key], dtype=np.float32) for key in ("mean", "std"))
    if (
        mean.shape != (13,)
        or std.shape != (13,)
        or not np.isfinite(np.r_[mean, std]).all()
        or np.any(std <= 0)
        or not np.isfinite([metadata["target_return"], metadata["return_scale"]]).all()
        or metadata["return_scale"] <= 0
    ):
        raise ValueError("Invalid public response normalization or return scale")
    return metadata


class PublicResponse(Learned):
    """Load an original response expert using only JSON and ``allow_pickle=False``.

    ``path`` may be an asset directory or a matching ``.npz``/``.json`` file.
    A directory defaults to seed 11; an explicit file infers its seed from JSON.
    CPU, the original 11 actions, and context 24 preserve the frozen contract.
    Inference uses the existing Learned call interface and controller unchanged.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        seed: int | None = None,
        actions=None,
        context: int = 24,
        device: str = "cpu",
    ):
        if seed is not None and (type(seed) is not int or seed not in CHECKPOINT_SHA256):
            raise ValueError("Choose a registered public response seed: 11, 23, or 37")
        if context != 24 or device != "cpu":
            raise ValueError("The public response expert requires context 24 and CPU inference")
        actions = np.asarray(ACTIONS if actions is None else actions, dtype=np.float64)
        if not np.array_equal(actions, np.asarray(ACTIONS)):
            raise ValueError("Inference action grid differs from the frozen response expert")
        path = DEFAULT_ASSET_DIRECTORY if path is None else Path(path)
        if path.is_dir():
            path = path / f"response-{11 if seed is None else seed}.npz"
        if path.suffix not in {".npz", ".json"}:
            raise ValueError("Public response input must be an asset directory or .npz/.json")
        weights_path = path.with_suffix(".npz")
        metadata = _metadata(path.with_suffix(".json"))
        if seed is not None and metadata["seed"] != seed:
            raise ValueError("Public response file belongs to a different seed")
        if _sha256(weights_path) != metadata["weights_sha256"]:
            raise ValueError("Public response weights SHA-256 mismatch")
        model = ResponseModel(13, len(actions), **ARCHITECTURE)
        expected = model.state_dict()
        with np.load(weights_path, allow_pickle=False) as arrays:
            if set(arrays.files) != set(expected):
                raise ValueError("Public response state dictionary keys differ")
            state = {}
            for key, tensor in expected.items():
                array = arrays[key]
                if (
                    array.shape != tuple(tensor.shape)
                    or array.dtype != tensor.numpy().dtype
                    or not np.isfinite(array).all()
                ):
                    raise ValueError(f"Invalid public response weight: {key}")
                state[key] = torch.from_numpy(array.copy())
        model.load_state_dict(state, strict=True)
        self.model = model.eval()
        self.kind = "response"
        self.actions, self.context, self.device = actions, context, device
        self.mean = np.asarray(metadata["mean"], dtype=np.float32)
        self.std = np.asarray(metadata["std"], dtype=np.float32)
        self.target, self.scale = metadata["target_return"], metadata["return_scale"]
        self.corrected, self.prior, self.hard_cpa = False, 3.0, False
        self.seed, self.metadata = metadata["seed"], metadata
        self.reset()
