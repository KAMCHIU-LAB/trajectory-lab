"""Portable inference, corrupted inputs, and optional original-checkpoint parity."""

import hashlib
import json
import os
import shutil
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from trajectorylab.bidding.learning import Learned, rollout
from trajectorylab.bidding.market import Tape
from trajectorylab.bidding_public_response import (
    ACTIONS,
    CHECKPOINT_SHA256,
    DEFAULT_ASSET_DIRECTORY,
    PublicResponse,
)

ROOT = Path(__file__).resolve().parents[1]


def copy_asset(tmp_path, seed=11):
    for suffix in (".npz", ".json"):
        shutil.copyfile(
            DEFAULT_ASSET_DIRECTORY / f"response-{seed}{suffix}",
            tmp_path / f"response-{seed}{suffix}",
        )
    return tmp_path / f"response-{seed}.npz"


@pytest.mark.parametrize("seed", [11, 23, 37])
def test_registered_experts_load_without_pickle(seed, monkeypatch):
    def reject_pickle(*args, **kwargs):
        raise AssertionError("Portable inference must not load pickle")

    monkeypatch.setattr(torch, "load", reject_pickle)
    policy = PublicResponse(seed=seed)
    registered = tomllib.loads((ROOT / "configs/bidding-confirmation.toml").read_text())
    assert policy.metadata["checkpoint_sha256"] == registered["checkpoint_sha256"][str(seed)]
    assert policy.seed == seed
    assert not policy.model.training
    assert not policy.corrected and not policy.hard_cpa
    state = np.zeros(13, dtype=np.float32)
    state[1:3] = 1.0
    action, prediction = policy([state], [], [], 2900.0, 100.0)
    assert action in ACTIONS
    assert np.isfinite(prediction["predicted_tick_cost"])
    assert np.isfinite(prediction["predicted_tick_conversions"])


def test_explicit_file_infers_seed_and_directory_honors_seed():
    path = DEFAULT_ASSET_DIRECTORY / "response-23.json"
    from_file = PublicResponse(path)
    from_directory = PublicResponse(DEFAULT_ASSET_DIRECTORY, seed=23)
    assert from_file.seed == from_directory.seed == 23
    for key, value in from_file.model.state_dict().items():
        torch.testing.assert_close(value, from_directory.model.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="different seed"):
        PublicResponse(path, seed=11)


@pytest.mark.parametrize(
    "options",
    [
        {"actions": [0.0, 1.0, 3.0]},
        {"context": 12},
        {"device": "cuda"},
        {"seed": 99},
    ],
)
def test_rejects_incompatible_expert_contract(options):
    with pytest.raises(ValueError):
        PublicResponse(**options)


def test_rejects_changed_weights(tmp_path):
    path = copy_asset(tmp_path)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        PublicResponse(path)


@pytest.mark.parametrize("corruption", ["object", "nonfinite", "shape"])
def test_rejects_invalid_arrays_even_with_updated_digest(tmp_path, corruption):
    path = copy_asset(tmp_path)
    with np.load(path, allow_pickle=False) as original:
        arrays = {key: original[key].copy() for key in original.files}
    key = next(iter(arrays))
    if corruption == "object":
        arrays[key] = np.asarray([object()], dtype=object)
    elif corruption == "nonfinite":
        arrays[key].flat[0] = np.nan
    else:
        arrays[key] = arrays[key].reshape(-1)[:1]
    np.savez_compressed(path, **arrays)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["weights_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        PublicResponse(path)


def test_rejects_metadata_outside_public_whitelist(tmp_path):
    path = copy_asset(tmp_path)
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    metadata["filesystem_path"] = "not-a-model-field"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata fields"):
        PublicResponse(path)


def test_original_immediate_rollout_matches_all_portable_experts():
    """Opt in with a trusted original main-v2 directory; no private path is stored."""
    original_root = os.environ.get("TRAJECTORYLAB_ORIGINAL_BIDDING_ROOT")
    if original_root is None:
        pytest.skip("Set TRAJECTORYLAB_ORIGINAL_BIDDING_ROOT for original-checkpoint parity")
    original_root = Path(original_root)
    tape = Tape(original_root / "markets/2001.npz")
    actions = np.asarray(ACTIONS)

    def measured(policy):
        predictions = []
        hook = policy.model.register_forward_hook(
            lambda model, inputs, output: predictions.append(
                output[0, -1].detach().cpu().numpy().copy()
            )
        )
        try:
            result = rollout(tape, 0, actions, "calibrated", policy, retain=True)
        finally:
            hook.remove()
        return result, np.stack(predictions)

    for seed in (11, 23, 37):
        checkpoint = original_root / f"models/response-{seed}/best.pt"
        assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256[seed]
        original = Learned(checkpoint, actions, 24, "cpu", corrected=False, hard_cpa=False)
        expected, expected_predictions = measured(original)
        actual, actual_predictions = measured(PublicResponse(seed=seed))
        for key in ("states", "actions", "rewards"):
            np.testing.assert_array_equal(actual.pop(key), expected.pop(key))
        assert actual == expected
        np.testing.assert_array_equal(actual_predictions, expected_predictions)
        print(
            f"seed={seed} market=2001 focal=0 action_mismatches=0 "
            f"max_prediction_absolute_error={np.max(np.abs(actual_predictions - expected_predictions))}"
        )
