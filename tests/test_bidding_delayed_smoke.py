"""Small wrapper contracts; the CLI itself supplies the public end-to-end probe."""
import importlib.util
from pathlib import Path

import pytest

from trajectorylab.bidding_delayed_experiment import load_config
from trajectorylab.utils import write_json

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('delayed_smoke', ROOT / 'scripts/run_bidding_delayed_smoke.py')
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def test_frozen_resume_and_tampering(tmp_path):
    path = tmp_path / 'selection-lock.json'
    record = dict(selection_sha256='abc', selection_identity='def')
    smoke.freeze_record(path, record)
    original = path.read_bytes()
    smoke.freeze_record(path, record)
    assert path.read_bytes() == original
    with pytest.raises(RuntimeError, match='Frozen smoke record changed'):
        smoke.freeze_record(path, dict(record, selection_sha256='changed'))
    assert path.read_bytes() == original
    write_json(path, dict(record, selection_sha256='tampered'))
    with pytest.raises(RuntimeError, match='Frozen smoke record changed'):
        smoke.freeze_record(path, record)


def test_smoke_has_complete_scenes_and_retains_business_sample_gate():
    config = load_config(ROOT / 'configs/bidding-delayed-smoke.toml')
    study, evaluation = config['study'], config['evaluation']
    markets = [study[f'{split}_markets'] for split in ('train', 'validation', 'confirmation')]
    assert all(len(ids) == 1 for ids in markets)
    assert len({m for ids in markets for m in ids}) == 3
    assert study['biases'] == [.75, 1., 1.5]
    assert evaluation['regimes'] == ['immediate', 'short', 'long']
    assert len(evaluation['methods']) == 5
    assert len(evaluation['gains']) == len(evaluation['priors']) == 2
    assert config['limits']['minimum_confirmation_markets'] == 24
