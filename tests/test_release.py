"""Release verification rejects changed sources and weakened replay evidence."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_release", Path(__file__).parents[1] / "scripts/verify_release.py"
)
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def test_release_identity_rejects_changed_or_missing_file(tmp_path):
    path = tmp_path / "public.py"
    path.write_text("value = 1\n")
    record = {"files_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}}
    assert verifier.verify_sources(tmp_path, record) == 1
    path.write_text("value = 2\n")
    with pytest.raises(ValueError, match="source changed"):
        verifier.verify_sources(tmp_path, record)
    path.unlink()
    with pytest.raises(ValueError, match="Missing release file"):
        verifier.verify_sources(tmp_path, record)


@pytest.mark.parametrize("name", ["../private.py", "/tmp/private.py"])
def test_release_inventory_cannot_escape_checkout(tmp_path, name):
    with pytest.raises(ValueError, match="Unsafe release path"):
        verifier.verify_sources(tmp_path, {"files_sha256": {name: "0" * 64}})


def test_empty_inventory_cannot_pass(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        verifier.verify_sources(tmp_path, {"files_sha256": {}})


@pytest.mark.parametrize("actual", [True, "1", None, 2])
def test_replay_requires_exact_value_and_type(actual):
    with pytest.raises(ValueError, match="independent_confirmation_markets"):
        verifier.verify_replay(
            {"independent_confirmation_markets": actual}, {"independent_confirmation_markets": 1}
        )


def test_replay_cannot_promote_software_probe_to_business_claim():
    expected = {"business_target_pass": False, "all_hard_budgets_respected": True}
    verifier.verify_replay(dict(expected), expected)
    with pytest.raises(ValueError, match="business_target_pass"):
        verifier.verify_replay(dict(expected, business_target_pass=True), expected)
    with pytest.raises(ValueError, match="all_hard_budgets_respected"):
        verifier.verify_replay({"business_target_pass": False}, expected)


def test_replay_only_does_not_rewrite_or_check_historical_sources(tmp_path, monkeypatch, capsys):
    manifest = tmp_path / "release.json"
    evidence = tmp_path / "replay.json"
    manifest.write_text(
        json.dumps(
            {
                "release": "0.2.0",
                "files_sha256": {"absent-historical-file.py": "0" * 64},
                "replay": {"expected": {"business_target_pass": False}},
            }
        )
    )
    evidence.write_text('{"business_target_pass": false}')
    monkeypatch.setattr(
        "sys.argv",
        ["verify_release", "--manifest", str(manifest), "--replay", str(evidence), "--replay-only"],
    )
    verifier.main()
    result = json.loads(capsys.readouterr().out)
    assert result["source_checked"] is False
    assert result["replay_checked"] is True and result["verified_files"] == 0


def test_replay_only_requires_actual_evidence(monkeypatch):
    monkeypatch.setattr("sys.argv", ["verify_release", "--replay-only"])
    with pytest.raises(SystemExit) as error:
        verifier.main()
    assert error.value.code == 2
