"""Recent result navigation must reject incomplete or modified publications."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "build_portal", Path(__file__).parents[1] / "scripts/build_portal.py"
)
portal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(portal)


def test_publication_navigation_checks_recorded_bytes(tmp_path):
    directory = tmp_path / "study"
    directory.mkdir()
    result = directory / "summary.json"
    result.write_text('{"passed": false}\n')
    (directory / "publication.json").write_text(
        json.dumps({"files": {result.name: hashlib.sha256(result.read_bytes()).hexdigest()}})
    )
    manifest_hash = hashlib.sha256((directory / "publication.json").read_bytes()).hexdigest()
    assert portal.publication_inventory(
        tmp_path, "study", "publication.json", "files", manifest_hash
    ) == {
        "manifest": "study/publication.json",
        "verified_files": 1,
    }
    result.write_text('{"passed": true}\n')
    assert (
        portal.publication_inventory(tmp_path, "study", "publication.json", "files", manifest_hash)
        is None
    )
    result.unlink()
    assert (
        portal.publication_inventory(tmp_path, "study", "publication.json", "files", manifest_hash)
        is None
    )


@pytest.mark.parametrize("name", ["../outside.json", "/tmp/outside.json"])
def test_publication_inventory_rejects_escape(tmp_path, name):
    directory = tmp_path / "study"
    directory.mkdir()
    (directory / "publication.json").write_text(json.dumps({"files": {name: "0" * 64}}))
    manifest_hash = hashlib.sha256((directory / "publication.json").read_bytes()).hexdigest()
    assert (
        portal.publication_inventory(tmp_path, "study", "publication.json", "files", manifest_hash)
        is None
    )


def test_changed_publication_manifest_cannot_relabel_results(tmp_path):
    directory = tmp_path / "study"
    directory.mkdir()
    (directory / "summary.json").write_text('{"passed": false}')
    summary_hash = hashlib.sha256((directory / "summary.json").read_bytes()).hexdigest()
    manifest = directory / "publication.json"
    manifest.write_text(json.dumps({"decision": "failed", "files": {"summary.json": summary_hash}}))
    accepted_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert portal.publication_inventory(tmp_path, "study", manifest.name, "files", accepted_hash)
    manifest.write_text(json.dumps({"decision": "passed", "files": {"summary.json": summary_hash}}))
    assert (
        portal.publication_inventory(tmp_path, "study", manifest.name, "files", accepted_hash)
        is None
    )


def test_public_portal_contains_current_research():
    root = Path(__file__).parents[1]
    markup, records = portal.recent_publications(root / "reports")
    assert len(records) == 3
    assert "未确认" in markup and "未晋级" in markup
    assert "+1.21%" in markup and "+0.30%" in markup
