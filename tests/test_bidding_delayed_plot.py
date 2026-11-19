import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "delayed_plot", Path(__file__).parents[1] / "scripts/plot_bidding_delayed_evidence.py"
)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def evidence(directory):
    (directory / "README.md").write_text("Public simulation evidence\n")
    files = {"README.md": hashlib.sha256((directory / "README.md").read_bytes()).hexdigest()}
    (directory / "export-manifest.json").write_text(json.dumps({"files": files}))
    for name in ("confirmation-effects.svg", "confirmation-stress.svg"):
        (directory / name).write_text("<svg/>")


def test_complete_publication_hashes(tmp_path):
    evidence(tmp_path)
    result = plot.seal_publication(tmp_path)
    assert len(result["files"]) == 4
    for name, digest in result["files"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert plot.seal_publication(tmp_path) == result


@pytest.mark.parametrize("failure", ["tamper", "unknown", "symlink", "traversal"])
def test_publication_rejects_invalid_inputs(tmp_path, failure):
    evidence(tmp_path)
    if failure == "tamper":
        (tmp_path / "README.md").write_text("Changed")
    elif failure == "unknown":
        (tmp_path / "local.log").write_text("Excluded")
    elif failure == "symlink":
        (tmp_path / "confirmation-effects.svg").unlink()
        (tmp_path / "confirmation-effects.svg").symlink_to(tmp_path / "README.md")
    else:
        (tmp_path / "export-manifest.json").write_text(json.dumps({"files": {"../x": "0"}}))
    with pytest.raises(ValueError):
        plot.seal_publication(tmp_path)
