import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "strategy_export",
    Path(__file__).resolve().parents[1] / "scripts/export_bidding_delayed_strategy.py",
)
e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def study(tmp_path, monkeypatch):
    source, output, history = tmp_path / "input", tmp_path / "public", tmp_path / "history"
    config = e.strategy.base.load_config("configs/bidding-delayed-strategy.toml")
    config["study"]["focal_agents"] = [0]
    config["evaluation"]["arrival_seeds"] = [11]
    recorded_source = dict(
        frozen_core={"source": "abc"}, extension={}, config_sha256="abc", registration_sha256="abc"
    )
    monkeypatch.setattr(e.strategy, "source_record", lambda path: recorded_source)
    monkeypatch.setattr(e.strategy.base, "load_config", lambda path: config)
    dump(history / "history.json", {"immediate": {"probabilities": [1.0]}})
    history_core = {"source": recorded_source["frozen_core"]}
    history_identity = e.fingerprint(history_core)
    history_hash = e.strategy.file_hash(history / "history.json")
    dump(
        history / "history-manifest.json",
        dict(identity=history_identity, core=history_core, history_sha256=history_hash),
    )
    provenance = dict(
        config=config,
        source=recorded_source,
        history_sha256=history_hash,
        training_history_identity=history_identity,
    )
    rows = []
    for v in e.strategy.variants():
        for m in config["study"]["validation_markets"]:
            for group in ["under", "nominal", "over"]:
                for delay in config["evaluation"]["regimes"]:
                    rows.append(
                        dict(
                            market=m,
                            focal=0,
                            bias_group=group,
                            delay=delay,
                            arrival_seed=11,
                            method=v["name"],
                            cost=5.0,
                            conversions=10,
                            score=10.0,
                            exposures=20,
                            budget=100.0,
                            target_cpa=10.0,
                            cpa=0.5,
                            violation=0,
                            budget_utilization=0.05,
                            posterior_pending_mae=0.2,
                            deployable=True,
                        )
                    )
    core = dict(
        **provenance,
        split="validation",
        methods=e.strategy.variants(),
        markets={},
        selection_identity=None,
    )
    run_id = e.fingerprint(core)
    summary = e.strategy.development_selection(rows, config)
    selection = dict(
        **summary,
        provenance=provenance,
        development_identity=run_id,
        development_results_sha256=e.fingerprint(rows),
    )
    selection["identity"] = e.fingerprint(selection)
    dump(source / "selection.json", selection)
    dump(source / "validation/manifest.json", dict(identity=run_id, core=core))
    dump(source / "validation/summary.json", dict(identity=run_id, **summary))
    dump(
        source / "validation/status.json",
        dict(identity=run_id, phase="complete", rows=len(rows), elapsed_seconds=1.0),
    )
    dump(source / "validation/results.json", rows)
    return source, output, tmp_path / "config.toml", history


def test_lossless_scientific_json_and_complete_scalar_export(study):
    counts = e.export_evidence(*study)
    source, output, _, _ = study
    assert (
        counts["files"] == 9
        and counts["campaigns"] == 648
        and counts["market_cells"] == 648
        and counts["cells"] == 108
    )
    for name in e.JSON_FILES:
        assert json.loads((source / name).read_text()) == json.loads((output / name).read_text())
    with (output / "validation.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 648 and set(rows[0]) == set(e.COLUMNS)
    manifest = json.loads((output / "export-manifest.json").read_text())
    assert len(manifest["files"]) == 8
    assert set(manifest["files"]) == set(e.OUTPUTS) - {"export-manifest.json"}
    assert e.export_evidence(*study) == counts


@pytest.mark.parametrize(
    "field,value",
    [
        ("unlisted_private_key", "plain"),
        ("stop_reason", "/Users/person/private"),
        ("stop_reason", "person@example.com"),
        ("stop_reason", "api_key=private"),
        ("stop_reason", "sk-abcdefghijklmnopqrstuv"),
        ("stop_reason", float("nan")),
    ],
)
def test_json_unknown_or_private_injection_rejected(study, field, value):
    source, output, _, _ = study
    payload = json.loads((source / "selection.json").read_text())
    payload[field] = value
    dump(source / "selection.json", payload)
    with pytest.raises(ValueError):
        e.export_evidence(*study)
    assert not output.exists()


def test_existing_private_file_and_symlink_rejected_without_deletion(study, tmp_path):
    _, output, _, _ = study
    output.mkdir()
    private = output / "private.log"
    private.write_text("preserve")
    with pytest.raises(ValueError, match="existing output"):
        e.export_evidence(*study)
    assert private.read_text() == "preserve"
    private.unlink()
    target = tmp_path / "private-source"
    target.write_text("preserve target")
    (output / "validation.csv").symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        e.export_evidence(*study)
    assert target.read_text() == "preserve target"
    assert (output / "validation.csv").is_symlink()


def test_campaign_injection_and_result_hash_rejected(study):
    source, output, _, _ = study
    path = source / "validation/results.json"
    rows = json.loads(path.read_text())
    rows[0]["actions"] = [1.0]
    dump(path, rows)
    with pytest.raises(ValueError, match="campaign field"):
        e.export_evidence(*study)
    del rows[0]["actions"]
    rows[0]["posterior_pending_mae"] = 0.9
    dump(path, rows)
    with pytest.raises(ValueError, match="result fingerprint"):
        e.export_evidence(*study)
    assert not output.exists()


def test_history_and_source_identity_checked(study):
    _, output, _, history = study
    dump(history / "history.json", {"changed": True})
    with pytest.raises(ValueError, match="history fingerprint"):
        e.export_evidence(*study)
    assert not output.exists()
