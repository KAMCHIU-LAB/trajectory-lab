"""Freeze and cache integrity checks without opening any confirmation market."""

import copy
import hashlib
from itertools import product

import pytest

import trajectorylab.bidding_delayed_experiment as runner
from trajectorylab.bidding_delayed_statistics import validate_rows
from trajectorylab.utils import fingerprint, write_json


def configuration():
    return {
        "study": {
            "focal_agents": [0],
            "pv_count": 10000,
            "actions": [0.0, 1.0],
            "biases": [0.75, 1.0, 1.5],
            "validation_markets": [2001],
        },
        "evaluation": {"regimes": ["immediate", "long"], "arrival_seeds": [11]},
    }


def cache(tmp_path, monkeypatch):
    config = configuration()
    source = {"upstream_revision": "public-test", "sha256": {"vendor.py": "original"}}
    monkeypatch.setattr(runner, "event_source_identity", lambda: copy.deepcopy(source))
    path = tmp_path / "markets" / "2001.npz"
    path.parent.mkdir()
    path.write_bytes(b"finite public tape fixture")
    study = {k: config["study"][k] for k in ("focal_agents", "pv_count", "actions", "biases")}
    meta = {
        "identity": fingerprint(dict(study=study, market=2001, source=source)),
        "source": source,
    }
    write_json(path.with_suffix(".json"), meta)
    return config, path, source, meta


def test_cache_provenance_binds_actual_bytes(tmp_path, monkeypatch):
    config, path, _, _ = cache(tmp_path, monkeypatch)
    original = runner.cache_provenance(config, tmp_path, "validation")
    assert original["2001"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    path.write_bytes(b"replaced tape")
    changed = runner.cache_provenance(config, tmp_path, "validation")
    assert changed["2001"]["identity"] == original["2001"]["identity"]
    assert changed["2001"]["sha256"] != original["2001"]["sha256"]
    assert fingerprint(changed) != fingerprint(original)


def test_changed_tape_cannot_resume_existing_evaluation(tmp_path, monkeypatch):
    config, path, _, _ = cache(tmp_path, monkeypatch)
    methods = [{"name": "frozen"}]
    monkeypatch.setattr(runner, "variants", lambda _: methods)
    monkeypatch.setattr(runner, "source_record", lambda: {"runner": "frozen"})
    history = tmp_path / "history.json"
    history.write_bytes(b"{}")
    core = dict(
        config=config,
        source=runner.source_record(),
        markets=runner.cache_provenance(config, tmp_path, "validation"),
        history_sha256=hashlib.sha256(history.read_bytes()).hexdigest(),
        methods=methods,
        split="validation",
    )
    write_json(tmp_path / "validation" / "manifest.json", {"identity": fingerprint(core)})
    path.write_bytes(b"replaced tape")
    with pytest.raises(RuntimeError, match="Frozen evaluation changed"):
        runner.evaluate(config, tmp_path, "validation")


@pytest.mark.parametrize("change", ["vendor", "metadata_source", "metadata_identity"])
def test_cache_provenance_rejects_source_drift(tmp_path, monkeypatch, change):
    config, path, source, meta = cache(tmp_path, monkeypatch)
    if change == "vendor":
        changed = copy.deepcopy(source)
        changed["sha256"]["vendor.py"] = "changed"
        monkeypatch.setattr(runner, "event_source_identity", lambda: changed)
    elif change == "metadata_source":
        meta["source"] = {"sha256": {"vendor.py": "swapped"}}
        write_json(path.with_suffix(".json"), meta)
    else:
        meta["identity"] = "wrong-market-or-configuration"
        write_json(path.with_suffix(".json"), meta)
    with pytest.raises(RuntimeError, match="source identity changed"):
        runner.cache_provenance(config, tmp_path, "validation")


@pytest.mark.parametrize("drift", ["none", "config", "source", "history"])
def test_frozen_selection_rejects_drift(tmp_path, monkeypatch, drift):
    config = configuration()
    source = {"public.py": "original"}
    monkeypatch.setattr(runner, "source_record", lambda: source)
    history = tmp_path / "history.json"
    history.write_bytes(b'{"training_history": "original"}')
    selection = {
        "source": copy.deepcopy(source),
        "config_identity": fingerprint(config),
        "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest(),
        "frozen_variants": [{"name": "frozen"}],
    }
    write_json(tmp_path / "selection.json", selection)
    if drift == "config":
        config["study"]["pv_count"] += 1
    elif drift == "source":
        source["public.py"] = "changed"
    elif drift == "history":
        history.write_bytes(b'{"training_history": "changed"}')
    if drift == "none":
        assert runner.check_frozen_selection(config, tmp_path) == selection
    else:
        with pytest.raises(RuntimeError, match="changed after"):
            runner.check_frozen_selection(config, tmp_path)


def test_equal_method_coverage_does_not_allow_missing_declared_cell():
    config = configuration()
    methods = [{"name": "base"}, {"name": "candidate"}]
    rows = [
        dict(
            market=2001,
            focal=0,
            bias_group=group,
            delay=delay,
            arrival_seed=11,
            method=method["name"],
            score=1.0,
            conversions=1,
            cost=1.0,
            budget=2.0,
            budget_utilization=0.5,
            violation=0,
        )
        for method, group, delay in product(
            methods, ("under", "nominal", "over"), ("immediate", "long")
        )
    ]
    assert runner.validate_grid(rows, config, "validation", methods)["rows"] == 12
    missing = [r for r in rows if (r["bias_group"], r["delay"]) != ("over", "long")]
    assert validate_rows(missing)["rows"] == 10
    with pytest.raises(RuntimeError, match="missing declared campaign cells"):
        runner.validate_grid(missing, config, "validation", methods)


def test_reselection_blocked_after_confirmation_opened(tmp_path):
    manifest = tmp_path / "confirmation" / "manifest.json"
    write_json(manifest, {"identity": "frozen"})
    with pytest.raises(RuntimeError, match="Cannot reselect"):
        runner.select_development(configuration(), tmp_path)
