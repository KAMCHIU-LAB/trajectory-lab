"""Publication boundaries using synthetic records; no study results are read."""

import csv
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "expert_export",
    Path(__file__).resolve().parents[1] / "scripts/export_bidding_expert_selection.py",
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)
CHECKED_DEVELOPMENT = exporter.checked_development
PUBLIC_PROVENANCE = exporter.public_provenance

PUBLIC_FILES = {
    "results.csv",
    "features.csv",
    "summary.json",
    "provenance.json",
    "report.md",
    "comparison.svg",
    "export.json",
}
BASE_METHODS = (
    "full-feedback",
    "full-response",
    "prefix-feedback",
    "prefix-response",
    "hindsight-two-experts",
)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def outcome(score):
    return dict(
        cost=10.0,
        conversions=2,
        score=score,
        violation=0,
        exposures=20,
        budget=100.0,
        target_cpa=10.0,
        budget_utilization=0.1,
        hard_budget_violation=False,
        final_arrived=2,
        event_ids=[123],
        actions=[1.0],
    )


@pytest.fixture
def study(tmp_path, monkeypatch):
    root, output = tmp_path / "repo", tmp_path / "input"
    destination = root / "reports/expert-selection"
    config = {
        "study": {
            "train_markets": [101],
            "validation_markets": [201],
            "confirmation_markets": [301],
        }
    }
    records = {
        split: [
            dict(
                split=split,
                market=market,
                focal=0,
                bias_group=group,
                features=[0.0] * len(exporter.FEATURE_NAMES),
                target=0.1,
                prefix_digest="a" * 64,
                outcomes={method: outcome(2.0 + i) for i, method in enumerate(BASE_METHODS)},
                raw_events=[{"positive": True}],
            )
            for group in ("under", "nominal", "over")
        ]
        for split, market in (("train", 101), ("validation", 201), ("confirmation", 301))
    }
    selector_calls = []

    def with_selector(rows, model, threshold, name):
        selector_calls.append(name)
        return [
            {
                **row,
                "outcomes": {**row["outcomes"], name: deepcopy(row["outcomes"]["prefix-feedback"])},
            }
            for row in rows
        ]

    def compare(rows, method, reference, config):
        score = float(rows[0]["outcomes"][method]["score"])
        return {
            group: {"score": {"candidate_mean": score, "effect": score / 2 - 1}}
            for group in ("all", "under", "nominal", "over")
        }

    monkeypatch.setattr(exporter, "ROOT", root)
    monkeypatch.setattr(exporter, "freeze", lambda config, output: {"identity": "frozen-public"})
    monkeypatch.setattr(
        exporter, "read_records", lambda output, split, config: deepcopy(records[split])
    )
    monkeypatch.setattr(exporter, "with_selector", with_selector)
    monkeypatch.setattr(exporter, "compare", compare)
    monkeypatch.setattr(
        exporter,
        "checked_development",
        lambda config, output: json.loads((output / "development.json").read_text()),
    )

    def provenance(config, output, frozen, confirmation):
        splits = (
            ("train", "validation", "confirmation") if confirmation else ("train", "validation")
        )
        markets = [market for split in splits for market in config["study"][f"{split}_markets"]]
        return {
            "inputs": [{"market": m} for m in markets],
            "markets": [{"market": m} for m in markets],
        }

    monkeypatch.setattr(exporter, "public_provenance", provenance)
    for filename in ("data-freeze.json", "execution-runtime.json"):
        dump(output / filename, {"identity": "public-input"})
    for market in (101, 201, 301):
        for directory in ("inputs", "markets"):
            dump(output / directory / f"{market}.json", {"market": market, "sha256": "a" * 64})
    (output / "private.log").write_text("never publish")
    (output / "weights.npz").write_bytes(b"not a public evidence file")
    return config, output, destination, selector_calls


def development(status):
    candidates = (
        []
        if status == "STOP_NO_HEADROOM"
        else [
            dict(
                name=f"ridge-a{alpha}-t{threshold}",
                model={"alpha": alpha},
                threshold=threshold,
                stats={"score": float(index)},
            )
            for index, (alpha, threshold) in enumerate(
                (a, t) for a in (1, 10, 100) for t in (0, 0.01)
            )
        ]
    )
    result = dict(status=status, reference="full-feedback", candidate_results=candidates)
    if candidates:
        result["selected"] = candidates[-1]["name"]
    return result


@pytest.mark.parametrize("status", ["STOP_NO_HEADROOM", "STOP_DEV_FAILURE"])
def test_stopped_study_exports_only_public_scalars_and_preserves_all_candidates(study, status):
    config, output, destination, selector_calls = study
    declared = development(status)
    dump(output / "development.json", declared)
    exporter.export(config, output, destination)
    assert {p.name for p in destination.iterdir()} == PUBLIC_FILES
    for path in destination.iterdir():
        content = path.read_bytes()
        assert b"\r" not in content
        assert all(line == line.rstrip() for line in content.splitlines())
    assert not any(p.is_symlink() for p in destination.iterdir())
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["status"] == status
    assert summary["confirmation"] is None
    assert summary["development"] == declared
    assert "Hindsight is nondeployable" in summary["interpretation"]
    report = (destination / "report.md").read_text()
    assert "uses future outcomes and is not deployable" in report
    assert "do not establish generalization" in report
    assert "CONFIRM_PASS" not in report
    assert summary["development"]["reference"] != "hindsight-two-experts"
    names = [candidate["name"] for candidate in declared["candidate_results"]]
    assert selector_calls == names
    assert set(summary["development_comparisons"]) == set(BASE_METHODS) | set(names)
    with (destination / "results.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert all(
        set(row) == {"split", "market", "focal", "bias_group", "method", *exporter.METRICS}
        for row in rows
    )
    assert {row["method"] for row in rows if row["split"] == "validation"} == set(
        BASE_METHODS
    ) | set(names)
    assert all(sum(row["method"] == name for row in rows) == 3 for name in names)
    with (destination / "features.csv").open() as stream:
        feature_rows = list(csv.DictReader(stream))
    assert len(feature_rows) == 6
    assert all("target" in row and "raw_events" not in row for row in feature_rows)
    assert "event_ids" not in (destination / "results.csv").read_text()
    manifest = json.loads((destination / "export.json").read_text())
    assert manifest["status"] == status
    assert set(manifest["files_sha256"]) == PUBLIC_FILES - {"export.json"}
    for name, sha256 in manifest["files_sha256"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == sha256
    assert (destination / "comparison.svg").read_text().lstrip().startswith("<?xml")


def test_ready_without_confirmation_refuses_publication(study):
    config, output, destination, _ = study
    dump(output / "development.json", development("READY_FOR_CONFIRMATION"))
    with pytest.raises((FileNotFoundError, RuntimeError)):
        exporter.export(config, output, destination)
    assert not destination.exists()


@pytest.mark.parametrize("status", ["CONFIRM_PASS", "CONFIRM_FAILURE"])
def test_ready_rechecks_confirmation_and_includes_confirmation_provenance(
    study, monkeypatch, status
):
    config, output, destination, selector_calls = study
    declared = development("READY_FOR_CONFIRMATION")
    candidate = declared["candidate_results"][-1]
    saved = dict(status=status, candidate=candidate["name"], reference=declared["reference"])
    dump(output / "development.json", declared)
    dump(output / "confirmation.json", saved)
    dump(output / "selection-freeze.json", {"candidate": candidate})
    dump(output / "confirmation-open.json", {"identity": "immutable-public-selection"})
    calls = []

    def confirm(config_arg, output_arg):
        calls.append((config_arg, output_arg))
        return deepcopy(saved)

    monkeypatch.setattr(exporter, "confirm", confirm)
    exporter.export(config, output, destination)
    assert calls == [(config, output)]
    assert selector_calls == [c["name"] for c in declared["candidate_results"]] + [
        candidate["name"]
    ]
    summary = json.loads((destination / "summary.json").read_text())
    assert summary["status"] == status
    assert summary["confirmation"] == saved
    provenance = json.loads((destination / "provenance.json").read_text())
    assert {row["market"] for row in provenance["inputs"]} == {101, 201, 301}
    assert {row["market"] for row in provenance["markets"]} == {101, 201, 301}


def test_ready_refuses_confirmation_that_fails_immutable_recheck(study, monkeypatch):
    config, output, destination, _ = study
    dump(output / "development.json", development("READY_FOR_CONFIRMATION"))
    dump(output / "confirmation.json", {"status": "CONFIRM_PASS"})

    def confirm(*args):
        raise RuntimeError("Immutable confirmation result changed")

    monkeypatch.setattr(exporter, "confirm", confirm)
    with pytest.raises(RuntimeError, match="Immutable confirmation"):
        exporter.export(config, output, destination)
    assert not destination.exists()


@pytest.mark.parametrize("status", ["RUNNING", "STOP_DEV_FAILURE", "CONFIRM_RUNNING"])
def test_ready_refuses_nonterminal_confirmation_before_rechecking(study, monkeypatch, status):
    config, output, destination, _ = study
    dump(output / "development.json", development("READY_FOR_CONFIRMATION"))
    dump(output / "confirmation.json", {"status": status})
    monkeypatch.setattr(
        exporter,
        "confirm",
        lambda *args: pytest.fail("A nonterminal confirmation must not be accepted"),
    )
    with pytest.raises(RuntimeError, match="terminal"):
        exporter.export(config, output, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "status", ["RUNNING", "STOP_NOT_REGISTERED", "CONFIRM_RUNNING", "CONFIRM_PASS"]
)
def test_unregistered_or_unconfirmed_development_status_refuses_publication(study, status):
    config, output, destination, _ = study
    dump(output / "development.json", development(status))
    with pytest.raises(RuntimeError):
        exporter.export(config, output, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "private",
    [
        b"/Users/example/private",
        b"/home/example/private",
        b"github_pat_abcdefghijklmnopqrstuv",
        b"ghp_abcdefghijklmnopqrstuv",
        b"sk-abcdefghijklmnopqrstuv",
        b"AKIAABCDEFGHIJKLMNOP",
        b"example@example.com",
        b"api_key=not-public",
        b'"access_token": "not-public"',
        b"password=not-public",
        b"-----BEGIN RSA PRIVATE KEY-----",
    ],
)
def test_safe_write_rejects_private_paths_and_tokens_without_overwrite(tmp_path, private):
    destination = tmp_path / "public.json"
    destination.write_bytes(b"preserve")
    with pytest.raises(RuntimeError, match="privacy"):
        exporter.safe_write(destination, private)
    assert destination.read_bytes() == b"preserve"


def test_destination_must_resolve_inside_repository_reports(study, tmp_path):
    config, output, destination, _ = study
    dump(output / "development.json", development("STOP_NO_HEADROOM"))
    with pytest.raises(RuntimeError, match="reports"):
        exporter.export(config, output, tmp_path / "outside")
    outside = tmp_path / "outside-existing"
    outside.mkdir()
    destination.parent.mkdir(parents=True)
    destination.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="reports|symlink"):
        exporter.export(config, output, destination)
    assert not list(outside.iterdir())


def test_existing_file_symlink_cannot_escape_output_root(study, tmp_path):
    config, output, destination, _ = study
    dump(output / "development.json", development("STOP_NO_HEADROOM"))
    destination.mkdir(parents=True)
    outside = tmp_path / "preserve.txt"
    outside.write_text("preserve external file")
    (destination / "results.csv").symlink_to(outside)
    with pytest.raises(RuntimeError, match="symlink|output|reports"):
        exporter.export(config, output, destination)
    assert outside.read_text() == "preserve external file"


def test_system_path_alias_outside_repository_does_not_block_publication(
    study, tmp_path, monkeypatch
):
    config, output, _, _ = study
    dump(output / "development.json", development("STOP_NO_HEADROOM"))
    # Model /var -> /private/var without depending on a particular host layout.
    root = exporter.ROOT
    root.mkdir()
    alias = tmp_path / "system-path-alias"
    alias.symlink_to(root, target_is_directory=True)
    monkeypatch.setattr(exporter, "ROOT", alias)
    destination = alias / "reports/expert-selection"
    exporter.export(config, output, destination)
    assert {p.name for p in destination.iterdir()} == PUBLIC_FILES


def test_symlink_within_reports_is_rejected_even_when_target_is_inside_reports(study):
    config, output, destination, _ = study
    dump(output / "development.json", development("STOP_NO_HEADROOM"))
    reports = destination.parent
    target = reports / "canonical"
    target.mkdir(parents=True)
    alias = reports / "alias"
    alias.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        exporter.export(config, output, alias / "evidence")
    assert not list(target.iterdir())


def test_unregistered_existing_output_is_rejected_and_preserved(study):
    config, output, destination, _ = study
    dump(output / "development.json", development("STOP_NO_HEADROOM"))
    destination.mkdir(parents=True)
    extra = destination / "unregistered.log"
    extra.write_text("preserve unregistered file")
    with pytest.raises(RuntimeError, match="existing|unexpected|registered|output"):
        exporter.export(config, output, destination)
    assert extra.read_text() == "preserve unregistered file"


@pytest.fixture
def decision_files(tmp_path):
    output = tmp_path / "decision"
    config = {"study": {"train_markets": [101], "validation_markets": [201]}}
    for name in ("experiment-freeze.json", "data-freeze.json"):
        dump(output / name, {"identity": "public-source"})
    for split, market in (("train", 101), ("validation", 201)):
        for name in (f"inputs/{market}.json", f"markets/{market}.json", f"{split}/{market}.json"):
            dump(output / name, {"market": market, "split": split})
        (output / "markets" / f"{market}.npz").write_bytes(b"synthetic-unopened-array")
    dump(output / "confirmation.json", {"must_not_be_opened": True})
    return config, output


def snapshot(directory):
    return {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}


def fake_development(monkeypatch, output, recomputed, *, selection=None, mirrors=None):
    def develop(config, mirror):
        assert mirror != output
        assert not (mirror / "confirmation.json").exists()
        assert not (mirror / "selection-freeze.json").exists()
        for split in ("train", "validation"):
            for market in config["study"][f"{split}_markets"]:
                assert (mirror / split / f"{market}.json").exists()
                assert (
                    mirror / "markets" / f"{market}.npz"
                ).read_bytes() == b"synthetic-unopened-array"
        dump(mirror / "development.json", recomputed)
        if selection is not None:
            dump(mirror / "selection-freeze.json", selection)
        if mirrors is not None:
            mirrors.append(mirror)
        return deepcopy(recomputed)

    monkeypatch.setattr(exporter, "develop", develop)


@pytest.mark.parametrize("status", ["STOP_NO_HEADROOM", "STOP_DEV_FAILURE"])
def test_checked_development_recomputes_without_mutating_saved_files(
    decision_files, monkeypatch, status
):
    config, output = decision_files
    declared = development(status)
    dump(output / "development.json", declared)
    before, mirrors = snapshot(output), []
    fake_development(monkeypatch, output, declared, mirrors=mirrors)
    assert CHECKED_DEVELOPMENT(config, output) == declared
    assert snapshot(output) == before
    assert len(mirrors) == 1 and not mirrors[0].exists()


@pytest.mark.parametrize(
    "field", ["status", "reference", "candidate_results", "model", "threshold"]
)
def test_checked_development_rejects_forged_stop_decision_without_rewriting_source(
    decision_files, monkeypatch, field
):
    config, output = decision_files
    recomputed = development("STOP_DEV_FAILURE")
    saved = deepcopy(recomputed)
    if field == "status":
        saved["status"] = "STOP_NO_HEADROOM"
    elif field == "reference":
        saved["reference"] = "hindsight-two-experts"
    elif field == "candidate_results":
        saved["candidate_results"] = []
    elif field == "model":
        saved["candidate_results"][0]["model"] = {"alpha": 999}
    else:
        saved["candidate_results"][0]["threshold"] = 999
    dump(output / "development.json", saved)
    before = snapshot(output)
    fake_development(monkeypatch, output, recomputed)
    with pytest.raises(RuntimeError, match="development decision"):
        CHECKED_DEVELOPMENT(config, output)
    assert snapshot(output) == before


def test_checked_development_rejects_forged_ready_selection(decision_files, monkeypatch):
    config, output = decision_files
    declared = development("READY_FOR_CONFIRMATION")
    dump(output / "development.json", declared)
    selection = {"candidate": declared["candidate_results"][-1], "reference": declared["reference"]}
    saved = {**selection, "reference": "hindsight-two-experts"}
    dump(output / "selection-freeze.json", saved)
    before = snapshot(output)
    fake_development(monkeypatch, output, declared, selection=selection)
    with pytest.raises(RuntimeError, match="selection"):
        CHECKED_DEVELOPMENT(config, output)
    assert snapshot(output) == before


def test_checked_development_cross_device_link_fallback_remains_read_only(
    decision_files, monkeypatch
):
    config, output = decision_files
    declared = development("STOP_NO_HEADROOM")
    dump(output / "development.json", declared)
    before = snapshot(output)
    fake_development(monkeypatch, output, declared)

    def unavailable_link(*args):
        raise OSError("Different filesystem")

    monkeypatch.setattr(exporter.os, "link", unavailable_link)
    assert CHECKED_DEVELOPMENT(config, output) == declared
    assert snapshot(output) == before


@pytest.fixture
def provenance_files(tmp_path):
    config = exporter.load_config(
        Path(__file__).resolve().parents[1] / "configs/bidding-expert-selection.toml"
    )
    config["study"].update(
        train_markets=[101], validation_markets=[201], confirmation_markets=[301]
    )
    config["limits"]["torch_threads"] = 1
    output = tmp_path / "metadata"
    digest = "a" * 64
    generator = {"source": "public-synthetic"}
    frozen = dict(
        config=deepcopy(config),
        source={"public.py": digest},
        weights={"model.npz": digest},
        generator=generator,
        protocol_sha256=digest,
        identity=digest,
    )
    dump(
        output / "data-freeze.json",
        {k: frozen[k] for k in ("config", "generator", "protocol_sha256", "identity")},
    )
    runtime = dict(
        device="cpu",
        identity=digest,
        numpy="2.0.0",
        pre_outcome_commit="abc1234",
        python="3.11.0",
        scipy="1.0.0",
        threads=1,
        torch="2.0.0",
    )
    dump(output / "execution-runtime.json", runtime)
    for market in (101, 201, 301):
        dump(
            output / "inputs" / f"{market}.json",
            dict(
                market=market, generator_identity=digest, data_identity=digest, tape_sha256=digest
            ),
        )
        dump(
            output / "markets" / f"{market}.json",
            dict(
                bytes=123,
                effective_actions=[0.0, 1.0],
                focal_agents=[0],
                identity=digest,
                market_id=market,
                pv_count=2,
                requested_pv_count=2,
                seconds=0.1,
                source=generator,
            ),
        )
    dump(output / "selection-freeze.json", {"candidate": "public-candidate"})
    dump(output / "confirmation-open.json", {"identity": digest})
    return config, output, frozen


@pytest.mark.parametrize("confirmed", [False, True])
def test_public_provenance_records_only_opened_splits_and_recorded_runtime(
    provenance_files, confirmed
):
    config, output, frozen = provenance_files
    result = PUBLIC_PROVENANCE(
        config, output, frozen, {"status": "CONFIRM_FAILURE"} if confirmed else None
    )
    markets = {101, 201, 301} if confirmed else {101, 201}
    assert {row["market"] for row in result["inputs"]} == markets
    assert {row["market_id"] for row in result["markets"]} == markets
    assert result["runtime"]["status"] == "recorded"
    assert ("confirmation_open" in result) is confirmed
    assert ("confirmation_selection" in result) is confirmed


def test_public_provenance_does_not_fabricate_missing_execution_runtime(provenance_files):
    config, output, frozen = provenance_files
    (output / "execution-runtime.json").unlink()
    result = PUBLIC_PROVENANCE(config, output, frozen, None)
    assert result["runtime"]["status"] == "not_recorded"
    assert not {"python", "numpy", "torch", "scipy"} & result["runtime"].keys()


@pytest.mark.parametrize(
    "filename",
    ["execution-runtime.json", "data-freeze.json", "inputs/101.json", "markets/101.json"],
)
def test_public_provenance_rejects_unregistered_notes(provenance_files, filename):
    config, output, frozen = provenance_files
    path = output / filename
    record = json.loads(path.read_text())
    record["notes"] = "Unregistered internal annotation"
    dump(path, record)
    with pytest.raises(RuntimeError, match="metadata fields"):
        PUBLIC_PROVENANCE(config, output, frozen, None)


@pytest.mark.parametrize(
    "field,value",
    [
        ("threads", True),
        ("threads", 1.0),
        ("threads", "1"),
        ("python", 311),
        ("device", "cuda"),
        ("pre_outcome_commit", "not-a-commit"),
    ],
)
def test_public_provenance_rejects_invalid_runtime_types_and_values(provenance_files, field, value):
    config, output, frozen = provenance_files
    path = output / "execution-runtime.json"
    record = json.loads(path.read_text())
    record[field] = value
    dump(path, record)
    with pytest.raises(RuntimeError, match="runtime"):
        PUBLIC_PROVENANCE(config, output, frozen, None)
