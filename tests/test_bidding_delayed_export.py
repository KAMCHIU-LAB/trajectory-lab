import csv
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "delayed_export",
    Path(__file__).resolve().parents[1] / "scripts/export_bidding_delayed_evidence.py",
)
exporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exporter)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def row(focal=0, seed=11, score=2.0):
    return dict(
        market=6101,
        focal=focal,
        bias_group="under",
        delay="short",
        arrival_seed=seed,
        method="posterior-g0.3-p8",
        estimator="posterior",
        gain=0.3,
        prior=8.0,
        cost=10.0,
        conversions=2,
        exposures=100,
        budget=100.0,
        target_cpa=10.0,
        cpa=5.0,
        violation=0,
        score=score,
        budget_utilization=0.1,
        pending_mae=0.2,
        final_arrived=2,
        zero_conversions=0,
        deployable=True,
        actions=[0.5, 1.0],
        private_id="secret-not-exported",
        private_path="/Users/private/cache",
    )


def fixture_input(path):
    for name in exporter.JSON_FILES:
        data = {"identity": "abc", "private_path": "/Users/private/cache"}
        if name.endswith("status.json"):
            data.update(phase="complete", rows=4)
        elif name.endswith("summary.json"):
            data.update(split=name.split("/")[0], final_metrics="mature at H+W")
        elif name == "history.json":
            data = {
                "short": {
                    "probabilities": [0.5, 1.0],
                    "training_sample_count": 2,
                    "training_markets": [1001],
                    "all_mature": True,
                }
            }
        elif name.endswith("manifest.json"):
            data["core"] = {
                "config": {"study": {"actions": [0, 1], "name": "public-study"}},
                "source": {"src/trajectorylab/bidding_pending_response.py": "abc"},
            }
        dump(path / name, data)
    campaigns = [
        row(focal, seed, score=float(focal + seed)) for focal in (0, 8) for seed in (11, 23)
    ]
    for split in ("validation", "confirmation"):
        dump(path / split / "results.json", campaigns)
    (path / "private.log").write_text("not public")
    (path / "weights.bin").write_bytes(b"not public")


def test_allowlist_csv_aggregation_and_deterministic_export(tmp_path):
    source, output = tmp_path / "input", tmp_path / "public"
    fixture_input(source)
    first = exporter.export_evidence(source, output)
    assert first == exporter.export_evidence(source, output)
    assert set(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()) == set(
        first["files"]
    ) | {"export-manifest.json"}
    assert not (output / "private.log").exists()
    with (output / "confirmation.csv").open() as stream:
        campaigns = list(csv.DictReader(stream))
    assert len(campaigns) == 4
    assert set(campaigns[0]) == set(exporter.COLUMNS)
    assert "actions" not in campaigns[0] and "private_id" not in campaigns[0]
    with (output / "confirmation-markets.csv").open() as stream:
        aggregate = list(csv.DictReader(stream))
    assert len(aggregate) == 1
    assert float(aggregate[0]["score"]) == 21.0
    assert aggregate[0]["campaign_count"] == "4"
    assert aggregate[0]["focal_count"] == aggregate[0]["arrival_seed_count"] == "2"
    assert json.loads((output / "history.json").read_text())["short"]["training_markets"] == [1001]
    assert "private_path" not in (output / "selection.json").read_text()
    readme = (output / "README.md").read_text()
    assert "not independent statistical" in readme and "No historical trained weights" in readme


@pytest.mark.parametrize(
    "unsafe",
    [
        "/Users/person/cache",
        "read /home/person/file",
        "/etc/passwd",
        "person@example.com",
        "api_key=hidden",
        "file:///private/cache",
        r"C:\Users\person\cache",
        "sk-abcdefghijklmnopqrstuv",
    ],
)
def test_forbidden_export_string_rejected_without_partial_output(tmp_path, unsafe):
    source, output = tmp_path / "input", tmp_path / "public"
    fixture_input(source)
    dump(source / "selection.json", {"history_contract": unsafe})
    with pytest.raises(ValueError, match="public export"):
        exporter.export_evidence(source, output)
    assert not output.exists()


def test_confirmation_required_and_complete(tmp_path):
    source, output = tmp_path / "input", tmp_path / "public"
    fixture_input(source)
    (source / "confirmation/summary.json").unlink()
    with pytest.raises(ValueError, match="confirmation summary"):
        exporter.export_evidence(source, output)
    assert not output.exists()
    fixture_input(source)
    dump(source / "confirmation/status.json", {"phase": "running"})
    with pytest.raises(ValueError, match="complete"):
        exporter.export_evidence(source, output)
    assert not output.exists()


def test_full_market_cell_count_and_null_cpa():
    campaigns = []
    for method in ["naive", "mature_only", "delay_corrected", "posterior", "oracle_past"]:
        for market in range(6101, 6125):
            for group in ["under", "nominal", "over"]:
                for delay in ["immediate", "short", "long"]:
                    for seed in [11, 23]:
                        item = row(seed=seed)
                        item.update(
                            method=method,
                            estimator=method,
                            market=market,
                            bias_group=group,
                            delay=delay,
                            cpa=None,
                        )
                        campaigns.append(item)
    result = exporter.market_rows(exporter.scalar_rows(campaigns))
    assert len(result) == 5 * 24 * 3 * 3
    assert all(
        r["campaign_count"] == 2 and r["cpa"] is None and r["cpa_nonnull_count"] == 0
        for r in result
    )
    assert exporter.public_json({"claim": "https://github.com/alimama-tech/AuctionNet"})[
        "claim"
    ].startswith("https://")


def test_optional_public_diagnostics_and_stale_output_rejection(tmp_path):
    source, output = tmp_path / "input", tmp_path / "public"
    fixture_input(source)
    benchmark = {
        "description": "CPU only",
        "environment": "Python",
        "results": {"posterior": {"median_ms": 0.5, "p95_ms": 1.0, "calls": 3}},
    }
    audit = {
        "aggregate": [
            {
                "method": "posterior",
                "coverage80": 0.8,
                "mean_crps": 0.2,
                "mae": 0.3,
                "decisions": 10,
            }
        ],
        "trajectories_verified": 2,
        "interpretation": "offline diagnostic",
    }
    dump(source / "benchmark.json", benchmark)
    dump(source / "offline-pending-audit/summary.json", audit)
    dump(
        source / "offline-pending-audit/manifest.json",
        {"core": {"audit_seed": 1, "ticks": [8], "posterior_samples": 256}},
    )
    (source / "offline-pending-audit/decisions.csv").write_text("private raw decisions")
    exporter.export_evidence(source, output)
    assert json.loads((output / "benchmark.json").read_text()) == benchmark
    assert json.loads((output / "offline-pending-audit/summary.json").read_text()) == audit
    assert not (output / "offline-pending-audit/decisions.csv").exists()
    stale = output / "private-cache.log"
    stale.write_text("preserve me")
    with pytest.raises(ValueError, match="unexpected existing"):
        exporter.export_evidence(source, output)
    assert stale.read_text() == "preserve me"


def test_url_keeps_credential_scan():
    with pytest.raises(ValueError):
        exporter.public_json({"claim": "https://github.com/public?api_key=hidden"})
