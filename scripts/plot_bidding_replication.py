"""Plot public discovery and fixed-model confirmation summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SCENARIOS = {
    "calibrated": "Nominal",
    "over_25": "Overestimate 25%",
    "over_50": "Overestimate 50%",
    "under_25": "Underestimate 25%",
    "mid_over_50": "Mid-period +50%",
}
STUDIES = ("Discovery (12 markets)", "Confirmation D (24 markets)")
COLORS = ("#4c78a8", "#f58518")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    return raw, json.loads(raw)


def discovery_points(summary: dict) -> list[dict]:
    rows = {
        (row["scenario"], row["metric"]): row
        for row in summary["paired_comparisons"]
        if row.get("method") == "response"
        and row.get("reference") == "feedback"
        and row.get("metric") in ("score", "violation")
    }
    return [
        {
            "study": STUDIES[0],
            "scenario": scenario,
            "metric": metric,
            "mean": rows[scenario, metric]["mean_delta"],
            "ci95": rows[scenario, metric]["market_bootstrap_95"],
        }
        for scenario in SCENARIOS
        for metric in ("score", "violation")
    ]


def confirmation_points(summary: dict) -> list[dict]:
    if summary.get("primary_scenario") != "over_50":
        raise RuntimeError("Confirmation D must identify over_50 as its sole primary scenario")
    effects = summary["scenario_effects"]
    points = []
    for scenario in SCENARIOS:
        effect = effects[scenario]
        for metric, key in (("score", "score_delta"), ("violation", "violation_delta")):
            points.append(
                {
                    "study": STUDIES[1],
                    "scenario": scenario,
                    "metric": metric,
                    "mean": effect[key]["mean"],
                    "ci95": effect[key]["market_bootstrap_95"],
                }
            )
    return points


def checked_confirmation(directory: Path) -> tuple[bytes, dict, bytes, dict, bytes]:
    summary_bytes, summary = read_json(directory / "summary.json")
    status_bytes, status = read_json(directory / "status.json")
    export_bytes, export = read_json(directory / "export.json")
    identity = summary.get("identity")
    hashes = export.get("files_sha256", {})
    expected_hashes = {
        "manifest.json",
        "status.json",
        "summary.json",
        "results.json",
        "results.csv",
        "report.html",
        "market-cache.json",
    }
    exported_hashes_match = set(hashes) == expected_hashes and all(
        (directory / name).is_file() and digest((directory / name).read_bytes()) == expected
        for name, expected in hashes.items()
    )
    legal_decisions = {"confirmed", "not-confirmed", "inconclusive"}
    if (
        status.get("phase") != "complete"
        or status.get("identity") != identity
        or status.get("decision") != summary.get("decision")
        or export.get("run_identity") != identity
        or export.get("decision") != summary.get("decision")
        or summary.get("decision") not in legal_decisions
        or not exported_hashes_match
        or summary.get("sample_size", {}).get("markets") != 24
        or summary.get("sample_size", {}).get("rows") != 4320
    ):
        raise RuntimeError("Confirmation D is incomplete or its public export hashes differ")
    return summary_bytes, summary, status_bytes, status, export_bytes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--main-summary", type=Path, default=Path("reports/bidding-main/summary.json")
    )
    parser.add_argument(
        "--confirmation-dir", type=Path, default=Path("reports/bidding-confirmation")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/bidding-replication"))
    args = parser.parse_args()

    main_bytes, main_summary = read_json(args.main_summary)
    d_summary_bytes, d_summary, d_status_bytes, d_status, d_export_bytes = checked_confirmation(
        args.confirmation_dir
    )
    points = discovery_points(main_summary) + confirmation_points(d_summary)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.hashsalt": "trajectory-lab-bidding-replication-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.20, top=0.83, wspace=0.35)
    positions = {
        (scenario, study): index * 2 + offset
        for index, scenario in enumerate(SCENARIOS)
        for offset, study in enumerate(STUDIES)
    }
    for ax, metric, title, xlabel, scale in (
        (axes[0], "score", "Score difference", "Response − feedback score", 1),
        (axes[1], "violation", "CPA violation difference", "Percentage-point difference", 100),
    ):
        ax.axvline(0, color="#333333", linewidth=1.1, zorder=0)
        for point in points:
            if point["metric"] != metric:
                continue
            y = positions[point["scenario"], point["study"]]
            mean = point["mean"] * scale
            lo, hi = (bound * scale for bound in point["ci95"])
            color = COLORS[STUDIES.index(point["study"])]
            ax.errorbar(
                mean,
                y,
                xerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=5,
                linewidth=1.6,
                label=point["study"],
            )
        ax.set_yticks([index * 2 + 0.5 for index in range(len(SCENARIOS))], SCENARIOS.values())
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    legend = dict(zip(labels, handles, strict=True))
    fig.suptitle("Simulation: discovery vs fixed-model confirmation", y=0.98, weight="bold")
    fig.legend(
        [legend[label] for label in STUDIES],
        STUDIES,
        loc="upper center",
        bbox_to_anchor=(0.62, 0.94),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.025,
        "Discovery is exploratory; D is fixed-model confirmation.\n"
        "95% intervals resample markets conditional on three checkpoints; they exclude training randomness and are not online outcomes.\n"
        "Only overestimate 50% is primary in D; nominal and underestimate results remain scenario-specific.",
        ha="center",
        fontsize=8.5,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    svg, png = args.output / "replication.svg", args.output / "replication.png"
    fig.savefig(
        svg,
        bbox_inches="tight",
        pad_inches=0.15,
        metadata={"Date": None, "Creator": "TrajectoryLab"},
    )
    fig.savefig(
        png, dpi=180, bbox_inches="tight", pad_inches=0.15, metadata={"Software": "matplotlib"}
    )
    plt.close(fig)
    record = {
        "studies": {
            STUDIES[0]: {
                "identity": main_summary["identity"],
                "summary_sha256": digest(main_bytes),
            },
            STUDIES[1]: {
                "identity": d_summary["identity"],
                "summary_sha256": digest(d_summary_bytes),
                "status_sha256": digest(d_status_bytes),
                "export_sha256": digest(d_export_bytes),
                "decision": d_summary["decision"],
            },
        },
        "uncertainty_scope": "95% market-bootstrap intervals conditional on three fixed checkpoints; not training-randomness intervals",
        "primary_scope": "Only over_50 is primary in confirmation D; other scenarios are descriptive secondary evidence",
        "points": points,
        "generator_sha256": digest(Path(__file__).read_bytes()),
        "figure_sha256": {svg.name: digest(svg.read_bytes()), png.name: digest(png.read_bytes())},
    }
    (args.output / "figure.json").write_text(
        json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
