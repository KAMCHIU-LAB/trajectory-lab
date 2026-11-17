"""Plot existing exploratory response-versus-feedback market bootstrap results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


SCENARIOS = {
    "calibrated": "Nominal",
    "over_25": "Overestimate 25%",
    "over_50": "Overestimate 50%",
    "under_25": "Underestimate 25%",
    "mid_over_50": "Mid-period +50%",
}
METRICS = ("score", "violation")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("reports/bidding-main/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/bidding-analysis"))
    args = parser.parse_args()

    summary_bytes = args.summary.read_bytes()
    summary = json.loads(summary_bytes)
    rows = {
        (row["scenario"], row["metric"]): row
        for row in summary["paired_comparisons"]
        if row.get("method") == "response"
        and row.get("reference") == "feedback"
        and row.get("metric") in METRICS
    }
    points = []
    for scenario in SCENARIOS:
        for metric in METRICS:
            row = rows[(scenario, metric)]
            points.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "mean_delta": row["mean_delta"],
                    "market_bootstrap_95": row["market_bootstrap_95"],
                    "seed_mean_deltas": row["seed_mean_deltas"],
                    "uncertainty_scope": row["uncertainty_scope"],
                }
            )

    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "svg.hashsalt": "trajectory-lab-bidding-tradeoffs-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.86, wspace=0.34)
    colors = ["#4c78a8", "#72b7b2", "#f58518", "#e45756", "#79706e"]
    for ax, metric, title, xlabel, scale in (
        (axes[0], "score", "Score difference", "Response − feedback score", 1),
        (
            axes[1],
            "violation",
            "CPA violation difference",
            "Percentage-point difference",
            100,
        ),
    ):
        selected = [
            next(p for p in points if p["metric"] == metric and p["scenario"] == s)
            for s in SCENARIOS
        ]
        means = [p["mean_delta"] * scale for p in selected]
        lower = [p["market_bootstrap_95"][0] * scale for p in selected]
        upper = [p["market_bootstrap_95"][1] * scale for p in selected]
        y = list(range(len(SCENARIOS)))
        ax.axvline(0, color="#333333", linewidth=1.1, zorder=0)
        for i, (mean, lo, hi) in enumerate(zip(means, lower, upper, strict=True)):
            ax.errorbar(
                mean,
                y[i],
                xerr=[[mean - lo], [hi - mean]],
                fmt="o",
                color=colors[i],
                capsize=3,
                markersize=6,
                linewidth=1.7,
            )
        ax.set_yticks(y, [SCENARIOS[s] for s in SCENARIOS])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.8)
    fig.suptitle("Exploratory response vs feedback across stress scenarios", weight="bold")
    fig.text(
        0.5,
        0.03,
        "12 test markets. 95% market-bootstrap intervals are conditional on three checkpoints. "
        "Exploratory comparison; not an online outcome.",
        ha="center",
        fontsize=9,
    )
    svg = args.output / "bidding-tradeoffs.svg"
    png = args.output / "bidding-tradeoffs.png"
    fig.savefig(
        svg,
        bbox_inches="tight",
        pad_inches=0.15,
        metadata={"Date": None, "Creator": "TrajectoryLab"},
    )
    fig.savefig(
        png,
        dpi=180,
        bbox_inches="tight",
        pad_inches=0.15,
        metadata={"Software": "matplotlib"},
    )
    plt.close(fig)

    script_path = Path(__file__)
    record = {
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "script_sha256": sha256(script_path),
        "source_identity": summary["identity"],
        "claim_scope": "exploratory; 12 markets; intervals conditional on three checkpoints; not online outcome",
        "points": points,
        "figure_sha256": {svg.name: sha256(svg), png.name: sha256(png)},
    }
    (args.output / "figure.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
