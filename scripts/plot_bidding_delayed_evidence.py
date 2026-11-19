#!/usr/bin/env python3
"""Plot recorded confirmation estimates without selecting or fitting a policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def seal_publication(directory: Path):
    """Bind scalar exports and both plots into one inspectable file inventory."""
    directory = Path(directory)
    if directory.is_symlink() or any(p.is_symlink() for p in directory.rglob("*")):
        raise ValueError("Public evidence must not contain symlinks")
    manifest = json.loads((directory / "export-manifest.json").read_text())
    files = dict(manifest["files"])
    for name, expected in files.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid public manifest path")
        path = directory / relative
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Scalar export changed before publication")
    for name in ("export-manifest.json", "confirmation-effects.svg", "confirmation-stress.svg"):
        path = directory / name
        if path.is_symlink():
            raise ValueError("Public evidence must not contain symlinks")
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual - {"publication-manifest.json"} != set(files):
        raise ValueError("Unlisted file in public evidence directory")
    result = {"schema": "delayed-feedback-publication-v1", "files": dict(sorted(files.items()))}
    (directory / "publication-manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def plot_evidence(directory: Path):
    directory = Path(directory)
    summary = json.loads((directory / "confirmation/summary.json").read_text())
    primary = summary["primary_comparison"]
    groups = ("all", "nominal", "under")
    labels = ("All stresses", "Nominal pCVR", "Underprediction")
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.8), layout="constrained")
    for ax, metric, title in zip(
        axes, ("score_ratio", "conversion_ratio", "violation_delta"),
        ("Mature score change (%)", "Mature conversions (%)", "CPA violation change (pp)"),
        strict=True,
    ):
        for position, group in enumerate(groups):
            result = primary["groups"][group]
            value = result[metric]
            low, high = result[f"{metric}_ci95"]
            offset = 0 if metric == "violation_delta" else 1
            value, low, high = (100 * (v - offset) for v in (value, low, high))
            ax.errorbar(value, position, xerr=[[value-low], [high-value]], fmt="o", color="#16697a", capsize=4)
            ax.annotate(f"{value:+.2f}", (value, position), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=8)
            target = 3 if metric == "violation_delta" else (-3 if group != "all" else (3 if metric == "score_ratio" else None))
            if target is not None:
                ax.plot([target, target], [position-.28, position+.28], color="#be6e46", linestyle="--", linewidth=1.5)
        ax.axvline(0, color="#87959a", linewidth=1)
        ax.set_yticks(range(3), labels if ax is axes[0] else ("", "", ""))
        ax.set_ylim(2.6, -.6)
        ax.set_xlabel(title)
        ax.grid(axis="x", alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Frozen feedback correction vs tuned arrived-only feedback\n"
                 "Bars: descriptive paired-market 95% intervals; dashed marks: practical thresholds.\n"
                 "Protection gates use separate multiplicity-adjusted one-sided bounds.", fontsize=10)
    fig.savefig(directory / "confirmation-effects.svg", metadata={"Date": None, "Creator": "TrajectoryLab"})
    plt.close(fig)

    selected = summary["strongest_point"]
    naive = next(v["name"] for v in summary["selection"]["frozen_variants"] if v["estimator"] == "naive")
    cells = {(v["method"], v["bias_group"], v["delay"]): v["score"] for v in summary["cells"]}
    regimes, biases = ("immediate", "short", "long"), ("under", "nominal", "over")
    values = np.array([
        [100*(cells[selected, group, delay]/cells[naive, group, delay]-1) for delay in regimes]
        for group in biases
    ])
    limit = max(1., float(np.max(np.abs(values))))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), layout="constrained")
    im = ax.imshow(values, cmap="BrBG", vmin=-limit, vmax=limit)
    for (r, c), value in np.ndenumerate(values):
        ax.text(c, r, f"{value:+.2f}%", ha="center", va="center", color="white" if abs(value) > .65*limit else "#16282d")
    ax.set_xticks(range(3), ("Immediate", "Short delay", "Long delay"))
    ax.set_yticks(range(3), ("Underprediction", "Nominal pCVR", "Overprediction"))
    ax.set_title("Mature score change by stress cell\nDescriptive; cells share the same 24 markets", fontsize=11)
    fig.colorbar(im, ax=ax, label="Change vs tuned arrived-only (%)", shrink=.8)
    fig.savefig(directory / "confirmation-stress.svg", metadata={"Date": None, "Creator": "TrajectoryLab"})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/bidding-delayed-feedback"))
    directory = parser.parse_args().input
    plot_evidence(directory)
    seal_publication(directory)
