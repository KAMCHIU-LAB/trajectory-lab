"""Portable reports made from complete, explicitly enumerated experiment runs."""

from __future__ import annotations

import base64
import csv
import html
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trajectorylab.data import Dataset, Normalizer
from trajectorylab.utils import fingerprint, read_json, write_json

COLORS = {"bc": "#3478C6", "iql": "#138A72", "dt": "#BD6542"}


def collect_results(directory: Path, manifest: dict) -> list[dict]:
    jobs = manifest["jobs"]
    if manifest["status"] != "complete" or len(jobs) != manifest["expected_runs"]:
        raise ValueError("Refusing to label an incomplete suite as a complete report")
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("Duplicate run identities")
    results = []
    for job in jobs:
        if job["status"] != "complete":
            raise ValueError(f"Incomplete run: {job['id']}")
        result = read_json(directory / "runs" / job["id"] / "result.json")
        for key in ("algorithm", "view", "seed"):
            if result[key] != job[key]:
                raise ValueError(f"Result metadata mismatch: {key}")
        if (
            result["run_id"] != job["id"]
            or result["updates"] != manifest["config"]["training"]["steps"]
        ):
            raise ValueError("Wrong result identity or training budget")
        if result["test"]["seeds"] != manifest["config"]["evaluation"]["test_seeds"]:
            raise ValueError("Final evaluation seeds differ from the declared protocol")
        values = np.asarray(result["test"]["returns"])
        if len(values) != len(result["test"]["seeds"]) or not np.isfinite(values).all():
            raise ValueError("Invalid final evaluation returns")
        if not np.isclose(values.mean(), result["test"]["return_mean"]):
            raise ValueError("Return summary does not match raw episode returns")
        results.append(result)
    return results


def aggregate_results(results: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["view"], r["algorithm"]].append(r)
    rows = []
    for (view, algorithm), runs in sorted(groups.items()):
        if len({r["seed"] for r in runs}) != len(runs):
            raise ValueError("Duplicate training seeds cannot be treated as independent evidence")
        returns = [r["test"]["return_mean"] for r in runs]
        rows.append(
            {
                "view": view,
                "algorithm": algorithm,
                "training_seeds": len(runs),
                "return_mean": float(np.mean(returns)),
                "return_std_across_training_seeds": float(np.std(returns, ddof=1))
                if len(runs) > 1
                else None,
                "seed_returns": returns,
                "support_distance_mean": float(
                    np.mean([r["test"]["support"]["distance_mean"] for r in runs])
                ),
                "support_above_validation_p95_fraction": float(
                    np.mean(
                        [r["test"]["support"].get("above_validation_p95_fraction", 0) for r in runs]
                    )
                ),
                "wall_seconds_mean": float(np.mean([r["wall_seconds"] for r in runs])),
                "success_rate": (
                    float(np.mean([r["test"]["success_rate"] for r in runs]))
                    if runs[0]["test"]["success_rate"] is not None
                    else None
                ),
            }
        )
    return rows


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plots(
    results: list[dict], rows: list[dict], manifest: dict, directory: Path, root: Path, output: Path
) -> list[tuple[str, Path]]:
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(max(8, len(rows) * 0.95), 4.5))
    for i, row in enumerate(rows):
        ax.bar(
            i,
            row["return_mean"],
            color=COLORS[row["algorithm"]],
            alpha=0.75,
            yerr=row["return_std_across_training_seeds"] or 0,
            capsize=4,
        )
        offsets = np.linspace(-0.12, 0.12, len(row["seed_returns"]))
        ax.scatter(i + offsets, row["seed_returns"], color="#17273D", s=18, zorder=3)
    for name, reference in manifest.get("references", {}).items():
        ax.axhline(reference["return_mean"], linestyle="--", linewidth=1, label=f"{name} reference")
    ax.set_xticks(
        range(len(rows)), [f"{r['algorithm'].upper()}\n{r['view']}" for r in rows], rotation=25
    )
    ax.set_ylabel("Final raw episode return (higher is better)")
    ax.set_title("Fixed-budget final checkpoints · dots = independent training seeds")
    if manifest.get("references"):
        ax.legend(fontsize=8)
    _save(fig, figures / "returns.png")
    views = manifest["config"]["data"]["views"]
    fig, axes = plt.subplots(1, len(views), figsize=(5 * len(views), 3.8), squeeze=False)
    for ax, view in zip(axes[0], views, strict=True):
        for algorithm in manifest["config"]["suite"]["algorithms"]:
            trajectories = []
            for r in results:
                if r["view"] == view and r["algorithm"] == algorithm:
                    history = read_json(directory / "runs" / r["run_id"] / "history.json")
                    series = {
                        x["step"]: x["validation_return"]
                        for x in history
                        if "validation_return" in x
                    }
                    trajectories.append(series)
            common = (
                sorted(set.intersection(*(set(s) for s in trajectories))) if trajectories else []
            )
            if common:
                values = np.asarray([[s[k] for k in common] for s in trajectories])
                mean, std = values.mean(0), values.std(0)
                ax.plot(common, mean, label=algorithm.upper(), color=COLORS[algorithm])
                ax.fill_between(common, mean - std, mean + std, color=COLORS[algorithm], alpha=0.12)
        ax.set_title(view)
        ax.set_xlabel("Gradient updates")
        ax.set_ylabel("Validation return")
        ax.legend(fontsize=8)
    _save(fig, figures / "learning.png")
    audit = manifest["data_audit"]
    dataset = Dataset.load(root / audit["dataset_directory"])
    by_id = {e.id: e for e in dataset.episodes}
    train_ids = audit["split_episode_ids"]["train"]
    train_states = np.concatenate([by_id[i].observations[:-1] for i in train_ids])
    normalizer = Normalizer.fit(train_states)
    rng = np.random.default_rng(2026)
    reference = normalizer(
        train_states[rng.choice(len(train_states), min(5000, len(train_states)), replace=False)]
    )
    center = reference.mean(0)
    _, _, vh = np.linalg.svd(reference - center, full_matrices=False)
    projection = vh[:2].T
    fig, axes = plt.subplots(1, len(views), figsize=(4.5 * len(views), 4), squeeze=False)
    for ax, view in zip(axes[0], views, strict=True):
        episodes = [by_id[i] for i in audit["views"][view]["episode_ids"]]
        states = np.concatenate([e.observations[:-1] for e in episodes])
        sample = states[rng.choice(len(states), min(1500, len(states)), replace=False)]
        xy = (normalizer(sample) - center) @ projection
        background = (reference - center) @ projection
        ax.scatter(background[:, 0], background[:, 1], s=3, alpha=0.1, color="#8595A8")
        ax.scatter(xy[:, 0], xy[:, 1], s=4, alpha=0.25, color="#138A72")
        ax.set_title(f"{view}\n{len(states):,} training transitions")
        ax.set_xlabel("Training-pool PCA 1")
        ax.set_ylabel("Training-pool PCA 2")
    _save(fig, figures / "coverage.png")
    plots = [
        ("Final policy evaluation", figures / "returns.png"),
        ("Validation learning curves", figures / "learning.png"),
        ("Data views in a common training-only projection", figures / "coverage.png"),
    ]
    # The example is specified by seed order, never selected for an attractive score.
    source = audit["source_metadata"]
    if source["env_id"] == "Pendulum-v1" or source["env_id"].startswith("PointMaze"):
        algorithms = manifest["config"]["suite"]["algorithms"]
        seed = manifest["config"]["suite"]["seeds"][0]
        fig, axes = plt.subplots(
            len(views),
            len(algorithms),
            figsize=(4 * len(algorithms), 3.4 * len(views)),
            squeeze=False,
        )
        for row_index, view in enumerate(views):
            for column, algorithm in enumerate(algorithms):
                ax = axes[row_index, column]
                result = next(
                    r
                    for r in results
                    if (r["view"], r["algorithm"], r["seed"]) == (view, algorithm, seed)
                )
                with np.load(
                    directory / "runs" / result["run_id"] / "test_states.npz", allow_pickle=False
                ) as stored:
                    states = stored["states"][: result["test"]["episode_lengths"][0]]
                trajectory = output / "evidence" / f"{result['run_id']}-example-trajectory.csv"
                with trajectory.open("w", newline="") as stream:
                    writer = csv.writer(stream, lineterminator="\n")
                    writer.writerow(["step"] + [f"observation_{i}" for i in range(states.shape[1])])
                    writer.writerows([[i, *state.tolist()] for i, state in enumerate(states)])
                if source["env_id"] == "Pendulum-v1":
                    xy = np.column_stack((np.arctan2(states[:, 1], states[:, 0]), states[:, 2]))
                    background = np.column_stack(
                        (np.arctan2(train_states[:, 1], train_states[:, 0]), train_states[:, 2])
                    )
                    ax.set(
                        xlabel="Angle (rad)",
                        ylabel="Angular velocity (rad/s)",
                        xlim=(-np.pi, np.pi),
                        ylim=(-8.3, 8.3),
                    )
                    ax.scatter([0], [0], marker="*", color="#D7A426", s=100, zorder=4)
                else:
                    # Adapter sorts the dictionary keys: achieved_goal, desired_goal, observation.
                    xy, background = states[:, :2], train_states[:, :2]
                    ax.scatter(
                        [states[0, 2]], [states[0, 3]], marker="*", color="#D7A426", s=100, zorder=4
                    )
                    ax.set(xlabel="Achieved goal x", ylabel="Achieved goal y", aspect="equal")
                subset = np.linspace(0, len(background) - 1, min(2000, len(background))).astype(int)
                ax.scatter(
                    background[subset, 0], background[subset, 1], s=3, color="#8595A8", alpha=0.12
                )
                ax.scatter(xy[:, 0], xy[:, 1], c=np.arange(len(xy)), cmap="viridis", s=7, alpha=0.7)
                ax.scatter([xy[0, 0]], [xy[0, 1]], marker="^", color="#17273D", s=45, zorder=5)
                ax.set_title(
                    f"{algorithm.upper()} · {view}\nreturn {result['test']['returns'][0]:.1f}",
                    fontsize=10,
                )
        fig.suptitle(
            f"First declared training seed {seed} / test seed {manifest['config']['evaluation']['test_seeds'][0]} · dark → light = time",
            fontsize=11,
        )
        fig.tight_layout()
        _save(fig, figures / "trajectories.png")
        plots.append(
            (
                "Predeclared example trajectories · gray = source training support; star = goal; triangle = start",
                figures / "trajectories.png",
            )
        )
    return plots


def generate_report(directory: Path, root: Path, output: Path) -> Path:
    manifest = read_json(directory / "suite.json")
    results = collect_results(directory, manifest)
    rows = aggregate_results(results)
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "summary.json",
        {
            "suite_id": manifest["suite_id"],
            "groups": rows,
            "results": results,
            "evidence_digest": fingerprint(results),
        },
    )
    evidence = output / "evidence"
    evidence.mkdir(exist_ok=True)
    shutil.copy2(directory / "suite.json", evidence / "suite.json")
    for r in results:
        for name in ("result.json", "history.json"):
            shutil.copy2(
                directory / "runs" / r["run_id"] / name, evidence / f"{r['run_id']}-{name}"
            )
    with (output / "runs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=(
                "algorithm",
                "view",
                "seed",
                "test_return",
                "wall_seconds",
                "support_distance",
            ),
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "algorithm": r["algorithm"],
                    "view": r["view"],
                    "seed": r["seed"],
                    "test_return": r["test"]["return_mean"],
                    "wall_seconds": r["wall_seconds"],
                    "support_distance": r["test"]["support"]["distance_mean"],
                }
            )
    plots = _plots(results, rows, manifest, directory, root, output)
    sections = []
    for title, path in plots:
        encoded = base64.b64encode(path.read_bytes()).decode()
        sections.append(
            f'<section><h2>{html.escape(title)}</h2><img alt="{html.escape(title)}" src="data:image/png;base64,{encoded}"></section>'
        )
    table_rows = []
    for row in rows:
        std = row["return_std_across_training_seeds"]
        spread = f" ± {std:.2f}" if std is not None else " (single seed)"
        success = f"{row['success_rate']:.1%}" if row["success_rate"] is not None else "—"
        table_rows.append(
            f"<tr><td>{html.escape(row['view'])}</td><td>{row['algorithm'].upper()}</td>"
            f"<td>{row['training_seeds']}</td><td>{row['return_mean']:.2f}{spread}</td>"
            f"<td>{success}</td><td>{row['support_distance_mean']:.3f}</td>"
            f"<td>{row['wall_seconds_mean']:.1f}s</td></tr>"
        )
    audit = manifest["data_audit"]
    source = audit["source_metadata"]
    audit_rows = []
    for name, view in audit["views"].items():
        quantiles = view["return_quantiles"]
        audit_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{view['episodes']}</td>"
            f"<td>{view['transitions']:,}</td><td>{view['budget_fraction']:.1%}</td>"
            f"<td>{quantiles[2]:.2f}</td><td>{quantiles[0]:.2f} / {quantiles[-1]:.2f}</td>"
            f"<td>{view['terminal_count']} / {view['timeout_count']}</td>"
            f"<td>{view['action_saturation_fraction']:.1%}</td></tr>"
        )
    diagnostic_rows = []
    for row in rows:
        matching = [
            r for r in results if (r["view"], r["algorithm"]) == (row["view"], row["algorithm"])
        ]
        saturation = np.mean([r["test"]["action_saturation_fraction"] for r in matching])
        ess = [
            r["final_training_metrics"]["weight_ess_fraction"]
            for r in matching
            if "weight_ess_fraction" in r["final_training_metrics"]
        ]
        ess_display = f"{np.mean(ess):.1%}" if ess else "—"
        diagnostic_rows.append(
            f"<tr><td>{html.escape(row['view'])}</td><td>{row['algorithm'].upper()}</td>"
            f"<td>{row['support_above_validation_p95_fraction']:.1%}</td>"
            f"<td>{saturation:.1%}</td><td>{ess_display}</td></tr>"
        )
    protocol = html.escape(
        __import__("json").dumps(
            {
                "config": manifest["config"],
                "source": source,
                "runtime": manifest["runtime"],
                "source_digest": audit["source_digest"],
                "implementation_digest": manifest["implementation_digest"],
            },
            indent=2,
        )
    )
    single_seed = len(manifest["config"]["suite"]["seeds"]) < 3
    warning = (
        "Integration smoke run: these results establish execution only, not an algorithm ranking."
        if single_seed
        else "Small-budget study: results apply to this task, data subset, configuration and seeds."
    )
    document = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrajectoryLab · {html.escape(manifest["config"]["suite"]["name"])}</title>
<style>
:root{{color-scheme:light;font-family:system-ui,-apple-system,sans-serif;color:#1d2b40;background:#f3f6f9}}
body{{max-width:1180px;margin:0 auto;padding:40px 24px}}header{{padding:28px 0 24px;border-bottom:1px solid #ccd6e1}}
h1{{font-size:44px;letter-spacing:-2px;margin:12px 0}}h2{{font-size:21px;margin:0 0 20px}}p,li{{line-height:1.65}}
.eyebrow{{font-size:12px;letter-spacing:2px;color:#138a72;font-weight:750}}.muted{{color:#637188}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}.card,section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:24px}}
.card strong{{display:block;font-size:26px;margin-top:6px}}section{{margin:20px 0}}img{{width:100%;height:auto}}
.notice{{background:#eaf3ed;border-left:3px solid #138a72;padding:14px 18px;margin:20px 0}}
.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:12px 10px;border-bottom:1px solid #e7ecf2;text-align:left;white-space:nowrap}}th{{color:#637188}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;line-height:1.5}}a{{color:#2162a1}}details{{margin-top:16px}}
@media(max-width:680px){{h1{{font-size:34px}}.cards{{grid-template-columns:1fr}}body{{padding:20px 12px}}section{{padding:16px}}}}
</style>
<header><div class="eyebrow">OFFLINE RL · DATA DIAGNOSTICS</div><h1>TrajectoryLab</h1>
<p>{html.escape(manifest["config"]["suite"]["name"])} · {html.escape(source["env_id"])}</p>
<p class="muted">A reproducible comparison of behavior cloning, implicit value learning and return-conditioned sequence modeling.</p></header>
<div class="cards"><div class="card">Verified completed runs<strong>{len(results)} / {manifest["expected_runs"]}</strong></div>
<div class="card">Training seeds per group<strong>{len(manifest["config"]["suite"]["seeds"])}</strong></div>
<div class="card">Updates per run<strong>{manifest["config"]["training"]["steps"]:,}</strong></div></div>
<div class="notice">{warning}</div>
<section><h2>Final results</h2><p class="muted">Mean ± sample standard deviation across independent training seeds. Each seed is first averaged over the same final evaluation episodes. Higher raw return is better; no D4RL-normalized scores are claimed.</p>
<div class="table"><table><thead><tr><th>Data view</th><th>Method</th><th>Seeds</th><th>Raw return</th><th>Success</th><th>Support distance</th><th>Mean run time</th></tr></thead><tbody>{"".join(table_rows)}</tbody></table></div>
<p><a href="runs.csv">Per-run CSV</a> · <a href="summary.json">Summary JSON</a> · <a href="evidence/suite.json">Full protocol and source manifest</a></p></section>
<section><h2>Training-data audit</h2><p class="muted">Whole episodes selected from the source training split. Return is an undiscounted trajectory total. Saturation is the fraction of action coordinates with absolute normalized value at least 0.99.</p>
<div class="table"><table><thead><tr><th>View</th><th>Episodes</th><th>Transitions</th><th>Budget filled</th><th>Median return</th><th>Min / max return</th><th>Terminals / timeouts</th><th>Action saturation</th></tr></thead><tbody>{"".join(audit_rows)}</tbody></table></div></section>
<section><h2>Policy and weighting diagnostics</h2><p class="muted">The support threshold is calibrated separately for each view from validation trajectories, using a common training-pool scale. ESS is the effective sample size divided by batch size, averaged over the final minibatch of each IQL seed; this snapshot is not a dataset-wide estimate.</p>
<div class="table"><table><thead><tr><th>View</th><th>Method</th><th>States above validation distance p95</th><th>Policy action saturation</th><th>IQL final-minibatch ESS</th></tr></thead><tbody>{"".join(diagnostic_rows)}</tbody></table></div>
<p class="muted">These diagnostics describe associations. They do not establish why one policy has a higher return.</p></section>
{"".join(sections)}
<section><h2>How to interpret this evidence</h2><ul>
<li>Complete trajectories are split before data selection. Normalizers and PCA use training data only.</li>
<li>Data views share a declared transition cap. Actual counts, whole-episode budget tolerance and source IDs are in the manifest.</li>
<li>Validation and test seeds are disjoint. The final checkpoint is evaluated; test returns never select a checkpoint.</li>
<li>Support distance uses a common training-pool scale and a fixed-size subsample. It is representation-dependent, not a calibrated OOD detector.</li>
<li>Three training seeds provide limited uncertainty evidence. Data-view comparisons alone do not establish a causal explanation.</li>
<li>{html.escape(source["benchmark_status"])}. These experiments do not establish production performance or SOTA.</li>
</ul><details><summary>Configuration, environment and provenance</summary><pre>{protocol}</pre></details></section>
<footer class="muted">Suite {manifest["suite_id"]} · Evidence {fingerprint(results)[:16]} · Generated from recorded runs.</footer></html>"""
    (output / "index.html").write_text(document)
    return output / "index.html"
