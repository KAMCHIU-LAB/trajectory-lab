"""Export aggregate public-simulator evidence after the registered study stops."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from trajectorylab.bidding_expert_experiment import (
    ROOT,
    compare,
    confirm,
    develop,
    freeze,
    load_config,
    read_records,
    with_selector,
)
from trajectorylab.bidding_expert_selection import FEATURE_NAMES
from trajectorylab.utils import read_json, write_json

matplotlib.rcParams["svg.hashsalt"] = "trajectorylab-expert-selection-v1"
PRIVATE = (
    r"/Users/",
    r"/home/",
    r"github_pat_",
    r"gh[pousr]_[A-Za-z0-9]{20}",
    r"-----BEGIN .*PRIVATE KEY-----",
    r"\bsk-[A-Za-z0-9_-]{20,}\b",
    r"\bAKIA[A-Z0-9]{16}\b",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)\s*[\"']?\s*[:=]",
)
PUBLIC_FILES = frozenset(
    (
        "results.csv",
        "features.csv",
        "summary.json",
        "provenance.json",
        "report.md",
        "comparison.svg",
        "export.json",
    )
)
DEVELOPMENT_STOPS = frozenset(
    (
        "STOP_NO_HEADROOM",
        "STOP_NO_OBSERVABLE_SIGNAL",
        "STOP_DEV_FAILURE",
        "STOP_INSUFFICIENT_BUDGET",
    )
)
METRICS = (
    "cost",
    "conversions",
    "score",
    "violation",
    "exposures",
    "budget",
    "target_cpa",
    "budget_utilization",
    "hard_budget_violation",
    "final_arrived",
)


def csv_bytes(rows, fields):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def safe_write(path, content):
    if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
        raise RuntimeError("Publication paths must not traverse a symlink")
    if any(re.search(pattern.encode(), content) for pattern in PRIVATE):
        raise RuntimeError(f"Publication privacy scan rejected {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def checked_development(config, output):
    """Recompute in a disposable mirror; never rewrite a saved research decision."""
    saved = read_json(output / "development.json")
    with tempfile.TemporaryDirectory(prefix="expert-decision-audit-") as directory:
        mirror = Path(directory)
        for name in ("experiment-freeze.json", "data-freeze.json"):
            shutil.copyfile(output / name, mirror / name)
        for split in ("train", "validation"):
            for market in config["study"][f"{split}_markets"]:
                for name in (
                    f"inputs/{market}.json",
                    f"markets/{market}.json",
                    f"{split}/{market}.json",
                ):
                    target = mirror / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(output / name, target)
                # Same filesystem is not guaranteed for the system temp directory.
                source = output / "markets" / f"{market}.npz"
                target = mirror / "markets" / source.name
                try:
                    os.link(source, target)
                except OSError:
                    target.symlink_to(source.resolve())
        recomputed = develop(config, mirror)
        if recomputed != saved:
            raise RuntimeError("Saved development decision differs from recomputed evidence")
        if recomputed["status"] == "READY_FOR_CONFIRMATION" and read_json(
            mirror / "selection-freeze.json"
        ) != read_json(output / "selection-freeze.json"):
            raise RuntimeError("Saved selection differs from recomputed development decision")
    return recomputed


def exact_fields(record, fields):
    if not isinstance(record, dict) or set(record) != set(fields):
        raise RuntimeError("Unregistered publication metadata fields")
    return {key: record[key] for key in fields}


def public_provenance(config, output, frozen, confirmation):
    """Accept explicit metadata schemas, not free-form local annotations."""

    def digest(value):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
            raise RuntimeError("Invalid publication digest")
        return value

    def config_schema(value, template):
        if isinstance(template, dict):
            selected = exact_fields(value, template)
            return {k: config_schema(v, template[k]) for k, v in selected.items()}
        if isinstance(template, list):
            if not isinstance(value, list):
                raise RuntimeError("Invalid configuration list")
            if template and isinstance(template[0], str):
                if any(item not in template for item in value):
                    raise RuntimeError("Unregistered configuration text")
                return list(value)
            return [config_schema(item, template[0]) for item in value]
        if isinstance(template, str):
            if value != template:
                raise RuntimeError("Unregistered configuration text")
        elif type(value) is not type(template) or (
            isinstance(value, (int, float)) and not np.isfinite(value)
        ):
            raise RuntimeError("Invalid configuration scalar")
        return value

    config_schema(config, load_config(ROOT / "configs/bidding-expert-selection.toml"))
    experiment = exact_fields(
        frozen, ("config", "source", "weights", "generator", "protocol_sha256", "identity")
    )
    data = exact_fields(
        read_json(output / "data-freeze.json"),
        ("config", "generator", "protocol_sha256", "identity"),
    )
    if data["config"] != config or data["generator"] != experiment["generator"]:
        raise RuntimeError("Data provenance differs from frozen public sources")
    for record in (experiment, data):
        for key in ("protocol_sha256", "identity"):
            digest(record[key])

    runtime_path = output / "execution-runtime.json"
    runtime = {"status": "not_recorded", "note": "Execution environment was not recorded."}
    if runtime_path.exists():
        runtime = exact_fields(
            read_json(runtime_path),
            (
                "device",
                "identity",
                "numpy",
                "pre_outcome_commit",
                "python",
                "scipy",
                "threads",
                "torch",
            ),
        )
        if (
            runtime["device"] != "cpu"
            or runtime["identity"] != frozen["identity"]
            or type(runtime["threads"]) is not int
            or runtime["threads"] <= 0
            or runtime["threads"] != config["limits"]["torch_threads"]
            or not isinstance(runtime["pre_outcome_commit"], str)
            or not re.fullmatch(r"[0-9a-f]{7,40}", runtime["pre_outcome_commit"])
            or any(
                not isinstance(runtime[k], str)
                or not re.fullmatch(r"[0-9][0-9A-Za-z.+_-]{0,63}", runtime[k])
                for k in ("python", "numpy", "torch", "scipy")
            )
        ):
            raise RuntimeError("Invalid execution runtime metadata")
        runtime = {"status": "recorded", **runtime}

    inputs, markets = [], []
    splits = ("train", "validation", "confirmation") if confirmation else ("train", "validation")
    for split in splits:
        for market in config["study"][f"{split}_markets"]:
            item = exact_fields(
                read_json(output / "inputs" / f"{market}.json"),
                ("market", "generator_identity", "data_identity", "tape_sha256"),
            )
            for key in ("generator_identity", "data_identity", "tape_sha256"):
                digest(item[key])
            meta = exact_fields(
                read_json(output / "markets" / f"{market}.json"),
                (
                    "bytes",
                    "effective_actions",
                    "focal_agents",
                    "identity",
                    "market_id",
                    "pv_count",
                    "requested_pv_count",
                    "seconds",
                    "source",
                ),
            )
            if (
                item["market"] != market
                or meta["market_id"] != market
                or meta["source"] != experiment["generator"]
                or digest(meta["identity"]) != item["generator_identity"]
                or any(
                    type(meta[k]) is not int or meta[k] < 0
                    for k in ("bytes", "pv_count", "requested_pv_count")
                )
                or not isinstance(meta["seconds"], (int, float))
                or not np.isfinite(meta["seconds"])
                or meta["seconds"] < 0
                or any(
                    not isinstance(meta[k], list)
                    or not all(type(x) in (int, float) and np.isfinite(x) for x in meta[k])
                    for k in ("effective_actions", "focal_agents")
                )
            ):
                raise RuntimeError("Invalid public market metadata")
            inputs.append(item)
            markets.append(meta)
    result = dict(
        experiment=experiment,
        data_generation=data,
        runtime=runtime,
        inputs=inputs,
        markets=markets,
        protocol_amendment="Source/selection-seal/zero-variance safeguards were reviewed and committed before any new market policy outcome analysis. The initial data-generation protocol hash is retained; the experiment freeze binds the final pre-outcome protocol.",
        excluded="Raw traffic, per-event labels, event IDs, pickle checkpoints, optimizer/RNG states, local paths, runtime logs and orchestration notes.",
    )
    if confirmation:
        result["confirmation_selection"] = read_json(output / "selection-freeze.json")
        result["confirmation_open"] = read_json(output / "confirmation-open.json")
    return result


def export(config, output, destination):
    root = ROOT.resolve()
    destination = destination.absolute()
    if not destination.is_relative_to(ROOT.absolute() / "reports"):
        raise RuntimeError("Public evidence must be exported under repository reports")
    relative = destination.relative_to(ROOT.absolute())
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink() or part == "..":
            raise RuntimeError("Publication paths must not traverse a symlink")
    destination = cursor
    if destination.exists() and any(
        p.is_symlink() or not p.is_file() or p.name not in PUBLIC_FILES
        for p in destination.iterdir()
    ):
        raise RuntimeError("Unregistered file or symlink in publication destination")
    frozen = freeze(config, output)
    development = checked_development(config, output)
    status = development["status"]
    confirmation = None
    if status == "READY_FOR_CONFIRMATION":
        saved_confirmation = read_json(output / "confirmation.json")
        if saved_confirmation.get("status") not in ("CONFIRM_PASS", "CONFIRM_FAILURE"):
            raise RuntimeError("Unregistered confirmation terminal state")
        confirmation = confirm(config, output)
        status = confirmation["status"]
    elif status not in DEVELOPMENT_STOPS:
        raise RuntimeError("Study has no registered terminal state")
    train = read_records(output, "train", config)
    dev = read_records(output, "validation", config)
    for candidate in development["candidate_results"]:
        dev = with_selector(dev, candidate["model"], candidate["threshold"], candidate["name"])
    reference = development["reference"]
    comparisons = {m: compare(dev, m, reference, config) for m in sorted(dev[0]["outcomes"])}
    records = train + dev
    if confirmation:
        selection = read_json(output / "selection-freeze.json")
        c = selection["candidate"]
        confirmed = read_records(output, "confirmation", config)
        records += with_selector(confirmed, c["model"], c["threshold"], c["name"])

    table = []
    features = []
    for row in records:
        key = {k: row[k] for k in ("split", "market", "focal", "bias_group")}
        features.append(
            {
                **key,
                "target": row["target"],
                "prefix_digest": row["prefix_digest"],
                **dict(zip(FEATURE_NAMES, row["features"], strict=True)),
            }
        )
        for method, outcome in sorted(row["outcomes"].items()):
            table.append({**key, "method": method, **{k: outcome[k] for k in METRICS}})
    provenance = public_provenance(config, output, frozen, confirmation)
    safe_write(
        destination / "results.csv",
        csv_bytes(table, ("split", "market", "focal", "bias_group", "method", *METRICS)),
    )
    safe_write(
        destination / "features.csv",
        csv_bytes(
            features,
            ("split", "market", "focal", "bias_group", "target", "prefix_digest", *FEATURE_NAMES),
        ),
    )
    evidence = dict(
        status=status,
        experiment_identity=frozen["identity"],
        development=development,
        development_comparisons=comparisons,
        confirmation=confirmation,
        scope="Unknown fixed pCVR bias, immediate feedback, one seed-11 response expert, fixed-opponent public simulation.",
        interpretation="Development-selected reference and exploratory intervals; no development gain is independent confirmation. Hindsight is nondeployable and includes future outcome noise.",
        validation_markets=len(config["study"]["validation_markets"]),
        training_markets=len(config["study"]["train_markets"]),
    )
    safe_write(
        destination / "summary.json",
        (json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    safe_write(
        destination / "provenance.json",
        (json.dumps(provenance, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )

    methods = sorted(
        comparisons, key=lambda m: comparisons[m]["all"]["score"]["candidate_mean"], reverse=True
    )
    lines = [
        "# One-shot expert selection: registered study",
        "",
        f"Terminal state: **`{status}`**.",
        "",
        "The study asks whether a visible 16-period prefix can select between two frozen experts under unknown prediction bias. Full terminal utility includes the prefix cost. The reference is the strongest cheap development control, including direct observable calibration.",
        "",
        f"Training: {len(config['study']['train_markets'])} new markets. Development: {len(config['study']['validation_markets'])} new markets, six focal advertisers and three bias groups per market. Statistical unit: market. Response seed: 11; feedback is immediate.",
        "",
        f"Frozen development reference: **`{reference}`**. Development results and intervals below are exploratory, selected on this same split; they do not establish generalization.",
        "",
        "| Method | Mean score | Relative to reference | Under | Nominal | Over |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in methods:
        stats = comparisons[method]
        lines.append(
            f"| `{method}` | {stats['all']['score']['candidate_mean']:.3f} | {stats['all']['score']['effect']:+.1%} | {stats['under']['score']['effect']:+.1%} | {stats['nominal']['score']['effect']:+.1%} | {stats['over']['score']['effect']:+.1%} |"
        )
    lines.extend(
        [
            "",
            "`hindsight-two-experts` uses future outcomes and is not deployable. It diagnoses this pair and this common prefix; it is not a general policy upper bound. All declared campaigns are included, including zero conversions and exhausted prefixes.",
            "",
            "![Development comparison](comparison.svg)",
            "",
            "## Audit and interpretation",
            "",
            "- The two continuations start from a copied queue, ledger, budget and actual history. Features are captured before continuations; hidden bias and final labels are never policy inputs.",
            "- The selector fits a full-cycle advantage target. Its rejection threshold is operational, not a confidence interval.",
            "- The simple calibration control observes arrived labels and accepted public probabilities; it does not receive a true correction factor.",
            "- Full-day controls expose the opportunity cost of waiting for the common prefix.",
            "- A failed development or budget gate leaves confirmation unopened. A registered stop closes this question without proving that all possible selectors fail.",
            "- This study does not prove generative distributional value, delayed-feedback robustness, shared-budget competition, or industrial revenue impact.",
            "",
            "[Protocol](../../docs/bidding-expert-selection-protocol.md) · [Reproduce](../../docs/bidding-expert-reproduction.md) · [All outcomes](results.csv) · [Public features and paired targets](features.csv) · [Statistics and candidates](summary.json) · [Source/input/weight provenance](provenance.json)",
            "",
        ]
    )
    safe_write(destination / "report.md", "\n".join(lines).encode())

    display = [
        reference,
        "full-response",
        "prefix-response",
        "prefix-feedback",
        "hindsight-two-experts",
    ]
    if development.get("selected"):
        display.insert(1, development["selected"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    values = [comparisons[m]["all"]["score"]["candidate_mean"] for m in display]
    axes[0].barh(
        display[::-1],
        values[::-1],
        color=["#9c6a14" if m == "hindsight-two-experts" else "#225b78" for m in display[::-1]],
    )
    axes[0].set_xlabel("Mean terminal score (development only)")
    axes[0].set_title("Prefix opportunity cost and optimistic headroom")
    x = np.arange(3)
    for method in display:
        values = [
            100 * comparisons[method][g]["score"]["effect"] for g in ("under", "nominal", "over")
        ]
        axes[1].plot(
            x,
            values,
            marker="o",
            label=method,
            linestyle="--" if method == "hindsight-two-experts" else "-",
        )
    axes[1].set_xticks(x, ["Under 25%", "Nominal", "Over 50%"])
    axes[1].set_ylabel("Score change vs development reference (%)")
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[1].legend(fontsize=7)
    axes[1].set_title("Trade-offs under the same unknown-bias distribution")
    svg = io.StringIO()
    fig.savefig(svg, format="svg", metadata={"Date": None})
    plt.close(fig)
    safe_write(
        destination / "comparison.svg",
        ("\n".join(line.rstrip() for line in svg.getvalue().splitlines()) + "\n").encode(),
    )
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(destination.iterdir())
        if p.name in PUBLIC_FILES and p.name != "export.json"
    }
    write_json(
        destination / "export.json",
        dict(status=status, experiment_identity=frozen["identity"], files_sha256=files),
    )
    print(f"EXPORTED status={status} campaigns={len(records)} outcome_rows={len(table)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/bidding-expert-selection.toml"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/bidding/expert-selection-v1"
    )
    parser.add_argument(
        "--destination", type=Path, default=ROOT / "reports/bidding-expert-selection"
    )
    args = parser.parse_args()
    export(load_config(args.config), args.output, args.destination)


if __name__ == "__main__":
    main()
