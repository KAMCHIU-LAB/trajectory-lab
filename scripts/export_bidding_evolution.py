"""Export two failed frozen bidding studies as a public scalar-only comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path

from export_bidding_control import PRIVACY_PATTERNS, atomic_write, digest, finite

ROOT = Path(__file__).resolve().parents[1]
TOOL = "bidding-evolution-scalar-export-v1"
SCENES = ("calibrated", "over_25", "over_50", "under_25", "mid_over_50")
E_ARMS = ("zero_E", "actual_E", "pseudo_E")
MIN_ARMS = ("FB", "minimum", "pacing_only", "quality_only")
OUTPUTS = {"summary.json", "results.csv", "comparison.svg", "export.json"}
STUDIES = ("information-value-v1", "feedback-coordination-v1")
E_FILES = {
    "manifest.json",
    "summary.json",
    "protocol.json",
    "resources.json",
    "join.json",
    *(
        f"held-{market}.{suffix}"
        for market in (1002, 1006, 1010, 1014)
        for suffix in ("json", "npz")
    ),
}
MIN_FILES = {"manifest.json", "summary.json", "terminals.json", "tick-logs.jsonl"}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(value):
    # Exactly the producer fingerprint/identity serialization, including default separators.
    return digest(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def file_sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            result.update(block)
    return result.hexdigest()


def no_symlinks(path):
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink rejected")


def read_json(path):
    no_symlinks(path)
    require(path.is_file(), "Expected a regular evidence file")
    return json.loads(path.read_bytes())


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), "Invalid numeric scalar")
    return value


def count(value):
    require(type(value) is int and value >= 0, "Invalid absolute count")
    return value


def boolean(value):
    require(type(value) is bool, "Invalid boolean scalar")
    return value


def same_number(left, right):
    require(
        math.isclose(number(left), number(right), rel_tol=1e-10, abs_tol=1e-10),
        "Summary differs from independent verifier evidence",
    )


def flags(value, names):
    require(set(value) == set(names), "Gate schema changed")
    return {name: boolean(value[name]) for name in names}


def checked_study(name, expected_files, seal_key, gate_key, candidate_key, frozen_key):
    folder = ROOT / "artifacts/bidding/numerical-audits" / name
    no_symlinks(folder)
    require(folder.is_dir(), "Missing sealed study")
    entries = list(folder.rglob("*"))
    require(
        all(p.is_file() and not p.is_symlink() for p in entries),
        "Sealed evidence must contain only regular files",
    )
    seal = read_json(folder / "seal.json")
    require(set(seal["files"]) == expected_files, "Unexpected seal file allowlist")
    require({p.name for p in entries} == expected_files | {"seal.json"}, "Sealed file set changed")
    for filename, expected in seal["files"].items():
        require(
            isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
            "Invalid sealed hash",
        )
        require(file_sha(folder / filename) == expected, "Sealed file hash changed")
    seal_identity = identity(seal["files"])
    require(seal[seal_key] == seal_identity, "Seal identity changed")
    manifest = read_json(folder / "manifest.json")
    summary = read_json(folder / "summary.json")
    verifier_path = (
        ROOT
        / ".omx/state/bidding-v2"
        / (name.removesuffix("-v1") + "-independent-verification.json")
    )
    verifier = read_json(verifier_path)
    freeze = identity(manifest["frozen"])
    require(
        manifest[frozen_key] is True
        and summary["status"] == "complete"
        and summary["input_source_integrity"] is True,
        "Study is not complete and frozen",
    )
    require(
        freeze == manifest["freeze"] == summary["freeze"] == verifier["freeze"],
        "Freeze differs between producer and verifier",
    )
    require(
        verifier["seal_identity"] == seal_identity
        and verifier["failures"] == []
        and verifier["integrity_verdict"] == "PASS"
        and verifier[gate_key] == "FAIL"
        and verifier[candidate_key] is False
        and summary[candidate_key] is False,
        "Expected independent integrity PASS and scientific FAIL",
    )
    require(
        summary["independent_confirmation"] is False
        and summary["generative_contribution"] is False,
        "Unexpected scientific scope",
    )
    snapshot = {folder / filename: expected for filename, expected in seal["files"].items()}
    snapshot[folder / "seal.json"] = file_sha(folder / "seal.json")
    snapshot[verifier_path] = file_sha(verifier_path)
    provenance = {
        "freeze_sha256": freeze,
        "seal_identity_sha256": seal_identity,
        "seal_file_sha256": snapshot[folder / "seal.json"],
        "verifier_sha256": snapshot[verifier_path],
        "integrity_verdict": "PASS",
        "scientific_gate_verdict": "FAIL",
    }
    return manifest, summary, verifier, provenance, snapshot


def scene_result(score, conversions, risks, cells, selected_key, feedback_key):
    cells = count(cells)
    require(cells > 0, "Empty scene")
    result = {
        "score_mean": number(score[selected_key]),
        "fb_score_mean": number(score[feedback_key]),
        "score_ratio": number(score["ratio"]),
        "conversion_mean": number(conversions[selected_key]),
        "fb_conversion_mean": number(conversions[feedback_key]),
        "conversion_ratio": number(conversions["ratio"]),
        "risk_count": count(risks[selected_key]),
        "fb_risk_count": count(risks[feedback_key]),
        "terminal_count": cells,
    }
    for prefix, key in (("", selected_key), ("fb_", feedback_key)):
        total = number(conversions[key]) * cells
        require(math.isclose(total, round(total), abs_tol=1e-8), "Noninteger conversion total")
        result[prefix + "conversion_count"] = count(round(total))
    require(
        result["risk_count"] <= cells and result["fb_risk_count"] <= cells,
        "Risk count exceeds terminal count",
    )
    same_number(result["score_mean"] / result["fb_score_mean"], result["score_ratio"])
    same_number(
        result["conversion_mean"] / result["fb_conversion_mean"], result["conversion_ratio"]
    )
    return result


def information_summary(manifest, source, verifier, provenance):
    protocol = manifest["frozen"]["protocol"]
    require(
        protocol["arms"] == list(E_ARMS)
        and protocol["scenes"] == list(SCENES)
        and len(protocol["markets"]) == 4
        and protocol["ticks"] == [0, 16, 32]
        and source["full_policy_rollout"] is False
        and set(source["arms"]) == set(E_ARMS),
        "Information study scope changed",
    )
    arms = {}
    for name in E_ARMS:
        arm, verified = source["arms"][name], verifier["evidence"]["arms"][name]
        require(set(arm["scenes"]) == set(SCENES), "Scene schema changed")
        scenes = {}
        for scene in SCENES:
            cell, risks = arm["scenes"][scene], arm["violation_counts"][scene]
            require(risks["context_terminals"] == 72, "Context terminal coverage changed")
            scenes[scene] = scene_result(
                cell["score"],
                cell["conversions"],
                risks,
                risks["context_terminals"],
                "selected",
                "feedback",
            )
            same_number(cell["score"]["ratio"], verified["scene_score_ratios"][scene])
            same_number(cell["conversions"]["ratio"], verified["scene_conversion_ratios"][scene])
            require(risks == verified["violation_counts"][scene], "Verifier risk counts differ")
        for key in ("five_scene_macro", "physical_delta_MSE", "switches"):
            same_number(arm[key], verified[key])
        arms[name] = {
            "role": {"zero_E": "reference", "actual_E": "candidate", "pseudo_E": "diagnostic"}[
                name
            ],
            "candidate_eligible": name == "actual_E",
            "five_scene_macro": number(arm["five_scene_macro"]),
            "physical_delta_mse": number(arm["physical_delta_MSE"]),
            "switch_count": count(arm["switches"]),
            "scenes": scenes,
        }
    gate_flags = flags(
        source["flags"],
        (
            "mse",
            "macro_FB",
            "macro_zero_E",
            "markets_FB",
            "markets_zero_E",
            "scene_protections",
            "nonfeedback",
        ),
    )
    require(
        gate_flags == verifier["evidence"]["flags"] and not all(gate_flags.values()),
        "Information gate evidence differs",
    )
    improvement = 1 - arms["actual_E"]["physical_delta_mse"] / arms["zero_E"]["physical_delta_mse"]
    same_number(improvement, source["mse_relative_improvement"])
    same_number(improvement, verifier["evidence"]["mse_relative_improvement"])
    return {
        "candidate": "actual_E",
        "candidate_verdict": "FAIL",
        "provenance": provenance,
        "scope": "Four seen training markets; leave-one-market fits; one intervention then FB.",
        "market_count": 4,
        "campaign_count": 120,
        "context_count": 360,
        "terminal_unit": "Overlapping intervention context; three contexts per campaign.",
        "full_policy_rollout": False,
        "independent_confirmation": False,
        "generative_contribution": False,
        "mse_relative_improvement": improvement,
        "positive_markets_vs_fb": count(source["positive_markets_FB"]),
        "positive_markets_vs_zero_e": count(source["positive_markets_zero_E"]),
        "gate_flags": gate_flags,
        "arms": arms,
        "limitations": [
            "Descriptive feasibility on four seen markets with overlapping training folds; no significance or generalization claim.",
            "The failed prediction MSE gate closes this candidate despite positive descriptive score ratios.",
            "Pseudo E uses random training donors; it is diagnostic and is never an eligible candidate or an independence test.",
            "Guard weights were not archived; verifier rebuilds matched saved predictions but reused the frozen optimizer helper.",
            "E was reconstructed from strictly prior saved aggregate predicted sums; individual exposure probabilities were not archived.",
            "Recorded artifact byte sampling was stale; the verifier separately counted sealed bytes.",
        ],
    }


def coordination_summary(manifest, source, verifier, provenance):
    protocol = manifest["frozen"]["protocol"]
    require(
        protocol["arms"] == list(MIN_ARMS)
        and protocol["scenes"] == list(SCENES)
        and set(source["splits"]) == {"train", "seen_development"}
        and protocol["independent_confirmation"] is False,
        "Coordination study scope changed",
    )
    splits = {}
    for split, markets in (("train", 4), ("seen_development", 6)):
        part = source["splits"][split]
        require(
            len(protocol["splits"][split]) == markets and set(part["arms"]) == set(MIN_ARMS),
            "Split coverage changed",
        )
        arms = {}
        for name in MIN_ARMS:
            arm = part["arms"][name]
            verified = verifier["evidence"]["effects"][split][name]
            require(set(arm["scenes"]) == set(SCENES), "Scene schema changed")
            scenes = {}
            for scene in SCENES:
                cell = arm["scenes"][scene]
                require(cell["violations"]["campaigns"] == markets * 6, "Campaign coverage changed")
                scenes[scene] = scene_result(
                    cell["score"],
                    cell["conversions"],
                    cell["violations"],
                    markets * 6,
                    "value",
                    "FB",
                )
                for metric in ("score", "conversions"):
                    for key in ("value", "FB", "ratio"):
                        same_number(cell[metric][key], verified["scenes"][scene][metric][key])
                require(
                    cell["violations"] == verified["scenes"][scene]["violations"],
                    "Verifier risk counts differ",
                )
            same_number(arm["five_scene_macro"], verified["five_scene_macro"])
            arms[name] = {
                "role": "candidate"
                if name == "minimum"
                else ("baseline" if name == "FB" else "diagnostic"),
                "candidate_eligible": name == "minimum",
                "five_scene_macro": number(arm["five_scene_macro"]),
                "physical_delta_mse": None,
                "scenes": scenes,
            }
        gate_flags = flags(
            part["flags"], ("over50", "under25", "macro", "protections", "market_direction")
        )
        require(
            part["split_pass"] is False
            and not all(gate_flags.values())
            and gate_flags == verifier["evidence"]["split_flags"][split],
            "Coordination gate evidence differs",
        )
        splits[split] = {
            "market_count": markets,
            "campaigns_per_arm": markets * 30,
            "candidate_verdict": "FAIL",
            "gate_flags": gate_flags,
            "arms": arms,
        }
    return {
        "candidate": "minimum",
        "candidate_verdict": "FAIL",
        "provenance": provenance,
        "scope": "Four training and six seen development markets; frozen full policy replay.",
        "terminal_unit": "Campaign terminal; six campaigns per market and scene.",
        "independent_confirmation": False,
        "generative_contribution": False,
        "minimum_non_fb_decision_count": count(source["minimum_non_FB_decisions"]),
        "splits": splits,
        "limitations": [
            "All ten markets are historically seen training or development data; seen development is not independent confirmation.",
            "Failed scientific gates close minimum; pacing-only and quality-only arms remain diagnostic.",
            "This is an ablation using the minimum of two existing feedback signals, not the complete min-pacing algorithm or its theorem.",
            "No statistical significance, new-market generalization, or generative contribution is established.",
            "Archived wall time starts after module import; full launcher wall duration is unavailable.",
        ],
    }


def result_rows(studies):
    rows = []
    for study, summary in studies.items():
        splits = {"seen_training_contexts": summary} if "arms" in summary else summary["splits"]
        for split, part in splits.items():
            for name, arm in part["arms"].items():
                for scene, cell in arm["scenes"].items():
                    rows.append(
                        {
                            "study": study,
                            "split": split,
                            "arm": name,
                            "arm_role": arm["role"],
                            "candidate_eligible": arm["candidate_eligible"],
                            "study_candidate_verdict": "FAIL",
                            "scene": scene,
                            "terminal_unit": "context" if study == STUDIES[0] else "campaign",
                            "five_scene_macro": arm["five_scene_macro"],
                            "physical_delta_mse": arm["physical_delta_mse"],
                            **cell,
                            "score_change_percent": 100 * (cell["score_ratio"] - 1),
                            "conversion_change_percent": 100 * (cell["conversion_ratio"] - 1),
                        }
                    )
    return rows


def comparison(studies):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"svg.hashsalt": TOOL, "svg.fonttype": "none", "font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.6), sharey=False)
        x = list(range(len(SCENES)))
        information, coordination = (studies[name] for name in STUDIES)
        series = [
            [
                (name, information["arms"][name], color)
                for name, color in zip(E_ARMS, ("#64748b", "#0072b2", "#d55e00"), strict=True)
            ],
            [
                ("minimum: train", coordination["splits"]["train"]["arms"]["minimum"], "#0072b2"),
                (
                    "minimum: seen development",
                    coordination["splits"]["seen_development"]["arms"]["minimum"],
                    "#d55e00",
                ),
            ],
        ]
        for axis, lines in zip(axes, series, strict=True):
            width = 0.78 / len(lines)
            for i, (label, arm, color) in enumerate(lines):
                values = [100 * (arm["scenes"][scene]["score_ratio"] - 1) for scene in SCENES]
                axis.bar(
                    [v + (i - (len(lines) - 1) / 2) * width for v in x],
                    values,
                    width=width,
                    label=label,
                    color=color,
                )
            axis.axhline(0, color="#202020", linewidth=0.8)
            axis.set_xticks(x, SCENES, rotation=25, ha="right")
            axis.set_ylabel("Score change vs own FB baseline (%)")
            axis.grid(axis="y", alpha=0.22)
            axis.set_axisbelow(True)
            axis.legend(loc="upper left", fontsize=8)
        axes[0].set_ylim(-1, 3.1)
        axes[1].set_ylim(-60, 15)
        for axis, lines in zip(axes, series, strict=True):
            lower, upper = axis.get_ylim()
            require(
                all(
                    lower < 100 * (arm["scenes"][scene]["score_ratio"] - 1) < upper
                    for _, arm, _ in lines
                    for scene in SCENES
                ),
                "Comparison axis would hide a result",
            )
        axes[0].set_title(
            "E ablation: three arms, four seen training markets\nOne intervention then FB; actual E candidate FAIL"
        )
        axes[1].set_title(
            "Minimum feedback coordination: candidate FAIL\nTrain and seen development; full policy replay"
        )
        fig.suptitle(
            "Two distinct frozen studies: descriptive results; both candidates FAIL", y=0.98
        )
        fig.text(
            0.5,
            0.035,
            "Independent panel scales and different units/protocols: do not pool or rank across studies.\n"
            "Pseudo E is diagnostic. Seen development is not independent confirmation. No generative gain established.",
            ha="center",
            fontsize=9,
        )
        fig.subplots_adjust(left=0.065, right=0.985, top=0.79, bottom=0.27, wspace=0.28)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="svg", metadata={"Date": None, "Creator": TOOL})
        plt.close(fig)
        return buffer.getvalue()


def privacy_scan(payload):
    for content in payload.values():
        text = content.decode("utf-8")
        require(
            not any(re.search(pattern, text) for pattern in PRIVACY_PATTERNS),
            "Potential private content withheld",
        )
        require(
            not re.search(r"(?:[\"'>=\s])(?:[A-Za-z]:[\\/]|/(?![/>]))[^\s<\"']+", text),
            "Potential absolute filesystem path withheld",
        )


def encode_json(value):
    finite(value)
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def check_destination(destination):
    no_symlinks(destination)
    if not destination.exists():
        return
    require(
        destination.is_dir() and {p.name for p in destination.iterdir()} == OUTPUTS,
        "Destination is not an existing export from this tool",
    )
    require(
        all(p.is_file() and not p.is_symlink() for p in destination.iterdir()),
        "Destination contains nonregular files",
    )
    previous = read_json(destination / "export.json")
    require(
        previous.get("tool") == TOOL
        and set(previous.get("output_sha256", {})) == OUTPUTS - {"export.json"},
        "Destination ownership record differs",
    )
    require(
        all(
            file_sha(destination / name) == expected
            for name, expected in previous["output_sha256"].items()
        ),
        "Destination has edited or unreviewed outputs",
    )


def export(destination):
    destination = destination.absolute()
    no_symlinks(destination)
    destination = destination.resolve()
    for source in (ROOT / "artifacts", ROOT / ".omx", ROOT / "scripts"):
        require(
            not destination.is_relative_to(source) and not source.is_relative_to(destination),
            "Destination overlaps private inputs or source code",
        )
    check_destination(destination)
    inputs, studies = {}, {}
    for name, files, seal_key, gate_key, candidate_key, frozen_key, summarize in (
        (
            STUDIES[0],
            E_FILES,
            "content_identity",
            "feasibility_gate_verdict",
            "feasibility_pass",
            "frozen_before_fit",
            information_summary,
        ),
        (
            STUDIES[1],
            MIN_FILES,
            "identity",
            "scientific_gate_verdict",
            "candidate_pass",
            "frozen_before_rollout",
            coordination_summary,
        ),
    ):
        manifest, source, verifier, provenance, snapshot = checked_study(
            name, files, seal_key, gate_key, candidate_key, frozen_key
        )
        studies[name] = summarize(manifest, source, verifier, provenance)
        inputs.update(snapshot)
    summary = {
        "schema_version": 1,
        "tool": TOOL,
        "candidate_verdict": "BOTH_FAIL",
        "scope": "Public descriptive scalar summaries from two sealed, independently checked studies.",
        "comparison_boundary": "Different protocols, populations and terminal units; do not pool across studies.",
        "independent_confirmation": False,
        "generative_contribution": False,
        "studies": studies,
        "export_validation": "All sealed file hashes, exact file sets, no symlinks, producer identities, freeze and verifier verdicts checked. Experiments were not rerun.",
    }
    rows = result_rows(studies)
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    payload = {
        "summary.json": encode_json(summary),
        "results.csv": table.getvalue().encode(),
        "comparison.svg": comparison(studies),
    }
    exporter = Path(__file__).resolve()
    exporter_sha = file_sha(exporter)
    record = {
        "schema_version": 1,
        "tool": TOOL,
        "exporter_sha256": exporter_sha,
        "input_sha256": {str(path.relative_to(ROOT)): sha for path, sha in sorted(inputs.items())},
        "output_sha256": {name: digest(content) for name, content in sorted(payload.items())},
        "record_hash_boundary": "This record is excluded from its own output hashes to avoid circular hashing.",
        "privacy_boundary": "Only explicitly selected scalar metrics, static explanatory text and provenance hashes are public. No implementation source, full manifests, machine paths, logs, weights, traffic or exposure records are copied.",
        "privacy_scan": "Existing PRIVACY_PATTERNS plus absolute filesystem path checks applied to every output; scanning is supplementary to the scalar allowlist.",
        "scientific_boundary": "Both candidates failed; diagnostics are ineligible; seen development is not independent confirmation; no generative gain is established.",
    }
    payload["export.json"] = encode_json(record)
    privacy_scan(payload)
    require(
        all(file_sha(path) == expected for path, expected in inputs.items())
        and file_sha(exporter) == exporter_sha,
        "Evidence changed during export",
    )
    check_destination(destination)
    for name in ("summary.json", "results.csv", "comparison.svg", "export.json"):
        atomic_write(destination / name, payload[name])
    require(
        all((destination / name).read_bytes() == content for name, content in payload.items()),
        "Export reread differs",
    )
    print(f"exported studies=2 rows={len(rows)} candidates=BOTH_FAIL privacy=PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path("reports/bidding-evolution"))
    export(parser.parse_args().destination)
