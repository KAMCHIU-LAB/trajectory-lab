"""Publish allowlisted text aggregates from the completed post-hoc validation audit."""

from __future__ import annotations

import argparse
import hashlib
import html
import itertools
import json
import math
import re
from pathlib import Path

from trajectorylab.bidding_onpolicy import GROUPS, ROOT, SEEDS, _inputs
from trajectorylab.utils import fingerprint, read_json


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def finite(value) -> None:
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("Nonfinite aggregate evidence")


def evidence(directory: Path, main_run: Path, control_run: Path) -> tuple[dict, dict, dict]:
    manifest, summary, status = [
        read_json(directory / f"{name}.json") for name in ("manifest", "summary", "status")
    ]
    core, _, _, selection, _, _, bound = _inputs(main_run, control_run)
    identity, summary_hash = fingerprint(core), digest((directory / "summary.json").read_bytes())
    if (
        {k: v for k, v in manifest.items() if k not in {"identity", "summary_sha256"}} != core
        or any(row.get("identity") != identity for row in (manifest, summary, status))
        or status.get("phase") != "complete"
        or any(row.get("summary_sha256") != summary_hash for row in (manifest, status))
        or any(
            summary[k] != core[k] for k in ("main_identity", "control_identity", "decision_curves")
        )
        or summary["sample_counts"]
        != {"rollouts": 1080, "ticks": 51840, "independent_validation_markets": 6}
    ):
        raise RuntimeError("Completed audit identity, hashes or counts differ")
    replay = summary["replay_agreement"]
    if (
        replay["all_rollouts_match"] is not True
        or replay["checked_rollouts"] != 1080
        or replay["max_absolute_difference"]
        != dict.fromkeys(("score", "cost", "conversions", "violation"), 0.0)
    ):
        raise RuntimeError("The complete replay must have zero differences in all four outcomes")
    axes = {
        "market": [int(m) for m in core["validation_markets"]],
        "scenario": core["scenarios"],
        "method": ["response", selection["candidate"]],
        "seed": list(SEEDS),
    }
    dimensions = {
        "by_method": ("method",),
        "by_scenario_method": ("scenario", "method"),
        "by_seed_method": ("seed", "method"),
        "by_market_method": ("market", "method"),
        "by_scenario_method_seed_market": ("scenario", "method", "seed", "market"),
    }
    for name, fields in dimensions.items():
        expected = set(itertools.product(*(axes[k] for k in fields)))
        groups = summary["aggregates"][name]
        if (
            len(groups) != len(expected)
            or {tuple(r[k] for k in fields) for r in groups} != expected
        ):
            raise RuntimeError("Missing or duplicated method/scenario/seed/market aggregates")
        for row in groups:
            histogram, frequency = row["nominal_action_histogram"], row["nominal_action_frequency"]
            if (
                row["campaigns"] != 1080 // len(expected)
                or row["ticks"] != row["campaigns"] * 48
                or set(row["curves"]) != set(GROUPS)
                or len(row["first_12_tick_cumulative_budget_utilization_mean"]) != 12
                or set(histogram) != {str(float(a)) for a in core["actions"]}
                or set(frequency) != set(histogram)
                or any(type(v) is not int or v < 0 for v in histogram.values())
                or sum(histogram.values()) != row["ticks"]
                or any(
                    not math.isclose(frequency[a], n / row["ticks"], rel_tol=1e-12, abs_tol=1e-12)
                    for a, n in histogram.items()
                )
            ):
                raise RuntimeError("Aggregate ticks, curves or action coverage differ")
    finite(summary)
    return manifest, summary, bound


def render(summary: dict, manifest: dict) -> str:
    columns = [
        "场景",
        "策略",
        "current logMSE",
        "remaining logMSE",
        "当前成本比",
        "当前转化比",
        "剩余成本比",
        "剩余转化比",
        "零动作",
        "tick1 预算",
        "tick12 预算",
        "预算截断",
    ]
    head = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
    rows = []
    for row in summary["aggregates"]["by_method"] + summary["aggregates"]["by_scenario_method"]:
        errors, early = (
            row["chosen_action_errors"],
            row["first_12_tick_cumulative_budget_utilization_mean"],
        )
        values = [row.get("scenario", "全部场景（描述性）"), row["method"]]
        values += [
            f"{errors[s]['chosen_action_log_mse']:.4f}"
            for s in ("current", "remaining_constant_effective_action")
        ]
        values += [
            "未定义"
            if row["curves"][g]["normalized_predicted_to_true_ratio"] is None
            else f"{row['curves'][g]['normalized_predicted_to_true_ratio']:.4f}"
            for g in GROUPS
        ]
        values += [
            f"{v:.2%}"
            for v in (
                row["nominal_action_frequency"]["0.0"],
                early[0],
                early[11],
                row["budget_blocked_fraction"],
            )
        ]
        rows.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in values) + "</tr>")
    paced = next(
        r
        for r in summary["aggregates"]["by_method"]
        if r["method"] == manifest["selected_controller"]["candidate"]
    )
    zero = paced["nominal_action_frequency"]["0.0"]
    limits = "".join(f"<li>{html.escape(item)}</li>" for item in summary["limitations"])
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>所选动作预测与预算节奏：事后验证诊断</title>
<style>body{{font:16px/1.7 system-ui;max-width:1200px;margin:32px auto;padding:0 20px}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #ddd;padding:7px}}.scroll{{overflow-x:auto}}</style>
<h1>所选动作预测与预算节奏</h1>
<p><strong>Post-hoc 验证集诊断。</strong>在查看 main/control 结果后制定，无拟合或调参，
不是独立确认、因果瓶颈证明或新的收益证据。只有 6 个独立验证市场。</p>
<p>已完成审计中的 1,080 个完整周期、51,840 个时段覆盖两策略及全部 5 个场景、3 个种子、6 个广告主。
1,080 次重放的 score、cost、conversions、violation 最大绝对差均为 0，与冻结 control 验证完全一致。
本导出核对身份、哈希、覆盖及该重放记录，没有再次执行实验。</p>
<h2>全时段误差不能直接比较预测器质量</h2>
<p>两策略访问不同状态并选择不同动作，所选动作误差同时受到这些分布影响。
current 全 tick MSE 接近，不能说明预测质量相同。paced 零动作占 {zero:.1%}；
零动作的四条预测曲线及匹配未截断标签均为零，会稀释全 tick 平均 MSE。
本报告没有独立验证非零动作条件 MSE，不据此报告该指标或反推预测器优劣。</p>
<p>原 response 用 remaining 曲线决策；paced 用 current 曲线及冻结的资源价格/校准。
下表审计的是原始模型曲线，全部预算截断时段仍以 evaluator-private 未截断标签计入；标签从不反馈策略。</p>
<h2>聚合与全部场景</h2><div class="scroll"><table>{head}{"".join(rows)}</table></div>
<p>四个“比”为归一化预测总量／归一化未截断真实总量：成本先除初始预算，转化先乘目标 CPA 再除初始预算。
它们不是原始金额/转化比，也不证明逐状态校准。current 的 log1p 尺度为 48，remaining 为 1；
成本和转化等权。tick1/tick12 是相应时段结算后的平均累计预算利用率。</p>
<p>remaining 标签固定当前有效动作直到第 47 tick；即使场景稍后切换，也不沿未来 bias 路径改动作。
它不是策略后续换动作或预算截断后的真实回报。完整四曲线 Smooth L1、动作直方图、逐种子/市场聚合见 JSON。</p>
<ul>{limits}</ul><p><a href="summary.json">完整聚合</a> · <a href="manifest.json">冻结来源</a> ·
<a href="export.json">导出校验与限制</a></p></html>"""


def export(directory: Path, main_run: Path, control_run: Path, destination: Path) -> None:
    manifest, summary, bound = evidence(directory, main_run, control_run)
    payload = {name: (directory / name).read_bytes() for name in ("manifest.json", "summary.json")}
    payload["report.html"] = render(summary, manifest).encode()
    patterns = (
        r"/Users/|/home/",
        r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b",
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    )
    for name, content in payload.items():
        if any(re.search(pattern, content.decode()) for pattern in patterns):
            raise RuntimeError(f"Potential private material in {name}; content withheld")
    if destination.exists() and any(destination.iterdir()):
        previous = destination / "manifest.json"
        if (
            not previous.is_file()
            or read_json(previous).get("identity") != manifest["identity"]
            or {p.name for p in destination.iterdir()} - (set(payload) | {"export.json"})
        ):
            raise RuntimeError("Destination contains different or unreviewed evidence")
    if (
        any(
            digest(path.read_bytes()) != manifest["files_sha256"][name]
            for name, path in bound.items()
        )
        or digest(payload["summary.json"]) != manifest["summary_sha256"]
    ):
        raise RuntimeError("Frozen audit sources or evidence changed before export")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in payload.items():
        (destination / name).write_bytes(content)
    record = {
        "run_identity": manifest["identity"],
        "exporter_sha256": digest(Path(__file__).read_bytes()),
        "files_sha256": {name: digest(content) for name, content in payload.items()},
        "scope": "post-hoc validation-only on-policy forecast and pacing aggregates; no new benefit or causal claim",
        "verification": "source/protocol/input identity, completion, summary hash, exact aggregate coverage and reported zero replay differences checked; no replay or independent forecast-metric recomputation",
        "privacy_gate": "allowlisted text scanned for home paths, emails and common credential patterns; not a universal disclosure proof",
        "excluded": "states, traffic, tapes, weights, raw tick labels, machine logs and runner state",
    }
    (destination / "export.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(
        f"exported post-hoc validation aggregates identity={manifest['identity']} summary_sha256={manifest['summary_sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run", type=Path, default=ROOT / "artifacts/bidding/onpolicy-v1")
    parser.add_argument("--main-run", type=Path, default=ROOT / "artifacts/bidding/main-v2")
    parser.add_argument("--control-run", type=Path, default=ROOT / "artifacts/bidding/control-v1")
    parser.add_argument("--destination", type=Path, default=ROOT / "reports/bidding-onpolicy")
    args = parser.parse_args()
    export(args.audit_run, args.main_run, args.control_run, args.destination)
