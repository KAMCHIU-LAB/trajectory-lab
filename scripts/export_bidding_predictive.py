"""Publish text-only evidence from completed validation forecast diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path

from trajectorylab import bidding_context, bidding_predictive
from trajectorylab.utils import fingerprint, read_json


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def evidence(directory: Path, *, ridge: bool) -> tuple[dict, dict]:
    manifest = read_json(directory / "manifest.json")
    summary = read_json(directory / "summary.json")
    status = read_json(directory / "status.json")
    excluded = {"identity", "summary_sha256"}
    if ridge:
        excluded |= {"model_sha256", "selected_lambda"}
    core = {key: value for key, value in manifest.items() if key not in excluded}
    identity = fingerprint(core)
    summary_hash = digest((directory / "summary.json").read_bytes())
    if (
        identity != manifest.get("identity")
        or summary.get("identity") != identity
        or status.get("identity") != identity
        or status.get("phase") != "complete"
        or summary_hash != manifest.get("summary_sha256")
        or summary_hash != status.get("summary_sha256")
    ):
        raise RuntimeError("Diagnostic evidence is incomplete or has inconsistent hashes")
    module = bidding_context if ridge else bidding_predictive
    if digest(Path(module.__file__).read_bytes()) != manifest["source_sha256"]:
        raise RuntimeError("Diagnostic source differs from the completed evidence")
    if ridge:
        model_hash = digest((directory / "model.npz").read_bytes())
        if model_hash != manifest.get("model_sha256") or model_hash != status.get("model_sha256"):
            raise RuntimeError("Ridge model hash mismatch")
    return manifest, summary


def render(summary: dict, ridge: dict) -> str:
    def number(value) -> str:
        return "未定义" if value is None else f"{value:.4f}"

    def table(columns, rows) -> str:
        head = "<tr>" + "".join(f"<th>{html.escape(c)}</th>" for c in columns) + "</tr>"
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>"
            for row in rows
        )
        return f"<div class='scroll'><table>{head}{body}</table></div>"

    current, remaining = [], []
    for method in summary["methods"]:
        name = method["method"]
        if method["seed"] is not None:
            name += f" / seed {method['seed']}"
        for scope, rows, groups in (
            ("current", current, ("current_cost", "current_conversion")),
            ("remaining", remaining, ("remaining_cost", "remaining_conversion")),
        ):
            if scope not in method["pooled"]:
                continue
            values = method["pooled"][scope]
            rows.append(
                [name, number(values["log_mse"]), number(values["smooth_l1_beta1"])]
                + [number(values["groups"][g]["aggregate_predicted_to_true_ratio"]) for g in groups]
            )
    columns = ["预测器 / 种子", "log MSE", "Smooth L1", "成本预测/真实", "转化预测/真实"]
    ridge_rows = [
        [
            str(candidate["ridge_lambda"]),
            number(candidate["train"]["log_mse"]),
            number(candidate["validation"]["log_mse"]),
        ]
        for candidate in ridge["candidates"]
    ]
    limits = "".join(f"<li>{html.escape(item)}</li>" for item in summary["limitations"])
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>出价响应预测：同口径验证审计</title>
<style>body{{font:16px/1.7 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#202735}}
table{{border-collapse:collapse;font-size:14px}}th,td{{border:1px solid #d8dce5;padding:8px;text-align:left}}
.scroll{{overflow-x:auto}}code{{overflow-wrap:anywhere}}h2{{margin-top:32px}}</style>
<h1>出价响应预测：同口径验证审计</h1>
<p>问题：小的预测误差，是否同时意味着成本与转化总量预测准确？
这是公开 AuctionNet 自定义固定对手模拟中的验证集诊断，不是出价收益评估。</p>
<p>6 个验证市场，36 个市场—广告主组合，360 条名义行为轨迹，17,280 行。
每行包含 11 个动作；行和动作不构成独立实验。训练市场为 16 个。
三个神经模型分别计算误差，没有集成预测。</p>
<h2>当前时段：所有预测器使用相同标签</h2>
{table(columns, current)}
<p>误差越小越好；总量比接近 1 表示该口径下总量更接近真实值，不代表逐状态校准。
这些是按初始预算／目标 CPA 归一化后的标签总量比，不是原始总花费或总转化的比率。
零动作仍以相同权重进入误差。</p>
<h2>剩余时段：神经模型与 ridge</h2>
{table(columns, remaining)}
<h2>Ridge 的验证选择记录</h2>
<p>选择 lambda = {ridge["selected_lambda"]}；下表包含所有已声明候选。</p>
{table(["lambda", "训练 log MSE", "验证 log MSE"], ridge_rows)}
<h2>如何使用这些结果</h2>
<p>模型输出是响应点预测。log 空间损失不保证得到已校准的条件算术均值；
控制器将它们用于成本／价值决策，是需要检验的近似。误差与总量比可以支持偏差诊断，
不能单独证明控制器瓶颈、序列模型价值或实际投放收益。</p>
<ul>{limits}</ul>
<p><a href="summary.json">完整指标与逐市场结果</a> · <a href="manifest.json">审计来源</a> ·
<a href="ridge-summary.json">Ridge 拟合记录</a> · <a href="ridge-manifest.json">Ridge 来源</a> ·
<a href="export.json">导出文件校验</a></p></html>"""


def export(audit_run: Path, ridge_run: Path, destination: Path) -> None:
    manifest, summary = evidence(audit_run, ridge=False)
    ridge_manifest, ridge_summary = evidence(ridge_run, ridge=True)
    if (
        manifest["main_identity"] != ridge_manifest["main_identity"]
        or summary.get("main_identity") != manifest["main_identity"]
        or summary.get("ridge_identity") != ridge_manifest["identity"]
        or manifest["ridge"]["identity"] != ridge_manifest["identity"]
        or manifest["ridge"]["files_sha256"]["summary.json"]
        != digest((ridge_run / "summary.json").read_bytes())
        or manifest["ridge"]["files_sha256"]["manifest.json"]
        != digest((ridge_run / "manifest.json").read_bytes())
    ):
        raise RuntimeError("Predictive and ridge evidence do not belong together")
    methods = {(row["method"], row["seed"]) for row in summary["methods"]}
    expected = {("empirical", None), ("ridge", None)} | {("response", s) for s in (11, 23, 37)}
    if methods != expected or len(summary["methods"]) != len(expected):
        raise RuntimeError("Incomplete predictive method coverage")
    counts = summary["sample_counts"]
    markets = manifest["data"]["validation"]["market_ids"]
    if (
        counts["rows"] != 17280
        or counts["behavior_trajectories"] != 360
        or counts["independent_validation_markets"] != 6
        or counts["training_markets_for_empirical"] != 16
        or counts["market_focal_pairs"] != 36
        or counts["per_market_rows"] != {str(m): 2880 for m in markets}
        or any(set(row["per_market"]) != {str(m) for m in markets} for row in summary["methods"])
    ):
        raise RuntimeError("Incomplete validation market/row coverage")
    payload = {
        "manifest.json": (audit_run / "manifest.json").read_bytes(),
        "summary.json": (audit_run / "summary.json").read_bytes(),
        "ridge-manifest.json": (ridge_run / "manifest.json").read_bytes(),
        "ridge-summary.json": (ridge_run / "summary.json").read_bytes(),
        "report.html": render(summary, ridge_summary).encode(),
    }
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
            or {path.name for path in destination.iterdir()} - (set(payload) | {"export.json"})
        ):
            raise RuntimeError("Destination contains different or unreviewed evidence")
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in payload.items():
        (destination / name).write_bytes(content)
    record = {
        "run_identity": manifest["identity"],
        "exporter_sha256": digest(Path(__file__).read_bytes()),
        "files_sha256": {name: digest(content) for name, content in payload.items()},
        "scope": "completed validation-only predictive diagnostics; no policy-value claim",
        "verification": "identity, evidence hashes and coverage checked; metrics are not independently recomputed",
        "privacy_gate": "allowlisted text checked for home paths, email addresses and common credential patterns; not a proof against every disclosure",
        "excluded": "model weights, traffic tapes, runner state, machine logs and absolute local paths",
    }
    (destination / "export.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"exported validation diagnostic files={len(payload)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-run", type=Path, required=True)
    parser.add_argument("--ridge-run", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    export(args.audit_run, args.ridge_run, args.destination)
