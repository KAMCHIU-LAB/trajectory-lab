"""Portable evidence with market-cluster uncertainty and explicit claim limits."""

from __future__ import annotations

import base64
import hashlib
import html
import io
from collections import defaultdict
from pathlib import Path

import numpy as np

from trajectorylab.utils import read_json, write_json


def aggregate(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    groups = defaultdict(list)
    index = {(r["method"], r["market"], r["focal"], r["scenario"], r["seed"]): r for r in rows}
    for row in rows:
        groups[row["scenario"], row["method"]].append(row)
    summaries, comparisons = [], []
    for (scenario, method), group in sorted(groups.items()):
        metrics = (
            "score",
            "conversions",
            "violation",
            "zero_conversions",
            "cpa_debt",
            "budget_utilization",
            "cost",
        )
        summary = {
            "scenario": scenario,
            "method": method,
            "markets": len({r["market"] for r in group}),
            "campaigns": len({(r["market"], r["focal"]) for r in group}),
            "training_seeds": len({r["seed"] for r in group if r["seed"] >= 0}),
        }
        summary.update({metric: float(np.mean([r[metric] for r in group])) for metric in metrics})
        summary["pooled_cpa"] = (
            sum(r["cost"] for r in group) / sum(r["conversions"] for r in group)
            if sum(r["conversions"] for r in group)
            else None
        )
        summaries.append(summary)
        references = [name for name in ("feedback", "fixed-tuned") if name != method]
        if method.endswith("+correction"):
            references.append(method.split("+")[0])
        if method == "response":
            references.append("response-hard")
        for reference in references:
            for metric in ("score", "conversions", "violation"):
                market_deltas, seed_deltas = defaultdict(list), defaultdict(list)
                for row in group:
                    seed = -1 if reference in ("feedback", "fixed-tuned") else row["seed"]
                    matched = index[reference, row["market"], row["focal"], scenario, seed]
                    delta = row[metric] - matched[metric]
                    market_deltas[row["market"]].append(delta)
                    seed_deltas[row["seed"]].append(delta)
                values = np.array([np.mean(market_deltas[k]) for k in sorted(market_deltas)])
                rng = np.random.default_rng(1407)
                bootstrap = rng.choice(values, (2000, len(values)), replace=True).mean(axis=1)
                lo, hi = np.quantile(bootstrap, [0.025, 0.975])
                comparisons.append(
                    {
                        "scenario": scenario,
                        "method": method,
                        "reference": reference,
                        "metric": metric,
                        "mean_delta": float(values.mean()),
                        "market_bootstrap_95": [float(lo), float(hi)],
                        "seed_mean_deltas": {
                            str(k): float(np.mean(v)) for k, v in seed_deltas.items()
                        },
                        "uncertainty_scope": "resample markets after averaging focal agents and training seeds; not a full training-seed confidence interval",
                    }
                )
    return summaries, comparisons


def render(directory: Path) -> None:
    manifest = read_json(directory / "manifest.json")
    rows = read_json(directory / "results.json")
    summaries, comparisons = aggregate(rows)
    write_json(
        directory / "summary.json",
        {
            "groups": summaries,
            "paired_comparisons": comparisons,
            "purpose": manifest["config"]["study"]["purpose"],
            "identity": manifest["identity"],
            "independent_unit": "market ID",
            "feedback_shared_across_training_seeds": True,
            "renderer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for method in sorted({r["method"] for r in summaries}):
        group = [r for r in summaries if r["method"] == method]
        axes[0].plot(
            [r["scenario"] for r in group], [r["score"] for r in group], marker="o", label=method
        )
        axes[1].scatter(
            [r["violation"] for r in group], [r["conversions"] for r in group], label=method
        )
    axes[0].set(ylabel="Mean CPA-penalized conversion score", title="Declared stress scenarios")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].set(
        xlabel="CPA violation frequency",
        ylabel="Mean realized conversions",
        title="Conversion / constraint trade-off",
    )
    axes[1].legend(fontsize=8)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=140)
    plt.close(figure)
    encoded = base64.b64encode(buffer.getvalue()).decode()

    def cell(value):
        if value is None:
            return "undefined"
        return f"{value:.3f}" if isinstance(value, float) else html.escape(str(value))

    columns = (
        "scenario",
        "method",
        "score",
        "conversions",
        "violation",
        "zero_conversions",
        "cpa_debt",
        "budget_utilization",
    )
    table = "<tr>" + "".join(f"<th>{x}</th>" for x in columns) + "</tr>"
    table += "".join(
        "<tr>" + "".join(f"<td>{cell(r[k])}</td>" for k in columns) + "</tr>" for r in summaries
    )
    study = manifest["config"]["study"]
    training = read_json(directory / "training.json")
    dataset = read_json(directory / "dataset-summary.json")
    content = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>公开生成式出价实验</title><style>body{{font:16px/1.6 system-ui;max-width:1180px;margin:32px auto;padding:0 20px;color:#202a36}}table{{border-collapse:collapse;font-size:13px}}td,th{{padding:7px;border:1px solid #dce0e6;text-align:right}}th{{background:#edf2f8}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}.note{{background:#f1f5fa;padding:16px;border-left:4px solid #47678c}}h1,h2{{line-height:1.3}}</style>
<h1>预测失准下的可控生成式出价</h1>
<p>阶段：<b>{html.escape(study["purpose"])}</b> · 实验标识 <code>{manifest["identity"][:16]}</code></p>
<div class="note">本报告使用公开 AuctionNet 拍卖内核的自定义固定对手配置。它不是官方排行榜复现，也不能推出线上收益。
{"本轮为小样本可行性实验，转化稀疏，不能作为方法有效性的证据。" if study["purpose"] == "feasibility-only" else "正式配置仍只支持本模拟设置内的结论。"}
预算上限由结算器执行；CPA 约束由实际转化检验。零转化时 CPA 未定义，仍计入违约和零转化频率。</div>
<h2>实验设计</h2><p>训练/验证/测试市场：{len(study["train_markets"])}/{len(study["validation_markets"])}/{len(study["test_markets"])}；
每市场 {len(study["focal_agents"])} 个独立受测广告主，{study["pv_count"]:,} 个请求 PV，48 个时段。
训练种子 {html.escape(str(study["training_seeds"]))}；训练轨迹 {dataset["train_trajectories"]}。
同一市场中的广告主与随机种子不视作新增独立市场。</p>
<p>DT 生成动作；response 预测各候选动作的成本/转化响应曲线，由约束控制器选动作。
二者的监督信息不同：response 使用模拟器的密集反事实标签，DT 使用行为轨迹。不能把结果解释为等数据条件下的架构胜负。
response 默认在预算可行范围内最大化预测的 CPA 惩罚转化分数；response-hard 复用同一检查点，额外硬筛预测 CPA。fixed-tuned 只在验证集选乘数。
correction 只复用既有检查点，参数仅在验证集选择。压力场景改变可见 pCVR 与相应出价，真实转化过程保持不变。</p>
<img alt="Stress results and conversion constraint tradeoff" src="data:image/png;base64,{encoded}">
<h2>实际结果</h2><p>violation 是 CPA 违约频率；cpa_debt = 花费 − 目标 CPA × 转化数。Score 为带平方 CPA 惩罚的转化数。
主判断同时看转化与约束，避免把少花钱误判为业务改进。</p><div style="overflow-x:auto"><table>{table}</table></div>
<h2>不确定性与可复核材料</h2><p>summary.json 中提供相对 feedback、fixed-tuned 的配对差异，
以及同训练种子下校正与未校正、软决策与硬筛选的配对比较。区间按市场聚类做 2,000 次 bootstrap，
以及各训练种子的均值差异。区间对种子先取平均，未包含完整训练随机性；少量市场的区间需谨慎解读。</p>
<p>主比较为 over_50 下 response+correction 相对 response 的 score 与 CPA 违约频率差异。其余比较为探索性分析，未进行多重比较校正。
summary.json 单独记录报告生成器哈希，可在保留原始运行身份的前提下复核后续分析版本。</p>
<p><a href="results.csv">逐广告主原始结果 CSV</a> · <a href="summary.json">聚合与配对区间</a> ·
<a href="selection.json">验证集选择</a> · <a href="manifest.json">配置与来源哈希</a> ·
<a href="training.json">训练资源</a> · <a href="kernel-audit.json">公开内核等价核验</a></p>
<p>累计模型训练时间 {sum(r["seconds"] for r in training) / 60:.1f} 分钟；
模型参数 {html.escape(str({r["kind"]: r["parameters"] for r in training}))}。
MPS 随机状态被保存，硬件或软件变化下不承诺逐位复现。</p>
<h2>结论边界</h2><ul><li>固定对手不适应、不执行自身预算；受测广告主分别评估，无多智能体共同学习。</li>
<li>模拟转化、缩放预算、有限动作集及预设失准情形，均与真实广告市场有差距。</li>
<li>响应预测为独立的小模型实现，参考 GRM 思路，没有复现其完整训练数据和规模。</li>
<li>不含公司数据、公司代码或线上指标。AI 辅助开发；实际实现与结论以代码和实验记录为准。</li></ul>
<p>来源：<a href="https://github.com/alimama-tech/AuctionNet">AuctionNet</a> ·
<a href="https://arxiv.org/abs/2605.27811">GRM</a> ·
<a href="https://arxiv.org/abs/2106.01345">Decision Transformer</a></p></html>"""
    (directory / "report.html").write_text(content)
