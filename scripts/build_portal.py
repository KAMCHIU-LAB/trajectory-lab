"""Build a bidding-first report index from public exports under reports/."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

from trajectorylab.utils import read_json, write_json


def bidding_export(reports: Path, slug: str, *, control: bool = False):
    """Trust completed exporter evidence, without repeating its statistical analysis."""
    directory = reports / slug
    try:
        if directory.is_symlink() or not directory.resolve().is_relative_to(reports.resolve()):
            return None
        if any(
            not (directory / name).resolve().is_relative_to(directory.resolve())
            for name in ("export.json", "status.json", "manifest.json")
        ):
            return None
        record = read_json(directory / "export.json")
        status = read_json(directory / "status.json")
        manifest = read_json(directory / "manifest.json")
        phase = status["phase"]
        if phase not in ({"complete", "complete-not-promoted"} if control else {"complete"}):
            return None
        if record["run_identity"] != manifest["identity"]:
            return None
        if not control:
            purpose = (
                "feasibility-only" if slug == "bidding-pilot" else "custom-fixed-opponent-study"
            )
            if manifest["config"]["study"]["purpose"] != purpose:
                return None
        required = {"manifest.json", "selection.json", "status.json"}
        report_files = {"report.html", "results.csv", "results.json", "summary.json"}
        if control:
            if record["phase"] != phase:
                return None
            required |= {"empirical-rates.json", "validation-report/manifest.json"}
            required |= {f"validation-report/{name}" for name in report_files}
            if phase == "complete":
                required |= report_files
        else:
            required |= report_files | {
                "training.json",
                "dataset-summary.json",
                "kernel-audit.json",
            }
        if not required <= record["files_sha256"].keys():
            return None
        for name, expected in record["files_sha256"].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                return None
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return None
        prefix = "validation-report/" if phase == "complete-not-promoted" else ""
        summary = read_json(directory / f"{prefix}summary.json")
        return {"manifest": manifest, "status": status, "summary": summary, "prefix": prefix}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def bidding_cards(reports: Path):
    cards, catalog = [], []
    for slug, title, control in (
        ("bidding-main", "第一阶段 · 正式研究", False),
        ("bidding-control", "第二阶段 · 控制器研究", True),
        ("bidding-pilot", "开发参考 · Pilot", False),
    ):
        evidence = bidding_export(reports, slug, control=control)
        if evidence is None:
            stage = "正式研究待完成" if slug == "bidding-main" else "等待完整公开导出"
            description = "尚无完整公开导出，暂不展示结果或推断完成状态。"
            cards.append(
                f'<div class="card"><span>{stage}</span><h2>{title}</h2><p>{description}</p></div>'
            )
            catalog.append({"study": slug, "stage": stage, "report": None})
            continue
        manifest, status = evidence["manifest"], evidence["status"]
        if control:
            layout = manifest["evidence_layout"]
            if status["phase"] == "complete-not-promoted":
                stage = "验证已完成 · 未晋级"
                description = "候选未达到预设晋级条件；本轮未开展独立确认。"
            else:
                confirmation = evidence["summary"]["confirmation"]
                if confirmation["accepted"]:
                    stage = "独立确认已完成 · 达到预设标准"
                elif not confirmation["conversion_ratio"]["defined_for_all_resamples"]:
                    stage = "独立确认已完成 · 结论不确定"
                else:
                    stage = "独立确认已完成 · 未达到接受标准"
                description = (
                    f"{len(layout['confirmation_markets'])} 个独立确认市场；"
                    f"主场景 {confirmation['scenario']}，结论以完整报告的预设判断为准。"
                )
            facts = (
                f"{len(layout['validation_markets'])} 个验证市场 · "
                f"{len(layout['training_seeds'])} 个固定模型训练种子"
            )
        else:
            study = manifest["config"]["study"]
            pilot = slug == "bidding-pilot"
            stage = "开发结果 · 可行性实验" if pilot else "正式研究已完成 · 已公开导出"
            description = (
                "小规模开发结果用于检查流程和失败边界，不能据此证明方法有效。"
                if pilot
                else "固定乘数、反馈控制、DT 与响应模型的完整比较；完成实验不等于证实收益。"
            )
            split = "开发评估" if pilot else "测试"
            facts = (
                f"{len(study['test_markets'])} 个{split}市场 · "
                f"{len(study['training_seeds'])} 个训练种子 · "
                f"每市场请求 {study['pv_count']:,} PV"
            )
        report = f"{slug}/{evidence['prefix']}report.html"
        summary = f"{slug}/{evidence['prefix']}summary.json"
        cards.append(
            f'<div class="card"><span>{html.escape(stage)}</span><h2>{title}</h2>'
            f'<p>{html.escape(description)}</p><p class="small">{html.escape(facts)}</p>'
            f'<strong><a href="{report}">完整报告与原始记录 ↗</a></strong>'
            f'<p class="small"><a href="{summary}">统计摘要</a> · '
            f'<a href="{slug}/manifest.json">实验来源</a> · '
            f'<a href="{slug}/export.json">导出校验</a></p></div>'
        )
        catalog.append(
            {"study": slug, "stage": stage, "identity": manifest["identity"], "report": report}
        )
    return "".join(cards), catalog


def forecast_audit(reports: Path):
    directory = reports / "bidding-forecast"
    try:
        if directory.is_symlink() or not directory.resolve().is_relative_to(reports.resolve()):
            return None
        for name in ("export.json", "manifest.json", "summary.json"):
            if not (directory / name).resolve().is_relative_to(directory.resolve()):
                return None
        record = read_json(directory / "export.json")
        manifest = read_json(directory / "manifest.json")
        summary = read_json(directory / "summary.json")
        required = {
            "manifest.json",
            "summary.json",
            "ridge-manifest.json",
            "ridge-summary.json",
            "report.html",
        }
        if (
            set(record["files_sha256"]) != required
            or record["run_identity"] != manifest["identity"]
            or summary["identity"] != manifest["identity"]
        ):
            return None
        for name, expected in record["files_sha256"].items():
            path = directory / name
            if not path.resolve().is_relative_to(directory.resolve()):
                return None
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                return None
        return {"identity": manifest["identity"], "report": "bidding-forecast/report.html"}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def tradeoff_figure(reports: Path):
    directory = reports / "bidding-analysis"
    names = {"bidding-tradeoffs.svg", "bidding-tradeoffs.png"}
    try:
        if directory.is_symlink() or not directory.resolve().is_relative_to(reports.resolve()):
            return None
        if any((directory / name).is_symlink() for name in names | {"figure.json"}):
            return None
        record = read_json(directory / "figure.json")
        summary = reports / "bidding-main/summary.json"
        if (
            set(record["figure_sha256"]) != names
            or hashlib.sha256(summary.read_bytes()).hexdigest() != record["summary_sha256"]
            or read_json(summary)["identity"] != record["source_identity"]
        ):
            return None
        for name, expected in record["figure_sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                return None
        return {"identity": record["source_identity"], "record": "bidding-analysis/figure.json"}
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build(root: Path) -> Path:
    reports = root / "reports"
    studies = [
        (
            "pendulum",
            "01 · 数据选择",
            "Pendulum controlled study",
            "3 views · 3 methods · 3 training seeds",
        ),
        (
            "pointmaze",
            "02 · 公开任务",
            "Public PointMaze subset",
            "20,000 training transitions · recovered evaluation environment",
        ),
        (
            "ablation",
            "03 · 机制校验",
            "IQL weighting ablation",
            "Zero advantage temperature · matched BC actor comparison",
        ),
    ]
    cards, tables, catalog = [], [], []
    for slug, eyebrow, title, description in studies:
        summary = read_json(reports / slug / "summary.json")
        manifest = read_json(reports / slug / "evidence/suite.json")
        if manifest["status"] != "complete" or len(summary["results"]) != manifest["expected_runs"]:
            raise ValueError(f"Incomplete export: {slug}")
        count = len(summary["results"])
        catalog.append({"study": slug, "suite_id": summary["suite_id"], "runs": count})
        cards.append(
            f'<a class="card" href="{slug}/"><span>{eyebrow}</span><h2>{title}</h2><p>{description}</p><strong>{count} completed runs ↗</strong></a>'
        )
        rows = []
        for row in summary["groups"]:
            deviation = row["return_std_across_training_seeds"]
            spread = f" ± {deviation:.2f}" if deviation is not None else " (one seed)"
            rows.append(
                f"<tr><td>{html.escape(row['view'])}</td><td>{html.escape(row['algorithm'].upper())}</td><td>{row['return_mean']:.2f}{spread}</td></tr>"
            )
        tables.append(
            f'<section><div class="section-title"><h2>{title}</h2><a href="{slug}/">完整报告与原始记录 ↗</a></div><div class="table"><table><thead><tr><th>Data view</th><th>Method</th><th>Raw return · mean ± seed std</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><p class="small">每组先对八个评测回合求均值，再对三个训练种子汇总。回报越高越好。</p></section>'
        )

    mechanism = read_json(reports / "ablation/mechanism-check.json")
    if mechanism["status"] != "passed":
        raise ValueError("The mechanism check has not passed")
    total = sum(study["runs"] for study in catalog)
    bidding, bidding_catalog = bidding_cards(reports)
    audit = forecast_audit(reports)
    audit_panel = (
        '<section><div class="section-title"><h2>预测审计 · 误差与总量偏差</h2>'
        '<a href="bidding-forecast/report.html">同口径验证报告 ↗</a></div>'
        "<p>在相同验证行为轨迹上比较三个神经模型种子、ridge 与训练集经验曲线。"
        "经验曲线仅有当前时段预测。"
        "分别查看当前／剩余响应的误差和预测总量比，检查预测是否适合资源决策。"
        "这是已用于模型选择的验证集诊断，不能单独证明出价收益或控制器瓶颈。</p></section>"
        if audit
        else ""
    )
    tradeoffs = tradeoff_figure(reports)
    tradeoff_panel = (
        '<section><div class="section-title"><h2>测试取舍 · 优势有场景边界</h2>'
        '<a href="bidding-analysis/figure.json">图表数据与来源 ↗</a></div>'
        '<img src="bidding-analysis/bidding-tradeoffs.svg" '
        'alt="Response minus feedback: score and CPA violation differences across five stress scenarios">'
        "<p>严重高估场景表现较好，名义及低估场景更差。"
        "这是 12 个测试市场上的探索性比较，区间条件于三个检查点；不是线上收益。</p></section>"
        if tradeoffs
        else ""
    )
    catalog_data = {
        "completed_runs": total,
        "completed_runs_scope": "offline RL foundation studies only",
        "studies": catalog,
        "bidding_studies": bidding_catalog,
        "bidding_forecast_audit": audit,
        "bidding_tradeoffs": tradeoffs,
        "mechanism_check": mechanism["status"],
    }
    write_json(reports / "catalog.json", catalog_data)
    results_link = (
        '<a href="results.md">基础实验结果与统计口径 ↗</a> · '
        if (reports / "results.md").is_file()
        else ""
    )
    page = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrajectoryLab · 生成式出价与公开实验</title>
<style>
:root{{font-family:system-ui,-apple-system,sans-serif;color:#1c2c40;background:#f4f6f8;color-scheme:light}}*{{box-sizing:border-box}}
body{{max-width:1160px;margin:auto;padding:32px 24px 60px}}nav{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #d9e1e7;padding:10px 0 20px;font-weight:700}}
nav span,.small{{color:#607086;font-weight:400;font-size:13px}}header{{padding:56px 0 30px;max-width:850px}}.eyebrow,.card>span{{font-size:12px;letter-spacing:1px;font-weight:750;color:#147f69}}
h1{{font-size:clamp(34px,5vw,56px);letter-spacing:-2px;line-height:1.15;margin:16px 0 24px}}header p{{font-size:17px;line-height:1.8;color:#53647a}}h2{{font-size:20px;margin:12px 0}}p{{line-height:1.7}}
.facts{{display:flex;gap:32px;padding:22px 0 32px;flex-wrap:wrap}}.facts strong{{font-size:30px;display:block;letter-spacing:-1px}}.facts span{{font-size:13px;color:#607086}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}a{{color:#226494;text-decoration:none}}a:hover{{text-decoration:underline}}.card{{display:block;padding:24px;border:1px solid #d6e0e7;border-radius:12px;background:#fff;color:inherit}}.card p{{font-size:13px;color:#607086;min-height:46px}}.card strong{{font-size:13px;color:#147f69}}
section{{margin-top:22px;padding:28px;background:white;border:1px solid #dbe3ea;border-radius:12px}}.section-title{{display:flex;align-items:center;justify-content:space-between;gap:18px}}.section-title a{{font-size:13px}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:12px 8px;border-bottom:1px solid #e8edf1;text-align:left;white-space:nowrap}}th{{font-size:12px;color:#607086}}li{{line-height:1.8;margin:7px 0}}.note{{padding:16px 20px;border-left:3px solid #147f69;background:#eaf3ef;margin:26px 0;font-size:14px;line-height:1.8}}img{{width:100%;height:auto}}footer{{margin-top:28px;line-height:1.8}}
.insights{{display:grid;grid-template-columns:1.2fr .8fr;gap:16px;margin-top:22px}}.insights section{{margin:0}}.insights h2{{margin-top:0}}.insight-lead{{font-size:17px;font-weight:650;line-height:1.6}}.insight-detail{{color:#53647a;font-size:14px}}.compact td,.compact th{{padding:7px 8px}}.transfer{{border-top:1px solid #e8edf1;margin-top:18px;padding-top:15px;font-size:14px;color:#53647a}}
@media(max-width:760px){{.cards,.insights{{grid-template-columns:1fr}}header{{padding-top:34px}}section{{padding:18px}}body{{padding:20px 14px}}.section-title{{display:block}}.card p{{min-height:0}}}}
</style><nav>TrajectoryLab <span>Generative bidding · Public experiments</span></nav>
<header><div class="eyebrow">公开模拟 · 预测误差 · 可追溯的实验</div><h1>生成式出价：预测失准时，反馈能改善决策吗？</h1><p>在公开 AuctionNet 的自定义固定对手环境中，比较简单出价、反馈控制与历史条件模型，检验转化、CPA 违约和预算使用的取舍。下面的阶段标签只依据已公开的完整导出；实验完成与预设接受标准是否达到分别标注。</p></header>
<div class="cards">{bidding}</div>
<div class="note">当前入口保留开发结果和未通过的研究阶段。尚无完整公开导出的阶段显示待完成，不能从数据生成或训练进度推断收益。所有出价结论均限于该固定对手模拟，不代表官方基准、线上效果或通用安全保证；calibrated 仅表示未额外注入预测偏差。</div>
{tradeoff_panel}
{audit_panel}
<section><div class="section-title"><h2>基础实验 · 离线 RL 与数据诊断</h2><a href="catalog.json">实验索引 / JSON ↗</a></div><p>以下三份已完成的基础实验研究数据选择、公开任务与机制校验，保留完整报告及原始记录。</p><div class="facts"><div><strong>{total}</strong><span>基础实验已完成运行</span></div><div><strong>3</strong><span>基础实验报告</span></div></div><div class="cards">{"".join(cards)}</div>
<details><summary>展开基础实验的分组回报与机制校验</summary>
{"".join(tables)}
<section><h2>受控实验总览</h2><img src="pendulum/figures/returns.png" alt="Pendulum three data views and three methods with training-seed error bars"></section>
<p class="small">{results_link}<a href="ablation/mechanism-check.json">IQL 权重消融的逐种子校验 ↗</a></p>
</details></section>
<footer class="small">独立公开数据项目 · 本页从公开 reports 导出读取阶段与来源，优先链接完整报告。<br>各报告的样本单位、训练预算、统计口径和限制以其原始记录为准；基础实验不能代替出价有效性验证。</footer></html>"""
    destination = reports / "index.html"
    destination.write_text(page)
    return destination


if __name__ == "__main__":
    print(build(Path.cwd()))
