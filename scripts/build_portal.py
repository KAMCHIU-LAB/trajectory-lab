"""Build an offline report index from the three completed, exported studies."""

from __future__ import annotations

import html
import shutil
from pathlib import Path

from trajectorylab.utils import read_json, write_json


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
    catalog_data = {
        "completed_runs": total,
        "studies": catalog,
        "mechanism_check": mechanism["status"],
    }
    write_json(reports / "catalog.json", catalog_data)
    if (root / "docs/results.md").is_file():
        shutil.copy2(root / "docs/results.md", reports / "results.md")
    page = f"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>TrajectoryLab · 离线 RL 实验与数据诊断</title>
<style>
:root{{font-family:system-ui,-apple-system,sans-serif;color:#1c2c40;background:#f4f6f8;color-scheme:light}}*{{box-sizing:border-box}}
body{{max-width:1160px;margin:auto;padding:32px 24px 60px}}nav{{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #d9e1e7;padding:10px 0 20px;font-weight:700}}
nav span,.small{{color:#607086;font-weight:400;font-size:13px}}header{{padding:56px 0 30px;max-width:850px}}.eyebrow,.card>span{{font-size:12px;letter-spacing:1px;font-weight:750;color:#147f69}}
h1{{font-size:clamp(34px,5vw,56px);letter-spacing:-2px;line-height:1.15;margin:16px 0 24px}}header p{{font-size:17px;line-height:1.8;color:#53647a}}h2{{font-size:20px;margin:12px 0}}p{{line-height:1.7}}
.facts{{display:flex;gap:32px;padding:22px 0 32px;flex-wrap:wrap}}.facts strong{{font-size:30px;display:block;letter-spacing:-1px}}.facts span{{font-size:13px;color:#607086}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}a{{color:#226494;text-decoration:none}}a:hover{{text-decoration:underline}}.card{{display:block;padding:24px;border:1px solid #d6e0e7;border-radius:12px;background:#fff;color:inherit}}.card p{{font-size:13px;color:#607086;min-height:46px}}.card strong{{font-size:13px;color:#147f69}}
section{{margin-top:22px;padding:28px;background:white;border:1px solid #dbe3ea;border-radius:12px}}.section-title{{display:flex;align-items:center;justify-content:space-between;gap:18px}}.section-title a{{font-size:13px}}.table{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:12px 8px;border-bottom:1px solid #e8edf1;text-align:left;white-space:nowrap}}th{{font-size:12px;color:#607086}}li{{line-height:1.8;margin:7px 0}}.note{{padding:16px 20px;border-left:3px solid #147f69;background:#eaf3ef;margin:26px 0;font-size:14px;line-height:1.8}}img{{width:100%;height:auto}}footer{{margin-top:28px;line-height:1.8}}
@media(max-width:760px){{.cards{{grid-template-columns:1fr}}header{{padding-top:34px}}section{{padding:18px}}body{{padding:20px 14px}}.section-title{{display:block}}.card p{{min-height:0}}}}
</style><nav>TrajectoryLab <span>Offline RL · Data diagnostics · v0.1</span></nav>
<header><div class="eyebrow">REPRODUCIBLE EXPERIMENTS</div><h1>离线 RL 的数据选择实验</h1><p>相同的数据预算，不同的轨迹选择，会怎样改变策略学习？<br>以 BC、IQL 和轻量 Decision Transformer 为对照，展示从数据审计到策略评测的完整证据。</p></header>
<div class="facts"><div><strong>{total}</strong><span>已完成的正式训练</span></div><div><strong>3</strong><span>每组独立训练种子</span></div><div><strong>10,000</strong><span>每次训练的梯度更新</span></div><div><strong>2</strong><span>公开环境 / 任务</span></div></div>
<div class="cards">{"".join(cards)}</div>
<div class="note">本版结论限于所列任务、子集、实现和预算。各算法使用相同更新数，计算量并不相同；完整报告记录实际耗时、原始回报与限制。最终评测使用预先声明的种子和最后一个 checkpoint。</div>
<section><h2>如何审阅这个项目</h2><ol><li>从数据选择实验看算法与数据的交互，检查均值与种子波动。</li><li>打开完整报告，查看训练数据审计、验证曲线、支持距离与固定种子的实际策略轨迹。</li><li>核对 <a href="ablation/mechanism-check.json">IQL 权重消融的逐位参数校验</a>，再下载每次运行的 CSV / JSON。</li></ol><p class="small"><a href="results.md">实验分析文稿 / Markdown</a> · <a href="catalog.json">实验索引 / JSON</a>。全部图表与记录可离线阅读。</p></section>
{"".join(tables)}
<section><h2>受控实验总览</h2><img src="pendulum/figures/returns.png" alt="Pendulum three data views and three methods with training-seed error bars"></section>
<footer class="small">独立公开数据项目 · CPU 本地实测 · 数据集和 checkpoint 单独复现，不包含在 Git 仓库中。<br>支持距离为描述性诊断；三个训练种子只能提供有限的不确定性信息。方法、数据来源和实现选择见仓库实验协议。</footer></html>"""
    destination = reports / "index.html"
    destination.write_text(page)
    return destination


if __name__ == "__main__":
    print(build(Path.cwd()))
