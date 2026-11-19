# TrajectoryLab

**可控生成式出价：响应预测、预算 / CPA 控制与可复核的压力评测。**

[English](README.md) · [研究定位](docs/bidding-research-map.md) ·
[出价实验协议](docs/bidding-protocol.md) · [架构](docs/architecture.md)

[CPU 自动检查](https://github.com/KAMCHIU-LAB/trajectory-lab/actions/workflows/ci.yml) ·
[v0.2.0 回放指南](docs/reproduce.md) · [版本记录](CHANGELOG.md)

## 先看这里

- **已确认的局部收益：**高估 50% 时，3 个冻结响应模型在 24 个新模拟市场相对冻结反馈的 score **+17.6%**、转化 **+8.4%**、CPA 违约频率 **−24.8 个百分点**。[统计口径及失败边界](docs/bidding-findings.md)。
- **生成式分支：**已实现条件 Flow Matching、分位筛选与反馈回退；第二训练种子未通过业务门槛，生成分布带来的额外决策价值尚未确认。[已完成的种子复核与机制诊断](docs/bidding-generative-findings.md)。
- **延迟反馈：**独立的 24 市场确认中，posterior 补回的 score **+1.21%**（95% 区间 **[+0.41%, +2.05%]**），未达预设 +3% 实用门槛；高估、长延迟格下降 4.25%。同轨迹审计进一步区分估计精度与控制收益。[结果与边界](docs/bidding-delayed-findings.md)。
- **最新研究决定：**未知偏差专家选择在拟合前停止；这对专家的 hindsight 相对强反馈仅有 +2.27% 开发空间，未达 +3% 门槛。[结果与边界](docs/bidding-expert-findings.md)。
- **问题如何推动演进：**[17 个索引条目的研究记录](docs/bidding-research-journal.md)串起问题、假设、对照、结果和取舍；后续研究按[记录模板](docs/experiment-record-template.md)持续追加。
- **实现入口：**[拍卖结算](src/trajectorylab/bidding/market.py)、[响应学习](src/trajectorylab/bidding/learning.py)、[Flow 采样器](src/trajectorylab/bidding_generative_learning.py)、[公开记录](reports/bidding-generative/README.md)。

## 先跑一个完整的小例子

在源码目录中，使用 Python 3.11 或 3.12 环境执行：

```bash
python -m pip install -e '.[dev]'
python scripts/run_bidding_delayed_smoke.py --output artifacts/public-replay-v0.2.0
python scripts/verify_release.py --replay artifacts/public-replay-v0.2.0/smoke-evidence.json --replay-only
```

研究 checkout 使用 `--replay-only` 检查回放不变量。验证历史
`releases/v0.2.0.json` 中的源码文件哈希时，须 checkout `v0.2.0` tag，
在该版本中运行不带 `--replay-only` 的验证命令。

安装依赖后，无需 GPU、账号或外部数据集。预期得到 90 条验证记录、45 条确认记录，
成熟结算与硬预算检查均为 `true`，业务门槛为 `false`。这条路线只有 1 个确认市场，
用于检查软件流程，不能复现或证明历史收益。再次执行同一命令可检查冻结恢复。
[环境设置、预期输出与故障排查](docs/reproduce.md)。

## 研究什么

出价策略看到的是预测转化率，花掉的是真实预算，收到的转化反馈又有噪声。
当预测发生偏移时，学习到的响应模型能否比调过参的固定乘数、简单反馈控制器，
取得更好的转化与 CPA 约束权衡？

本项目用公开 AuctionNet 流量生成器和拍卖代码研究这个问题。
小型因果序列模型预测候选动作对应的成本 / 转化曲线，独立的数值控制器选择出价。
把预测与控制分开，才能在同一模型检查点下判断反馈修正、决策规则各自的作用。
当前实现中没有 LLM 规划器。

## 当前证据

| 研究 | 状态与可查看的证据 |
| --- | --- |
| [出价可行性实验](reports/bidding-pilot/report.html) | 已完成：1 个训练种子，480 条广告主周期评估记录。**尚未证明学习策略更优**；固定出价有竞争力，硬筛预测 CPA 会压低消耗。 |
| [正式出价实验](reports/bidding-main/report.html) | 已完成：16 / 6 / 12 个训练 / 验证 / 测试市场，3 个训练种子，6,480 条记录。历史修正未证实 score 改善；响应模型的优势取决于压力场景。 |
| [控制器诊断](reports/bidding-control/validation-report/report.html) | 已完成验证：11,340 条候选评估记录；新控制器未通过预设晋级门槛，没有开展新市场确认。 |
| [同口径预测审计](reports/bidding-forecast/report.html) | 已完成：相同验证轨迹上比较神经模型、上下文 ridge 和经验曲线。预测误差更小，不保证转化总量准确或出价收益更高。 |
| [所选动作与预算节奏审计](reports/bidding-onpolicy/report.html) | 已完成：1,080 个周期精确重放。检查控制器实际选择的动作、预测误差方向和早期预算消耗；属于事后验证诊断，不是因果证明。 |
| [新市场独立确认](reports/bidding-confirmation/report.html) | 已完成：24 个新市场、4,320 条记录。高估 50% 场景下，原 response 通过冻结的 score、转化保留、违约和种子一致性门槛；失败控制器的门槛仍未通过。 |
| [生成式响应开发复核](docs/bidding-generative-findings.md) | 复用 6 个开发市场，按规则在训练种子 23 后停止；业务复现与生成增量均未通过，附已完成的零拟合机制审计。 |
| [延迟反馈估计与控制](docs/bidding-delayed-findings.md) | 已完成：24 个新市场、19,440 条记录；7 个保护端点通过，+3% 实用门槛未通过。附 1,296 条同轨迹 pending 审计、源码哈希与小规模 CPU 重放。 |
| [Pending 使用策略对照](docs/bidding-delayed-strategy-findings.md) | 已完成：6 个旧开发市场比较 12 个配置。50% 补回的折中最好，但相对更强固定基线仅 +0.30%，没有候选通过晋级门槛，未开启新确认。 |
| [信息价值与反馈协调消融](docs/bidding-evolution-audits.md) | 已完成并独立复核：4 个训练市场上的历史赢家信息对照，以及 4 个训练 / 6 个已见开发市场上的控制机制对照。两个候选均未通过冻结门槛，公开摘录保留对照表、图和来源哈希。 |
| [动作解码器结构对照](docs/bidding-decoder-findings.md) | 三组各2,000步对照已完成并复核；乘积特征候选未胜过强对照，预测门失败，未进入策略评估。 |
| [未知偏差专家选择](docs/bidding-expert-findings.md) | `STOP_NO_HEADROOM`：8 个新训练 / 6 个新开发市场，252 个上下文、2,880 行记录，开发比较 24 种方法。不可部署的 hindsight 相对强反馈为 +2.27%，未达 +3%；未拟合 ridge，未生成或开启确认市场。 |
| [离线策略学习基础实验](docs/results.md) | 已完成：Pendulum、公开 PointMaze 子集和 IQL 机制消融，合计 39 次 BC / IQL / DT 训练。 |

同一市场内的广告主、不同方法和重复种子之间存在相关性，记录条数不是独立市场数。
[实测发现](docs/bidding-findings.md)保留负结果、计算成本和待验证的机制解释。

**一个获得独立支持的局部结果：**在 24 个全新模拟市场的 pCVR 高估 50% 场景，
冻结 response 相对冻结反馈基线的 **score 高 17.6%、转化高 8.4%、CPA 违约频率低
24.8 个百分点**。score 差为 +2.617，市场 bootstrap 95% 区间为 [1.801, 3.424]；
三个固定模型的 score 差均为正。新市场方案在旧结果发现后、新数据生成前冻结，
区间条件于这三个检查点。名义场景 score 低 3.8%（差值区间跨零），低估场景低
17.3%，不能宣传全面领先或线上收益；历史修正和新控制器的负结果仍然保留。
[查看完整统计口径和失败边界](docs/bidding-findings.md)。

![五种场景的旧测试与冻结模型独立确认：score 和 CPA 违约频率差异及市场区间](reports/bidding-replication/replication.svg)

## 值得检查的实现

| 部分 | 实现及作用 |
| --- | --- |
| 精确拍卖响应缓存 | [市场适配器](src/trajectorylab/bidding/market.py)：在声明的动作网格上缓存公开内核输出，各策略复用相同随机市场结果 |
| 学习策略 | [训练与执行](src/trajectorylab/bidding/learning.py)：因果 DT、单调响应模型；归一化只拟合训练市场 |
| 预测与控制的分离 | [逐时段控制研究](src/trajectorylab/bidding_control.py)：同一检查点比较控制器、经验响应曲线，以及完整的简单策略参数网格 |
| 反馈信息边界 | 策略只接收当前观测和过去实际反馈；发生预算截断的反馈不进入未截断预测的比例修正 |
| 延迟回传 | [事件账本](src/trajectorylab/bidding_delayed_experiment.py)与[pending 估计](src/trajectorylab/bidding_pending_response.py)：严格回传时间边界、成熟历史校准与最终标签结算 |
| 证据与恢复 | 源码 / 配置指纹、优化器与随机状态检查点、按市场配对统计、原始 CSV / JSON 和可公开的文本导出 |

响应模型使用模拟器干预产生的密集反事实标签，DT 使用行为策略轨迹，两者监督不同，
不能直接把分数差归因于架构。各方法能看到什么、主比较是什么、哪些分析属于探索，
都写在[实验协议](docs/bidding-protocol.md)中。

## 运行原始神经出价研究

以下是原始训练配置，区别于前面的 CPU 软件回放。在仓库根目录执行。
参考环境是 Apple M1、Python 3.12，出价配置使用 MPS。
其他设备需在独立本地配置中明确声明，形成新的实验身份。

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m trajectorylab.bidding --config configs/bidding-pilot.toml
```

这会真实生成数据、训练并评估，输出目录在日志中打印。Pilot 用于检查链路，不能证明算法优势。
完整实验、分阶段执行、断点恢复和低电量后台运行见[运行指南](docs/bidding-runbook.md)。

只查看已提交的报告，无需安装机器学习依赖：

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory reports
# 打开 http://127.0.0.1:8765
```

报告不依赖账号、API Key 或在线后端。原始训练 checkpoint、优化器 / RNG 状态、
流量数组、本机日志和运行器状态仍不公开。三份[公开响应专家权重](assets/bidding-response-v1/README.md)
是特定例外：它们从公开模拟训练的 checkpoint 转换为 numeric NPZ / JSON，加载无需 pickle。
GitHub 显示 HTML 源码，查看页面请本地启动上述服务。

小规模延迟回传 CPU 重放：

```bash
PYTHONPATH=src python scripts/run_bidding_delayed_smoke.py
```

该入口检查公开链路和恢复身份，不能复现正式收益；完整分阶段命令、假设和证据清单见
[延迟反馈协议](docs/bidding-delayed-protocol.md)。

## 离线策略学习基础

这部分展示轨迹选择、算法实现和实验判断能力，分数只描述对应模拟任务。

- **Pendulum：**高回报筛选使 BC / IQL 的三个配对种子都上升；DT 平均差为 +34.55，
  但差值标准差为 67.85，仅 2/3 种子上升，不能认为收益稳定。
- **PointMaze：**DT 平均回报较高，但所有方法的“是否曾到达目标”成功率都是 100%，指标已经饱和。
- **IQL 消融：**优势温度设为零时，在声明的实现和种子下恢复匹配的 BC actor 更新。

[结果与限制](docs/results.md) · [复现实验](docs/offline-policy-guide.md) ·
[离线学习协议](docs/protocol.md)

## 适用范围与来源

这是从 AuctionNet 派生的自定义固定对手模拟，采用自定义硬预算结算规则。
对手不适应策略，也不执行自身预算；项目不是官方排行榜复现。
模拟分数不能直接证明生产收益或业务因果效果。仓库不包含雇主代码、数据、内部集成或线上指标。

- [AuctionNet 来源与保留许可](third_party/auctionnet/README.md)。
- [GRM](https://arxiv.org/abs/2605.27811)：响应建模路线的研究参考，本项目未复现完整论文。
- [AIGB-R1](https://arxiv.org/abs/2607.17281)：规划 / 执行式出价 Agent 的研究背景，当前没有实现其 LLM Planner、PromptDT 或 D-GRPO。
- [IQL](https://arxiv.org/abs/2110.06169)、[Decision Transformer](https://arxiv.org/abs/2106.01345)、[Gymnasium Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) 与 [Minari PointMaze](https://minari.farama.org/main/datasets/D4RL/pointmaze/umaze-v2/)。

算法来自公开方法，项目贡献是独立实现和可检查的实验；开发使用了 AI 编程辅助。
代码使用 [MIT](LICENSE)，AuctionNet 子集保留独立的 [Apache-2.0 条款](third_party/auctionnet/README.md)，
上游数据遵循各自条款，不在仓库重新分发。

[代码、数据来源与发布边界](docs/code-and-data-provenance.md)。
