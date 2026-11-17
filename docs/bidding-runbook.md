# 生成式出价实验运行指南

本实验研究：当广告主看到的转化概率（pCVR）失准时，基于历史反馈的数值控制器、Decision Transformer（DT）和响应预测模型，在转化与 CPA 约束之间表现如何。实验基于公开 AuctionNet 拍卖代码及自定义的固定对手设置；细节和统计口径见[实验协议](bidding-protocol.md)。项目不包含商业资料、公司代码或线上指标。

## 环境与启动

在仓库根目录使用已有的 `.venv` 安装项目依赖并执行模块：

```bash
.venv/bin/python -m pip install -e .
.venv/bin/python -m trajectorylab.bidding --config configs/bidding-pilot.toml
```

默认阶段为 `all`。完成后，运行日志会打印输出目录；默认位于 `artifacts/bidding/` 下，目录名包含实验名和来源/配置指纹。也可显式指定输出目录：

```bash
.venv/bin/python -m trajectorylab.bidding --config configs/bidding-main.toml --output artifacts/bidding/my-main
```

`artifacts/` 已列入 `.gitignore`，数据、模型、状态和报告等运行产物不会作为普通 Git 文件跟踪。相同配置、代码和输出目录可用于续跑；若输出目录已有不同身份的 manifest，程序会拒绝混用，应为新身份选择新目录。

## 阶段、恢复与产物

可用阶段如下：

- `--stage data`：执行内核等价核验，并生成 train、validation、test 三组市场数据。
- `--stage train`：使用已生成的全部市场数据创建训练轨迹、训练 DT 与 response 模型，并保存训练记录。
- `--stage evaluate`：使用现有模型与市场数据进行验证集选择、测试集评估并渲染报告。要求数据和模型均已存在。
- `--stage all`（默认）：依次执行数据生成、训练和评估。

例：

```bash
.venv/bin/python -m trajectorylab.bidding --config configs/bidding-main.toml --stage data
.venv/bin/python -m trajectorylab.bidding --config configs/bidding-main.toml --stage train
.venv/bin/python -m trajectorylab.bidding --config configs/bidding-main.toml --stage evaluate
```

训练会按配置间隔保存 `checkpoint.pt`，其中包括模型、优化器、训练步数和随机状态；再次运行相同配置与输出目录时从该 checkpoint 继续。每个模型的 `best.pt` 和训练历史也保存在对应模型目录。中断后可重跑原命令；如需单独运行后续阶段，先确保它所需的市场数据和模型已经齐备。`status.json` 记录最近阶段或中断/失败状态。身份不匹配时不要复用旧输出目录。
同一输出目录只运行一个进程；程序使用文件锁拒绝重复占用。此实现面向 macOS/POSIX 本地运行。

评估完成后主要产物包括：

- `report.html`：可直接用浏览器打开的报告入口。
- `results.csv`、`results.json`：逐市场、广告主、场景和方法的测试结果。
- `summary.json`：汇总及相对反馈控制器的配对比较。
- `selection.json`：只根据验证市场选择的控制器及校正参数。
- `manifest.json`、`kernel-audit.json`、`dataset-summary.json`、`training.json`：身份/来源、内核核验、数据规模和训练记录。
- `markets/`、`models/`：生成的市场数据及模型 checkpoint。

## 配置与比较范围

| 配置 | 用途与市场划分 | 训练参数 |
| --- | --- | --- |
| `bidding-pilot.toml` | feasibility-only；训练市场 1001–1004，验证 2001–2002，测试 3001–3004；种子 11；每市场 50,000 PV；受测广告主 ID 0、16、32 | MPS，2 线程；宽度 64、2 层、4 heads、上下文 16；1,500 步、batch 32、学习率 0.0003；每 500 步 checkpoint、每 100 步记录 |
| `bidding-main.toml` | custom-fixed-opponent-study；训练市场 1001–1016，验证 2001–2006，测试 3101–3112（与已查看过的 pilot 测试市场分开）；种子 11、23、37；每市场 500,000 PV；受测广告主 ID 0、8、16、24、32、40 | MPS，2 线程；宽度 128、2 层、4 heads、上下文 24；15,000 步、batch 32、学习率 0.0003；每 1,000 步 checkpoint、每 250 步记录 |

两份配置都使用相同的离散动作集合 `[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]`。`[0.75, 1, 1.25, 1.5]` 是施加于广告主可见 pCVR 的失准倍率，不是对手出价倍率。固定对手使用基于市场种子的异质 bid multiplier（范围 15–145），并随时段波动；对手策略不适应受测广告主。评估包含校准和四种失准压力场景。设备配置为 MPS；若运行环境没有可用 MPS，程序会报错，需在本地配置中明确改用支持的设备后再运行，并使用新的输出身份。

比较的方法包括固定 multiplier 1（`fixed-1`）、在验证集选择 multiplier 的固定策略（`fixed-tuned`）、数值反馈控制器、DT、DT 加数值校正、response 模型及其校正版本，以及复用同一 response checkpoint 的 `response-hard`。response 默认在预算可行的候选动作中最大化预测的 CPA 惩罚转化分数；`response-hard` 对预测 CPA 做硬筛选，并在可行候选中最大化预测转化，若没有可行候选则选择预测 CPA 债务较小的预算可行候选。校正版本复用对应基础模型的 checkpoint。**response 模型的监督包含模拟器干预生成的密集候选动作反事实成本/转化标签；DT 的监督来自固定/反馈行为策略的轨迹。两者监督信息和学习目标不同，结果差异不能直接归因于架构优劣。**

## 低电量后台运行

仓库提供 `scripts/run_bidding_background.py`，可在仓库根目录用现有虚拟环境启动实验并指定配置与输出目录：

```bash
.venv/bin/python scripts/run_bidding_background.py --config configs/bidding-main.toml --output artifacts/bidding/background-main
```

脚本只为其启动的实验子进程运行 `caffeinate`，不修改 `pmset` 设置；电池电量降至 20% 或以下时会中断子进程，让实验通过 checkpoint 恢复，并在接通电源且电脑唤醒后自动续跑。状态写入输出目录的 `runner.json`。不能据此承诺合盖期间继续运行。
上述命令在当前终端启动运行器；脚本本身不注册后台服务。一次性 launchd 调度应使用 `KeepAlive=false`，避免实验完成后被反复重启。

## 导出可公开证据

完整运行结束后，可导出明确定义的文本文件集合：

```bash
.venv/bin/python scripts/export_bidding_report.py --run artifacts/bidding/my-main --destination reports/bidding-main
```

导出器核对任务完整性与汇总一致性，检查常见凭据、电子邮件和本机用户目录路径，保留文件校验摘要。
它排除模型、流量数据、本机日志和运行器状态；规则扫描不能证明不存在任何形式的隐私泄漏，公开前仍需人工审阅。
受控运行器支持 `--export reports/bidding-main`，在完成后执行该导出流程；不会自动提交或上传到 GitHub。

Pilot 是小规模可行性配置，转化稀疏，只用于检查数据生成、训练、评估和报告链路，**不支持方法效果结论**。主实验虽使用更大的市场和 PV 配置，仍是派生自 AuctionNet 的自定义固定对手实验，不是官方 trained-opponent benchmark 或排行榜复现；对手不适应且其预算不执行，结论限于所声明的模拟设置。
