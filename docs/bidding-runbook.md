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

两份配置都使用相同的离散动作集合 `[0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3]`。`[0.75, 1, 1.25, 1.5]` 是施加于广告主可见 pCVR 的失准倍率，不是对手出价倍率。固定对手使用基于市场种子的异质 bid multiplier（范围 15–145），并随时段波动；对手策略不适应受测广告主。评估包含未额外注入偏差的基准场景（`calibrated`，不代表已验证严格概率校准）和四种失准压力场景。设备配置为 MPS；若运行环境没有可用 MPS，程序会报错，需在本地配置中明确改用支持的设备后再运行，并使用新的输出身份。

比较的方法包括固定 multiplier 1（`fixed-1`）、在验证集选择 multiplier 的固定策略（`fixed-tuned`）、数值反馈控制器、DT、DT 加数值校正、response 模型及其校正版本，以及复用同一 response checkpoint 的 `response-hard`。response 默认在预算可行的候选动作中最大化预测的 CPA 惩罚转化分数；`response-hard` 对预测 CPA 做硬筛选，并在可行候选中最大化预测转化，若没有可行候选则选择预测 CPA 债务较小的预算可行候选。校正版本复用对应基础模型的 checkpoint。**response 模型的监督包含模拟器干预生成的密集候选动作反事实成本/转化标签；DT 的监督来自固定/反馈行为策略的轨迹。两者监督信息和学习目标不同，结果差异不能直接归因于架构优劣。**

## 低电量后台运行

仓库提供 `scripts/run_bidding_background.py`，可在仓库根目录用现有虚拟环境启动实验并指定配置与输出目录：

```bash
.venv/bin/python scripts/run_bidding_background.py --config configs/bidding-main.toml --output artifacts/bidding/background-main
```

脚本只为其启动的实验子进程运行 `caffeinate`，不修改 `pmset` 设置；电池电量降至 20% 或以下时会中断子进程，让实验通过 checkpoint 恢复，并在接通电源且电脑唤醒后自动续跑。状态写入输出目录的 `runner.json`。锁屏后进程可继续运行；合盖行为取决于设备和系统电源策略，不能保证实验继续。
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

## 条件接续控制器研究

`configs/bidding-control.toml` 指向已完成的第一轮目录，复用其三个响应模型检查点。
候选设置、经验预测对照和晋级条件见[第二轮协议](bidding-control-protocol.md)。
先做验证集筛选；不达标时保留所有候选结果并结束该轮，不生成新的确认数据。
达标后才生成 3201–3224 号独立市场，确认结果不用于重调本轮参数。

第一轮完成后可直接运行：

```bash
.venv/bin/python -m trajectorylab.bidding_control --config configs/bidding-control.toml
```

也可提前启动等待运行器；路径必须与配置一致：

```bash
.venv/bin/python scripts/run_bidding_control_background.py --config configs/bidding-control.toml --output artifacts/bidding/control-v1 --export reports/bidding-control
```

该运行器等待第一轮成功完成后再启动计算，沿用低电量暂停和过程绑定唤醒逻辑。
等待期间不持有唤醒断言；一次性 launchd 作业仍应设置 `KeepAlive=false`。
配置或受监督源码发生变化时拒绝继续；需要改实验时使用新配置和输出目录。
完整市场评估结果逐市场保存，可在同身份下恢复；被中断的单个市场需重做。

`complete-not-promoted` 表示验证集未达到晋级条件，并非运行失败。
公开导出器 `scripts/export_bidding_control.py` 同时支持此状态和完成独立确认的 `complete` 状态，
只复制声明的文本证据。它核对覆盖、原始记录与统计一致性，并检查常见隐私标识。
监督器在成功导出后退出并释放计算进程的唤醒断言。

## 独立确认 D

确认 D 使用 `configs/bidding-confirmation.toml` 中冻结的 24 个新市场（3301–3324），预计输出 4,320 条策略评估记录。主比较为 `over_50` 场景的原始 `response` 对 `feedback`；四项固定通过门槛和统计口径见[确认协议](bidding-confirmation-protocol.md)。结果生成中不读取部分策略结果，不因中间走势提前停止或调整门槛。

在仓库根目录前台启动或以相同身份续跑：

```bash
.venv/bin/python -m trajectorylab.bidding_confirmation --config configs/bidding-confirmation.toml --stage all
```

也可分开执行数据生成和评估；评估要求全部 24 个市场缓存及身份匹配：

```bash
.venv/bin/python -m trajectorylab.bidding_confirmation --config configs/bidding-confirmation.toml --stage data
.venv/bin/python -m trajectorylab.bidding_confirmation --config configs/bidding-confirmation.toml --stage evaluate
```

中断后重跑相同命令和配置，继续使用 `artifacts/bidding/confirmation-v1`；不要删除、替换或混用不同身份的输出。

低电量后台运行器会按配置生成市场、等待完整市场缓存后评估，并可在低电量暂停后恢复：

```bash
.venv/bin/python scripts/run_bidding_confirmation_background.py --config configs/bidding-confirmation.toml
```

只有执行状态为 `complete` 后才导出公开报告和聚合 campaign 记录：

```bash
.venv/bin/python scripts/export_bidding_confirmation.py --config configs/bidding-confirmation.toml --destination reports/bidding-confirmation
```

执行 `phase=complete` 表示流程完整结束；科学判定仍可能是 `confirmed`、`not-confirmed` 或 `inconclusive`。`phase=complete` 且 `decision=not-confirmed` 是按预设门槛得到的完整负结果，不是运行失败，仍可导出；数据、身份或执行错误导致的 `phase=failed`/未完成状态不应导出为确认报告。

## 复现范围

1. **公开证据核查：**报告公开汇总和 campaign 级记录，可检查记录覆盖、配对统计、判定门槛和导出哈希；不包含权重或市场缓存。
2. **重新训练：**公开代码和数据生成/训练流程可用于训练新模型，应保存在独立运行目录；改变配置、数据划分、源码或运行时会产生新实验身份。本轮 D 绑定既有检查点的精确哈希，不能直接换入重训产物并沿用确认结论。
3. **精确冻结检查点重放：**需要尚未公开的本地候选权重包、匹配的仓库源码与固定运行时；打包命令为：

   ```bash
   .venv/bin/python scripts/package_bidding_replay.py --output artifacts/bidding/publication/bidding-frozen-inputs.tar.gz
   ```

   该包是本地发布候选，不是公开报告附件。固定运行时为 Python 3.12.13、NumPy 2.5.5、PyTorch 2.15.0；哈希核验要求见包内说明及[确认协议](bidding-confirmation-protocol.md)。公开报告不含权重，因此不能声称外部读者已能精确重放这三个冻结检查点；跨平台数值等价也未建立。
