# Reproduce the public software path

[中文说明](#中文速览) · [Research results](bidding-findings.md) · [Change log](../CHANGELOG.md)

Choose a route according to what you want to check:

| Route | Requirements | What it establishes |
| --- | --- | --- |
| Read the committed reports | A browser; Python's local HTTP server renders HTML | The published measurements and their declared scope are inspectable. |
| Run the delayed-feedback CPU replay | A source checkout, Python 3.11 or 3.12, core dependencies | Public market generation, training-history fitting, selection, settlement and frozen resume work end to end. |
| Recreate a research study | Its protocol, complete inputs and compute budget | A new experiment with its own identity. Historical frozen model bundles are not all published. |

The small replay contains **one** confirmation market. It cannot reproduce the
historical +17.6% result or establish a business gain. The delayed-feedback
research used 24 independent confirmation markets; these are different runs.

## Install from source

Run these commands in a terminal. No account, API key, employer service, model
download or production dataset is required. Installing Python packages requires
internet access; the replay itself uses the vendored public simulator locally.

```bash
git clone https://github.com/KAMCHIU-LAB/trajectory-lab.git
cd trajectory-lab
git checkout v0.2.0
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Python 3.11 can be substituted for 3.12. Windows users need an environment with
POSIX file locking, such as WSL2; native Windows is not validated. The public
bidding runners locate configurations and the vendored simulator relative to
the source tree, so use an editable source checkout or extracted source archive.
A standalone installed wheel is not the supported bidding reproduction route.

`requirements.lock` is a snapshot of the earlier full research environment,
including optional simulator packages. The small replay installs core
dependencies through `pyproject.toml`; it does not require MuJoCo or Minari.

## Run and inspect the CPU replay

```bash
python scripts/run_bidding_delayed_smoke.py --output artifacts/public-replay-v0.2.0
python scripts/verify_release.py --replay artifacts/public-replay-v0.2.0/smoke-evidence.json
```

The first command prints a JSON evidence record and saves it under the chosen
output directory. It generates synthetic markets locally, fits mature history
and evaluates the declared bias/delay grid. There is no neural-model training in
this small path. Reference timing and package versions are recorded in
[`releases/v0.2.0.json`](../releases/v0.2.0.json); installation time is separate.
Hardware and package versions can change runtime and numeric outcomes.

The checked contract is:

| Evidence field | Expected value | Meaning |
| --- | --- | --- |
| `scope` | `software-reproducibility-only` | No business-effect claim |
| `business_target_pass` | `false` | The independent-market requirement remains enforced |
| `independent_confirmation_markets` | `1` | One software probe, not 24 independent confirmations |
| `validation_rows` | `90` | Complete development grid after deduplicating identical gain/prior candidates |
| `confirmation_rows` | `45` | Five frozen methods × three biases × three delay regimes |
| `all_final_arrivals_mature` | `true` | Final settlement includes every accepted conversion report |
| `all_hard_budgets_respected` | `true` | No campaign spends beyond its hard budget |

The oracle-past method is a nondeployable diagnostic. Row counts are correlated
campaign records, not independent samples. The validation configuration includes repeated prior values, which collapse to
one candidate per gain. Its 90 rows are five methods × two unique gain/prior
candidates × three biases × three delay regimes, not 90 independent observations.

Run the first command again with the same output directory to exercise resume.
Source, configuration, market, history and selection identities must match. A
changed source revision intentionally refuses an old output; use a **new output
directory** for a new experiment. Do not edit or delete identity locks to force
resume. Elapsed time is observational and is not an exact reproducibility target.

## Check source identity and correctness

```bash
python scripts/verify_release.py
python -m ruff check src tests scripts
python -m ruff format --check src tests scripts
python -m pytest -q
python -m trajectorylab run --config configs/smoke.toml
```

`verify_release.py` checks the file hashes recorded for v0.2.0. It works with the
standard library and fails if a recorded file changes or disappears. An optional
`--replay` additionally checks the invariant fields in a locally generated
evidence record. This verifies the software contract, not statistical efficacy
or the truth of an arbitrary JSON file. The tests and actual replay provide the
execution evidence; hosted CI reports its own run status.

The end-to-end replay test in routine CI uses `--replay-only` to check the declared replay fields on newer
commits. Source hashes are checked when publishing the release and when running
the default command on its tag; they are not rewritten to make later code pass.

The last command is the earlier offline-policy smoke suite. Its output is
printed on completion and is separate from the bidding replay. CI runs both
paths on CPU: pytest executes the actual delayed-feedback CLI and its frozen
resume, followed by the earlier offline-policy command. Public dataset downloads
and expensive studies are excluded.

## Browse results without ML dependencies

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory reports
# Open http://127.0.0.1:8765
```

Start with the report index, then use these deeper references:

- [Confirmed response result and all stress scenarios](bidding-findings.md).
- [Flow implementation, stopped seed replication and failed value gates](bidding-generative-findings.md).
- [Delayed-feedback results](bidding-delayed-findings.md) and [full staged reproduction](bidding-delayed-protocol.md).
- [Twelve pending-use strategies and the failed promotion gate](bidding-delayed-strategy-findings.md).
- [Fourteen research stages](bidding-research-journal.md), linking each question to evidence and a decision.
- [Public-source provenance and omitted inputs](code-and-data-provenance.md).

## Troubleshooting

- **`No module named trajectorylab`:** activate the environment and run
  `python -m pip install -e '.[dev]'` from the checkout root.
- **Frozen source/history/selection changed:** keep that output as historical
  evidence and choose a fresh `--output` directory for the current revision.
- **Release hash mismatch:** check that you are on the intended tag. A later
  code revision has a different identity; do not overwrite the release manifest.
- **HTML appears as source on GitHub:** use the local HTTP server above.
- **A full historical study requests missing checkpoints or authority files:**
  consult its reproduction limits. The small replay deliberately avoids those
  omitted historical inputs; disabling a preflight check is not reproduction.

## 中文速览

先在源码目录创建 Python 3.11 / 3.12 环境并安装，再运行上面的两条 CPU 回放命令。
检查输出是否为 90 条验证记录、45 条确认记录、1 个独立确认市场，成熟结算和预算
检查均为 `true`，业务门槛仍为 `false`。各条记录存在相关性，不能按行数放大样本量。

这条路线验证公开代码能完成生成数据、拟合成熟历史、选择参数、确认与恢复的流程。
历史收益、Flow 增量与线上效果各有独立证据要求；版本记录保留了它们的适用边界。
