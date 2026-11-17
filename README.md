# TrajectoryLab

**Controllable generative bidding: response forecasts, budget/CPA control and reproducible stress evaluation.**

[中文](README.zh-CN.md) · [Research position](docs/bidding-research-map.md) ·
[Bidding protocol](docs/bidding-protocol.md) · [Architecture](docs/architecture.md)

## The question

A bidder sees predicted conversion probabilities, spends a finite budget and
receives noisy conversion feedback. When those predictions drift, does a learned
response model help it recover a better conversion–CPA trade-off than a tuned
fixed multiplier or a simple feedback controller?

This project studies that question with public AuctionNet traffic and auction
code. A compact causal model predicts action-conditioned cost/conversion curves;
a separate numerical controller chooses the bid. Keeping those pieces separate
allows matched-checkpoint experiments on forecast correction and control rules.
There is currently no LLM planner in the implemented system.

## Evidence so far

| Study | Status and evidence |
| --- | --- |
| [Bidding feasibility study](reports/bidding-pilot/report.html) | Completed: 480 campaign-level evaluations, one training seed. It did **not** establish a learned-policy advantage; fixed bidding was competitive and hard predicted-CPA constraints suppressed spend. |
| [Formal bidding study](reports/bidding-main/report.html) | Complete: 16 / 6 / 12 training / validation / test markets, three training seeds and 6,480 rows. History correction did not establish a score gain; response-model performance depended on the stress scenario. |
| [Controller refinement](reports/bidding-control/validation-report/report.html) | Validation complete: 11,340 candidate evaluations. The new controller failed the predeclared promotion gate; no new-market confirmation was conducted. |
| [Matched forecast audit](reports/bidding-forecast/report.html) | Complete: neural, contextual ridge and empirical forecasts on identical validation trajectories. Lower prediction error does not guarantee accurate conversion totals or better bidding. |
| [Offline policy-learning foundation](docs/results.md) | Completed: 39 BC/IQL/DT training runs across Pendulum, a public PointMaze subset and an IQL mechanism ablation. |

Campaigns within a market and repeated methods/seeds are correlated. Row counts
are not independent-market counts. The [findings](docs/bidding-findings.md) retain
negative results, compute measurements and unresolved mechanisms.

**A useful trade-off:** under 50% pCVR overprediction, response scored 18.3%
higher than the feedback baseline, with 7.5% more conversions and an 18.1
percentage-point lower CPA violation rate; all three seed score differences
were positive. This comparison is exploratory: its nominal-scenario score was
8.1% lower, so it does not establish an overall advantage. In the primary
comparison, history correction's score difference was −0.577, with a
market-clustered 95% interval of [−1.275, 0.095].
[Full statistics and failure boundaries](docs/bidding-findings.md).

![Exploratory score and CPA violation differences across all five stress scenarios](reports/bidding-analysis/bidding-tradeoffs.svg)

## What to inspect

| Part | Implementation and purpose |
| --- | --- |
| Exact auction response cache | [Market adapter](src/trajectorylab/bidding/market.py): public-kernel outcomes on a declared action grid, reused across policies with common random outcomes |
| Learned policies | [Training and rollout](src/trajectorylab/bidding/learning.py): a causal DT and a monotone response model; normalization fitted on training markets only |
| Forecast versus controller | [Paced control study](src/trajectorylab/bidding_control.py): same neural checkpoints, alternative controllers, empirical response curves and full simple-baseline grids |
| Feedback boundary | Current observations and past realized outcomes only; budget-censored outcomes are excluded from unthrottled forecast-ratio updates |
| Evidence and recovery | Source/configuration fingerprints, optimizer/RNG checkpoints, market-paired comparisons, raw CSV/JSON and text-only publication exports |

The response model receives dense simulated intervention labels; DT learns from
behavior trajectories. Their comparison does not isolate architecture alone.
The [protocol](docs/bidding-protocol.md) specifies the information available to
each method, the primary comparison and the exploratory analyses.

## Run locally

Start from the repository root. The reference environment is Python 3.12 on an
Apple M1; bidding configurations request MPS. A different device must be declared
in a separate local configuration and produces a different experiment identity.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m trajectorylab.bidding --config configs/bidding-pilot.toml
```

This performs real data generation, training and evaluation. The pilot is for
feasibility, not a performance claim. Its output directory is printed in the log.
For full studies, staged execution, checkpoint recovery and battery-aware
background operation, see the [bidding runbook](docs/bidding-runbook.md).

To inspect the committed reports without installing ML dependencies:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory reports
# Open http://127.0.0.1:8765
```

Reports need no account, API key or live backend. Public exports omit model
weights, traffic arrays, local logs and runner state. GitHub displays HTML source;
serve reports locally to view them.

## Earlier policy-learning studies

These are supporting experiments in trajectory selection and implementation
correctness, rather than evidence of advertising performance.

- **Pendulum:** high-return selection improved BC/IQL in all three paired seeds.
  DT's mean difference was +34.55 with a paired-difference SD of 67.85 and only
  2/3 positive seeds; this does not establish a stable DT gain.
- **PointMaze:** DT had higher mean return, while all methods reached a saturated
  100% “ever reached the goal” metric.
- **IQL ablation:** zero advantage temperature recovers the matched BC actor
  under the declared implementation and seeds.

[Results and limitations](docs/results.md) ·
[Reproduce these studies](docs/offline-policy-guide.md) ·
[Offline-learning protocol](docs/protocol.md)

## Scope and attribution

This is a custom fixed-opponent simulation derived from AuctionNet, with a custom
hard-budget settlement rule. It is not an official leaderboard reproduction;
opponents do not adapt or exhaust budgets. Simulator scores do not establish
production revenue or causal business impact. No employer code, data, internal
integrations or online metrics are included.

## References and attribution

- [AuctionNet source and retained license](third_party/auctionnet/README.md).
- [GRM: Constrained Auto-Bidding via Generative Response Modeling](https://arxiv.org/abs/2605.27811), the response-model research reference. This compact implementation does not reproduce the full paper.
- [AIGB-R1](https://arxiv.org/abs/2607.17281), context for planner/executor bidding agents. Its LLM planner, PromptDT and D-GRPO are not implemented here.
- [IQL](https://arxiv.org/abs/2110.06169), [Decision Transformer](https://arxiv.org/abs/2106.01345), [Gymnasium Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/) and the [Minari PointMaze dataset](https://minari.farama.org/main/datasets/D4RL/pointmaze/umaze-v2/).

The algorithms are public methods; the project contributes an independent
implementation and inspectable experiments. Development used AI-assisted coding.
Code uses [MIT](LICENSE); the AuctionNet subset retains its separate
[Apache-2.0 terms](third_party/auctionnet/README.md). Upstream datasets retain
their own terms and are not redistributed.
