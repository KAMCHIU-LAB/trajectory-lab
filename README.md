# TrajectoryLab

**Public, inspectable experiments in generative bidding and offline policy learning.**

[中文说明](README.zh-CN.md) · [Experimental protocol](docs/protocol.md) ·
[Results and limitations](docs/results.md) · [Architecture](docs/architecture.md)

## Generative bidding under prediction error — in progress

The current research track asks whether historical outcome feedback can improve
response-model bidding when observed pCVR is miscalibrated. It uses the public
AuctionNet traffic generator and auction kernel, independent compact models,
fixed/tuned multipliers, numerical feedback, DT, and matched-checkpoint
response/correction/hard-constraint ablations.

- [Question, information boundaries and primary comparison](docs/bidding-protocol.md)
- [Run and resume locally](docs/bidding-runbook.md)
- [Completed feasibility report](reports/bidding-pilot/report.html) · [Development findings](docs/bidding-findings.md)

The pilot completes real data generation, MPS training and 480 campaign-level
evaluations. It **does not establish an advantage for learned bidding**: a simple
fixed multiplier is strong, and hard forecast constraints can suppress spending.
The larger three-seed study uses new test markets and is still pending final
results. This is a custom fixed-opponent simulation, not the official AuctionNet
benchmark. No employer code, data or online metrics are used. The independent
response-model variant is inspired by GRM; it is not a full paper reproduction.

## Earlier offline policy studies

TrajectoryLab asks how trajectory selection changes offline policy learning under
a small, declared training budget. It independently implements **BC, IQL, and a
compact Decision Transformer baseline**, then exposes the data split, losses,
validation curves, final simulator returns, and descriptive support diagnostics.

The useful artifact is a traceable experiment, including results that fail to
support a hypothesis. This is an independent public-data project. No employer
code, datasets, integrations, or business metrics are included.

## What is here

| Component | What can be inspected |
| --- | --- |
| Controlled study | Pendulum-v1; uniform, high-return and return-stratified whole-trajectory views, each with 12,000 transitions |
| External check | A seeded subset of public Minari `D4RL/pointmaze/umaze-v2`; 20,000 training transitions, recovered evaluation environment |
| Baselines | Shared-actor BC/IQL; causal return/state/action transformer, with explicit implementation choices |
| Mechanism check | Zero-temperature IQL weighting recovers the BC actor under matched seeds and settings |
| Evidence | Three training seeds, fixed final checkpoints, raw per-episode returns, CSV/JSON, portable HTML reports |
| Engineering | Content-grouped episode splits, train-only normalization, atomic optimizer/RNG checkpoints, result-integrity gates, correctness tests |

This is **not** a reproduction of full-dataset published scores or a general
algorithm ranking. Equal updates do not mean equal FLOPs. Read the
[protocol](docs/protocol.md) before interpreting the results.

## What the results say, and what they do not

On Pendulum, paired by training-seed ID, top-return sampling raises BC and IQL
returns on all three seeds. For DT it raises the mean by 34.55, but the sample
standard deviation of the three paired differences is 67.85 and only two
differences are positive. Return-stratified sampling does not beat uniform in
this split. In the public PointMaze subset, DT has a higher mean return, but all
three methods hit the same saturated 100% “ever reached the goal” success metric.
The useful question is why these patterns differ across tasks; the available
runs do not support a universal algorithm ranking.

For generative bidding, the useful connection is the experiment design: test
whether selecting high-value logged trajectories reduces coverage of less common
traffic; align a return target with the bidding and budget horizon; and treat
offline support diagnostics as investigation signals, not launch guarantees.
The new public-simulator track investigates a narrower prediction-error question;
business conclusions still require independent evaluation. The measured scores
below describe the two earlier simulator tasks. This repository contains no
employer auction data and makes no production or revenue claim. See the
[results and next experiments](docs/results.md).

## Read the measured experiments

- [Pendulum data-selection study](reports/pendulum/index.html)
- [Public PointMaze subset](reports/pointmaze/index.html)
- [IQL weighting ablation](reports/ablation/index.html)
- [Findings, failure analysis and actual compute](docs/results.md)

GitHub displays the HTML source. Download/open a report locally, or serve the
included report portal:

```bash
python3 -m http.server 8765 --bind 127.0.0.1 --directory reports
# Open http://127.0.0.1:8765
```

Reports embed their plots and require no CDN, account, API key, or live backend.
CSV, JSON, manifests, and per-run histories are included alongside each report.
Upstream datasets and model checkpoints are excluded from Git.

## Quick start

Python 3.11+; the local reference uses Python 3.12 on an Apple M1, CPU / one thread
per learner. Start from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
trajectorylab doctor
make check
make smoke
make report
make serve
```

The smoke run performs **real training and evaluation**, with only 100 updates.
It validates integration; its scores are not evidence of algorithm quality.
Checkpoints let an interrupted invocation resume from its last durable update.
A repeated completed invocation reuses the same results.

`make report` and `make serve` use `reports/latest`, the most recently invoked
suite. To browse the three included formal studies together, use `make demo`
instead. `make portal` rebuilds their index from the exported study evidence.

## Reproduce the studies

```bash
make study       # 27 runs: 3 data views × 3 methods × 3 training seeds
make ablation    # 3 runs: IQL with advantage_beta = 0

python -m pip install -e '.[benchmark]'
make pointmaze   # 9 runs; first call downloads the public Minari dataset
```

All formal runs use 10,000 gradient updates, batch size 128, and eight final
environment seeds per training seed. The public dataset download is approximately
530 MiB; allow several GB for the environment, source data and checkpoints.
Transformer runs dominate CPU time. Actual measured run times and contention
are recorded in the reports; these studies are longer than the smoke test.

For a stopped, already initialized **CPU** suite, an optional recovery driver can
finish independent runs in parallel without changing their training settings:

```bash
# Stop the serial runner first. Replace the path with its printed suite directory.
python scripts/complete_suite.py --suite artifacts/suites/<suite-name-id> --workers 3
```

Do not run both drivers on the same suite concurrently. The recovery driver checks
source, implementation and runtime identity, and records its own source hash and
worker count in the evidence. Each learner still uses its configured thread count.
Parallel wall times include shared-machine contention.

`requirements.lock` records the full known-good local environment. Other Python
versions and operating systems may need platform-appropriate dependency resolution;
bitwise equivalence across versions or accelerators is not promised.

## Why these details matter

- **A trajectory is the split unit.** Content duplicates are grouped before
  splitting, and sequence windows cannot cross an episode boundary.
- **Termination is not truncation.** Only true termination removes the IQL
  Bellman bootstrap; a time limit retains it.
- **The DT action label is causally hidden.** State token t predicts action t
  before reading its action token. Perturbation tests check this boundary.
- **A seed is the aggregation unit.** Error bars describe training-seed means,
  rather than treating correlated evaluation episodes as independent training runs.
- **A report must be complete.** Missing jobs, duplicate identities, changed test
  seeds, and summaries inconsistent with raw returns fail the report gate.
- **Support distance is descriptive.** Common train-only scaling makes views
  comparable; nearest-neighbor distance is not a calibrated OOD or safety score.

See [architecture](docs/architecture.md), [protocol](docs/protocol.md),
[design decisions](docs/design-decisions.md),
[maintenance roadmap](docs/roadmap.md), and [contribution guidelines](CONTRIBUTING.md).
The GitHub Actions workflow is supplied; local checks do not imply hosted CI has run.

## References and attribution

- Kostrikov et al. [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169).
- Chen et al. [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).
- [Gymnasium Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/),
  [Minari PointMaze dataset](https://minari.farama.org/main/datasets/D4RL/pointmaze/umaze-v2/),
  [Gymnasium Robotics](https://robotics.farama.org/envs/maze/point_maze/).

BC, IQL and DT are public methods, not new algorithms contributed by this project.
The implementations here are compact independent variants; important differences
are documented in the protocol. Development used AI-assisted coding. Claims are
limited to the provided implementation and recorded experiments.

Code: [MIT](LICENSE). Public source datasets retain their upstream terms and are
downloaded separately rather than redistributed.
The vendored AuctionNet subset retains its separate
[Apache-2.0 license and attribution](third_party/auctionnet/README.md).
