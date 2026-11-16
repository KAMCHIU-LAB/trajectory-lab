# Measured results and limitations

This release contains **39 completed training runs**, with 10,000 updates each:
27 controlled data-selection runs, nine public-subset runs, and three mechanism
checks. These are measured simulator results from this implementation.

## Reading the numbers

- Each table cell is the mean ± sample standard deviation of **three training-seed
  means**. Each seed mean contains eight final simulator episodes. Higher raw
  return is better; scores are not D4RL-normalized.
- Training seeds are 0, 1 and 2. Validation environment seeds are 10000–10002;
  final environment seeds are 20000–20007. The final checkpoint is evaluated
  without selecting it by test performance.
- Batch size is 128, network width 64. DT uses context length 15, two layers and
  four heads. All methods use the same number of optimizer updates, which does
  not equalize tokens, FLOPs, training time or hyperparameter tuning.
- The local reference is Apple M1 / 16 GB, Python 3.12, CPU, one thread per
  learner. Exact installed versions are in [requirements.lock](../requirements.lock).
  The source, configuration and runtime identifiers are exported with each study.

## 1. Controlled trajectory selection

Pendulum-v1 source data contains 300 generated episodes / 60,000 transitions from
a documented controller, noisy-control and random mixture. Content-grouped
80/10/10 splitting precedes data selection. Each view contains 60 complete
training episodes / **12,000 transitions**.

| Training view | BC | IQL | Compact DT |
| --- | ---: | ---: | ---: |
| Uniform | -500.54 ± 49.49 | -338.98 ± 15.52 | -179.54 ± 63.31 |
| Top return | -131.85 ± 11.09 | -131.25 ± 5.75 | -144.99 ± 8.42 |
| Return stratified | -619.17 ± 103.72 | -347.90 ± 23.37 | -261.63 ± 57.39 |

The non-learned controller reference returns -128.32; random actions return
-1201.84 on the same final environment seeds. The controller is a reference,
not a theoretical optimum.

**Paired comparison:** subtract the uniform return from the top-return return
for each matching training-seed ID. The table reports the mean and sample
standard deviation of those three differences; it does not report a confidence
interval or significance test.

| Method | Paired mean difference ± across-seed SD | Positive seeds |
| --- | ---: | ---: |
| BC | +368.69 ± 55.09 | 3/3 |
| IQL | +207.73 ± 15.05 | 3/3 |
| Compact DT | +34.55 ± 67.85 | 2/3 |

**Observed:** top-return means are higher for all three methods in this split,
but the seed-level pattern differs: BC and IQL are positive on each paired seed,
while DT's mean difference is smaller than its across-seed SD. That is a reason
to add seeds before describing the DT effect as stable. DT has the highest mean
on the uniform view, with substantial between-seed variation. Return-stratified
sampling provides no improvement over uniform here.

**Interpretation:** the useful result is the interaction between method and
training data. A view that helps cloning can reduce the apparent advantage of
more complex learners. This task does not establish that filtering always helps:
the source includes a competent controller, episodes have equal length, views
overlap, and only one source-data seed and split are used.

[Full report](../reports/pendulum/index.html) ·
[Aggregates and individual test returns](../reports/pendulum/summary.json) ·
[Run table](../reports/pendulum/runs.csv)

## 2. Public PointMaze subset

The source is Minari `D4RL/pointmaze/umaze-v2`. A seeded uniform subset of 600
source episodes is split before training-view construction. The uniform training
view contains **20,000 transitions / 279 complete episodes**. No company data or
private benchmark is involved.

| Method | Raw return | Ever reached goal |
| --- | ---: | ---: |
| BC | 79.04 ± 10.35 | 100% |
| IQL | 80.71 ± 6.77 | 100% |
| Compact DT | 201.62 ± 17.66 | 100% |

Random actions return 3.25. DT obtains the highest mean under this subset and
budget; BC and IQL are close. Success is an OR over the episode's native success
flags: reaching the goal once counts. It therefore saturates and cannot explain
the return differences. The continuing, sparse-reward evaluation task rewards
time spent at the goal, making total return informative beyond first arrival.

**A protocol issue worth investigating:** all selected source episodes have total
return 1. The recovered evaluation environment has a fixed goal and 300-step
continuing horizon, whereas source trajectories stop at goal-transition
boundaries. DT's declared training-return quantile consequently starts at 1;
after additional goal rewards its remaining target becomes negative, outside
the source return-to-go support. This is an identifiable conditioning and horizon
shift. Its effect on performance has **not** been isolated by an RTG ablation;
it is not evidence for a causal explanation of the ranking.

The report's support distances are descriptive. For example, a strong return
does not require the smallest nearest-neighbor distance. State scaling,
goal representation and reference subsampling influence this statistic.

[Full report](../reports/pointmaze/index.html) ·
[Aggregates and individual test returns](../reports/pointmaze/summary.json) ·
[Run table](../reports/pointmaze/runs.csv)

## 3. IQL actor-weighting mechanism check

With `advantage_beta = 0`, matched IQL and BC runs on the uniform Pendulum view
produce **bitwise-identical actor parameters in all three training seeds** and
exactly identical final test returns. Their aggregate is -500.54 ± 49.49.

The equality follows from the shared actor, sampler and likelihood loss when
advantage weights are one. This checks an implementation invariant; it is not
an algorithmic discovery. The verifier checks the actor-relevant settings and
compares the saved checkpoint tensors, rather than relying on rounded scores.

[Ablation report](../reports/ablation/index.html) ·
[Per-seed equality checks](../reports/ablation/mechanism-check.json)

## How these results transfer to bidding research

The findings describe simulated control tasks. A bidding study can reuse the
experimental questions: an offline dataset also records trajectories generated by
a behavior policy, and return, coverage and decision horizon all shape what a
learned policy can support. The following hypotheses need a separately governed,
public or authorized dataset before they can support conclusions about bidding.

| Observation here | Question for a bidding study | Evidence needed before a product claim |
| --- | --- | --- |
| On Pendulum, top-return minus uniform is positive on 3/3 BC and IQL seeds and 2/3 DT seeds; return stratification does not beat uniform | Does selecting high-value logged trajectories help the aggregate while weakening coverage for low-frequency traffic or budget regimes? | A predeclared cohort/traffic split, fixed training budget, independent holdouts, and both aggregate and per-cohort outcomes |
| PointMaze training fragments have return 1, while evaluation has a longer continuing horizon | Does the learned return target describe the same horizon over which bids spend budget and conversions mature? | Explicit episode/truncation semantics, a horizon-matched target, delayed-outcome treatment, and a new test-seed set |
| Support distance changes with state representation and reference sampling | Are policy recommendations concentrated in market states that are rare in the logs? | Domain-defined features, policy-to-log coverage diagnostics, hard budget/auction constraints and a separate safety evaluation |

The transferable capability is experiment design and interpretation. The measured
returns describe Pendulum and PointMaze. A production bidding study additionally
needs data authorization, counterfactual assumptions, business-specific constraints
and independent evaluation.

## Compute and reproducibility

The sum of the run-recorded wall times is approximately **133.4 minutes** for the
Pendulum study, **49.7 minutes** for PointMaze, and **0.9 minutes** for the ablation.
DT dominates the compute. These are sums across learners, not elapsed project
time: downloads, installation and time before recovered checkpoints are not
included. Some runs shared the machine with other learners, and resumed timing
has checkpoint granularity. These figures are not isolated throughput benchmarks.

The evidence includes data/configuration fingerprints, training histories,
per-episode returns, update counts and execution-driver records. The four plot
types cover returns, validation learning curves, training-state projections and
actual simulator trajectories. Example trajectories always use the first declared
training seed and final environment seed; they are not selected by return.

Reproduction commands and CPU recovery instructions are in the [README](../README.md).
Dataset downloads and model checkpoints are excluded from Git; locally completed
runs can be reused or resumed. Hosted CI is configured but has not been run as
part of these local measurements.

## What would strengthen the evidence next?

1. Isolate the PointMaze return-conditioning question with declared RTG targets,
   then add training seeds without tuning on the final test seeds.
2. Add another task and an additional strong baseline before making broader
   method comparisons. Repeat data-generation and split seeds to measure data
   uncertainty, which these error bars do not contain.
3. Measure a separate compute-controlled comparison and independently held-out
   environment variations before claiming efficiency or generalization.

Three training seeds, two small tasks, one public-data subset and no systematic
hyperparameter search limit the conclusions. This release demonstrates an
auditable local experimental workflow; it provides no evidence of industrial
revenue, deployment robustness or state-of-the-art performance.
