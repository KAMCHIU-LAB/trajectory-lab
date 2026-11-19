# Conditional generative response: completed findings

**截至 2026-11-18：响应模型的局部确认收益成立；Flow 在低估场景出现开发收益，但跨种子业务门槛及生成式增量均未闭环。**

This page records a completed, stopped development study. It is separate from the [24-market fixed-response confirmation](bidding-findings.md). Results from later or ongoing research are not included.

## Question and implementation

Can a conditional distribution over action-dependent costs and conversions improve budget/CPA decisions beyond a point response model or a controller using distribution means?

The [Flow Matching sampler](../src/trajectorylab/bidding_generative_learning.py) generates joint response bundles conditioned on observed history. It generates possible responses, not bid actions. The [relative controller](../src/trajectorylab/bidding_generative_relative.py) compares candidate actions with feedback, projects forecasts to the remaining budget, screens paired score-improvement quantiles, and falls back to feedback. Model quantiles are not statistical confidence bounds or safety guarantees. All samples are integrated; the policy cannot choose a favorable realized future.

Comparators include a point predictor, a mean-response/plugin controller, disabled quantile screening, and shuffled cross-action samples. Controller parameters are fixed before the training-seed replication. The code and [configuration](../configs/bidding-generative-replication.toml) retain the original stop rules.

## Fixed-policy seed replication

- Reuses six development markets (2001–2006), six focal advertisers, and five scenarios; discovery seed 11 is included. These are correlated evaluations, not new independent evidence.
- Seeds 11 and 23 completed the 15,000-step training budget. The registered third seed, 37, was **not run** after seed 23 failed the stop gate. This is not a completed three-seed success.
- Business gate: each primary scenario must have score ratio ≥1.03, conversion ratio ≥0.97, and violation-rate increase ≤3 percentage points; nominal score ratio must be ≥0.98.
- Flow controller: radius 1, quantile 0.5; plugin radius 0.25; point radius 1. Sampling: 64 samples, 8 integration steps, fixed sampling seed 91347. Controllers are selected by their registered development procedures, so attribution is a practical comparator result rather than an architecture-only intervention.

### Relative to the frozen feedback controller

Each row contains 36 advertiser-period observations per method and training seed. Percentages are scenario mean ratios; violation differences are percentage points.

| Scenario | Seed | Score change | Conversion change | CPA violation change |
| --- | ---: | ---: | ---: | ---: |
| calibrated | 11 | +8.07% | +10.53% | +0.00 pp |
| calibrated | 23 | +4.67% | +9.92% | +2.78 pp |
| over_25 | 11 | +6.99% | +12.68% | -2.78 pp |
| over_25 | 23 | +0.82% | +4.62% | +0.00 pp |
| over_50 | 11 | +12.52% | +12.69% | +0.00 pp |
| over_50 | 23 | +11.68% | +7.56% | -2.78 pp |
| under_25 | 11 | +5.40% | +9.23% | +2.78 pp |
| under_25 | 23 | +4.70% | +8.08% | -2.78 pp |
| mid_over_50 | 11 | +3.23% | +12.40% | +2.78 pp |
| mid_over_50 | 23 | -0.30% | +3.43% | +2.78 pp |

Seed 11 passed the business gate; seed 23 did not. Its over_25 score gain was +0.82%, **2.18 percentage points short of the +3% requirement**. Positive under_25 means (+5.40% / +4.70%) are development signals, not a confirmed solution to underprediction.

### Distribution fitting is not decision value

Both completed seeds passed the registered distribution diagnostic, but empirical 80% coverage was about 67–69% and 95% coverage about 87–88%; this does not establish perfect calibration.

Against point prediction, the arithmetic mean of the three primary scenario score ratios was **−1.32% / −0.97%**, below the +1% gate. Against plugin, over_50 score was **−4.58% / −5.48%**, below the −2% per-scenario floor. Shuffling and disabling quantile screening did not establish a uniform decision advantage. **Generative value remains unconfirmed.**

## Zero-fit mechanism audits

### Forecast target versus feedback continuation

The audit reused 540 development prefixes (six markets × six advertisers × five scenes × three time points). The two frozen model seeds share the same contexts and random numbers. No fresh market or fitted model was added.

| Layer | Quantity |
| --- | --- |
| M | Model forecast for a constant action, followed by proportional budget projection |
| P | True constant-action response under the same projection |
| C | Constant action with exact hard-budget settlement |
| F | One action intervention, then the fixed feedback policy for the remainder |

Each layer compares its own action-minus-feedback improvement. At P, generated responses did not consistently outperform point predictions across the registered scenario, market-direction and seed gates. C and F also failed to establish the required joint advantage. Thus these diagnostics do not support attributing the failure entirely to the future constant-action assumption. M−P includes conditional unpredictability; C−F changes future actions and is not an isolated causal attribution.

### Immediate impact and later compensation

The completed decomposition reused 1,440 train and 540 development contexts; model-seed duplication does not increase that count. In the **4,107 non-feedback development action × context cells**, terminal score improvement was positive in 1,188, zero in 1,391 and negative in 1,528.

Of the 1,391 zeros, **895 (64.3%)** came from identical immediate settlement; **496 (35.7%)** came from equal positive integer conversions with both outcomes on the CPA-feasible score plateau. No zero arose from different immediate outcomes followed by exactly identical terminal cost and conversions. This does not exclude partial compensation in other cells. Cost and integer conversions decompose additively; score does not.

Checks reconciled all 1,980 contexts: maximum cost-decomposition residual 8.49e−13, zero integer-conversion residual, maximum score-recomputation residual 2.14e−14, and zero violation-label mismatches. These are existing completed audit outputs, not a newly executed experiment in this publication.

Hindsight oracle headroom is diagnostic only: it depends on future outcomes unavailable to a deployed policy. It proves neither learnability nor repeated-replanning gains. The audits themselves authorize no new fitting.

## Evidence and reproduction limits

- [Exact stopped replication summary](../reports/bidding-generative/summary.json): preserves `business_replicated=false`, `generative_value_confirmed=false`, `complete=false`, and the third-seed stop.
- [Registered horizon interpretation](../reports/bidding-generative/horizon-interpretation.json).
- [Compact compensation record](../reports/bidding-generative/feedback-compensation.json): overall train/development groups and reconciliation checks, extracted from the completed summary.
- [Publication manifest](../reports/bidding-generative/publication-manifest.json): source/configuration identity, original evidence hashes and exported-file hashes.

The replication source at [research revision 43fc0ac](https://github.com/KAMCHIU-LAB/trajectory-lab/tree/43fc0acf61e4ad13b01c95492c45e94ce711b384) matches all 17 recorded source hashes. The v0.2.0 delivery pass formats source files; the historical publication manifest continues to identify the original bytes. Checkpoints, raw trajectory/traffic arrays, detailed decision traces and local runner state are excluded. The aggregates and implementation can be inspected publicly; the original frozen runs cannot be exactly replayed from this repository alone. Earlier [runbook](bidding-runbook.md) instructions describe the underlying simulation pipeline. No new training, evaluation, tests or market generation was performed to prepare the original research export; later release software checks are recorded separately.

This is a custom AuctionNet-derived fixed-opponent simulator with custom hard-budget settlement. It is not an official leaderboard result, employer system, or evidence of online business gains.
