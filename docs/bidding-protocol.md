# Generative bidding under prediction error

Status: implementation and feasibility experiments. No advertising result has yet
been established. This study uses public simulator code and an independent
experimental implementation; it includes no employer material.

## Question and scope

When the conversion probability shown to a bidder is miscalibrated, does a
history-conditioned response forecast with feedback correction improve the
trade-off between conversions and CPA violations?

Compare fixed multipliers (including validation tuning), a numerical feedback controller, a compact Decision Transformer (DT),
DT with the same numerical correction, and a response model with/without
correction. Reuse checkpoints for controller ablations. Language-model planning
is a later experiment, conditional on establishing this numerical core.

## Public environment and deliberate differences

Source: `alimama-tech/AuctionNet`, revision
`c20de631a40e24669d1d6a8e2acefd8f492f172d` (Apache-2.0).
Use its NeurIPS traffic generator and three-slot auction kernel with 48
advertisers and 48 ticks. Reserve price is 0.0001, matching upstream test.gin.
The vendored auction file removes gin's import and decorator and adds a
modification/license notice; constructor arguments are supplied explicitly.
The generator is unmodified.

This is a **custom fixed-opponent study derived from AuctionNet**, not the
official trained-opponent benchmark or a reproduction of published scores.
Opponents use seeded, heterogeneous bid multipliers and do not adapt to the
studied agent. Their budgets are not enforced. Each focal advertiser is evaluated
separately against that market. These assumptions limit strategic conclusions.

The focal bidder has a hard budget. Inside a tick, exposed opportunities are
settled in PV order until its remaining budget cannot pay the next opportunity.
This differs from upstream's iterative random bid dropping. Exact cumulative
costs and realized conversion counts implement this settlement, with no
fractional or interpolated conversions.

Candidate multipliers form a declared finite action set. Counterfactual response
tapes are generated with the public auction kernel for every effective action
used by the declared stress scenarios. Lookup is exact; there is no interpolation
or clipping an unseen action to a cached endpoint. The evaluator keeps these
counterfactuals private from policies. Reusing conversion samples across actions
is a common-random-number variance reduction device, not independent evidence.

Caching the upstream latent value and potential-conversion matrices within a
tick is exact: upstream generates them independently of bids from fixed seeds.
Allocation, payment, exposure and conversion masking remain in the upstream
kernel. A runtime equivalence check precedes data generation.

## Information boundary

At tick t, a policy receives its current observed pCVR summary, opportunity count,
budget, target CPA, and past actions/outcomes. It receives no future traffic,
future price, latent unbiased pCVR, scenario ID, or counterfactual response tape.
For stress, multiply only its observed pCVR; leave true conversion generation and
opponents unchanged. Training uses calibrated observations. Validation scenarios
choose controller settings; held-out market IDs provide the final evaluation.

Dense training targets are **realized** costs/conversions from simulator
interventions, not latent expected conversions. The response model forecasts the
current tick and the unthrottled remaining horizon at each candidate action.
Future quantities are labels only. This is a compact independent response-model
variant inspired by GRM, not a full GRM reproduction or a foundation world model.

The response controller maximizes predicted terminal CPA-penalized conversions
among budget-affordable constant-action forecasts. The response-hard ablation
instead screens predicted terminal CPA as a hard condition. Both reuse the same
checkpoint. This decision was made after a development-only pilot showed that
hard screening could collapse spending under sparse conversions. The main test
market IDs (3101–3112) are distinct from all viewed pilot test IDs (3001–3004).
Feedback correction adjusts historical cost/conversion forecast ratios for the
response model; DT uses a historical realized-CPA action adjustment. These are
different numerical corrections, with validation-selected shrinkage priors.
Ticks truncated by the hard-budget settlement are excluded from response-ratio
updates: an unthrottled response forecast must not be calibrated against a
budget-censored outcome. The executor exposes only the observed budget-stop
event, not the unobserved counterfactual cost or conversions.

DT uses complete behavior trajectories, return/state/action token order, and
whole-trajectory windows. Return targets and normalization use training data
only. Behavior includes fixed multipliers and feedback policies; returns
condition on a terminal conversion score with CPA penalty. The desired return is
decremented only by observed per-tick score increments.

## Evidence and acceptance

- Split by market ID before fitting; all focal agents in a market share a split.
- Persist configurations, source hashes, runtime, training histories, RNG and
  optimizer checkpoints. Do not mix evidence after source/configuration changes.
- First run a small feasibility configuration; it cannot support effectiveness
  claims because conversions are sparse. Then retain 500,000 requested PVs per
  period for the main configuration, subject to resource measurements.
- Tune numerical controller parameters on validation markets only. Never select
  a checkpoint or safety margin from test performance.
- Report conversions, CPA debt (cost - target * conversions), violation rate,
  zero-conversion frequency, budget utilization, score and compute. Define CPA
  as undefined for zero conversions; do not silently discard those campaigns.
- Aggregate training-seed differences and market-cluster uncertainty. Multiple
  focal agents/ticks from one market are not independent market samples.
- Primary comparison, fixed before main test evaluation: response+correction
  versus response under over_50, reporting paired score and CPA-violation
  differences jointly. Other scenarios, fixed/feedback baselines, and hard/soft
  decision ablations are exploratory; intervals have no multiplicity correction.
- A useful improvement must show its conversion cost and compete with tuned
  feedback control. Hard budget compliance is enforced by settlement and is not
  evidence that a learned model discovered safety.
- No production lift, universal safety, SOTA, or originality claim follows from
  this custom simulation. Publish negative findings and scope limitations.

## Execution and stop conditions

1. Establish source provenance, exact cached-kernel equivalence and a complete
   small data/train/evaluate/report run.
2. Check data size, realized conversion counts, forecast fit and runtime before
   increasing the budget. Stop invalid runs on nonfinite values or identity
   mismatch; preserve checkpoints for valid interruption recovery.
3. Run the main comparisons, then spend extra compute only on informative
   ablations and unresolved uncertainty. Avoid an unbounded parameter search.
4. Export portable reports and a claim ledger. Audit tracked artifacts for
   credentials, local paths and employer material before any publication.

References: [AuctionNet](https://github.com/alimama-tech/AuctionNet),
[GRM](https://arxiv.org/abs/2605.27811),
[Decision Transformer](https://arxiv.org/abs/2106.01345),
[AIGB-R1](https://arxiv.org/abs/2607.17281).
