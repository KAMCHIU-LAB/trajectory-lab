# Checkpoint-matched bidding control study

Status: specified before reading the first formal study's held-out outcomes.
The implementation is `trajectorylab.bidding_control`; its declared settings
are in `configs/bidding-control.toml`. This is a second, conditional experiment,
not a change to the frozen first study or its primary comparison.

## Question and proposed contribution

Does a history-conditioned response forecast become more useful when the bidder
prices budget consumption one tick at a time, rather than requiring a constant
action's whole remaining forecast to fit the remaining budget? Does forgetting
old forecast errors help after a change in observed pCVR?

The contribution being investigated is a diagnosis of the interaction between
forecasting, control and feedback. The project does not claim a new primal-dual
algorithm, a full GRM reproduction, or a language-model bidding agent. A response
model supplies forecasts; a numerical controller selects bids. The experiment
must establish the value of this core before adding an LLM layer.

The public simulator, exact response cache, hard-budget executor and information
boundary are the same as in the [first protocol](bidding-protocol.md). No employer
data, code or metrics are used. Fixed opponents and the custom settlement rule
still limit the result's applicability.

## What changes and what stays matched

Reuse the first study's three response checkpoints, training markets, six
validation markets, action set and five stress scenarios. Do not retrain a
network, change the first study's code, or select settings using its test scores.
The new experiment records its own implementation hash, checkpoint hashes, first
study identity and a hash/snapshot of the first study's baseline selection.

At a tick, let `c(a)` be predicted cost divided by initial budget, and `q(a)` be
predicted conversions times target CPA divided by initial budget. Choose the
lowest-index maximizer of:

```
(1 + quality_price) * q(a) - (budget_price + quality_price) * c(a)
```

Both nonnegative prices start at zero. After actual settlement, update:

```
allowance = remaining_budget_fraction_before_tick / remaining_ticks
budget_price  = max(0, budget_price  + eta_budget  * (actual_c - allowance))
quality_price = max(0, quality_price + eta_quality * (actual_c - actual_q))
```

Only current and past observations are available. The linear conversion objective
is a surrogate, not an exact optimizer of the nonlinear CPA-penalized evaluation
score. Prices may react too late to an early overspend; the settlement mechanism
still enforces the hard budget. Zero action remains available. Hard-budget
compliance is not an algorithmic improvement attributed to the model.

For cumulative or discounted correction, maintain realized and uncorrected
predicted cost/conversion sums. Form separate shrinkage ratios, clipped to
`[0.25, 4]`, using a fixed prior of three conversions and their target-CPA cost.
Discounted sums have an eight-tick half-life. Decay occurs once per observed
tick, including a budget-censored tick. Censored feedback updates actual resource
prices but does not update unthrottled forecast ratios. No scenario ID, future
labels or counterfactual unchosen-action feedback reaches a policy.

## Bounded validation selection

1. Evaluate four declared `(eta_budget, eta_quality)` pairs:
   `(10,10)`, `(10,50)`, `(50,10)`, `(50,50)`. Select the raw learned controller's
   best pair by mean validation score, with equal weight across scenarios and
   advertisers and averaged training seeds.
2. At that pair, evaluate cumulative correction, discounted correction, and a
   budget-only ablation with `eta_quality=0`. Select one of these or the raw
   controller by the same validation score. Ties follow declared candidate order.
3. Fit a pooled empirical response curve using **training markets only**: sum
   cost/budget and conversions*CPA/budget per observed opportunity and nominal
   action. At evaluation, scale by current opportunity count and the campaign's
   budget/CPA. This deliberately simple model does not use the neural model's
   full history or pCVR covariates. It is not a fully matched model class.
4. Give that empirical forecast all four price pairs and all four modes above.
   This is a stronger search allowance than the neural controller receives,
   since these deterministic evaluations are cheap. Retain the empirical
   variant matching the selected neural controller's pair and correction mode
   for a separate forecast-value diagnostic.
5. Also evaluate every nominal fixed multiplier and the first study's full
   declared feedback grid (three gains × three shrinkage priors). Preserve the
   first study's pooled-tuned settings as named aliases and omit matching grid
   duplicates. Select a pooled simple reference by pooled validation score and
   a primary simple reference by validation `over_50` score across these and all
   empirical controllers. Both references are frozen before confirmation. This
   prevents a scenario-specific result from benefiting from an under-tuned simple
   comparator. The baseline search uses no further neural model fits.
6. Preserve the original response and response+correction policies. Their
   settings remain those selected in the first study; they are controller
   ablations, not additional checkpoint searches.

Every evaluated candidate appears in the validation report. Validation estimates
are selection-biased and cannot establish effectiveness.

## Promotion and independent confirmation

A selected neural candidate advances only if, in **both pooled validation and
the primary over_50 scenario**:

- Its score is at least 10% above both the corresponding strongest simple
  reference and the original response controller.
- It retains at least 95% of that simple reference's conversions.
- Its CPA violation rate rises by no more than five percentage points.
- The reference has positive score and conversions, so these ratios are defined.

These thresholds are project-specific resource-allocation rules, not established
business acceptance thresholds. If a candidate fails, publish the validation
failure and skip the expensive confirmation data generation. Do not relax the
gate to manufacture a positive result.

If promoted, generate 24 new market IDs, **3201–3224**, disjoint from training,
validation, viewed development evaluation and first-study test markets. Freeze
candidate, references and settings before generating or inspecting their outcomes.
Evaluate the selected policy, references, matched empirical forecast and declared
controller ablations on all five scenarios. No confirmation result is used to
retune this experiment.

The primary inference compares the selected neural candidate against the
validation-selected primary simple reference in `over_50`. Average advertiser
and training-seed results within each market, then bootstrap the 24 paired
market vectors 10,000 times with a fixed random seed. Record raw score difference,
conversion ratio and CPA-violation difference jointly.

The declared confirmation decision requires:

- The score difference's 95% bootstrap lower bound is above zero.
- The conversion ratio's lower bound is at least 0.95.
- The violation difference's upper bound is at most 0.05.
- All three training seeds have a positive mean score difference.

An undefined conversion ratio makes the decision inconclusive. These intervals
are conditional on the three checkpoints and one training-data split; they do
not quantify full training-seed or simulator-distribution uncertainty. Three
positive seeds are a consistency check, not a high-powered seed-level test.
Other scenarios and ablations are exploratory and have no multiplicity correction.
The matched empirical comparison establishes at most a difference from this
pooled forecast, not superiority to all non-neural response models.

## Compute, recovery and publication

This stage adds no gradient training. Validation policy evaluation comes first;
24 further 500,000-requested-PV markets are generated only after promotion.
Report measured wall time, rather than promising a completion time from nominal
PV count. Cache complete market evaluations atomically; reject a different
source/configuration identity in the same output directory.

The waiting supervisor runs this stage after the first study completes, uses a
process-bound wake assertion during computation, and pauses below 21% battery
when disconnected from AC power. It changes no system sleep preferences.
Text-only exports contain manifests, raw metrics, all candidate results and the
decision; exclude checkpoints, traffic, local paths, runner state and machine
logs. A privacy scan is necessary but cannot prove every possible disclosure
absent. Publication needs a final human-readable claim review.

Useful outcomes include a reproducible advantage, a controller bottleneck that
survives matched ablations, or evidence that a simpler feedback method suffices
in this simulator. A negative result is retained; it does not become a resume
claim of measured lift.
