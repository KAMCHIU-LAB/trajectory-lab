# Delayed-feedback control extension: registered decision rules

## Why a separate extension

The first delayed-feedback confirmation is complete and sealed. Its posterior
pending correction improved aggregate score by 1.21%, below the registered 3%
practical target. The overprediction/long-delay cell declined by 4.25%.
Those observations motivate this extension; they are not independent evidence
for a new policy. Markets 6101–6124 cannot select or confirm this extension.

The working hypothesis is that estimating past unreported conversions and using
that estimate to change the current bid are different problems. Increasing the
conversion estimate raises the existing feedback controller's continuous action
at fixed cost. Recovering past conversions may therefore remove conservative
pressure without correcting the current probability scale.

This protocol is written before extension development is run. It is a bounded
follow-up, not a revision of the first study's gates or claims.

## Information and fixed quantities

Use only the existing `PolicyView`: time, observed spend, arrived conversion
count, budget, target CPA, and public accepted-event cohorts. Reuse mature
training history from markets 1001–1004; use only markets 2001–2006 for selection.
No hidden prediction multiplier, future outcome, final label, test scenario
name, or test delay parameter enters the strategy.

Keep the original market source, hard-budget settlement, 48 decision ticks,
48-tick maturity window, 11-action grid, three prediction stresses, three delay
regimes and three arrival seeds. Fix controller gain 0.3 and prior 8. The
original four delayed-feedback modules, market and learning modules, historical
fit, original configuration and original exports remain frozen.

Let O be arrived conversions, U the existing posterior pending mean, w_j the
posterior weight on calibration grid k_j, L = sum(w_j log(k_j)), and L0 the
corresponding frozen prior mean. Let J = sum(n_c p_c F(age_c)), a publicly
observable reporting-opportunity proxy. J is not a statistical confidence level.

## Exactly twelve configurations

| Family | Rule | Count |
| --- | --- | ---: |
| Fixed references | Conversion input V=O or V=O+U | 2 |
| Partial correction | V=O+lambda U, lambda in {0.25, 0.5} | 2 |
| Calibration gate | If J>=8 and Pr(k<0.9)>=tau, use lambda=0.25; otherwise lambda=1. tau in {0.8, 0.95} | 2 |
| Relative action calibration | Multiply continuous feedback by clip(exp(eta*(L-L0)), 2/3, 1.5); eta in {0.5,1}, crossed with V=O or O+U | 4 |
| Absolute action calibration | Multiply continuous feedback by clip(exp(eta*J/(J+8)*L), 2/3, 1.5); eta in {0.5,1}, V=O+U | 2 |

Let r be the raw product of pacing and quality terms from the original feedback
formula, before its original [0.05, 3] clipping. Compute desired=clip(r*f, 0.05, 3)
and then quantize once to the closest declared action; ties choose the lower
action. Thus calibration precedes clipping, including when r is saturated.
For fixed references f=1. Preserve the original float32 state reconstruction for
their numeric equivalence. With no reporting opportunity, correction factors are one.
Relative calibration is centered on the prior, not an absolute inverse of the
hidden stress. The truncated prior generally has L0 unequal to zero. The two
absolute-calibration controls check whether prior centering creates an offset.
Calibration of selected accepted events need not equal calibration of future
traffic. This is not a guaranteed estimate of the hidden prediction multiplier.

The cheap partial-correction controls are essential: a more complex candidate
must earn its value against them. No new neural model or dependency is needed.

## Development selection and stop condition

All scene cells and markets receive equal weight, as in the original study.
The stronger fixed reference is the one with higher aggregate development score
among arrived-only and full posterior correction. A candidate must satisfy all:

1. Aggregate mature score at least 3% above that fixed reference.
2. Nominal and underprediction aggregate score and conversions each retain at
   least 97% of the fixed arrived-only reference.
3. Each of the three prediction groups increases CPA violation frequency by no
   more than 3 percentage points versus the arrived-only reference.
4. Each overprediction delay cell retains at least 99% score and 97% conversions,
   with violation increase no greater than 3 percentage points versus arrived-only.
5. Aggregate score improves in at least four of the six paired development
   markets versus the stronger fixed reference.

Choose the highest aggregate score among eligible candidates. Among eligible
candidates with score >= 0.995 times that maximum, prefer the simplest by a frozen ordering:
fixed partial correction, calibration gating, then action calibration; use the
configuration declaration order to resolve remaining ties. Six development
markets provide a selection screen, not confirmatory evidence.

If all twelve configurations fail, stop the extension and publish the failed
gates. Do not append thresholds, change the target, or open new confirmation
markets. This budget prevents an open-ended search for a favorable holdout.

## One possible fresh confirmation

Only a passing development candidate can open markets 7101–7124. Freeze the
candidate, stronger reference, source/configuration/history identities and
development summary before generating or reading those markets. Evaluate the
candidate and both fixed references once. Do not reuse markets 6101–6124.

Primary comparison: at least 3% aggregate mature-score improvement relative to
the development-selected stronger fixed reference, with paired-market 95%
bootstrap lower bound above zero improvement. Use 10,000 resamples and seed
20261119. The independent unit is the market, not a campaign or arrival seed.

Ten protection endpoints use Bonferroni-adjusted one-sided bounds (alpha=0.05/10)
relative to the fixed arrived-only reference: score and conversion retention
of at least 97% for each of underprediction, nominal and overprediction (six),
and violation increase at most 3 percentage points for all scenes and each of
those three groups (four). At least 24 independent markets are required.
Publish the nine delay/stress cells descriptively even when aggregate gates pass.
The full-posterior comparison is also reported; it cannot replace a failed
primary comparison. No individual-cell or industry-leading claim follows from
passing these aggregate gates.

## Validation and limits

Tests cover zero evidence, gate boundaries, correction direction, prior-centering
offset, correction before quantization, accounting edge cases, policy information
boundaries, fixed-reference equivalence, exactly twelve configurations, split
separation, source/configuration/history identity and gated confirmation entry.
Independent review recomputes market aggregates and any confirmation intervals.

Matched stable reporting history remains an assumption. There is no online delay
drift discovery, production validation or joint future cost/conversion risk model.
Pending posteriors alone do not imply a CPA chance guarantee. A positive result
would support this lightweight control extension in the declared simulation;
it would not establish a generative-model advantage.
