# Pending estimates and bid control: a bounded follow-up

[Registered rules](bidding-delayed-strategy-protocol.md) ·
[Development evidence](../reports/bidding-delayed-strategy/README.md) ·
[Original independent confirmation](bidding-delayed-findings.md)

## Decision

All twelve registered configurations completed on the six existing development
markets, producing 11,664 campaign rows. None passed the complete promotion
screen. No candidate was selected and no fresh confirmation market was opened.
This is a completed development study, not a confirmed new policy improvement.

The most useful development trade-off was fixed 50% pending correction. It
improved score 1.5986% versus arrived-only feedback, but just 0.3008% versus the
stronger full-posterior reference, below the required 3%. It passed the remaining
development protections and improved in four of six markets. Those selection
results do not establish its generalization advantage.

## Complete comparison

Every row uses controller gain 0.3, prior 8 and the same public action grid.
The stronger fixed reference, selected by development score, was full posterior
correction (14.999058). Both fixed references exactly reproduce all 972 matching
campaigns each from the original development run.

| Configuration | Mean mature score | Change vs stronger fixed reference | Markets won / 6 |
| --- | ---: | ---: | ---: |
| Arrived-only reference | 14.807459 | −1.2774% | 1 |
| Full posterior reference | 14.999058 | 0.0000% | 0 |
| Partial 0.25 | 14.832196 | −1.1125% | 1 |
| Partial 0.5 | 15.044173 | +0.3008% | 4 |
| Calibration gate 0.8 | 14.993927 | −0.0342% | 4 |
| Calibration gate 0.95 | 14.978603 | −0.1364% | 1 |
| Relative calibration eta 0.5, no pending | 14.437311 | −3.7452% | 3 |
| Relative calibration eta 0.5, full pending | 14.450462 | −3.6575% | 1 |
| Relative calibration eta 1, no pending | 13.862563 | −7.5771% | 1 |
| Relative calibration eta 1, full pending | 13.925911 | −7.1548% | 1 |
| Absolute calibration eta 0.5, full pending | 14.676887 | −2.1479% | 2 |
| Absolute calibration eta 1, full pending | 14.328661 | −4.4696% | 1 |

These are descriptive development comparisons after searching the registered
family. They are not twelve independent discoveries, and no unadjusted
development interval would turn the selected maximum into a confirmed gain.

## What the simple trade-off achieved

For 50% correction, relative to arrived-only feedback on these six markets:

| Mixture | Score | Mature conversions | CPA violation frequency |
| --- | ---: | ---: | ---: |
| All scenes | +1.5986% | +1.3509% | −0.3086 pp |
| Underprediction | +2.3848% | +2.4817% | +0.9259 pp |
| Nominal | +0.8547% | +0.7599% | −1.5432 pp |
| Overprediction | +1.7152% | +1.1205% | −0.3086 pp |

Compared with full posterior correction, it retained 99.6834% of conversions,
improved score 0.3008%, and reduced violations by 1.4403 pp. Its three
overprediction delay cells passed the registered development retention/risk
screen. Full correction increased underprediction violations by 5.2469 pp in
development; half correction increased them by 0.9259 pp.

The original 24-market confirmation found a different, smaller underprediction
violation change for full correction, and a loss in the overprediction/long-delay
cell. This contrast is a reason to respect market-level uncertainty and split
separation. It does not permit claiming that half correction fixes the sealed
confirmation failure. Half correction has not been tested on fresh markets.

## What was learned about the method

1. **Outcome estimation is separate from control.** The original common-trajectory
   audit reduced long-delay pending MAE by 55.81%, yet its independent score gain
   was only 1.21%. This extension shows that using less than the full estimate can
   be competitive within the same heuristic controller. It is a development
   observation, not a general theorem that biased estimates are better.
2. **Probability calibration is not a CPA-optimal bid rule.** Relative and
   evidence-weighted absolute calibration factors both failed here. Selected-event
   calibration, action-dependent auction prices, budget pacing and the existing
   feedback loop interact. The ablations reject these specific plug-in rules;
   they do not uniquely identify which interaction caused every loss.
3. **The simple control must be a serious comparator.** A posterior gate did not
   add value over a fixed partial correction. Additional logic needs measurable
   decision value, a stable information boundary and a comparison against strong
   cheap controls before it earns a place in the policy.
4. **A trade-off must be explicit.** Score, conversions and violation frequency
   are reported separately. A score gain cannot hide a material risk increase,
   and losing a small amount of conversions can be acceptable only under a
   declared objective and retention floor. Passing this development screen is
   not a production risk guarantee.

## What would justify a different next study

The unresolved question concerns future action response, not just past pending
counts: how do candidate bids change joint future cost and conversions under
the current observed state? A further generative study would need an
action-conditioned joint response distribution and an explicit constrained
decision rule. A point head and a distribution head should share training data,
state information and a matched backbone; their decisions should be compared
against tuned scalar controls and a strong point-response controller.

Calibration/coverage should be checked separately from final utility and
violations. The selected policy then needs a frozen fresh-market confirmation.
Drawing samples of past pending counts or multiplying bids by an inferred
calibration factor does not establish that distributional decision value.

No such new generator was trained in this extension. The existing Flow Matching
branch and its failed seed-replication gate remain separate evidence. Current
results support a reproducible investigation and a conservative candidate for
further study; they do not complete the claim of generative bidding superiority.

## Integrity and resources

Development took 249.79 seconds in the recorded Apple M1 environment. The run
used existing public market caches and mature training history, with no new
dependency, neural training, employer data or employer code. The independent
review recomputed every development gate from raw rows without importing the
selection implementation. All accounting fields of both fixed references
matched the original development run exactly.

Selection identity:
`49bf3a74ecacdcd014766ce45911bddb1c60134ea6139c80c7c9d3a0ac381329`.
Development identity:
`e17064ba73b111df13d7f8877963d2876a21b8db38ed571092d2267fd0ddde43`.

To reproduce after preparing the original training/development caches:

```bash
PYTHONPATH=src python scripts/run_bidding_delayed_strategy.py --stage develop
PYTHONPATH=src python scripts/export_bidding_delayed_strategy.py
```

`--stage confirm` rejects the recorded failed selection before accessing new
market data. Markets 7101–7124 remain unused. Do not change the frozen selection
to bypass that gate; a different scientific question requires its own protocol.
