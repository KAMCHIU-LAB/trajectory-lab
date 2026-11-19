# Why this expert-selection study has these boundaries

This record explains decisions made before the new development outcomes were
analysed. The [protocol](bidding-expert-selection-protocol.md) defines the gates;
the [reproduction guide](bidding-expert-reproduction.md) defines execution.
These design choices are not evidence that selection improves bidding.

## 1. Predict a decision advantage, rather than a scenario name

The earlier response expert helped under severe overprediction and lost under
underprediction. A classifier trained to recognize a true simulator multiplier
would not answer whether the expert helps from the state actually reached.
Instead, the target compares two feasible continuations from the same visible
history. At deployment the selector sees only that history and predicts the
response expert's normalized terminal-score advantage over feedback.

True bias labels are retained for evaluator subgroup reporting. They are absent
from the policy feature contract, as are market/focal identifiers and future
labels. The visible-prefix digest, rather than a hidden campaign identifier,
also determines the seeded mixture control.

Implementation: [public feature contract and paired continuations](../src/trajectorylab/bidding_expert_selection.py);
checks: [visibility, clone isolation and timing](../tests/test_bidding_expert_selection.py).

## 2. Keep the cost of obtaining observations

All campaigns first execute 16 periods of the same feedback policy. Both
continuations inherit its spent budget, conversions, previous action, queue and
actual model context. No expert receives a free restart or observations from
the unchosen branch. One switch limits the state-distribution changes introduced
by the selector; repeated switching is outside this study.

The utility is nonlinear in total cost and conversions. Therefore the training
label is `score(prefix + response) - score(prefix + feedback)`, with both branches
settled to maturity. A difference of suffix-only scores would generally be a
different objective. Full-day controls show whether waiting for observations
costs more than selection can recover.

Checks: [full-cycle target and method coverage](../tests/test_bidding_expert_experiment.py).

## 3. Isolate unknown bias before adding unknown delay

The primary experiment uses immediate feedback, visible at the next decision.
That preserves the legacy response expert's observation semantics, which are
checked period by period against the original rollout. The delayed environment
adapter exists for interface checks, but no delayed-performance claim follows.

With delayed labels, a small observed conversion count can reflect either a
probability error or slow reporting. Joint identification needs additional
assumptions or mature history. Adding that uncertainty while changing expert
inputs would make the cause of a success or failure ambiguous. A delayed expert
would need its own frozen identity and validation.

This is a deliberate narrowing of the broader roadmap, not a solution to its
delay-identification question.

## 4. Let direct calibration challenge model complexity

Knowing an injected error of +50% would make a fixed reciprocal correction an
unfairly informed baseline. The deployable calibration controls instead estimate
a ratio from arrived conversions and the sum of public predicted probabilities
on accepted exposures, with declared smoothing and clipping. The ratio changes
with each control's own history; the true multiplier is never supplied.

Six calibration configurations match the six selector configurations. Established
feedback and fixed-action grids are also retained. The development reference is
the strongest cheap control, not the legacy baseline that made the earlier
response result look favorable. Choosing it on development still introduces
selection effects; only an unopened confirmation set could establish a new gain.

## 5. Check attainable space before fitting a selector

Hindsight chooses the better of the two continuations after seeing their full
outcomes. It includes future noise and cannot be deployed. It diagnoses only
this expert pair after this prefix, not all policies or all prefix lengths.

If even that optimistic choice lacks the registered +3% headroom over the cheap
reference, fitting a selector cannot rescue this particular pair on these
development outcomes. The registered decision is to stop before model fitting.
If headroom exists, train-only ridge fits and leave-one-market-out diagnostics
test one small family. No new family or prefix length is added after observing
the result under the same identity.

## 6. Keep statistical and business acceptance separate

Campaigns share market randomness, so the independent unit is a market. A large
CSV does not create a large independent sample. The +3% practical threshold,
positive paired interval and nine subgroup protections answer different
questions. Development intervals remain exploratory after selecting a reference
and candidate on that split.

The budget screen uses doubled development variance and conservative multiple
endpoint planning. It is not a guarantee of joint 80% power. Zero or near-zero
observed standard errors are inconclusive, rather than automatic significance.
The paired-t protection tests retain finite-sample and rare-event limitations.

## 7. Publish runnable inference without a private checkpoint

Trusted local response checkpoints were converted once into numeric NPZ arrays
and allowlisted JSON metadata. The public loader uses `allow_pickle=False`;
optimizer state, RNG state and local paths are excluded. All three converted
experts were checked for exact action and prediction parity on a legacy market.
Seed 11 was declared for the new study before opening its outcomes.

The [asset record](../assets/bidding-response-v1/README.md) distinguishes input
hashes, conversion provenance and inference parity from a full bitwise training
reproduction. A source checkout contains everything needed to generate the new
study's markets and execute its inference path.

## 8. Make recovery preserve the experiment

Per-market results bind source identity, exact input bytes and complete method
coverage. An immutable confirmation-opening seal binds the selected candidate,
reference and market allocation. Resume and final reporting verify that seal.
Independent review found and repaired gaps in the seal, result-loading
provenance and zero-variance handling before new policy outcomes were analysed.

The first data-generation seal preceded those review repairs. It is preserved;
the final experiment seal records the amended protocol and source identity.
Retaining both makes the timing visible instead of rewriting the earlier seal.

After the scientific stop, delivery checks found that the sealed
`bidding_expert_selection.py` did not match Ruff's formatter. Its exact path has
one disclosed formatting exclusion in `pyproject.toml`; lint and behavioral
tests still cover it. Its test file was formatted normally. Formatting the sealed
source after observing results would change its byte identity and invalidate
reproduction against the preserved seal. This delivery-only exception leaves the
old source hash, protocol, results and seals intact; a future study should format
that source before freezing its new identity.

Only aggregate simulator outcomes, public features, paired training targets and
allowlisted provenance are exported. Raw traffic and per-event labels remain
outside publication. Pattern scans supplement this boundary; they cannot prove
the absence of every possible secret or determine legal ownership.
