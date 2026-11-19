# One-shot expert selection under unknown prediction bias

Status: protocol frozen before opening new train/development markets; no result is implied.

## Question and scope

Can one decision, using only the first 16 opportunity periods' observed feedback,
improve terminal constrained bidding utility over a strong cheap controller when
the direction of a campaign-fixed pCVR error is unknown?

The environment is the repository's custom fixed-opponent AuctionNet mechanism,
not AuctionNet's full shared-budget competition. Each market has 500,000 requested
PVs, 48 periods, six focal advertisers, and equal-weight prediction multipliers
0.75, 1, 1.5. Campaigns share market randomness: the market, not campaign, is the
independent statistical unit. True multipliers and final labels are environment
state and never policy inputs. The first study uses **immediate feedback only**:
labels generated at period j are available before decision j+1. This preserves
the original response expert's observation semantics. Delayed feedback, LLM
planning, and generative distributional value are separate unanswered questions.

## Frozen experts and common prefix

The common prefix is feedback gain .3, prior 8, periods 0–15 inclusive. Immediately
before decision 16, copy the environment's budget, queue, visible ledger, previous
action and actual history into both continuations. Continue either the same
feedback expert or the uncorrected response checkpoint with training seed 11,
context 24, CPU reference inference. Seed 11 is the first declared legacy seed;
no seed search is performed. Public numeric weights retain the original trusted
checkpoint's SHA-256 and are independently compared with legacy inference.

The response expert receives the actual common-prefix history, without a fresh
start or free observations. Its pCVR mean scales with injected prediction error;
the sigma/mean feature retains the original expert's definition. The entire
campaign, including prefix spending and conversions, enters terminal score:

`score = conversions * min(1, target_cpa * conversions / spend)^2`

The paired target is **full response terminal score minus full feedback terminal
score**, normalized by initial budget / target CPA. It is not a suffix score
difference. Include every registered campaign, including zero conversions,
exhausted prefix budgets, equal experts, and failed evaluations. Simulation state
may contain private truth for settlement; policies receive public arrays only.

## Frozen data and model budget

Exact IDs, feature construction, candidates and limits are in
[`configs/bidding-expert-selection.toml`](../configs/bidding-expert-selection.toml).
Pilot markets 1001, 1002, 2001, 2002 are historical diagnostics only. New training
markets 120001–120008 and development markets 121001–121006 have not appeared in
the repository's previous configured studies. Reserved confirmation IDs
122001–122024 remain unopened until a candidate, reference, sample size, and
source identity are frozen. No old confirmation market is reused for selection.

One regularized linear advantage family has three penalties and two rejection
thresholds (six candidates). Features comprise the legacy 13 visible state
values, cumulative accepted count/probability and a smoothed observed calibration
ratio, plus accepted-count/probability/arrivals in four age buckets. No IDs,
regime names, true multipliers, unchosen response curves, or final labels enter
features. Scaling is fitted on training markets only. Features and targets are
stored separately. Zero variance features remain zero after centering.

A positive predicted advantage above the frozen threshold chooses response;
otherwise choose feedback. The threshold is an operational rejection rule, not
a calibrated confidence bound. Leave-one-market-out training diagnostics and
development terminal utility are reported; prediction MSE is not the endpoint.

## Cheap controls and prefix cost

Report full-day feedback (.3/.7/1.2 gains × 3/8 priors), five fixed multipliers,
full-day response, and six direct observable ratio-calibration controls (three
base controllers × two priors). Calibration uses `(arrived + prior) /
(accepted_public_probability_sum + prior)`, clipped to [.5, 2], then snaps the
desired corrected bid to the declared action grid. It never knows the true
multiplier. Feedback and fixed controls get their own full-day natural history.

Also report same-prefix always-feedback/always-response, deterministic seeded
50/50 campaign mixture, and two visible-ratio thresholds. Calibration gets six
candidates, equal to the selector candidate count, in addition to the established
feedback/fixed controls. Choose the deployable cheap reference with the highest
development mean score; its confirmation identity cannot change afterwards.
Full-day controls explicitly expose opportunity cost of the common prefix.

## Sequential gates and statistics

1. Validate clone isolation, as-of timing, privacy boundary, exact terminal-score
   target, original inference parity, and budget/coverage invariants.
2. Compute the within-pair hindsight best continuation on all registered new
   development campaigns. It uses future outcomes and is not deployable or a
   bound on general policies. If its mean score has less than 3% headroom over
   the strongest cheap development control, stop the selector study.
3. If headroom exists, fit the six training-only linear candidates and select
   once on development terminal utility. If the selected candidate has less than
   3% gain, or any development protection point estimate fails, stop. No new
   model family or prefix search is added to this identity.
4. Estimate required independent-market count from development paired variance,
   conservatively considering both raw and doubled variance. Minimum 24. Require
   sufficient budget for primary superiority and nine protection endpoints;
   use Bonferroni planning as a conservative bound for Holm testing. If the
   estimated requirement exceeds 24, stop for insufficient confirmation budget.
5. Freeze one candidate, one reference, all sources/weights/configuration and
   confirmation size before generating reserved markets. Confirm once. Require
   mean score gain ≥3% and paired market bootstrap 95% lower bound >0. For each
   under/nominal/over group, score ratio ≥.99, conversion ratio ≥.98, violation
   increase ≤.01 are tested with one-sided paired market tests and Holm family
   correction across nine endpoints. Expose all effects and intervals.

The protection tests use a paired-market Student t approximation; finite-sample
coverage for skewed or rare-event markets is not guaranteed. Zero/near-zero
observed variance is inconclusive and never produces a passing p=0. Planning
with doubled development variance is a resource screen for individual tests,
not an assurance that the joint gain/protection rule will pass with 80% power.

No per-market confirmation scores are used to adjust models or thresholds. An
interrupted run resumes only with matching run identity; a code fix starts a new
identity and new reserved markets if confirmation has been opened. The opening
seal binds a separate candidate fingerprint, experiment fingerprint and exact
market list; resume, result loading and final reporting verify that same seal.
Result loading also verifies input-byte and generator attestations, registered
method coverage, source identity and per-market result digests. Preserve
failures and report missingness instead of silently dropping markets.

## Resource and stop contract

Use one CPU inference process, two Torch threads, sequential market generation,
peak RSS cap 10 GiB, new cache cap 5 GiB, and six-hour preflight cap. Pilot measured
generation/evaluation time determines whether the cap is feasible. Only this
study's disposable new market caches may be removed after their hashes and
complete per-campaign results are recorded. Never remove historical evidence.
Process-scoped wakefulness ends automatically when the experiment process exits.

Legal terminal states: `STOP_NO_HEADROOM`, `STOP_NO_OBSERVABLE_SIGNAL`,
`STOP_DEV_FAILURE`, `STOP_INSUFFICIENT_BUDGET`, `INVALID_INFORMATION_OR_STATE`,
`CONFIRM_FAILURE`, `CONFIRM_PASS`. Scientific stops are complete results for this
question, not evidence of universal failure or of a positive industrial gain.
Any subsequent adapter, generative or competitive extension needs a distinct
protocol and evidence. No job-grade or production-performance claim follows.
