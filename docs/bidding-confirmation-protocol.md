# Independent confirmation D: frozen original response versus feedback

## Timing and scope

This is a new study declared **after** discovery in main-v2 and after the failed
control-v1 validation gate. The exploratory main result suggested an original
response advantage over feedback in `over_50`. This study does not reopen the
paced/correction gate, retrain models, tune policies, or convert the old discovery
into a preregistered result. This protocol and `configs/bidding-confirmation.toml`
must be locked before generating any market 3301–3324. Their hashes and the
implementation hash enter the output identity before generation starts.

Only public AuctionNet-derived synthetic markets are used. No main test outcomes
or confirmation outcomes are read as study inputs. Frozen main/control manifests,
selection/status records and the three already trained response checkpoints are
identity inputs; the original records remain unchanged. The control selection must
name `feedback` as its primary reference, and its status must remain
`complete-not-promoted`.

## Fixed evidence and policies

- Exactly 24 new market seeds: **3301–3324**, disjoint from every main split and
  the earlier control confirmation allocation. No expansion or replacement.
- Each market uses 500,000 requested PVs, focal advertisers `[0,8,16,24,32,40]`,
  the frozen 11 nominal actions and four bias values in the config, and the main
  market/auction implementation. Market construction remains unchanged.
- Five complete scenario strata: `calibrated`, `over_25`, **`over_50`**,
  `under_25`, `mid_over_50`. `calibrated` is the nominal scenario.
- Six policy slots per campaign/scenario: original uncorrected, soft-CPA
  `response` at seeds **11,23,37**; frozen `feedback` gain **0.3**, prior **8**;
  `fixed-1` at **1.0**; `fixed-tuned` at **1.25**, selected by the old main
  validation and frozen here. Simple policies have seed `-1`.
- Response uses each seed's frozen main `best.pt`, checkpoint normalization and
  causal 24-tick context, with CPU inference and two Torch threads. There is no
  response ensemble policy, new training, fitting, tuning, or parameter search.
- Expected final evidence is **24 × 6 × 5 × 6 = 4,320 rows**. Each declared
  `(market,focal,scenario,method,seed)` must occur exactly once and be finite.
  Undefined realized CPA for a zero-conversion row is represented as JSON `null`,
  as in the frozen evaluator; numerical metrics remain finite.

## Data barrier and execution

`--stage data` only constructs market caches. `--stage evaluate` requires all 24
market NPZ/metadata pairs with matching declared identities before opening any
market tape, reading any cached policy results, or evaluating any policy.
`--stage all` applies the same barrier after generation. No interim policy scores,
outcome inspection, sequential decision, early stop for efficacy/futility, or
partial-market inference is allowed. Operational interruption/resume is allowed;
it never changes the market allocation, policies or decision rule.

The output is `artifacts/bidding/confirmation-v1`. An exclusive run lock and
atomic market construction/result JSON replacement support safe resumption.
Config, protocol, own/dependency source, frozen main/control records and checkpoint
hashes bind the manifest identity. Existing output with a different identity is
rejected. Each policy-result cache carries that identity and its row digest;
complete reuse verifies published file hashes and exact evidence coverage.
External bounded prefetch may construct disjoint markets under the same previously
created manifest; the module itself performs no parallel spawning. All 24 cache
hashes are sealed before the evaluation barrier opens. A changed cache cannot be
silently substituted on resume.

## One primary decision

The primary contrast is **original response minus feedback in `over_50`**.
Reuse `bidding_control.paired` with **10,000** market bootstrap resamples and RNG
**29117**. Within each market, first average the three response seeds per focal,
then average six paired focal effects. Bootstrap the 24 market units with
replacement and report the 2.5%/97.5% quantiles. Conversion retention is the ratio
of mean response conversions to mean feedback conversions, including within each
bootstrap draw; it is not an average of individual ratios.

Confirmation passes only if all four conditions hold:

1. Score-difference 95% interval lower bound **> 0**.
2. Conversion mean-ratio 95% interval lower bound **≥ 0.95**, with the observed
   ratio and **every** bootstrap ratio defined (positive feedback denominator).
3. CPA-violation-rate difference 95% interval upper bound **≤ 0.05**.
4. Each of the **3/3** separately reported response seed mean score differences
   versus paired feedback is **> 0**.

An undefined primary conversion ratio or bootstrap denominator yields
**inconclusive**, never a pass. Otherwise the decision is **confirmed** or
**not-confirmed**. Every terminal scientific decision has execution phase
`complete`; a failed statistical gate is not an operational failure. Do not add
markets, change thresholds, replace seeds, retune, or select a different primary
scenario after seeing results.

## Reporting and limits

Report all five strata and all four methods' score, conversions, cost, violation,
budget utilization and zero-conversion rates, including individual response seed
means and paired response/feedback intervals in each stratum. The primary gate
uses `over_50` alone; no pooled-scenario result substitutes for it. The other
strata, including nominal, and fixed baselines are secondary descriptive checks;
they do not create additional confirmation claims or a new selection opportunity.

Intervals are conditional on **these three fixed trained models**, this fixed
auction/opponent construction and advertiser allocation. Advertisers and repeated
model seeds within a market are correlated, not extra independent market units.
The interval does not estimate general training-seed uncertainty, real-market
deployment performance, a causal architecture/history/controller bottleneck, or
the failure mechanism of control-v1. A successful confirmation supports only the
declared frozen original-response versus frozen-feedback contrast in this new
market allocation and primary scenario.

The standalone report uses this study's confirmation title and semantics. It links
text-only manifest, summary and raw aggregate campaign rows; no states, tapes,
traffic records, weights, logs or absolute machine paths are included in those
reports. Market NPZs and per-market operational caches remain local artifacts.
