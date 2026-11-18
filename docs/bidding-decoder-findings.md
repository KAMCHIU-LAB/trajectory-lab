# Action-decoder comparison: a closed candidate

Status, 2026-11-18: three training arms and numerical verification completed.
The prediction gate failed; policy evaluation was skipped. This is a single-seed
point-response structure experiment, not evidence of stochastic generation value.

## Hypothesis and controls

Could explicitly sharing the product of public predicted conversion probability
and action improve response prediction? The [module](../src/trajectorylab/bidding_action_decoder.py)
and [runner](../scripts/run_bidding_action_decoder.py) compare:

| Arm | Parameterization | Trainable parameters |
| --- | --- | ---: |
| I | Retrained original indexed-action structure | 13,720 |
| A | Shared action-interval head | 13,700 |
| P | A plus public current pmean × interval midpoint | 13,700 |

All arms use the same frozen robust seed-11 encoder/normalizer, sampler stream,
original log SmoothL1 loss, optimizer and 2,000-update budget. A/P share the same
initial weights. All receive deployment-style last-24-step contexts; this common
context change was recorded before fitting. The product is a structural feature,
not a claim of sufficient information across different markets.

Data: 16 training markets and six previously used development markets, five
scenarios, six focal advertisers and ten behavior trajectories. There are 230,400
training and 86,400 development rows, which are not independent market samples.
No reserved confirmation markets were opened.

## Frozen gate and results

Each of the two future heads had to reduce MSE by at least 5% against both A and I,
along with nominal-scene protection and paired consistency conditions. Evaluation
uses the final checkpoint and original normalized label units, not training log
scales. Markets receive equal weight.

| P relative to comparator | Future cost MSE change | Future conversion MSE change |
| --- | ---: | ---: |
| A | +1.85% | −1.34% |
| I | +3.18% | +1.69% |

All four required 5% improvements failed. On 240 same-physical-bid diagnostic
pairs, P improved future cost/conversion gap MSE versus A by 31.66%/20.79%, but
worsened versus I by 14.45%/57.58%. Selecting only the weaker A comparison would
misrepresent the outcome. Six of the 12 gate conditions failed.

The paired labels describe fixed-action open-loop tails, not subsequent closed-loop
feedback decisions. These diagnostics do not identify the cause of business
performance. No policy gain or loss was measured for this candidate.

## Verification and evidence identities

A separate local verification pass reconciled all 316,800 label/metadata/state
rows and the complete 2,000 × 64 sampler sequence. It checked 7,920 encoder contexts
and recomputed all development/paired predictions from final weights on CPU within
the declared CPU/MPS tolerance. Summary arithmetic and all 12 gate decisions agreed.
The 63 sealed files and input/source bindings passed integrity checks.

- Freeze: `c4fb96892a56befa34465121ee75e653eee975c9ce696744c1528638b1a2363e`
- Seal identity: `a3ef15ff9a5fbbc7a7f07a352b19f0cad68c94503e2af84f5617c6836d91967d`
- Local verifier SHA-256: `b84153a6edd4862561cd868706cd059c967739efcf1677219409a7cfb54217f0`

The local audit bundles are not public artifacts. This page is a curated scalar
readout, not a substitute for all original audit inputs. Optimizer updates were
not independently replayed, encoder vectors were sampled, and interruption/recovery
was not dynamically exercised. Historical development reuse and one training seed
preclude a new-market generalization claim.

Recorded training/cache/evaluation wall time was 137.79 seconds; peak process RSS
was about 2.12 GiB and sealed output about 307.5 MiB. Those figures exclude design,
implementation, verification and final sealing overhead.

The candidate is closed. It does not change the earlier
[scoped 24-market confirmation](bidding-findings.md), solve underprediction or prove
that structural features never help. See [publication provenance](code-and-data-provenance.md)
for source and reproduction boundaries.
