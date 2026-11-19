# Reproduce the frozen expert-selection study

This guide executes the [registered protocol](bidding-expert-selection-protocol.md)
with [`configs/bidding-expert-selection.toml`](../configs/bidding-expert-selection.toml).
It tests one choice between frozen feedback and response experts after a common
16-period prefix, under an unknown campaign-fixed pCVR bias. It does not imply a
positive result. A registered scientific `STOP_*` is a completed answer to this
question, and keeps confirmation inputs unopened.

## Source checkout and assets

Run from the repository root with Python 3.11 or 3.12. Keep the checkout's
`src/`, `configs/`, `docs/`, `assets/bidding-response-v1/` and
`third_party/auctionnet/` together. Loaders resolve assets and the vendored public
generator relative to the source tree; a wheel-only install is not the supported
reproduction path. Dependency wheel availability varies by Python, operating
system and architecture; successful installation on every platform is not
promised.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Use `python3.12` in the first command if that is your supported interpreter.
No account, external traffic dataset, private checkpoint or GPU is needed for
this study. Inputs are generated locally from the pinned public AuctionNet
source subset, using this repository's custom fixed-opponent mechanism, not the
full shared-budget competition.

The [public response assets](../assets/bidding-response-v1/README.md) convert the
three original response checkpoints into numeric NPZ plus JSON, without pickle.
Each JSON records the original checkpoint hash and distributed weight hash.
This study uses seed **11**, context 24 and CPU inference; do not search seeds.
Loading `PublicResponse(seed=11)` as shown in the asset guide checks the asset
contract before generating markets. The assets reproduce frozen inference,
not complete bitwise retraining.

## Information and comparison boundaries

The config declares 8 training, 6 validation/development and 24 reserved
confirmation markets, each with 500,000 requested PVs, 48 periods, six focal
advertisers and bias multipliers 0.75, 1 and 1.5. Campaigns share market randomness:
the independent statistical unit is **market**, not campaign, method or row.

This first study uses immediate feedback: a label generated at period `j` is
available before decision `j+1`. At decision 16 the selector sees only the actual
common-prefix history and public arrived feedback. It cannot see the true bias,
final labels, market IDs or outcomes of unchosen continuations. Simulation files
contain settlement truth for the evaluator; do not pass their raw arrays to a
policy. Features are captured before continuations are evaluated; training
targets remain separate. Normalization and the ridge model fit training markets
only. Validation selects once among six candidates and chooses the cheap
reference; confirmation cannot change either choice.

Validation and confirmation register 24 comparison methods:

| Methods | Count |
| --- | ---: |
| Full-day feedback: 3 gains × 2 priors | 6 |
| Full-day fixed multipliers | 5 |
| Observable calibration: 3 base controllers × 2 priors | 6 |
| Full-day response | 1 |
| Same-prefix always-feedback / always-response | 2 |
| Seeded 50/50 mixture | 1 |
| Visible-ratio thresholds | 2 |
| Hindsight best of the two continuations, diagnostic only | 1 |

The hindsight diagnostic uses future outcomes and is not deployable. Training
stores the two paired continuations, rather than all 24 comparison methods.
The target is the difference in **full terminal scores**, including prefix
spending and conversions, normalized by initial budget / target CPA; it is not a
suffix-only reward or prediction-MSE endpoint.

## Generate, freeze, evaluate and develop

Use a new output directory for a new experiment identity. The following commands
are sequential; the subshell stops at the first command error.

```bash
EXPERT_CONFIG=configs/bidding-expert-selection.toml
EXPERT_RUN=artifacts/bidding/expert-selection-v1
(
  set -e
  python -m trajectorylab.bidding_expert_experiment prepare --config "$EXPERT_CONFIG" --output "$EXPERT_RUN" --split train
  python -m trajectorylab.bidding_expert_experiment prepare --config "$EXPERT_CONFIG" --output "$EXPERT_RUN" --split validation
  python -m trajectorylab.bidding_expert_experiment freeze --config "$EXPERT_CONFIG" --output "$EXPERT_RUN"
  python -m trajectorylab.bidding_expert_experiment evaluate --config "$EXPERT_CONFIG" --output "$EXPERT_RUN" --split train
  python -m trajectorylab.bidding_expert_experiment evaluate --config "$EXPERT_CONFIG" --output "$EXPERT_RUN" --split validation
  python -m trajectorylab.bidding_expert_experiment develop --config "$EXPERT_CONFIG" --output "$EXPERT_RUN"
)
```

`develop` writes `development.json` and prints its status. It first checks
hindsight headroom against the strongest cheap validation reference, then checks
the selected candidate's gain and protections, then screens confirmation sample
requirements against the 24-market budget. These are registered gates, not
instructions to add candidates until one passes. Read a `STOP_*` result as the
scientific stopping point and report it with its reason.

## Open confirmation only after readiness

Run this block only for the same unchanged config, source tree and output
directory. Its first command requires `READY_FOR_CONFIRMATION`; otherwise the
subshell exits before opening or generating confirmation markets.

```bash
(
  set -e
  python -c 'import json, sys; status = json.load(open(sys.argv[1]))["status"]; print(status); sys.exit(status != "READY_FOR_CONFIRMATION")' "$EXPERT_RUN/development.json"
  python -m trajectorylab.bidding_expert_experiment open-confirmation --config "$EXPERT_CONFIG" --output "$EXPERT_RUN"
  python -m trajectorylab.bidding_expert_experiment prepare --config "$EXPERT_CONFIG" --output "$EXPERT_RUN" --split confirmation
  python -m trajectorylab.bidding_expert_experiment evaluate --config "$EXPERT_CONFIG" --output "$EXPERT_RUN" --split confirmation
  python -m trajectorylab.bidding_expert_experiment confirm --config "$EXPERT_CONFIG" --output "$EXPERT_RUN"
)
```

`confirmation.json` records `CONFIRM_PASS` or `CONFIRM_FAILURE`, effects,
intervals and the nine group protection tests. The protocol requires at least
3% mean score gain, a positive paired-market bootstrap lower bound and all
Holm-corrected protections. This is a scoped simulated result, not evidence of
production gain or a job grade. Do not inspect per-market confirmation scores to
retune a model, threshold, reference or sample size.

## Audit files and recovery

| Record under `EXPERT_RUN` | What it binds |
| --- | --- |
| `data-freeze.json`, `markets/{id}.npz` and matching JSON, `inputs/{id}.json` | Config, public generator identity, market identity and exact input bytes |
| `experiment-freeze.json` | Config, protocol, experiment sources, generator and selected expert assets |
| `train/{id}.json`, `validation/{id}.json`, `confirmation/{id}.json` | Complete registered coverage, input hash, experiment identity and result digest |
| `development.json`, `selection-freeze.json` | Gate outcome; when ready, one candidate, reference and confirmation allocation |
| `confirmation-open.json`, `confirmation.json` | Immutable opening seal and final decision under that seal |

After interruption, restore the same `EXPERT_CONFIG` and `EXPERT_RUN` values and
rerun the interrupted command, for example:

```bash
python -m trajectorylab.bidding_expert_experiment prepare --config "$EXPERT_CONFIG" --output "$EXPERT_RUN" --split validation
python -m trajectorylab.bidding_expert_experiment evaluate --config "$EXPERT_CONFIG" --output "$EXPERT_RUN" --split validation
```

Matching input caches and completed per-market evaluations are reused and
checked. Keep caches and audit files for resume. Once `selection-freeze.json`
exists, do not rerun `develop`; resume the gated confirmation block instead.
Opening confirmation and final reporting verify the same immutable selection
and seal. A source/config/asset change requires a new identity and output
directory; after confirmation has opened, a code fix also requires new reserved
market IDs. Preserve failed or incomplete records rather than dropping campaigns
or editing hashes to force a resume.

## Compute measurements and limits

The registered limits are one CPU inference process, two Torch threads, 10 GiB
peak RSS, 5 GiB new market cache and a six-hour preflight budget. Generation is
sequential; neural training is disabled. On an Apple M1 with 16 GB RAM and two
CPU threads, pilot measurements in this iteration were approximately **7.6 s**
for prototype evaluation on an old cached market and **188 s** for generation of
the first fresh registered market. These measure different workloads and are
pilot observations, not a total-runtime estimate or platform guarantee. Use the
`GENERATED` / `EVALUATED` timing logs and recorded resource use to assess your run.
