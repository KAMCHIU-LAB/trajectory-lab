# Contextual response baseline: supplementary diagnostic

Specified while the first formal study was still training, before inspecting
its held-out outcomes. This is an independent diagnostic; it does not change
either frozen bidding study, its primary comparison or its promotion gate.

## Question

The controller study's pooled empirical curve uses opportunity count, while the
neural forecast uses pCVR and state history. That comparison cannot identify the
value of sequence modeling. Add a cheap predictor that receives the same current
13-dimensional state, including cumulative and previous-tick feedback. It has no
sequence input, but it is **not history-free**: the state summarizes history.

The immediate question is how well this small contextual model fits the four
response curves on validation markets, and whether its aggregate predictions
show systematic underestimation or overestimation. This run does not compute
matched pooled-curve errors, so it cannot rank the two predictors. A predictive
result alone is not evidence of better bidding or a causal benefit from history.

## Data and fitting

- Reuse the first study's 16 training and 6 validation markets only. Do not load
  test or confirmation markets, policy-selection files or held-out result files.
- Use the existing ten behavior trajectories per focal campaign, with their
  current and remaining-horizon response labels. Each of the four curves has
  the same eleven nominal actions as the first study.
- Fit state mean/std on training rows only, with a standard-deviation floor of
  0.05. Clip standardized inputs to `[-8,8]`.
- Construct 105 features: intercept, 13 linear terms and 91 quadratic products
  `z_i*z_j`, for `i <= j`.
- Transform labels with `log1p(label * scale)`, where the four curve scales are
  `[48,48,1,1]`. Fit multivariate ridge regression using
  `X.T @ X / N + lambda * diag(0,1,...,1)`; the intercept is not penalized.
- Try `lambda = [1e-6,1e-4,1e-2,1]`. Select by validation mean squared error in
  transformed label space **after the prediction projection below**. Ties follow
  that declared order. No test-driven parameter changes are permitted.

At prediction, clip log outputs to `[0,20]` for numerical protection, invert the
label transform, set zero-action output to zero, and apply a cumulative maximum
over actions separately to each curve. The cap is fixed before validation and
is not inferred from validation labels. The predictor exposes raw-unit current
cost/conversion curves through the existing numerical-controller interface.

## Evidence and interpretation

Record all candidate train/validation log-space errors and aggregate
predicted/observed ratios for each of the four curves. Include input, source and
model identities, market counts and row counts. Repeated behavior trajectories
share response labels; the resulting rows are not independent experiments.

The selected predictor's validation error is selection-biased. A comparison
against the sequence model also changes model class, loss and fitting method;
it does not isolate attention, history or parameter count. This baseline is a
stronger covariate-aware diagnostic than a pooled curve, not a claim to be the
best non-neural predictor.

Any later policy-value comparison requires its own declared controller-selection
rules and untouched evaluation markets. These predictive diagnostics cannot
retroactively become a primary result in either frozen study.

## Local artifacts

The standalone module is `trajectorylab.bidding_context`. It writes a separate
source/configuration manifest, a model NPZ loaded without pickle, summary and
status. Loading also checks the main implementation hashes that define the
state and label semantics. Identity and model hashes prevent accidental mixing; a file lock guards
the output directory. No GPU fitting, new data generation or external model API
is required. Fitted weights stay in ignored local artifacts.

## Matched forecast audit

This extension was specified after reading the ridge validation summary, while
the formal neural fits were still running and before reading formal held-out
outcomes. It is a descriptive follow-up, not a new primary hypothesis or a
change to either frozen experiment.

After all three response fits finish, compare their predictions separately with
the selected ridge model and the training-only pooled empirical curve. Rebuild
the same 360 validation behavior trajectories: six markets, six focal campaigns,
ten behavior policies and 48 ticks, for 17,280 rows. No test, confirmation or
policy-selection file is an input. These are nominal (`calibrated`) behavior
trajectories; this audit does not measure stressed-scenario prediction errors.

For every neural row, use the exact causal window ending at that tick, with a
maximum length of 24, absolute timesteps, saved training normalization and
evaluation mode. Batched inference must take the last valid position, including
short early windows. Ridge receives the same row's state; the empirical curve
receives its current opportunity count. All targets are unthrottled response
labels, rather than budget-truncated realized feedback.

Report log-space MSE, Smooth L1 with beta 1 (the neural training definition), and
aggregate predicted/observed ratios. Use the same action weights, four label
scales and row weights for every applicable predictor. The two current curves
are directly comparable across predictors; remaining-horizon curves are only
available for neural and ridge models. Keep per-market and per-seed results.
Compute each neural seed's errors before any across-seed summary; do not average
predictions into an undeclared ensemble.

Validation has already selected neural checkpoints and ridge regularization,
so these are selection-biased diagnostics. Behavior-policy states and all-action
errors need not represent states or chosen actions visited by a new controller.
Prediction differences cannot establish policy value, isolate history, or prove
that the controller is the principal bottleneck. Do not select new checkpoints,
retune frozen policies or change confirmation gates using this audit.

### Point predictions and decision expectations

These losses fit response point predictions, not guaranteed calibrated
conditional means. For an unrestricted log-MSE predictor of nonnegative `Y`,
the population optimum is `(exp(E[log(1+sY)|x])-1)/s`, which is at most
`E[Y|x]` by Jensen's inequality. The neural log-Smooth-L1 objective likewise has
no conditional arithmetic-mean guarantee; this does **not** prove that a fitted
neural model must underestimate. Regularization, model capacity, sparse labels
and distribution shift can also affect aggregate ratios. Using these predictions
as expected costs/conversions is a controller approximation to evaluate, rather
than an established calibration property.

### Reproduce the local diagnostics

All three response fits must have completed before the matched audit. These
commands use existing train/validation caches and local checkpoints:

```bash
python -m trajectorylab.bidding_context \
  --main-run artifacts/bidding/main-v2 --output artifacts/bidding/context-v1
python -m trajectorylab.bidding_predictive \
  --main-run artifacts/bidding/main-v2 --ridge-run artifacts/bidding/context-v1 \
  --output artifacts/bidding/predictive-v1
python scripts/export_bidding_predictive.py \
  --audit-run artifacts/bidding/predictive-v1 --ridge-run artifacts/bidding/context-v1 \
  --destination reports/bidding-forecast
```

The exporter checks completed evidence identities, hashes and coverage, then
copies an explicit text allowlist and renders the metric table. It does not
independently recompute predictions or statistics. Model files, traffic caches,
runner state and machine paths are excluded. The output records the privacy
pattern checks and their limits.
