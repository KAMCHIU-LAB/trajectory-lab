# Design decisions

## A bounded experiment instead of a broad framework

The first release must run on a 16 GB Apple M1 and provide evidence that a reviewer
can inspect without an account. Distributed training, cloud dashboards and model
APIs would add setup cost without answering the data-selection question. The
result is a small Python package, explicit TOML files, local artifacts and static
reports. The package has no service dependency.

## Independent small baselines instead of an unmodified framework wrapper

BC provides a supervised reference. IQL adds implicit value learning and
advantage-weighted extraction without optimizing actions outside the dataset.
DT provides return-conditioned sequence modeling. Together they expose different
responses to the same training data. The implementation stays small enough to
read, but this introduces variant and tuning differences from larger reference
codebases. Correctness invariants and attribution are therefore more defensible
than claiming paper-score parity.

BC and IQL deliberately share the same bounded-mean Gaussian actor. This enables
a mechanism check: setting IQL's advantage temperature to zero must reproduce BC
actor updates. This is an implementation invariant, not an empirical discovery.

## Controlled data first; an external public task second

Pendulum permits cheap source generation and exact equal-transition comparisons
with complete fixed-length episodes. It does not establish general RL capability.
Minari PointMaze checks a different simulator, dictionary observations, goal
conditioning, and a separately defined evaluation environment. The limited subset
keeps the experiment local; published full-dataset rankings remain out of scope.

All three selected PointMaze training-view return quantiles can be identical
because a source episode ends at a successful goal transition. This is why its
first release uses only uniform selection. Applying a high-return filter anyway
would fail to create a meaningful return-quality intervention.

## Fixed final checkpoints instead of test-driven model selection

The final model is the last update under a declared budget. Validation curves help
diagnose training and generalization, but do not choose a lucky final checkpoint.
The final environment seeds are declared in advance. Further tuning should create
a new development study and preserve the original report.

## One-thread learners and an optional parallel recovery driver

Single-thread CPU learners make small runs predictable and avoid each process
claiming every CPU core. Independent runs can be scheduled concurrently to use
the machine, without changing optimizer updates or seed assignments. The recovery
driver uses the same training function and records its own source digest. Shared
machine contention means these wall times are operational evidence, not a clean
microbenchmark. The serial runner remains the simple reference path.

## Descriptive diagnostics instead of an OOD detector claim

Nearest-neighbor distance and PCA expose some support differences with minimal
dependencies. They depend on scaling, sampling, and representation; successful
policies may legitimately enter states absent from the data. These plots should
motivate hypotheses and additional experiments, rather than certify safety or
prove the cause of a return difference.

## Traceable artifacts instead of a screenshot-only result

Reports include per-run raw returns, training histories, configurations, source
episode IDs, budgets, software versions and digests. They reject incomplete
evidence. The data and model binaries remain local, while compact measurements
are suitable for version control. Readers can recompute aggregates without
installing the simulator or trusting a rendered chart.

Algorithm/model/data/evaluation/configuration/reference-runner files form the
core implementation digest. The optional parallel driver's source hash is stored
separately. The report renderer can evolve without changing the trained model
identity. Digests bind recorded artifacts; they are not signatures or independent
third-party verification.

## Transferable questions instead of transferred scores

Public control tasks let the experiment isolate data selection without private
auction logs. Their results describe the implemented control experiments. The
connection to bidding is methodological: logged trajectory selection can trade
off aggregate quality against coverage; reward horizons must match the policy's
decision period; and offline support checks should inform investigation rather
than certify release.
The results document lists the separate measurements a bidding study would need.
