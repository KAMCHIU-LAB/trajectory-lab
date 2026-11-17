# Reproduce the offline policy studies

This guide covers the earlier Pendulum, PointMaze and IQL ablation studies.
For the current generative-bidding track, use the [bidding runbook](bidding-runbook.md).

## Environment and smoke run

Python 3.11+; the local reference uses Python 3.12 on an Apple M1, CPU / one thread
per learner. Start from the repository root.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
trajectorylab doctor
make check
make smoke
make report
make serve
```

The smoke run performs **real training and evaluation**, with only 100 updates.
It validates integration; its scores are not evidence of algorithm quality.
Checkpoints let an interrupted invocation resume from its last durable update.
A repeated completed invocation reuses the same results.

`make report` and `make serve` use `reports/latest`, the most recently invoked
suite. To browse the three included formal studies together, use `make demo`
instead. `make portal` rebuilds their index from the exported study evidence.

## Reproduce the studies

```bash
make study       # 27 runs: 3 data views × 3 methods × 3 training seeds
make ablation    # 3 runs: IQL with advantage_beta = 0

python -m pip install -e '.[benchmark]'
make pointmaze   # 9 runs; first call downloads the public Minari dataset
```

All formal runs use 10,000 gradient updates, batch size 128, and eight final
environment seeds per training seed. The public dataset download is approximately
530 MiB; allow several GB for the environment, source data and checkpoints.
Transformer runs dominate CPU time. Actual measured run times and contention
are recorded in the reports; these studies are longer than the smoke test.

For a stopped, already initialized **CPU** suite, an optional recovery driver can
finish independent runs in parallel without changing their training settings:

```bash
# Stop the serial runner first. Replace the path with its printed suite directory.
python scripts/complete_suite.py --suite artifacts/suites/<suite-name-id> --workers 3
```

Do not run both drivers on the same suite concurrently. The recovery driver checks
source, implementation and runtime identity, and records its own source hash and
worker count in the evidence. Each learner still uses its configured thread count.
Parallel wall times include shared-machine contention.

`requirements.lock` records the full known-good local environment. Other Python
versions and operating systems may need platform-appropriate dependency resolution;
bitwise equivalence across versions or accelerators is not promised.

## Why these details matter

- **A trajectory is the split unit.** Content duplicates are grouped before
  splitting, and sequence windows cannot cross an episode boundary.
- **Termination is not truncation.** Only true termination removes the IQL
  Bellman bootstrap; a time limit retains it.
- **The DT action label is causally hidden.** State token t predicts action t
  before reading its action token. Perturbation tests check this boundary.
- **A seed is the aggregation unit.** Error bars describe training-seed means,
  rather than treating correlated evaluation episodes as independent training runs.
- **A report must be complete.** Missing jobs, duplicate identities, changed test
  seeds, and summaries inconsistent with raw returns fail the report gate.
- **Support distance is descriptive.** Common train-only scaling makes views
  comparable; nearest-neighbor distance is not a calibrated OOD or safety score.

See [architecture](architecture.md), [protocol](protocol.md),
[design decisions](design-decisions.md),
[maintenance roadmap](roadmap.md), and [contribution guidelines](../CONTRIBUTING.md).
The GitHub Actions workflow is supplied; local checks do not imply hosted CI has run.
