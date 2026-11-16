# Maintenance and research roadmap

The durable assets are the dataset contract, evaluation protocol and diagnostic
tooling. Maintaining those assets can support further studies; no fixed useful
lifetime or hiring outcome is promised.

## First release

- Independently implemented BC, IQL and a compact DT-style baseline.
- Controlled data-view study, public Minari subset check, one weighting ablation.
- Reproducible local runs, portable reports, correctness tests and source attribution.

## Next substantive increment

The next experiment should resolve one observed ambiguity before adding another
algorithm. The current PointMaze training fragments all have return 1, while the
recovered evaluation environment continues for 300 steps. The DT target is reduced
by observed reward and can go below the training return-to-go range.

### Study 1: PointMaze target-horizon check

- Compare the current return-to-go update with a zero-clamped target. Keep the
  dataset, split, model, optimizer, update count and evaluation horizon fixed.
- Add training seeds and reserve new environment seeds for final evaluation; the
  seeds already reported in v0.1 must not become a tuning set.
- Record raw return, goal occupancy after first arrival, support diagnostics and
  action behavior before and after reaching the goal. These metrics help explain
  a difference; they do not establish deployment safety.
- Publish both outcomes and exact configs, even if the target correction does not
  improve the policy.

After that result, choose one next question from this list:

1. Test whether return quality or state-space coverage matters more in another
   public task with varied episode lengths.
2. Repeat source-data and split seeds to measure data uncertainty alongside
   training-seed variation.
3. Add TD3+BC as a public baseline with a paper-referenced implementation check.
4. Compare a generative policy against DT only after defining a compute budget,
   test-seed protocol and a specific question the comparison can answer.

## Ongoing maintenance

- Keep a known-good dependency lock and update dependencies in isolated changes.
- Add regression cases for real failures; do not manufacture issue or commit activity.
- Keep old reports readable and tie new results to new implementation fingerprints.
- Publish experiments and engineering notes with limitations and upstream citations.
- Accept useful external reproduction reports and focused contributions.

Favor one well-supported new finding per release. A new algorithm, language-model
integration, or web interface is justified only when it answers a concrete question.
