# Maintenance and research roadmap

The durable assets are the dataset contract, evaluation protocol and diagnostic
tooling. Maintaining those assets can support further studies; no fixed useful
lifetime or hiring outcome is promised.

## First release

- Independently implemented BC, IQL and a compact DT-style baseline.
- Controlled data-view study, public Minari subset check, one weighting ablation.
- Reproducible local runs, portable reports, correctness tests and source attribution.

## Next substantive increment

Choose one research question, not several new frameworks:

1. Add another public continuous-control or navigation task and test whether the
   data-selection observations persist.
2. Expand training-seed count and separate tuning tasks from final reporting tasks.
3. Add TD3+BC as a public baseline, with a paper-referenced implementation check.
4. Compare a current generative policy against DT under an explicit compute budget,
   only after the baseline and data pipeline are validated.

## Ongoing maintenance

- Keep a known-good dependency lock and update dependencies in isolated changes.
- Add regression cases for real failures; do not manufacture issue or commit activity.
- Keep old reports readable and tie new results to new implementation fingerprints.
- Publish experiments and engineering notes with limitations and upstream citations.
- Accept useful external reproduction reports and focused contributions.

Favor one well-supported new finding per release. A new algorithm, language-model
integration, or web interface is justified only when it answers a concrete question.
