# Change log

## 0.2.0 — Public replay and evidence navigation

- Make the public CPU replay's commands, expected invariant outputs and resume
  boundaries explicit; add a versioned source/evidence record.
- Correct the test that treated every research configuration as an offline-RL
  suite. Keep configuration syntax checks separate from schema validation.
- Apply the existing formatter and exercise both small CPU paths in CI.
- Surface delayed-feedback findings and the research decision journal beside
  the original confirmation and generative-model limits.
- Retain the public simulator attribution in source distributions.

This release changes software delivery and navigation. It adds no new research
performance claim. Historical experiment hashes still refer to their recorded
source revisions; formatting changes do not replace those records.

## 0.1.0 — Initial public research implementation

Public offline-policy and bidding implementations, study protocols and exported
evidence. Earlier repository history contains the individual research stages;
there was no corresponding GitHub Release.
