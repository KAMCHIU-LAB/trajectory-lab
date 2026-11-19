# Code, data and publication provenance

Publication review: 2026-11-19. This page distinguishes public upstream code,
local implementation and the data excluded from the repository.

## Sources

| Component | Source and boundary |
| --- | --- |
| Auction and traffic kernels | [AuctionNet](https://github.com/alimama-tech/AuctionNet), pinned to `c20de631a40e24669d1d6a8e2acefd8f492f172d`; the two vendored Python files retain [Apache-2.0](../third_party/auctionnet/LICENSE.txt) terms. |
| Bidding models, controllers and research scripts | Local implementation of public methods and explicitly scoped experimental variants, developed with AI-assisted coding. The source review found no employer SDK, company endpoint, internal integration or private credential reader. This finding concerns the inspected repository; it is not a legal determination of ownership. |
| Bidding examples and reported measurements | Public-simulator-generated markets under the declared custom fixed-opponent and settlement protocol; no production advertising logs or online business metrics. |
| Delayed-feedback study and control extension | New event-level caches generated from the published AuctionNet-derived kernels, with synthetic event-keyed reporting delays. New estimator/ledger/control code uses public simulation interfaces; mature training history is generated within the study. No old private runtime bundle or trained checkpoint is imported. |
| Frozen expert-selection study | New synthetic markets generated from the same public AuctionNet-derived, fixed-opponent kernels, with immediate feedback. The [protocol](bidding-expert-selection-protocol.md) fixes response seed 11 and separates training, validation and gated confirmation. The [source-checkout guide](bidding-expert-reproduction.md) documents generation, freezing and recovery; no production traffic or private checkpoint is required. |
| Public response expert assets | Three frozen experts trained in this repository's public simulation, converted to numeric NPZ plus allowlisted JSON in [assets/bidding-response-v1](../assets/bidding-response-v1/README.md). Original checkpoint hashes and distributed weight hashes record conversion provenance. Loading uses `allow_pickle=False`; these assets preserve frozen inference, not complete bitwise retraining. |
| Earlier control tasks | Gymnasium Pendulum and the public Minari PointMaze subset named in the [offline study](offline-policy-guide.md). Dataset files retain upstream terms and are not redistributed. |

The study is not an official AuctionNet benchmark reproduction. See the
[experimental protocol](bidding-protocol.md) and [findings](bidding-findings.md).

## Vendored modifications

- `NeurIPSPvGen.py`: 23 additional blank lines in the original local copy, plus
  the modification/license header added for this publication. Its executable AST
  is identical to the pinned upstream generator.
- `BiddingEnv.py`: the `gin` import and `gin.configurable` decorator were removed,
  and a modification/license notice was added. Its AST matches upstream after
  those two documented removals; auction rules were not changed in this file.
- The upstream license text is retained. Attribution checks do not imply that
  the upstream authors endorse these experiments.

Adding the generator header changed source bytes after the experiments. Original
local experiment files and seals were preserved. Their manifests refer to the
historical source bytes, not to the later publication header. The original local
generator SHA-256 is
`1887bd7f4eea497672d1665e1e5b71e8cee843b6bb18ff94acda24733a0767c4`;
the pinned upstream generator SHA-256 is
`2ac50f6f5c42be0d329fd9106c35ef067120989cfa4e4852e9cd1680a835d368`.
The parent publication commit retains the original vendored file.

## Publication boundary

Included: reviewed source/configurations, documented protocols, scalar experiment
records and generated research figures. The three numeric NPZ/JSON response
experts in `assets/bidding-response-v1` are a specific weight-publication
exception. Original training checkpoints, optimizer/RNG state, traffic/trajectory
arrays, local orchestration state, private runner logs, credentials and environment
files are excluded. The checked source does not require a company account or service.

The publication review screened the candidate tree and reachable Git history for
credential patterns, private filesystem paths, nonpublic email addresses, private
network addresses and workplace integrations. Apparent phone-number matches were
decimal simulation metrics; path-pattern matches in exporters are detection rules.
Published PNGs have plotting-software/dpi metadata and no EXIF. Git attribution
uses public GitHub aliases and noreply addresses.

These are bounded source, metadata and pattern checks. They cannot establish the
absence of every possible secret or resolve the legal provenance of material
that has no visible source trail. No unresolved sensitive finding was accepted
for this publication.

## Reproduction limits

The source and exported aggregates are inspectable. Some research runners bind
to local protocol, verifier, input and checkpoint hashes under ignored directories.
Those original private runtime bundles are not shipped, so the frozen runs cannot
be exactly replayed from this repository alone. Missing authority/input files are
supposed to cause preflight rejection, rather than silently bypass a guard.

Use the [runbook](bidding-runbook.md) for the underlying public-simulator pipeline.
A newly generated experiment needs its own complete inputs and recorded identity;
it must not be described as a replay of an omitted frozen bundle. Configurations
and planned entry points are not evidence that a run completed. In particular,
`bidding-generative-final.toml` is an unexecuted plan; the stopped replication and
[decoder comparison](bidding-decoder-findings.md) retain their failed gates.

The new [delayed-feedback study](bidding-delayed-protocol.md) is different in
reproduction scope: its public source/configuration can generate all required
market tapes and mature training history without earlier trained weights. The
small CPU replay exercises this route. Exact numeric reproduction still requires
the recorded runtime, seeds and source bytes; scalar exports are not substitutes
for auction tapes. Its [control extension](bidding-delayed-strategy-protocol.md)
reuses only those generated development inputs. All twelve extension candidates
failed promotion, so the reserved fresh confirmation markets remain unopened.

The [expert-selection study](bidding-expert-selection-protocol.md) also supports
generation of all its new markets from a source checkout containing the repository
assets and `third_party` code. It uses the published seed-11 response asset for
frozen inference; it does not promise a complete rerun of the original expert
training. Training and validation inputs precede the frozen evaluation; reserved
confirmation inputs may be generated only after `READY_FOR_CONFIRMATION` and
the explicit opening step. The [reproduction guide](bidding-expert-reproduction.md)
defines these boundaries and valid scientific stops. This describes reproduction
scope, not an experiment outcome.

## v0.2.0 software delivery

The delivery pass applies the existing formatter and repairs configuration-test
coverage. It does not rewrite historical experiment records or claim new research
gains. Historical hashes describe the source at their recorded revisions; the
[pre-delivery research revision](https://github.com/KAMCHIU-LAB/trajectory-lab/tree/43fc0acf61e4ad13b01c95492c45e94ce711b384)
remains available. Fresh software replay outputs use new identities and output
directories. See the [reproduction guide](reproduce.md) and versioned source
record for the current supported public path.

In later research checkouts, the README replay command uses `--replay-only` to
check software replay invariants. The historical source hashes in
`releases/v0.2.0.json` must be verified from a checkout of the `v0.2.0` tag with
the verifier's full source checks enabled, without `--replay-only`.
