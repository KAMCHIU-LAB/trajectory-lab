# Delayed-feedback public evidence

This is a new, public mechanistic study in a custom fixed-opponent auction
simulation. The public AuctionNet generator and settlement reference originate
from [AuctionNet](https://github.com/alimama-tech/AuctionNet); source revisions
and hashes are recorded in the manifests. Synthetic feedback delays are a
controlled observation mechanism, not measured industrial conversion delays.
The repository's upstream license and attribution requirements still apply.

Campaign CSVs contain only public simulation indices and scalar diagnostics;
action traces, event caches, logs, local paths and historical weights are absent.
Validation includes the declared tuning candidates; confirmation contains the
frozen selections. Oracle-past is an unavailable-information diagnostic and is
not a deployable policy or a global optimum. No neural generator is established
by this export.

Business outcomes use fully matured labels at H+W, with no new auction
opportunities after H. Pending MAE is an offline diagnostic against settled
truth, not an input available to the policy. Each stable deployment context uses
only its matched mature TRAIN history to fit delay; this assumes stable delay
and does not establish identification under cold start or drift.

Raw campaign rows and repeated focal/arrival seeds are not independent statistical
units. Independent uncertainty is clustered by market. The `*-markets.csv` files
average each metric across focal/seed rows within (method, market, bias_group,
delay), retaining sample counts. CPA averages exclude undefined zero-conversion
CPA and include `cpa_nonnull_count`; this is mean campaign CPA, not pooled
cost/conversions. Violation is the mean campaign violation indicator.

Exact new-study replay requires the recorded seed/configuration, source revision
and source hashes (including the public upstream generator), plus matching
runtime dependencies. Use the repository delayed-feedback experiment commands
and smoke replay; these exported scalar tables alone are not auction tapes.
No historical trained weights are needed for this new study. Manifests preserve
source-run identities, not a claim that exported JSON after allowlisting has the
same content hash as its source file. The export manifest hashes the actual
public files. Numeric findings and uncertainty belong to the completed study
report, not to this attribution note.
