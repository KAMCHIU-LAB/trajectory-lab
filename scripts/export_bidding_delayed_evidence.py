#!/usr/bin/env python3
"""Export completed delayed-feedback evidence through explicit public allowlists."""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean

OPTIONAL_JSON_FILES = ('benchmark.json', 'offline-pending-audit/manifest.json', 'offline-pending-audit/summary.json')
JSON_FILES = (
    'history.json', 'history-manifest.json', 'selection.json',
    'validation/manifest.json', 'validation/status.json', 'validation/summary.json',
    'confirmation/manifest.json', 'confirmation/status.json', 'confirmation/summary.json',
)
COLUMNS = (
    'market', 'focal', 'bias_group', 'delay', 'arrival_seed', 'method', 'estimator',
    'gain', 'prior', 'cost', 'conversions', 'exposures', 'budget', 'target_cpa',
    'cpa', 'violation', 'score', 'budget_utilization', 'pending_mae',
    'final_arrived', 'zero_conversions', 'deployable',
)
GROUP_COLUMNS = ('method', 'market', 'bias_group', 'delay')
MEAN_COLUMNS = (
    'gain', 'prior', 'cost', 'conversions', 'exposures', 'budget', 'target_cpa',
    'cpa', 'violation', 'score', 'budget_utilization', 'pending_mae',
    'final_arrived', 'zero_conversions',
)
# Keys not listed here are omitted, even if a new upstream artifact adds them.
# Dynamic keys are restricted to public market IDs, method names and source files.
JSON_KEYS = frozenset('''
description environment results median_ms p95_ms calls
aggregate coverage80 decisions mae mean_crps mean_error mean_truth_u tick
trajectories_verified interpretation audit_seed confirmation_identity input_cache_manifest
posterior_samples results_sha256 scope script_sha256 selection_identity selection_sha256 ticks
core identity history_sha256 behavior config_identity context_contract markets source
sha256 event_adapter_sha256 upstream_revision all_mature probabilities
training_sample_count training_markets maturity_window split total_delay_pseudocount
config evaluation history limits study name purpose train_markets validation_markets
confirmation_markets focal_agents pv_count actions biases arrival_seeds arrival_seed
bootstrap_draws bootstrap_seed gains methods priors probability_bin_edges regimes
calibration_prior_log_sd regime_assumption maximum_neural_configurations
maximum_neural_seeds maximum_neural_training_hours minimum_confirmation_markets
oracle_diagnostic_action_change oracle_diagnostic_gain claim phase completed_markets
rows paired_campaigns elapsed_seconds peak_rss_native validation_identity
frozen_variants strongest_point diagnostic_oracle mean_scores
oracle_action_change_fraction generative_investment_gate diagnostic_scope
selection_scope history_contract mature_only_scope estimator gain prior
coverage primary_estimand primary_comparison other_comparisons_scope selection
comparisons cells final_metrics independent_market_count method baseline
independent_unit protection_endpoints groups minimum_markets
sufficient_independent_markets business_target_pass protection_pass uplift_target_pass
status score_ratio conversion_ratio score_ratio_ci95 conversion_ratio_ci95
score_ratio_protection_lower conversion_ratio_protection_lower violation_delta
violation_delta_ci95 violation_delta_protection_upper method_means baseline_means
score conversions violation budget_utilization bias_group delay campaigns pending_mae
'''.split())
METHOD = re.compile(r'(?:naive|mature_only|delay_corrected|posterior|oracle_past)(?:-g[0-9.]+-p[0-9.]+)?\Z')
SOURCE = re.compile(r'(?:src/trajectorylab/(?:bidding/)?[A-Za-z0-9_]+\.py|BiddingEnv\.py|NeurIPSPvGen\.py|market\.py)\Z')
PUBLIC_LABELS = {'immediate', 'short', 'long', 'all', 'nominal', 'under', 'over'}
SUSPICIOUS = re.compile(
    r'(?i)(?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}'
    r'|(?:^|[\s\"\'=(:])/(?:Users|home|tmp|private|var|Volumes|mnt|root|opt)/'
    r'|\b[A-Z]:[\\/]|(?:^|[\s\"\'=])~/'
    r'|\b(?:api[_ -]?key|access[_ -]?token|password|secret[_ -]?key)\s*[:=]'
    r'|\b(?:sk-[A-Za-z0-9_-]{12,}|gh[pousr]_[A-Za-z0-9]{20,})'
    r'|-----BEGIN [A-Z ]*PRIVATE KEY-----)'
)


def check_string(value: str):
    inspected = re.sub(r'https://[^\s\"<>]+', '', value)
    if SUSPICIOUS.search(value) or re.search(r'(?:^|[\s\"\'=(:])/[A-Za-z0-9._-]+/', inspected) or '\x00' in value:
        # Do not print the sensitive value in an error or public log.
        raise ValueError('public export contains a suspicious path, email, or credential')
    # A standalone absolute POSIX path is also forbidden beyond known prefixes.
    if value.startswith('/') or 'file://' in value.lower():
        raise ValueError('public export contains an absolute local path')


def public_json(value):
    """Recursive allowlist, retaining public dynamic map keys only."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in JSON_KEYS or key in PUBLIC_LABELS or key.isdecimal() or METHOD.fullmatch(key) or SOURCE.fullmatch(key):
                check_string(key)
                result[key] = public_json(item)
        return result
    if isinstance(value, list):
        return [public_json(item) for item in value]
    if isinstance(value, str):
        check_string(value)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('public JSON contains a nonfinite number')
    elif value is not None and not isinstance(value, (int, float, bool)):
        raise ValueError('unsupported public JSON value')
    return value


def read_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def scalar_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('campaign results must be a nonempty list')
    result = []
    seen = set()
    string_columns = {'bias_group', 'delay', 'method', 'estimator'}
    for original in rows:
        row = {column: original[column] for column in COLUMNS}
        for column, value in row.items():
            if column in string_columns:
                if not isinstance(value, str):
                    raise ValueError('invalid campaign label')
                check_string(value)
            elif value is None and column == 'cpa':
                continue
            elif not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                raise ValueError('campaign contains invalid scalar')
        if row['bias_group'] not in {'under', 'nominal', 'over'} or row['delay'] not in {'immediate', 'short', 'long'} or not METHOD.fullmatch(row['method']) or not METHOD.fullmatch(row['estimator']):
            raise ValueError('unknown public campaign label')
        key = tuple(row[k] for k in ('method', 'market', 'focal', 'bias_group', 'delay', 'arrival_seed'))
        if key in seen:
            raise ValueError('duplicate public campaign')
        seen.add(key)
        result.append(row)
    return sorted(result, key=lambda r: tuple(r[k] for k in ('method', 'market', 'bias_group', 'delay', 'focal', 'arrival_seed')))


def market_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in GROUP_COLUMNS)].append(row)
    result = []
    for key, campaigns in sorted(grouped.items()):
        row = dict(zip(GROUP_COLUMNS, key, strict=True))
        row.update(campaign_count=len(campaigns), focal_count=len({r['focal'] for r in campaigns}), arrival_seed_count=len({r['arrival_seed'] for r in campaigns}))
        row['cpa_nonnull_count'] = sum(r['cpa'] is not None for r in campaigns)
        for column in MEAN_COLUMNS:
            values = [r[column] for r in campaigns if r[column] is not None]
            row[column] = fmean(values) if values else None
        result.append(row)
    return result


README = '''# Delayed-feedback public evidence

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
'''


def write_csv(path, rows, columns):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def export_evidence(input_dir: Path, output_dir: Path):
    """Validate all permitted payloads before creating public output files."""
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if not (input_dir / 'confirmation/summary.json').is_file():
        raise ValueError('confirmation summary is required before public export')
    selected_json = (*JSON_FILES, *(name for name in OPTIONAL_JSON_FILES if (input_dir / name).is_file()))
    payloads = {name: public_json(read_json(input_dir / name)) for name in selected_json}
    for split in ('validation', 'confirmation'):
        if payloads[f'{split}/status.json'].get('phase') != 'complete':
            raise ValueError('both evaluation splits must be complete')
    campaigns = {split: scalar_rows(read_json(input_dir / split / 'results.json')) for split in ('validation', 'confirmation')}
    aggregates = {split: market_rows(rows) for split, rows in campaigns.items()}
    # Inspect the destination without deleting anything. Unknown stale files or
    # symlinks could otherwise make a clean allowlisted export look publishable.
    fixed_outputs = {'validation.csv', 'confirmation.csv', 'validation-markets.csv', 'confirmation-markets.csv', 'README.md', 'export-manifest.json'}
    selected_outputs = set(selected_json) | fixed_outputs
    owned_dirs = {str(Path(name).parent) for name in selected_outputs} - {'.'}
    if output_dir.is_symlink():
        raise ValueError('output directory must not be a symlink')
    if output_dir.exists():
        for existing in output_dir.rglob('*'):
            relative = existing.relative_to(output_dir).as_posix()
            if existing.is_symlink() or (existing.is_file() and relative not in selected_outputs) or (existing.is_dir() and relative not in owned_dirs):
                raise ValueError('output contains an unexpected existing file or directory')
    # No directory copying: every output name is owned by this exporter.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        destination = output_dir / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for split in ('validation', 'confirmation'):
        write_csv(output_dir / f'{split}.csv', campaigns[split], COLUMNS)
        columns = (*GROUP_COLUMNS, 'campaign_count', 'focal_count', 'arrival_seed_count', 'cpa_nonnull_count', *MEAN_COLUMNS)
        write_csv(output_dir / f'{split}-markets.csv', aggregates[split], columns)
    (output_dir / 'README.md').write_text(README, encoding='utf-8')
    import hashlib
    files = [*selected_json, 'validation.csv', 'confirmation.csv', 'validation-markets.csv', 'confirmation-markets.csv', 'README.md']
    manifest = {'schema': 'delayed-feedback-public-export-v1', 'files': {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in sorted(files)}}
    (output_dir / 'export-manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('artifacts/bidding/delayed-feedback'))
    parser.add_argument('--output', type=Path, default=Path('reports/bidding-delayed-feedback'))
    args = parser.parse_args()
    export_evidence(args.input, args.output)


if __name__ == '__main__':
    main()
