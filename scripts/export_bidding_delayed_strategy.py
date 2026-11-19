#!/usr/bin/env python3
"""Strict lossless public export of the sealed, failed development extension."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean

from trajectorylab import bidding_delayed_strategy as strategy
from trajectorylab.utils import fingerprint, read_json

_SPEC = importlib.util.spec_from_file_location('delayed_public_string_checks', Path(__file__).with_name('export_bidding_delayed_evidence.py'))
_CHECKS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CHECKS)
check_string = _CHECKS.check_string

JSON_FILES = ('selection.json', 'validation/manifest.json', 'validation/summary.json', 'validation/status.json')
COLUMNS = ('market', 'focal', 'bias_group', 'delay', 'arrival_seed', 'method', 'cost', 'conversions', 'score',
           'exposures', 'budget', 'target_cpa', 'cpa', 'violation', 'budget_utilization', 'posterior_pending_mae', 'deployable')
METRICS = ('cost', 'conversions', 'score', 'exposures', 'budget', 'target_cpa', 'cpa', 'violation', 'budget_utilization', 'posterior_pending_mae')
OUTPUTS = (*JSON_FILES, 'validation.csv', 'validation-markets.csv', 'validation-cells.csv', 'README.md', 'export-manifest.json')
KEYS = frozenset('''
actions arrival_seeds bias_violation bias_vs_naive biases bootstrap_draws bootstrap_seed
budget_utilization calibration_prior_log_sd calibration_threshold config config_sha256
confirmation_markets confirmation_scope conversion_ratio conversions development_identity
development_results_sha256 eta evaluation extension factor_lower factor_upper family focal_agents
frozen_core gain gate gates history history_sha256 identity information_threshold macro_uplift
macro_vs_stronger market_means market_wins maturity_window means minimum_confirmation_markets
minimum_dev_wins name nominal_under_retention over_cell_retention over_cells_vs_naive passed
pending prior probability_bin_edges protection_endpoints provenance purpose pv_count regimes
registration_sha256 score score_ratio selected source stop_reason strategy stronger_fixed_reference
study table threshold train_markets training_history_identity validation_markets variant violation
violation_delta vs_fixed_references core event_adapter_sha256 markets methods selection_identity
sha256 split upstream_revision elapsed_seconds phase rows
'''.split())
DYNAMIC = frozenset(['under', 'nominal', 'over', 'immediate', 'short', 'long', 'fixed-naive', 'fixed-posterior',
                     *map(str, range(2001, 2007)),
                     'scripts/run_bidding_delayed_strategy.py', 'BiddingEnv.py', 'NeurIPSPvGen.py', 'market.py',
                     'src/trajectorylab/bidding/learning.py', 'src/trajectorylab/bidding/market.py',
                     *[f'src/trajectorylab/{name}.py' for name in ('bidding_delayed_experiment', 'bidding_delayed_feedback', 'bidding_delayed_statistics', 'bidding_pending_response', 'bidding_delayed_strategy')]])


def strict_public(value):
    """Reject unknown fields; preserve every accepted scientific field exactly."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key not in KEYS and key not in DYNAMIC:
                raise ValueError('unknown public JSON key')
            check_string(key)
            strict_public(item)
    elif isinstance(value, list):
        for item in value:
            strict_public(item)
    elif isinstance(value, str):
        check_string(value)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite public number')
    elif value is not None and not isinstance(value, (int, float, bool)):
        raise ValueError('unsupported public value')
    return value


def public_rows(rows):
    labels = {'method', 'bias_group', 'delay'}
    methods = {v['name'] for v in strategy.variants()}
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty campaign list required')
    for row in rows:
        if set(row) != set(COLUMNS):
            raise ValueError('unknown or missing scalar campaign field')
        for key, value in row.items():
            if key in labels:
                if not isinstance(value, str):
                    raise ValueError('invalid campaign label')
                check_string(value)
            elif value is None and key == 'cpa':
                continue
            elif not isinstance(value, (int, float, bool)) or not math.isfinite(value):
                raise ValueError('nonfinite or nonscalar campaign value')
        if row['method'] not in methods or row['bias_group'] not in {'under', 'nominal', 'over'} or row['delay'] not in {'immediate', 'short', 'long'}:
            raise ValueError('unknown campaign label')
    return sorted(rows, key=lambda r: tuple(r[k] for k in ('method', 'market', 'bias_group', 'delay', 'focal', 'arrival_seed')))


def aggregate(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    result = []
    for group, items in sorted(grouped.items()):
        row = dict(zip(keys, group, strict=True))
        row['campaign_count'] = len(items)
        row['independent_market_count'] = len({r['market'] for r in items})
        row['focal_count'] = len({r['focal'] for r in items})
        row['arrival_seed_count'] = len({r['arrival_seed'] for r in items})
        row['cpa_nonnull_count'] = sum(r['cpa'] is not None for r in items)
        for metric in METRICS:
            # Balanced registered campaign coverage makes these equal-market
            # means for fully defined metrics; CPA is conditional on being defined.
            # Grid verification below rejects missing campaigns.
            values = [r[metric] for r in items if r[metric] is not None]
            row[metric] = fmean(values) if values else None
        result.append(row)
    return result


def verify_run(payloads, rows, config_path, history_dir):
    selection = payloads['selection.json']
    manifest, summary, status = (payloads[f'validation/{name}.json'] for name in ('manifest', 'summary', 'status'))
    if selection['identity'] != fingerprint({k:v for k,v in selection.items() if k != 'identity'}):
        raise ValueError('selection identity mismatch')
    if manifest['identity'] != fingerprint(manifest['core']):
        raise ValueError('run manifest identity mismatch')
    run_id = manifest['identity']
    if selection['development_identity'] != run_id or summary['identity'] != run_id or status['identity'] != run_id:
        raise ValueError('run identities disagree')
    if status['phase'] != 'complete' or status['rows'] != len(rows):
        raise ValueError('development run is incomplete')
    if fingerprint(rows) != selection['development_results_sha256']:
        # Results are fingerprinted in original run order, before CSV sorting.
        raise ValueError('source result fingerprint mismatch')
    if selection['gate'] or selection['selected'] is not None or len(selection['table']) != 12 or any(r['passed'] for r in selection['table']):
        raise ValueError('this exporter requires the sealed twelve-failure development study')
    provenance = selection['provenance']
    if any(manifest['core'].get(k) != v for k,v in provenance.items()):
        raise ValueError('selection and run provenance disagree')
    if provenance['source'] != strategy.source_record(config_path) or provenance['config'] != strategy.base.load_config(config_path):
        raise ValueError('current source/config/protocol differs from frozen run')
    history_manifest = read_json(history_dir/'history-manifest.json')
    if history_manifest['identity'] != fingerprint(history_manifest['core']) or history_manifest['identity'] != provenance['training_history_identity'] or history_manifest['core']['source'] != provenance['source']['frozen_core']:
        raise ValueError('training history manifest mismatch')
    if strategy.file_hash(history_dir/'history.json') != provenance['history_sha256'] or history_manifest['history_sha256'] != provenance['history_sha256']:
        raise ValueError('training history fingerprint mismatch')
    expected_summary = {k:v for k,v in selection.items() if k not in {'identity', 'provenance', 'development_identity', 'development_results_sha256'}}
    if summary != dict(identity=run_id, **expected_summary):
        raise ValueError('selection and scientific summary disagree')
    config = provenance['config']
    strategy.validate_grid(rows, config, config['study']['validation_markets'], strategy.variants())


README = '''# Bounded feedback-use development extension

All twelve preregistered configurations failed the development gate. No candidate
was selected, and no fresh confirmation was opened for this extension. These are
selection-stage observations from six previously seen development markets, not
independent confirmation or a claim of a winning policy. No additional search
or new performance claim follows from this package.

The lossless JSON preserves selection, protocol/source/history and run identities.
The CSV retains all scalar campaign rows; market/cell tables provide means and
counts for independent checks. Focal agents and arrival seeds are repeated
conditions, not independent statistical units; the market is the independent unit.
Business metrics use fully matured labels at H+W, with no new opportunities after
H. Posterior pending MAE is an offline diagnostic, not future input to the policy.
CPA means omit undefined zero-conversion CPA, with cpa_nonnull_count recorded;
they are campaign CPA means, not pooled cost/conversions.

The experiment uses the same public-source fixed-opponent AuctionNet-based
simulation and matched mature TRAIN delay history as the first study. It is a
controlled synthetic-delay mechanism study, not industrial validation. See the
repository source attribution and license, and the registered extension protocol.
No old trained weights, event arrays, action traces, raw caches or private logs
are included. Seeds, frozen configuration and the recorded source revisions are
required for replay; scalar exports alone are not auction tapes. Calibration
posteriors assume stable matched delay and do not imply CPA chance guarantees.

Original identities refer to the source experiment artifacts. The export manifest
hashes the actual public files, independently of those source-run identities.
'''


def export_evidence(input_dir, output_dir, config_path, history_dir):
    input_dir, output_dir, config_path, history_dir = map(Path, (input_dir, output_dir, config_path, history_dir))
    payloads = {name:strict_public(read_json(input_dir/name)) for name in JSON_FILES}
    raw_rows = read_json(input_dir/'validation/results.json')
    rows = public_rows(raw_rows)
    verify_run(payloads, raw_rows, config_path, history_dir)
    market_keys, cell_keys = ('method', 'market', 'bias_group', 'delay'), ('method', 'bias_group', 'delay')
    markets, cells = aggregate(rows, market_keys), aggregate(rows, cell_keys)
    allowed_dirs = {'validation'}
    if output_dir.is_symlink():
        raise ValueError('output symlink forbidden')
    if output_dir.exists():
        for path in output_dir.rglob('*'):
            relative = path.relative_to(output_dir).as_posix()
            if path.is_symlink() or (path.is_file() and relative not in OUTPUTS) or (path.is_dir() and relative not in allowed_dirs):
                raise ValueError('unexpected existing output contents or symlink')
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        target = output_dir/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    counts = ('campaign_count', 'independent_market_count', 'focal_count', 'arrival_seed_count', 'cpa_nonnull_count')
    for name, data, columns in [('validation.csv', rows, COLUMNS), ('validation-markets.csv', markets, (*market_keys, *counts, *METRICS)), ('validation-cells.csv', cells, (*cell_keys, *counts, *METRICS))]:
        with (output_dir/name).open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    (output_dir/'README.md').write_text(README, encoding='utf-8')
    manifest = dict(schema='delayed-strategy-development-export-v1', selection_identity=payloads['selection.json']['identity'],
                    run_identity=payloads['validation/manifest.json']['identity'],
                    files={name:hashlib.sha256((output_dir/name).read_bytes()).hexdigest() for name in OUTPUTS if name != 'export-manifest.json'})
    (output_dir/'export-manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n', encoding='utf-8')
    return dict(files=len(OUTPUTS), campaigns=len(rows), market_cells=len(markets), cells=len(cells),
                bytes=sum((output_dir/name).stat().st_size for name in OUTPUTS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('artifacts/bidding/delayed-strategy'))
    parser.add_argument('--output', type=Path, default=Path('reports/bidding-delayed-strategy'))
    parser.add_argument('--config', type=Path, default=Path('configs/bidding-delayed-strategy.toml'))
    parser.add_argument('--history', type=Path, default=Path('artifacts/bidding/delayed-feedback'))
    args = parser.parse_args()
    print(json.dumps(export_evidence(args.input, args.output, args.config, args.history), sort_keys=True))


if __name__ == '__main__':
    main()
