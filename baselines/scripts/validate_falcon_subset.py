#!/usr/bin/env python3
"""Independently validate a completed Falcon subset run using stdlib only.

Requires exact episode coverage, one evaluation per key, finite scalar metrics,
action counts, completion marker, and independently recomputed aggregates.
Produces a failed report for incomplete runs instead of silently averaging them.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

REQUIRED_METRICS = ('success', 'spl', 'psc', 'human_collision')


def validate(output):
    output = Path(output)
    failures, rows = [], []

    def load(name):
        try:
            return json.loads((output / name).read_text())
        except (OSError, ValueError) as exc:
            failures.append(name + ': ' + str(exc))
            return {}

    expected_document = load('expected_episodes.json')
    expected_keys = [tuple(key) for key in expected_document.get('episode_keys', [])]
    expected = set(expected_keys)
    if not expected or len(expected_keys) != len(expected):
        failures.append('Expected episode list is empty or contains duplicate keys')
    try:
        for i, line in enumerate((output / 'episode_metrics.jsonl').read_text().splitlines(), 1):
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('record is not an object')
                rows.append(row)
            except ValueError as exc:
                failures.append('episode log line %d: %s' % (i, exc))
    except OSError as exc:
        failures.append('episode_metrics.jsonl: ' + str(exc))
    seen = Counter()
    metric_values = {}
    actions = load('actions.json')
    expected_action_keys = set()
    for i, row in enumerate(rows):
        key = (row.get('scene_id'), row.get('episode_id'))
        seen[key] += 1
        if key not in expected:
            failures.append('Unexpected episode: ' + repr(key))
        if row.get('eval_index') != 1:
            failures.append('Episode eval_index must be 1: ' + repr(key))
        steps = row.get('action_steps')
        if type(steps) is not int or not 1 <= steps <= 500:
            failures.append('Invalid action step count: ' + repr(key))
        action_key = '%s|%s|0' % key
        expected_action_keys.add(action_key)
        trace = actions.get(action_key)
        if not isinstance(trace, list) or len(trace) != steps:
            failures.append('Action trace count does not match episode: ' + repr(key))
        metrics = row.get('metrics', {})
        if not isinstance(metrics, dict):
            failures.append('Invalid metrics object: ' + repr(key))
            continue
        for name in REQUIRED_METRICS:
            if name not in metrics:
                failures.append('Missing metric %s for %r' % (name, key))
        for name, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                failures.append('Nonfinite/nonnumeric metric %s for %r' % (name, key))
                continue
            if name in REQUIRED_METRICS and not 0 <= value <= 1:
                failures.append('Metric outside [0,1]: %s for %r' % (name, key))
            metric_values.setdefault(name, []).append(value)
    missing = sorted(expected - set(seen))
    duplicate = [list(key) for key, count in seen.items() if count != 1]
    if missing:
        failures.append('%d expected episodes missing' % len(missing))
    if duplicate:
        failures.append('%d episode keys repeated' % len(duplicate))
    if set(actions) != expected_action_keys:
        failures.append('Action trace keys differ from completed episode keys')
    recomputed = {key: math.fsum(values) / len(values) for key, values in metric_values.items()}
    aggregate = load('aggregate_metrics.json')
    if set(aggregate) != set(recomputed):
        failures.append('Aggregate metric names differ from per-episode metric union')
    for key, value in recomputed.items():
        other = aggregate.get(key)
        if not isinstance(other, (int, float)) or not math.isclose(value, other, rel_tol=1e-9, abs_tol=1e-9):
            failures.append('Aggregate mismatch: ' + key)
    reported = load('result.json')
    if all(k in recomputed for k in REQUIRED_METRICS):
        rounded = {label: round(recomputed[name], 4) for label, name in
                   [('SR', 'success'), ('SPL', 'spl'), ('PSC', 'psc'), ('H-Coll', 'human_collision')]}
        rounded['Total'] = round(0.4 * recomputed['success'] + 0.3 * recomputed['spl'] + 0.3 * recomputed['psc'], 4)
        if reported != rounded:
            failures.append('Official rounded result.json does not match independently recomputed means')
    completion = load('evaluation_complete.json')
    for field in ('completed_episode_records', 'requested_episode_count'):
        if completion.get(field) != len(expected_keys):
            failures.append('Completion count mismatch: ' + field)
    if completion.get('evals_per_episode') != 1:
        failures.append('Completion evals_per_episode is not 1')
    provenance = load('run_manifest.json')
    if provenance.get('seed') != 100 or provenance.get('policy_deterministic') is not False:
        failures.append('Run provenance lacks required seed100/stochastic protocol')
    if provenance.get('max_episode_steps') != 500:
        failures.append('Run provenance max_episode_steps changed')
    observation_manifest = load('instrumentation/observability_manifest.json')
    module = output / 'instrumentation/vln_falcon_evaluator.py'
    if not module.is_file() or hashlib.sha256(module.read_bytes()).hexdigest() != observation_manifest.get('derived_sha256'):
        failures.append('Instrumented evaluator SHA mismatch or missing')
    result = {'status': 'PASS_SUBSET_COMPLETE' if not failures else 'FAIL_INCOMPLETE_OR_INCONSISTENT',
              'scope': 'Falcon official checkpoint, specified val subset; not full-paper or training-seed reproduction',
              'expected_episode_count': len(expected_keys), 'observed_records': len(rows),
              'observed_unique_episodes': len(seen), 'missing_episode_keys': missing,
              'duplicate_episode_keys': duplicate, 'failures': failures,
              'recomputed_aggregate_metrics': recomputed,
              'num_environments': provenance.get('num_environments'),
              'seed': provenance.get('seed'), 'policy_deterministic': False}
    (output / 'validation.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'PASS_SUBSET_COMPLETE' else 2)
