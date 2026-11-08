#!/usr/bin/env python3
"""Re-read real shard episodes and independently verify a full-val aggregate."""
import argparse
from collections import Counter
from fractions import Fraction
import json
import math
from pathlib import Path

from falcon_full_val_integrity import CONFIG_PATH_EXCEPTIONS, LIMITATIONS, collect, digest, load, require


def validate(full_manifest, eval_outputs, output):
    output = Path(output)
    failures, official = [], None
    try:
        # Re-run exact shard coverage/protocol checks and original validators;
        # none of their collected validation.json files are modified.
        data = collect(full_manifest, eval_outputs)
        manifest = load(output / 'aggregation_manifest.json')
        require(manifest['schema'] == 'falcon_full_val_shard_aggregate_v1', 'Unexpected aggregate schema')
        require(manifest['full_paper_reproduction'] is False and manifest['checkpoint_count'] == 1 and
                manifest['independent_training_seeds'] == 0 and manifest['limitations'] == LIMITATIONS,
                'Aggregate evidence scope or required limitations changed')
        require(manifest['config_comparison_exceptions'] == list(CONFIG_PATH_EXCEPTIONS) and
                manifest['total_definition'] == '0.4*SR + 0.3*SPL + 0.3*PSC',
                'Config comparison exception list or official Total formula changed')
        for name in ('full_manifest', 'shards', 'protocol', 'protocol_sha256'):
            require(manifest[name] == data[name], 'Input/protocol evidence changed: ' + name)
        require(load(output / 'coverage.json') == data['coverage'], 'Reported full coverage differs from source episodes')
        expected = sorted(data['rows'], key=lambda row: (row['scene_id'], row['episode_id']))
        actual = [json.loads(line) for line in (output / 'episode_metrics.jsonl').read_text().splitlines()]
        require(Counter((r['scene_id'], r['episode_id']) for r in actual) ==
                Counter((r['scene_id'], r['episode_id']) for r in expected), 'Merged output key multiset changed')
        require(actual == expected, 'Merged per-episode values/provenance differ from original shard rows')
        n = len(expected)
        columns = sorted(expected[0]['metrics'])
        reported = load(output / 'aggregate_metrics.json')
        require(set(reported) == set(columns), 'Aggregate metric columns differ')
        # Sum exact binary values as rational numbers, independently of the
        # writer's fsum reduction. Permit only final floating rounding noise.
        exact_means = {key: float(sum((Fraction(row['metrics'][key]) for row in expected), Fraction()) / n)
                       for key in columns}
        for key, value in exact_means.items():
            require(type(reported[key]) in (int, float) and math.isfinite(reported[key]) and
                    math.isclose(reported[key], value, rel_tol=1e-12, abs_tol=1e-12),
                    'Per-episode weighted metric mismatch: ' + key)
        official = {label: math.fsum(row['metrics'][key] for row in expected) / n for label, key in
                    [('SR', 'success'), ('SPL', 'spl'), ('PSC', 'psc'), ('H-Coll', 'human_collision')]}
        official['Total'] = 0.4 * official['SR'] + 0.3 * official['SPL'] + 0.3 * official['PSC']
        require(load(output / 'official_metrics.json') == official, 'Official scalar metrics or Total mismatch')
        require(load(output / 'result.json') == {key: round(value, 4) for key, value in official.items()},
                'Official four-decimal result mismatch')
        names = ('aggregate_metrics.json', 'official_metrics.json', 'result.json', 'coverage.json', 'episode_metrics.jsonl')
        records = manifest['aggregate_artifacts']
        require(len(records) == len(names) and {Path(r['path']).name for r in records} == set(names), 'Aggregate artifact set differs')
        for entry in records:
            file = output / Path(entry['path']).name
            require(entry['bytes'] == file.stat().st_size and entry['sha256'] == digest(file), 'Aggregate artifact digest changed')
    except Exception as error:
        failures.append(type(error).__name__ + ': ' + str(error))
    result = {'status': 'FAIL_FULL_VAL_AGGREGATE' if failures else 'PASS_FULL_VAL_SHARD_AGGREGATE',
              'failures': failures, 'full_paper_reproduction': False,
              'expected_episode_count': 1087, 'expected_scene_count': 56,
              'official_metrics': official if not failures else None, 'limitations': LIMITATIONS}
    (output / 'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-manifest', type=Path, required=True)
    p.add_argument('--eval-output', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = validate(a.full_manifest, a.eval_output, a.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS_FULL_VAL_SHARD_AGGREGATE' else 2)
