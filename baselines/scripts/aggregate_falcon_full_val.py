#!/usr/bin/env python3
"""Aggregate only a complete, validated disjoint union of full-val Falcon shards."""
import argparse
import json
import math
from pathlib import Path

from falcon_full_val_integrity import CONFIG_PATH_EXCEPTIONS, LIMITATIONS, collect, record


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--full-manifest', type=Path, required=True)
    p.add_argument('--eval-output', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    output = a.output.resolve()
    if output.exists():
        p.error('Refusing to overwrite an existing aggregate directory')
    if any(Path(path).resolve() in output.parents for path in a.eval_output):
        p.error('Aggregate output must be outside the immutable shard evidence directories')
    output.mkdir(parents=True)
    try:
        data = collect(a.full_manifest, a.eval_output)
        rows = sorted(data['rows'], key=lambda row: (row['scene_id'], row['episode_id']))
        count = len(rows)
        aggregate = {name: math.fsum(row['metrics'][name] for row in rows) / count
                     for name in sorted(rows[0]['metrics'])}
        official = {label: aggregate[name] for label, name in
                    [('SR', 'success'), ('SPL', 'spl'), ('PSC', 'psc'), ('H-Coll', 'human_collision')]}
        official['Total'] = 0.4 * official['SR'] + 0.3 * official['SPL'] + 0.3 * official['PSC']
        save(output / 'aggregate_metrics.json', aggregate)
        save(output / 'official_metrics.json', official)
        save(output / 'result.json', {key: round(value, 4) for key, value in official.items()})
        save(output / 'coverage.json', data['coverage'])
        with (output / 'episode_metrics.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        manifest = {key: data[key] for key in ('full_manifest', 'shards', 'protocol', 'protocol_sha256')}
        manifest.update(schema='falcon_full_val_shard_aggregate_v1',
                        scope='One released Falcon checkpoint, complete 1087-episode/56-scene val coverage through independently restarted shards',
                        full_paper_reproduction=False, independent_training_seeds=0,
                        checkpoint_count=1, episode_weighting='Each unique episode has weight 1/1087; shard means are never averaged',
                        total_definition='0.4*SR + 0.3*SPL + 0.3*PSC', limitations=LIMITATIONS,
                        config_comparison_exceptions=list(CONFIG_PATH_EXCEPTIONS),
                        aggregate_artifacts=[record(output / name) for name in
                            ('aggregate_metrics.json', 'official_metrics.json', 'result.json', 'coverage.json', 'episode_metrics.jsonl')])
        save(output / 'aggregation_manifest.json', manifest)
        from validate_falcon_full_val_aggregate import validate
        result = validate(a.full_manifest, a.eval_output, output)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'PASS_FULL_VAL_SHARD_AGGREGATE' else 2
    except Exception as error:
        result = {'status': 'FAIL_NO_VALID_FULL_VAL_AGGREGATE', 'error_type': type(error).__name__,
                  'error': str(error), 'full_paper_reproduction': False}
        save(output / 'aggregation_failure.json', result)
        print(json.dumps(result, indent=2))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
