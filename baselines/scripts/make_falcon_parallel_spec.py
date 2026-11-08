#!/usr/bin/env python3
"""Prepare a 2-GPU/18-CPU managed spec for two disjoint 8-env Falcon shards.

No SSH, environment installation, freezing, job allocation or submission occurs.
The original single-shard generator audits the same inputs for each shard.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

from make_falcon_eval_spec import EVAL_OUTPUTS, DEFAULT_REMOTE, LOCAL


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest-a', type=Path, required=True)
    p.add_argument('--manifest-b', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', required=True)
    p.add_argument('--remote-root', default=DEFAULT_REMOTE)
    p.add_argument('--remote-python')
    p.add_argument('--env-json', type=Path)
    p.add_argument('--readiness-file', type=Path, action='append', default=[])
    p.add_argument('--wall-seconds', type=int, default=3600)
    p.add_argument('--output-budget-mib', type=int, default=416)
    p.add_argument('--min-available-memory-gib', type=int, default=48)
    a = p.parse_args()
    output = a.output.absolute()
    plan_path = output.with_name(output.stem + '_pair_plan.json')
    if output.exists() or plan_path.exists():
        p.error('Refusing to overwrite a spec or pair plan')
    if a.output_budget_mib < 384:
        p.error('Two-shard output budget must be at least 384MiB')
    if a.min_available_memory_gib < 48:
        p.error('Current pair plan requires at least 48GiB available memory')
    if not a.readiness_file:
        p.error('Supply at least one environment/readiness evidence file to freeze')
    validation_seconds = 240
    science_seconds = a.wall_seconds - 300
    if science_seconds < 1:
        p.error('Insufficient wall time for the pair scientific unit')
    paths = [a.manifest_a.absolute(), a.manifest_b.absolute()]
    docs = [json.loads(path.read_text()) for path in paths]
    ids = [doc.get('shard_id') for doc in docs]
    if any(not isinstance(i, str) or not re.fullmatch(r'balanced4_\d{2}', i) for i in ids) or len(set(ids)) != 2:
        p.error('Supply two distinct balanced4 manifests')
    keysets = [set(map(tuple, doc['episode_keys'])) for doc in docs]
    if keysets[0] & keysets[1]:
        p.error('Input shard episode keys overlap')
    if set(docs[0]['content_scenes']) & set(docs[1]['content_scenes']):
        p.error('Input shards must not share scenes or navmesh cache paths')
    singles = []
    with tempfile.TemporaryDirectory(prefix='falcon-pair-spec-') as tmp:
        for index, path in enumerate(paths):
            target = Path(tmp) / ('single_%d.json' % index)
            argv = [sys.executable, str(LOCAL / 'baselines/scripts/make_falcon_eval_spec.py'),
                    '--manifest', str(path), '--output', str(target), '--deadline', a.deadline,
                    '--num-envs', '8', '--cpus', '9', '--remote-root', a.remote_root,
                    '--wall-seconds', str(a.wall_seconds), '--science-timeout-seconds', str(science_seconds),
                    '--output-budget-mib', '192']
            if a.remote_python:
                argv += ['--remote-python', a.remote_python]
            if a.env_json:
                argv += ['--env-json', str(a.env_json.absolute())]
            for evidence in a.readiness_file:
                argv += ['--readiness-file', str(evidence.absolute())]
            generated = subprocess.run(argv, capture_output=True, text=True)
            if generated.returncode:
                raise RuntimeError('Single-shard input audit failed: ' + generated.stderr)
            singles.append(json.loads(target.read_text()))
    first, second = singles
    for field in ['remote_root', 'remote_python', 'gpu_binding', 'remote_inputs']:
        if first[field] != second[field]:
            p.error('Pair shard common input contract differs: ' + field)
    if first['science']['env'] != second['science']['env']:
        p.error('Pair runtime environments differ')
    runtime = {'python': first['remote_python'], 'repo': first['science']['cwd']}
    child_argv = first['science']['argv']
    for flag, name in [('--checkpoint', 'checkpoint'), ('--pretrained', 'pretrained'),
                       ('--data-path', 'data_path'), ('--scenes-dir', 'scenes_dir')]:
        runtime[name] = child_argv[child_argv.index(flag) + 1]
        other = second['science']['argv']
        if runtime[name] != other[other.index(flag) + 1]:
            p.error('Pair runtime path differs: ' + name)
    run_id = ('vln_falcon_pair' + '_'.join(i.removeprefix('balanced4_') for i in ids) + '_'
              + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    scripts = ['run_falcon_subset.py', 'prepare_falcon_observed.py', 'validate_falcon_subset.py',
               'run_falcon_parallel.py', 'validate_falcon_parallel.py']
    input_by_path = {row['path']: row for row in first['remote_inputs']}
    plan = {'schema': 'falcon_parallel_pair_v1', 'run_id': run_id, 'node_list': ['10-0-1-11'],
            'remote_output_directory': str(Path(first['remote_root']) / ('result_' + run_id)),
            'runtime': runtime, 'num_environments_per_child': 8,
            'episode_count': sum(d['episode_count'] for d in docs),
            'scene_count': sum(d['scene_count'] for d in docs),
            'runtime_script_sha256': {name: sha(LOCAL / 'baselines/scripts' / name) for name in scripts},
            'runtime_inputs': {key: input_by_path[runtime[key]] for key in ['checkpoint', 'pretrained']},
            'children': [{'shard_id': doc['shard_id'], 'manifest_file': 'episode_manifest_' + doc['shard_id'] + '.json',
                          'manifest_sha256': sha(path), 'episode_count': doc['episode_count'],
                          'scene_count': doc['scene_count']} for doc, path in zip(docs, paths)],
            'scope': 'Two disjoint original full-val scene shards, each with official 8 environments; pair completion is not full1087 completion'}
    output.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    budget = a.output_budget_mib << 20
    spec = dict(first)
    spec.update(run_id=run_id, min_available_memory_bytes=a.min_available_memory_gib << 30,
                slurm={**first['slurm'], 'gpus': 2, 'cpus': 18, 'node_list': ['10-0-1-11']},
                output_budget_bytes=budget, temporary_transfer_bytes=budget,
                remote_peak_additional_bytes=2 * budget + (8 << 20))
    spec['uploads'] = [{'file': str(LOCAL / 'baselines/scripts' / name), 'remote_name': name} for name in scripts]
    spec['uploads'] += [{'file': str(path), 'remote_name': child['manifest_file']}
                        for path, child in zip(paths, plan['children'])]
    spec['uploads'].append({'file': str(plan_path), 'remote_name': 'parallel_pair_plan.json'})
    scientific = first['scientific_files'] + second['scientific_files'] + [str(Path(__file__).resolve()), str(plan_path)]
    spec['scientific_files'] = list(dict.fromkeys(scientific))
    required = ['shards/' + child['shard_id'] + '/' + name for child in plan['children'] for name in EVAL_OUTPUTS]
    required += ['logs/' + child['shard_id'] + '.log' for child in plan['children']]
    required += ['parallel_plan.json', 'parallel_status.json', 'parallel_validation.json']
    spec['science'] = {**first['science'],
        'argv': [first['remote_python'], '{source}/run_falcon_parallel.py',
                 '--plan', '{source}/parallel_pair_plan.json', '--output', '{output}'],
        'validator_argv': [first['remote_python'], '{source}/validate_falcon_parallel.py',
                          '--plan', '{source}/parallel_pair_plan.json', '--output', '{output}'],
        'timeout_seconds': science_seconds, 'validation_timeout_seconds': validation_seconds,
        'required_outputs': required}
    spec['evaluation_plan'] = {**first['evaluation_plan'], 'mode': 'parallel_pair', 'shard_ids': ids,
        'episode_count': plan['episode_count'], 'scene_count': plan['scene_count'],
        'num_environments_per_child': 8, 'num_environments_total': 16, 'gpus': 2,
        'gpu_mapping': 'Parent worker verifies two allocated UUIDs; each child inherits exactly one distinct UUID',
        'scope_note': plan['scope'], 'memory_gate': 'Require >=48GiB available before model load; estimate, not measured 2x8-env peak',
        'output_estimate_rule': '192MiB per shard + 32MiB parent/status/log allowance by default',
        'parent_input_manifests': [str(path) for path in paths]}
    output.write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps({'status': 'SPEC_ONLY_NOT_FROZEN_NOT_SUBMITTED', 'spec': str(output),
                      'pair_plan': str(plan_path), 'run_id': run_id, 'shards': ids,
                      'episodes': plan['episode_count'], 'gpus': 2, 'cpus': 18,
                      'num_envs_each': 8, 'node_list': spec['slurm']['node_list'],
                      'required_outputs': len(required), 'output_budget_mib': a.output_budget_mib}, indent=2))


if __name__ == '__main__':
    main()
