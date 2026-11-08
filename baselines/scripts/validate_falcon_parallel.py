#!/usr/bin/env python3
"""Validate two Falcon shards independently and require their exact disjoint union."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from validate_falcon_subset import validate


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_pair(output, plan_path, require_final_wrapper=True):
    output, plan_path = Path(output).absolute(), Path(plan_path).absolute()
    failures, reports, source_union, runtime_union, all_rows = [], [], set(), set(), []
    plan = json.loads(plan_path.read_text())
    wrapper = json.loads((output / 'parallel_status.json').read_text())
    remote_output = Path(plan['remote_output_directory'])
    if (not remote_output.is_absolute()
            or wrapper.get('output_directory') != str(remote_output)):
        failures.append('Recorded remote output root differs from frozen pair plan')
    allowed = ['PARALLEL_COMPLETED'] if require_final_wrapper else ['CHILDREN_COMPLETED']
    if wrapper.get('status') not in allowed:
        failures.append('Wrapper did not reach the required completion state')
    if wrapper.get('plan_sha256') != sha(plan_path) or sha(output / 'parallel_plan.json') != sha(plan_path):
        failures.append('Frozen pair plan differs from executed plan')
    if len(plan.get('children', [])) != 2 or len(wrapper.get('children', [])) != 2:
        failures.append('Exactly two planned and executed children required')
    expected_devices = wrapper.get('allocated_gpu_uuids', [])
    if len(expected_devices) != 2 or len(set(expected_devices)) != 2:
        failures.append('Two distinct allocated GPU UUIDs required')
    for index, row in enumerate(plan.get('children', [])):
        if not re.fullmatch(r'balanced4_\d{2}', row.get('shard_id', '')):
            failures.append('Unsafe shard ID in frozen plan')
            continue
        child_dir = output / 'shards' / row['shard_id']
        child_failures = []
        try:
            # The frozen plan's digest anchors the original manifest bytes.
            # Its recovered copy is sufficient; runtime upload names need not
            # exist beside a locally relocated plan or result directory.
            manifest_path = child_dir / 'source_episode_manifest.json'
            if sha(manifest_path) != row['manifest_sha256']:
                raise ValueError('Recovered source manifest differs from frozen input SHA')
            doc = json.loads(manifest_path.read_text())
            expected_source = [tuple(k) for k in doc['episode_keys']]
            expected_runtime = [(str(Path(plan['runtime']['scenes_dir']) /
                                     scene.removeprefix('data/scene_datasets/')), ep)
                                for scene, ep in expected_source]
            if len(set(expected_source)) != row['episode_count'] or len(expected_source) != row['episode_count']:
                child_failures.append('Frozen source count/uniqueness differs from pair plan')
            if source_union & set(expected_source) or runtime_union & set(expected_runtime):
                child_failures.append('Episode keys overlap another shard')
            source_union.update(expected_source)
            runtime_union.update(expected_runtime)
            expected = json.loads((child_dir / 'expected_episodes.json').read_text())
            if (Counter(map(tuple, expected.get('episode_keys', []))) != Counter(expected_runtime)
                    or Counter(map(tuple, expected.get('original_episode_keys', []))) != Counter(expected_source)
                    or expected.get('source_manifest_sha256') != row['manifest_sha256']):
                child_failures.append('Output expected keys differ from independent input mapping')
            child_report = validate(child_dir)
            if child_report['status'] != 'PASS_SUBSET_COMPLETE':
                child_failures.extend(child_report['failures'])
            process = wrapper['children'][index]
            if (process.get('shard_id') != row['shard_id'] or process.get('exit_code') != 0
                    or process.get('gpu_uuid') != expected_devices[index]
                    or process.get('cuda_visible_devices') != expected_devices[index]
                    or process.get('num_environments') != 8
                    or process.get('output') != str(remote_output / 'shards' / row['shard_id'])
                    or process.get('process_group') != wrapper.get('process_group')):
                child_failures.append('Child process exit, GPU assignment, output or process group mismatch')
            status = json.loads((child_dir / 'run_status.json').read_text())
            if status.get('status') != 'PASS_SUBSET_COMPLETE':
                child_failures.append('Child run_status is not PASS_SUBSET_COMPLETE')
            provenance = json.loads((child_dir / 'run_manifest.json').read_text())
            if (provenance.get('num_environments') != 8
                    or provenance.get('episode_count') != row['episode_count']
                    or provenance.get('runner_sha256') != plan['runtime_script_sha256']['run_falcon_subset.py']
                    or provenance.get('validator_sha256') != plan['runtime_script_sha256']['validate_falcon_subset.py']
                    or provenance.get('slurm_job_id') != wrapper.get('slurm_job_id')
                    or provenance['inputs']['source_manifest']['sha256'] != row['manifest_sha256']):
                child_failures.append('Child runtime provenance differs from frozen protocol')
            for input_name, contract_name in [('checkpoint', 'checkpoint'),
                                                ('pointnav_initialization', 'pretrained')]:
                got = provenance['inputs'][input_name]
                want = plan['runtime_inputs'][contract_name]
                if any(got.get(k) != want[k] for k in ('path', 'bytes', 'sha256')):
                    child_failures.append('Frozen model input mismatch: ' + input_name)
            rows = [json.loads(s) for s in (child_dir / 'episode_metrics.jsonl').read_text().splitlines()]
            observed = [(r.get('scene_id'), r.get('episode_id')) for r in rows]
            if Counter(observed) != Counter(expected_runtime):
                child_failures.append('Observed episode keys are not the exact input-defined multiset')
            all_rows.extend(rows)
            reports.append({'shard_id': row['shard_id'], 'expected_episodes': row['episode_count'],
                            'observed_episodes': len(rows), 'original_validator': child_report,
                            'failures': child_failures})
        except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
            child_failures.append(type(error).__name__ + ': ' + str(error))
            reports.append({'shard_id': row.get('shard_id'), 'failures': child_failures})
        failures.extend(row['shard_id'] + ': ' + f for f in child_failures)
    observed_keys = [(r.get('scene_id'), r.get('episode_id')) for r in all_rows]
    if len(observed_keys) != plan['episode_count'] or len(set(observed_keys)) != len(observed_keys):
        failures.append('Pair count or global observed key uniqueness failed')
    if set(observed_keys) != runtime_union:
        failures.append('Pair actual keys differ from exact frozen-input union')
    values = {}
    for row in all_rows:
        for metric, value in row.get('metrics', {}).items():
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.setdefault(metric, []).append(value)
    aggregate = {metric: math.fsum(nums) / len(nums) for metric, nums in values.items()}
    report = {'status': 'FAIL_PAIR_INCOMPLETE_OR_INCONSISTENT' if failures else 'PASS_PAIR_COMPLETE',
              'scope': 'Two disjoint Falcon official-val shards; not full1087 unless later exact-union audit passes',
              'expected_episode_count': plan['episode_count'], 'observed_records': len(all_rows),
              'observed_unique_keys': len(set(observed_keys)), 'expected_original_key_count': len(source_union),
              'shards': reports, 'failures': failures, 'recomputed_episode_weighted_metrics': aggregate,
              'plan_sha256': sha(plan_path), 'runtime_scene_root': plan['runtime']['scenes_dir']}
    (output / 'parallel_validation.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--plan', type=Path, required=True)
    a = p.parse_args()
    result = validate_pair(a.output, a.plan)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS_PAIR_COMPLETE' else 2)
