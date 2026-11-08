#!/usr/bin/env python3
"""Synthetic, CPU-only checks of pair validation and process-group cleanup.

These are lifecycle fixtures, never model or benchmark results. No SSH/Slurm,
CUDA/Torch import or resource allocation occurs.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
UUIDS = ['GPU-11111111-1111-1111-1111-111111111111', 'GPU-22222222-2222-2222-2222-222222222222']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def write_child(output, manifest_path, plan, gpu):
    """Intentionally tiny synthetic output accepted only by offline tests."""
    output.mkdir(parents=True, exist_ok=True)
    doc = json.loads(manifest_path.read_text())
    source = doc['episode_keys']
    runtime = [[str(Path(plan['runtime']['scenes_dir']) / s), e] for s, e in source]
    write(output / 'expected_episodes.json', {'episode_keys': runtime, 'original_episode_keys': source,
          'episode_count': len(source), 'source_manifest_sha256': sha(manifest_path)})
    shutil.copyfile(manifest_path, output / 'source_episode_manifest.json')
    metrics = {'success': 1.0, 'spl': .5, 'psc': .8, 'human_collision': 0.0}
    rows = [{'scene_id': s, 'episode_id': e, 'eval_index': 1, 'action_steps': 1, 'metrics': metrics} for s, e in runtime]
    (output / 'episode_metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    write(output / 'actions.json', {'%s|%s|0' % (s, e): [0] for s, e in runtime})
    write(output / 'aggregate_metrics.json', metrics)
    write(output / 'result.json', {'SR': 1.0, 'SPL': .5, 'PSC': .8, 'H-Coll': 0.0, 'Total': .79})
    write(output / 'evaluation_complete.json', {'completed_episode_records': len(rows),
          'requested_episode_count': len(rows), 'evals_per_episode': 1})
    instrument = output / 'instrumentation/vln_falcon_evaluator.py'
    instrument.parent.mkdir(exist_ok=True)
    instrument.write_text('# synthetic offline evaluator fixture only\n')
    write(output / 'instrumentation/observability_manifest.json', {'derived_sha256': sha(instrument)})
    (output / 'instrumentation/observability.patch').write_text('synthetic offline fixture\n')
    write(output / 'run_manifest.json', {'seed': 100, 'policy_deterministic': False, 'max_episode_steps': 500,
          'num_environments': 8, 'episode_count': len(rows), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
          'runner_sha256': plan['runtime_script_sha256']['run_falcon_subset.py'],
          'validator_sha256': plan['runtime_script_sha256']['validate_falcon_subset.py'],
          'inputs': {'source_manifest': {'sha256': sha(manifest_path)},
                     'checkpoint': plan['runtime_inputs']['checkpoint'],
                     'pointnav_initialization': plan['runtime_inputs']['pretrained']}})
    write(output / 'run_status.json', {'status': 'PASS_SUBSET_COMPLETE'})
    write(output / 'resolved_config.json', {'synthetic_offline_fixture': True})
    (output / 'resolved_config.yaml').write_text('synthetic_offline_fixture: true\n')
    write(output / 'gpu_assignment.json', {'cuda_visible_devices': gpu})


FAKE_RUNNER = '''import json,os,pathlib,signal,subprocess,sys,time
from fixture_support import write_child
args=sys.argv[1:]; get=lambda flag: args[args.index(flag)+1]
output=pathlib.Path(get('--output')); manifest=pathlib.Path(get('--manifest'))
plan=json.loads((pathlib.Path(__file__).parent/'parallel_pair_plan.json').read_text())
mode=json.loads(manifest.read_text()).get('offline_mode','success')
if mode=='fail':
 time.sleep(.8);sys.exit(7)
if mode in ('ignore','orphan'):
 signal.signal(signal.SIGTERM,signal.SIG_IGN)
 proc=subprocess.Popen([sys.executable,'-c','import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(60)'])
 output.mkdir(parents=True,exist_ok=True)
 (output/'offline_descendant_pid.txt').write_text(str(proc.pid))
 if mode=='ignore':time.sleep(60)
 elif mode=='orphan':sys.exit(0)
write_child(output,manifest,plan,os.environ['CUDA_VISIBLE_DEVICES'])
print('synthetic offline child completed',flush=True)
'''


def fixture(root, modes=('success', 'success')):
    source = root / 'source'
    source.mkdir(parents=True)
    for name in ['run_falcon_parallel.py', 'validate_falcon_parallel.py', 'validate_falcon_subset.py']:
        shutil.copyfile(HERE / name, source / name)
    shutil.copyfile(__file__, source / 'fixture_support.py')
    (source / 'run_falcon_subset.py').write_text(FAKE_RUNNER)
    children = []
    for i, mode in enumerate(modes):
        shard = 'balanced4_%02d' % i
        manifest = source / (shard + '.json')
        write(manifest, {'episode_keys': [['hm3d/val/scene%d/scene%d.basis.glb' % (i, i), '0']],
                        'episode_count': 1, 'offline_mode': mode})
        children.append({'shard_id': shard, 'manifest_file': manifest.name,
                         'manifest_sha256': sha(manifest), 'episode_count': 1, 'scene_count': 1})
    model = {'path': '/synthetic/checkpoint', 'bytes': 1, 'sha256': 'a' * 64}
    plan = {'schema': 'falcon_parallel_pair_v1', 'run_id': 'OFFLINE_SYNTHETIC_NO_GPU',
            'node_list': ['10-0-1-11'], 'children': children, 'episode_count': 2,
            'remote_output_directory': str(root / 'output'),
            'runtime': {'python': sys.executable, 'repo': str(root), 'checkpoint': '/synthetic/checkpoint',
                        'pretrained': '/synthetic/pretrained', 'data_path': '/synthetic/episodes',
                        'scenes_dir': '/synthetic/alias/scene_datasets'},
            'runtime_inputs': {'checkpoint': model, 'pretrained': model},
            'runtime_script_sha256': {name: sha(source / name) for name in
                ['run_falcon_subset.py', 'run_falcon_parallel.py', 'validate_falcon_parallel.py', 'validate_falcon_subset.py']}}
    plan_path = source / 'parallel_pair_plan.json'
    write(plan_path, plan)
    return source, plan_path, plan


def live_members(pgid):
    result = subprocess.run(['ps', '-axo', 'pid=,pgid=,stat='], capture_output=True, text=True, check=True)
    return [int(fields[0]) for line in result.stdout.splitlines()
            if len(fields := line.split()) >= 3 and int(fields[1]) == pgid and not fields[2].startswith('Z')]


def launch(root, modes):
    source, path, plan = fixture(root, modes)
    output = root / 'output'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(UUIDS), 'SLURM_JOB_ID': '99999999',
           'SLURM_JOB_NAME': plan['run_id'], 'SLURMD_NODENAME': '10-0-1-11'}
    proc = subprocess.Popen([sys.executable, str(source / 'run_falcon_parallel.py'), '--plan', str(path),
                             '--output', str(output)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=25)
    except BaseException:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        raise
    status = json.loads((output / 'parallel_status.json').read_text())
    time.sleep(.3)
    assert not live_members(proc.pid), 'Live scientific descendants remain after wrapper exit'
    return source, path, plan, output, proc.returncode, status, stderr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    checks = []
    with tempfile.TemporaryDirectory(prefix='falcon-pair-offline-') as tmp:
        root = Path(tmp)
        source, path, plan, output, code, status, err = launch(root / 'success', ('success', 'success'))
        assert code == 0 and status['status'] == 'PARALLEL_COMPLETED', (code, status, err)
        for i, child in enumerate(plan['children']):
            device = json.loads((output / 'shards' / child['shard_id'] / 'gpu_assignment.json').read_text())
            assert device['cuda_visible_devices'] == UUIDS[i]
        checks.append({'name': 'two_child_success_exact_disjoint_keys_and_gpu_environment', 'pass': True})
        sys.path.insert(0, str(source))
        module_spec = importlib.util.spec_from_file_location('pair_validator_offline', source / 'validate_falcon_parallel.py')
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        report = module.validate_pair(output, path)
        assert report['status'] == 'PASS_PAIR_COMPLETE'
        # Recovered artifacts must validate independently of remote file paths
        # and runtime source-upload filenames. Keep a frozen SHA anchor, move
        # the result tree and plan, and omit every source upload from the copy.
        relocated = root / 'relocated/vm'
        shutil.copytree(output, relocated)
        relocated_plan_path = root / 'relocated/local_plan/pair_plan.json'
        relocated_plan = deepcopy(plan)
        remote_root = '/mnt/home/liyuqin/USERDIR/vln_baselines_20261103/runs/result_OFFLINE_SYNTHETIC_NO_GPU'
        relocated_plan['remote_output_directory'] = remote_root
        write(relocated_plan_path, relocated_plan)
        write(relocated / 'parallel_plan.json', relocated_plan)
        remote_status = deepcopy(status)
        remote_status['output_directory'] = remote_root
        remote_status['plan_sha256'] = sha(relocated_plan_path)
        for child in remote_status['children']:
            child['output'] = remote_root + '/shards/' + child['shard_id']
            child['log'] = remote_root + '/logs/' + child['shard_id'] + '.log'
        write(relocated / 'parallel_status.json', remote_status)
        assert all(not (relocated_plan_path.parent / child['manifest_file']).exists()
                   for child in relocated_plan['children'])
        relocated_report = module.validate_pair(relocated, relocated_plan_path)
        assert relocated_report['status'] == 'PASS_PAIR_COMPLETE', relocated_report
        checks.append({'name': 'remote_mnt_paths_local_vm_relocation_without_source_uploads_passes', 'pass': True})
        remote_status['children'][0]['output'] = remote_root + '/wrong_shard'
        write(relocated / 'parallel_status.json', remote_status)
        assert module.validate_pair(relocated, relocated_plan_path)['status'].startswith('FAIL')
        checks.append({'name': 'recorded_remote_output_path_tamper_rejected_after_relocation', 'pass': True})
        remote_status['children'][0]['output'] = remote_root + '/shards/' + remote_status['children'][0]['shard_id']
        write(relocated / 'parallel_status.json', remote_status)
        recovered_manifest = relocated / 'shards/balanced4_00/source_episode_manifest.json'
        recovered = json.loads(recovered_manifest.read_text()); recovered['episode_keys'][0][1] = 'FORGED'
        write(recovered_manifest, recovered)
        bad_source = module.validate_pair(relocated, relocated_plan_path)
        assert any('frozen input SHA' in failure for failure in bad_source['failures'])
        checks.append({'name': 'recovered_source_manifest_tamper_rejected_by_frozen_plan_sha', 'pass': True})
        # A signal exit cannot be hidden by otherwise valid scientific JSON.
        changed = deepcopy(status);changed['children'][0]['exit_code'] = -6
        write(output / 'parallel_status.json', changed)
        assert module.validate_pair(output, path)['status'].startswith('FAIL')
        write(output / 'parallel_status.json', status)
        checks.append({'name': 'signal_exit_rejected_despite_complete_fixture_outputs', 'pass': True})
        # Forging the self-reported expected list must not change frozen input keys.
        expected_path = output / 'shards/balanced4_00/expected_episodes.json'
        expected = json.loads(expected_path.read_text());expected['original_episode_keys'][0][1] = 'FORGED'
        write(expected_path, expected)
        bad = module.validate_pair(output, path)
        assert any('independent input mapping' in s for s in bad['failures'])
        checks.append({'name': 'expected_key_tamper_rejected_against_frozen_manifest', 'pass': True})
        for label, modes in [('child_failure', ('fail', 'ignore')), ('success_with_orphan', ('orphan', 'success'))]:
            _, _, _, _, code, status, err = launch(root / label, modes)
            assert code != 0 and status['status'] == 'FAILED_RETAINED', (code, status, err)
            assert status['cleanup']['escalation'] == 'SIGKILL_GROUP'
            checks.append({'name': label + '_terminates_entire_group_including_term_ignoring_grandchild',
                           'pass': True, 'wrapper_exit_code': code, 'retained_status': status['status']})
    report = {'status': 'PASS_OFFLINE_ONLY', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Synthetic subprocess and validator fixtures; no SSH, Slurm, GPU, Torch or model execution',
              'checks': checks, 'scripts': {name: sha(HERE / name) for name in
                ['run_falcon_parallel.py', 'validate_falcon_parallel.py', 'make_falcon_parallel_spec.py']}}
    write(a.output, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
