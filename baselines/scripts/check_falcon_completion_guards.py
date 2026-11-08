#!/usr/bin/env python3
"""Offline adverse process/collection fixtures; no metrics, models or GPU."""
import copy
import json
from pathlib import Path
import tempfile

from falcon_full_val_integrity import INPUT_FILES, digest, load, revalidate_without_writing_input
from falcon_shard_completion_evidence import completion_evidence, parent_outcome


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    root = Path(__file__).resolve().parents[2]
    actual_manifest = load(root / 'baselines/prep/falcon_full_val_balanced4_shard_00_20261103_manifest.json')
    checks = []
    with tempfile.TemporaryDirectory(prefix='falcon-checkpoint-adverse-') as temporary:
        managed = Path(temporary) / 'managed'
        vm = managed / 'run/vm'
        shard_id, other_id = 'balanced4_00', 'balanced4_02'
        directory = vm / 'shards' / shard_id
        for name in INPUT_FILES:
            save(directory / name, {})  # Empty diagnostic fixture; never scientific metrics.
        save(directory / 'source_episode_manifest.json', actual_manifest)
        save(directory / 'run_status.json', {'status': 'PASS_SUBSET_COMPLETE'})
        save(directory / 'validation.json', {'status': 'PASS_SUBSET_COMPLETE'})
        (managed / 'worker.py').write_text('# OFFLINE FIXTURE, NOT A SCIENTIFIC WORKER\n')
        job, rid = '900001', 'OFFLINE_FIXTURE_PAIR_TIMEOUT'
        remote, source = '/fixture/result', '/fixture/source'
        runtime = {'python': '/fixture/python', 'repo': '/fixture/repo', 'checkpoint': '/fixture/checkpoint',
                   'pretrained': '/fixture/pretrained', 'data_path': '/fixture/val.json.gz', 'scenes_dir': '/fixture/scenes'}
        model_inputs = {key: {'path': runtime[key], 'bytes': 1, 'sha256': '0' * 64} for key in ('checkpoint', 'pretrained')}
        script_names = ['run_falcon_subset.py', 'validate_falcon_subset.py', 'prepare_falcon_observed.py',
                        'run_falcon_parallel.py', 'validate_falcon_parallel.py']
        script_hashes = {name: '1' * 64 for name in script_names}
        msha = digest(directory / 'source_episode_manifest.json')
        selected = {'shard_id': shard_id, 'manifest_file': 'episode_manifest_' + shard_id + '.json',
                    'manifest_sha256': msha, 'episode_count': actual_manifest['episode_count'],
                    'scene_count': actual_manifest['scene_count']}
        plan = {'schema': 'falcon_parallel_pair_v1', 'run_id': rid, 'node_list': ['fixture-node'],
                'remote_output_directory': remote, 'runtime': runtime, 'runtime_script_sha256': script_hashes,
                'runtime_inputs': model_inputs, 'children': [selected, {**selected, 'shard_id': other_id}]}
        save(vm / 'parallel_plan.json', plan)
        argv = [runtime['python'], source + '/run_falcon_subset.py', '--repo', runtime['repo'],
                '--manifest', source + '/' + selected['manifest_file'], '--checkpoint', runtime['checkpoint'],
                '--pretrained', runtime['pretrained'], '--data-path', runtime['data_path'],
                '--scenes-dir', runtime['scenes_dir'], '--num-envs', '8', '--output', remote + '/shards/' + shard_id]
        uuids = ['GPU-fixture-A', 'GPU-fixture-B']
        child = {'shard_id': shard_id, 'exit_code': 0, 'gpu_uuid': uuids[0], 'cuda_visible_devices': uuids[0],
                 'num_environments': 8, 'process_group': 111, 'output': remote + '/shards/' + shard_id,
                 'log': remote + '/logs/' + shard_id + '.log', 'argv': argv}
        wrapper = {'status': 'FAILED_RETAINED', 'error_type': 'RuntimeError', 'error': 'WORKER_SIGNAL_15',
                   'plan_sha256': digest(vm / 'parallel_plan.json'), 'wrapper_sha256': script_hashes['run_falcon_parallel.py'],
                   'slurm_job_name': rid, 'slurm_job_id': job, 'slurmd_nodename': 'fixture-node',
                   'output_directory': remote, 'process_group': 111, 'allocated_gpu_uuids': uuids,
                   'children': [child, {**child, 'shard_id': other_id, 'exit_code': None}]}
        save(vm / 'parallel_status.json', wrapper)
        provenance = {'slurm_job_id': job, 'argv': argv[1:], 'episode_count': selected['episode_count'],
                      'scene_count': selected['scene_count'], 'runner_sha256': script_hashes['run_falcon_subset.py'],
                      'validator_sha256': script_hashes['validate_falcon_subset.py'],
                      'inputs': {'source_manifest': {'sha256': msha}, 'checkpoint': model_inputs['checkpoint'],
                                 'pointnav_initialization': model_inputs['pretrained']}}
        save(directory / 'run_manifest.json', provenance)
        save(managed / 'worker.json', {'run_id': rid, 'output_directory': remote, 'gpu_count': 2})
        uploads = [{'remote_name': name, 'sha256': sha} for name, sha in script_hashes.items()]
        uploads += [{'remote_name': 'worker.py', 'sha256': digest(managed / 'worker.py')},
                    {'remote_name': 'worker.json', 'sha256': digest(managed / 'worker.json')},
                    {'remote_name': 'parallel_pair_plan.json', 'sha256': digest(vm / 'parallel_plan.json')},
                    {'remote_name': selected['manifest_file'], 'sha256': msha}]
        save(managed / 'CONTROL.json', {'run': {'session_name': rid}, 'uploads': uploads,
                                      'remote': {'source_directory': source, 'output_directory': remote}})
        save(managed / 'CONTROL_FREEZE.json', {'status': 'FROZEN_BEFORE_ALLOCATION',
              'files': {str(managed / 'CONTROL.json'): digest(managed / 'CONTROL.json')}})
        save(managed / 'run/DRIVER_RECEIPT.json', {'job_id': job, 'session': rid, 'released': True,
              'control_freeze_sha256': digest(managed / 'CONTROL_FREEZE.json')})
        scheduler = {'squeue': '', 'sacct': job + '|' + rid + '|fixture-user|FAILED|5:0',
                     'state': 'FAILED', 'identity_verified': True, 'released': True}
        save(managed / 'run/SCHEDULER_LATEST.json', scheduler)
        worker = {'run_id': rid, 'slurm_job_id': job, 'status': 'FAILED_RETAINED', 'scientific_success': False,
                  'error_type': 'TimeoutError', 'error': 'SCIENTIFIC_UNIT_TIMEOUT',
                  'worker_sha256': digest(managed / 'worker.py'), 'config_sha256': digest(managed / 'worker.json'),
                  'gpu_binding': {'selected_uuids': uuids, 'cuda_actual_uuids': uuids}}
        save(vm / 'WORKER_RECEIPT.json', worker)
        (vm / 'logs').mkdir()
        (vm / 'logs' / (shard_id + '.log')).write_text('OFFLINE PROCESS EVIDENCE FIXTURE ONLY\n')

        def remanifest():
            files = [{'path': p.relative_to(vm).as_posix(), 'bytes': p.stat().st_size, 'sha256': digest(p)}
                     for p in sorted(vm.rglob('*')) if p.is_file()]
            save(managed / 'run/REMOTE_OUTPUT_MANIFEST.json', {'run_id': rid, 'slurm_job_id': job, 'files': files})
            save(managed / 'run/COLLECTED.json', {'files': files})

        remanifest()
        baseline = {p: p.read_bytes() for p in managed.rglob('*') if p.is_file()}
        accepted = completion_evidence(directory, job, managed)
        assert accepted['interrupted_parent'] is True and accepted['parent_batch_success'] is False
        checks.append({'name': 'isolated_timeout_process_evidence_recognized_not_scientific_completion', 'pass': True})
        # Process evidence is necessary but insufficient: the existing metric
        # validator must reject this empty fixture even with forged PASS labels.
        try:
            revalidate_without_writing_input(directory)
        except (ValueError, KeyError):
            checks.append({'name': 'empty_scientific_evidence_rejected_despite_process_proof', 'pass': True})
        else:
            raise AssertionError('Empty scientific fixture accepted')

        mutations = {
            'non_timeout_environment_failure': (vm / 'WORKER_RECEIPT.json', lambda j: j.update(error_type='RuntimeError', error='ALLOCATED_GPU_DIFFERS')),
            'child_signal_after_report': (vm / 'parallel_status.json', lambda j: j['children'][0].update(exit_code=-6)),
            'child_unobserved_exit': (vm / 'parallel_status.json', lambda j: j['children'][0].update(exit_code=None)),
            'wrong_gpu': (vm / 'parallel_status.json', lambda j: j['children'][0].update(gpu_uuid=uuids[1])),
            'wrong_output': (vm / 'parallel_status.json', lambda j: j['children'][0].update(output=remote + '/shards/' + other_id)),
            'non_timeout_wrapper_failure': (vm / 'parallel_status.json', lambda j: j.update(error='CHILD_FAILED: other exit=1')),
            'missing_child_validation': (directory / 'validation.json', lambda j: j.update(status='FAIL_INCOMPLETE_OR_INCONSISTENT')),
            'changed_checkpoint_input': (directory / 'run_manifest.json', lambda j: j['inputs']['checkpoint'].update(sha256='f' * 64)),
            'wrong_sacct_job_identity': (managed / 'run/SCHEDULER_LATEST.json', lambda j: j.update(sacct=job + '|DIFFERENT_JOB|fixture-user|FAILED|5:0')),
        }
        for name, (path, mutate) in mutations.items():
            for f, content in baseline.items():
                f.write_bytes(content)
            doc = load(path); mutate(doc); save(path, doc); remanifest()
            try:
                completion_evidence(directory, job, managed)
            except (ValueError, KeyError) as error:
                checks.append({'name': name, 'pass': True, 'rejection': str(error)})
            else:
                raise AssertionError(name + ' accepted')
        for f, content in baseline.items():
            f.write_bytes(content)
        (directory / 'actions.json').write_text('TAMPER_WITHOUT_UPDATED_COLLECTED_SHA\n')
        try:
            completion_evidence(directory, job, managed)
        except ValueError as error:
            checks.append({'name': 'collected_sha_tamper', 'pass': True, 'rejection': str(error)})
        else:
            raise AssertionError('SHA tamper accepted')
        for code in (1, -6):
            test = {**worker, 'error_type': 'RuntimeError', 'error': 'SCIENTIFIC_COMMAND_FAILED', 'scientific_exit_code': code}
            try:
                parent_outcome(test, 'TIMEOUT')
            except ValueError:
                checks.append({'name': 'slurm_timeout_does_not_excuse_exit_' + str(code), 'pass': True})
            else:
                raise AssertionError('Unrelated failed exit accepted under TIMEOUT')
    result = {'status': 'PASS_OFFLINE_ADVERSE_FIXTURES_ONLY', 'metrics_generated': False,
              'scientific_completion_fixture_created': False, 'gpu_execution': False, 'checks': checks}
    path = root / 'baselines/audits/falcon_shard_completion_guards_20261103.json'
    save(path, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
