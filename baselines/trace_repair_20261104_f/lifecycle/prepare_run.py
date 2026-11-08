#!/usr/bin/env python3
"""Freeze a new managed run from an explicit scientific specification; no SSH."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import uuid
from transport import read, save, sha
from managed_run import SOURCE, load_plan

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--python', default=sys.executable)
    p.add_argument('--profile', type=Path, default=Path('/Users/yuqinli/.codex/skills/tacc-3090/references/profile.json'))
    a = p.parse_args()
    spec = read(a.spec)
    root = a.directory.absolute()
    root.mkdir(parents=True, exist_ok=False)
    run_id = spec['run_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,63}', run_id):
        raise ValueError('UNSAFE_RUN_ID')
    slurm = spec['slurm']
    if slurm['gpus'] not in [1, 2] or slurm['cpus'] < 1:
        raise ValueError('INVALID_RESOURCE_COUNT')
    node_list = slurm.get('node_list', [])
    if not isinstance(node_list, list) or any(not isinstance(n, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', n) for n in node_list):
        raise ValueError('INVALID_EXPLICIT_NODE_LIST')
    duration = [int(v) for v in slurm['wall_time'].split(':')]
    if len(duration) != 3 or not 0 < duration[0] * 3600 + duration[1] * 60 + duration[2] <= 3600:
        raise ValueError('TACC_ONE_HOUR_LIMIT')
    if not slurm.get('deadline'):
        raise ValueError('SERVER_SIDE_DEADLINE_REQUIRED')
    wall_seconds = duration[0] * 3600 + duration[1] * 60 + duration[2]
    science = spec['science']
    validation_seconds = science.get('validation_timeout_seconds', 90)
    if science['timeout_seconds'] + validation_seconds + 30 > wall_seconds:
        raise ValueError('INSUFFICIENT_WALLTIME_FOR_VALIDATION')
    if not science.get('validator_argv') or not science.get('required_outputs') or not science.get('argv'):
        raise ValueError('SCIENTIFIC_ENTRY_AND_VALIDATOR_REQUIRED')
    if any(re.search(r'(TOKEN|PASSWORD|SECRET|API_KEY)', key, re.I) for key in science.get('env', {})):
        raise ValueError('DO_NOT_FREEZE_SECRET_ENV_VALUES')
    if any(key in science.get('env', {}) for key in ['CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS']):
        raise ValueError('SCIENTIFIC_ENV_MUST_NOT_OVERRIDE_ALLOCATED_GPUS')
    profile = read(a.profile)
    remote_root = Path(spec['remote_root'])
    if Path(profile['remote_userdir']) not in remote_root.parents:
        raise ValueError('REMOTE_ROOT_NOT_WITHIN_ACCOUNT_USERDIR')
    source = remote_root / ('source_' + run_id)
    output = remote_root / ('result_' + run_id)
    reserve = spec.get('reserve_bytes', 4 << 30)
    if reserve < 4 << 30:
        raise ValueError('FOUR_GIB_RESERVE_REQUIRED')
    output_budget = spec['output_budget_bytes']
    if output_budget < 1 or spec['remote_peak_additional_bytes'] < output_budget:
        raise ValueError('INVALID_STORAGE_ESTIMATE')
    names = ['managed_run.py', 'owner.py', 'worker.py', 'incremental_saver.py', 'transport.py', 'preflight.py']
    for name in names:
        shutil.copyfile(HERE / name, root / name)
    shutil.copyfile(a.profile, root / 'profile.json')
    save(root / 'SPEC.json', spec)
    (root / 'storage').mkdir()
    shutil.copyfile(HERE / 'manage_storage.py', root / 'storage/manage.py')
    policy = {'source_line': SOURCE, 'planned_primary_units': 1,
              'max_prediction_bytes_per_unit': output_budget, 'metadata_allowance_bytes': 8 << 20,
              'temporary_transfer_allowance_bytes': spec.get('temporary_transfer_bytes', output_budget),
              'reserve_free_bytes': reserve, 'emergency_free_bytes': reserve,
              'automatic_reclaim_on_low_space': False,
              'max_monitor_seconds': spec['monitor_timeout_seconds'] + 180,
              'monitor_interval_seconds': 5}
    save(root / 'storage/POLICY.json', policy)
    worker = {**science, 'run_id': run_id, 'output_directory': str(output), 'gpu_count': slurm['gpus'],
              'gpu_binding': spec.get('gpu_binding', 'original'),
              'expected_nodes': node_list,
              'min_available_memory_bytes': spec.get('min_available_memory_bytes', 0),
              'reserve_bytes': reserve, 'output_budget_bytes': output_budget}
    # Output paths can be referenced explicitly via {output}; env also carries VLN_RUN_OUTPUT.
    for key in ['argv', 'validator_argv']:
        worker[key] = [str(v).replace('{output}', str(output)).replace('{source}', str(source)) for v in worker[key]]
    worker['cwd'] = str(worker['cwd']).replace('{source}', str(source))
    save(root / 'worker.json', worker)
    script = '#!/bin/bash\nset -eu\nexport PYTHONNOUSERSITE=1\nexport PYTHONDONTWRITEBYTECODE=1\nexec ' + shlex.join([spec['remote_python'], str(source / 'worker.py'), '--config', str(source / 'worker.json')]) + '\n'
    (root / 'job.sh').write_text(script)
    uploads = []
    for row in [{'file': str(root / name), 'remote_name': name} for name in ['worker.py', 'worker.json', 'job.sh']] + spec.get('uploads', []):
        path = Path(row['file']).absolute()
        rel = Path(row['remote_name'])
        if rel.is_absolute() or '..' in rel.parts or not rel.parts:
            raise ValueError('UNSAFE_UPLOAD_DESTINATION')
        uploads.append({'file': str(path), 'remote_name': str(rel), 'bytes': path.stat().st_size, 'sha256': sha(path)})
    if len({r['remote_name'] for r in uploads}) != len(uploads):
        raise ValueError('DUPLICATE_UPLOAD_DESTINATION')
    ref = lambda f: {'file': str(Path(f).absolute()), 'sha256': sha(f)}
    control = {'run': {'session_name': run_id, 'attempt_id': uuid.uuid4().hex, 'local_run_directory': 'run'},
               'profile_file': 'profile.json', 'slurm': slurm,
               'remote': {'source_directory': str(source), 'output_directory': str(output),
                          'peak_additional_bytes': spec['remote_peak_additional_bytes']},
               'storage': {'module': ref(root / 'storage/manage.py'), 'policy': ref(root / 'storage/POLICY.json')},
               'storage_budget': {'reserve_bytes': reserve, 'output_bytes': output_budget},
               'remote_inputs': spec.get('remote_inputs', []), 'uploads': uploads,
               'monitor_timeout_seconds': spec['monitor_timeout_seconds']}
    save(root / 'CONTROL.json', control)
    files = [root / n for n in names] + [root / n for n in ['CONTROL.json', 'SPEC.json', 'profile.json', 'worker.json', 'job.sh', 'storage/manage.py', 'storage/POLICY.json']]
    files += [Path(r['file']) for r in uploads] + [Path(f).absolute() for f in spec.get('scientific_files', [])]
    control_freeze = {'status': 'FROZEN_BEFORE_ALLOCATION', 'source_line': SOURCE,
                      'files': {str(f): sha(f) for f in files}}
    save(root / 'CONTROL_FREEZE.json', control_freeze)
    managed = {'schema': 'managed_scientific_run_v1', 'source_line': SOURCE,
               'control': ref(root / 'CONTROL.json'), 'control_freeze': ref(root / 'CONTROL_FREEZE.json'),
               'python': ref(Path(os.path.abspath(a.python))), 'owner': ref(root / 'owner.py'),
               'incremental_saver': ref(root / 'incremental_saver.py'),
               'record_directory': 'managed_records', 'startup_timeout_seconds': 30,
               'poll_seconds': 0.5, 'deduplicate_admission_rgb': False}
    save(root / 'MANAGED.json', managed)
    files += [root / 'CONTROL_FREEZE.json', root / 'MANAGED.json', Path(os.path.abspath(a.python))]
    save(root / 'MANAGED_FREEZE.json', {'status': 'FROZEN_BEFORE_MANAGED_RUN', 'source_line': SOURCE,
                                      'files': {str(f): sha(f) for f in files}})
    # Validate by importing the copied entry so the freeze binds its actual __file__.
    import importlib.util
    module_spec = importlib.util.spec_from_file_location('_prepared_manager', root / 'managed_run.py')
    manager = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(manager)
    manager.load_plan(root / 'MANAGED.json', root / 'MANAGED_FREEZE.json')
    print(json.dumps({'status': 'FROZEN_NOT_SUBMITTED', 'directory': str(root),
                      'launch_argv': [a.python, str(root / 'managed_run.py'), '--config', str(root / 'MANAGED.json'), '--freeze', str(root / 'MANAGED_FREEZE.json')]}, indent=2))


if __name__ == '__main__':
    main()
