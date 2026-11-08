#!/usr/bin/env python3
"""Strict process/collection evidence for a completed Falcon shard.

A pair timeout may retain a normally completed child; it never becomes a
successful pair. All inputs are read-only and bound to the managed freeze.
"""
from pathlib import Path, PurePosixPath
import re

from falcon_full_val_integrity import INPUT_FILES, digest, load, record, require


def owning_root(directory):
    for parent in [directory] + list(directory.parents)[:4]:
        if (parent / 'WORKER_RECEIPT.json').is_file():
            return parent
    raise ValueError('Missing owning WORKER_RECEIPT.json; a hard timeout without collected evidence is not a checkpoint')


def scheduler_state(document, job_id, run_id):
    require(document.get('released') is True and document.get('identity_verified') is True and
            not str(document.get('squeue', '')).strip(), 'Scheduler release/identity evidence is incomplete')
    rows = [line.split('|') for line in document.get('sacct', '').splitlines() if line]
    main = [row for row in rows if row[0] == job_id]
    require(len(main) == 1 and len(main[0]) >= 5 and main[0][1] == run_id, 'Raw sacct main job identity differs')
    state = main[0][3].split()[0].rstrip('+')
    require(state in ('COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED'),
            'Parent scheduler record is not terminal')
    require(document.get('state') == state, 'Parsed and raw scheduler states differ')
    return state


def parent_outcome(worker, state):
    normal = (state == 'COMPLETED' and worker.get('status') == 'SCIENTIFIC_OUTPUTS_VALIDATED' and
              worker.get('scientific_success') is True and worker.get('scientific_exit_code') == 0 and
              worker.get('validation_exit_code') == 0 and
              type(worker.get('scientific_exit_code')) is int and type(worker.get('validation_exit_code')) is int)
    if normal:
        return False, 'PARENT_COMPLETED_VALIDATED'
    require(worker.get('status') == 'FAILED_RETAINED' and worker.get('scientific_success') is False and
            state in ('FAILED', 'TIMEOUT'), 'Parent failure is not a supported timeout checkpoint')
    explicit = worker.get('error_type') == 'TimeoutError' and worker.get('error') == 'SCIENTIFIC_UNIT_TIMEOUT'
    # A scheduler timeout can kill the scientific process before the worker's
    # own deadline. Only the expected TERM/KILL exit is accepted, never an
    # ordinary environment/model error occurring before a TIMEOUT record.
    scheduler_timeout = (state == 'TIMEOUT' and worker.get('error_type') == 'RuntimeError' and
                         worker.get('error') == 'SCIENTIFIC_COMMAND_FAILED' and
                         worker.get('scientific_exit_code') in (-15, -9))
    require(explicit or scheduler_timeout, 'Parent error is not explicit scientific timeout or scheduler-confirmed TERM/KILL timeout')
    return True, 'WORKER_SCIENTIFIC_TIMEOUT' if explicit else 'SLURM_TIMEOUT_WITH_SIGNALLED_SCIENTIFIC_PROCESS'


def check_collected(vm, managed, worker, required):
    manifest_path = managed / 'run/REMOTE_OUTPUT_MANIFEST.json'
    manifest, collected = load(manifest_path), load(managed / 'run/COLLECTED.json')
    require(manifest.get('run_id') == worker['run_id'] and manifest.get('slurm_job_id') == worker['slurm_job_id'],
            'Collected manifest job identity differs')
    entries = manifest['files']
    require(collected.get('files') == entries, 'Collection receipt differs from remote SHA manifest')
    by_name = {}
    for row in entries:
        rel = PurePosixPath(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts and row['path'] not in by_name, 'Unsafe/duplicate collected path')
        path = vm / row['path']
        require(path.is_file() and path.stat().st_size == row['bytes'] and digest(path) == row['sha256'],
                'Collected bytes/SHA differ: ' + row['path'])
        by_name[row['path']] = row
    require(set(required) <= set(by_name), 'Required complete-shard or parent evidence is absent from collected manifest')
    return record(manifest_path)


def completion_evidence(directory, job_id, managed_plan=None):
    directory = Path(directory).resolve()
    vm = owning_root(directory)
    managed = Path(managed_plan).resolve() if managed_plan else vm.parent.parent
    control_path, freeze_path = managed / 'CONTROL.json', managed / 'CONTROL_FREEZE.json'
    control, freeze = load(control_path), load(freeze_path)
    bound_control = [value for key, value in freeze['files'].items() if Path(key).resolve() == control_path]
    require(freeze.get('status') == 'FROZEN_BEFORE_ALLOCATION' and bound_control == [digest(control_path)],
            'Managed control is not bound to the original freeze')
    driver = load(managed / 'run/DRIVER_RECEIPT.json')
    worker_path = vm / 'WORKER_RECEIPT.json'
    worker, cfg = load(worker_path), load(managed / 'worker.json')
    require(worker.get('slurm_job_id') == job_id == driver.get('job_id') and
            worker.get('run_id') == control['run']['session_name'] == driver.get('session') and
            driver.get('released') is True and driver.get('control_freeze_sha256') == digest(freeze_path),
            'Worker/driver/freeze job identity or release differs')
    uploads = {row['remote_name']: row for row in control['uploads']}
    require(len(uploads) == len(control['uploads']), 'Duplicate frozen upload names')
    for name, field in [('worker.py', 'worker_sha256'), ('worker.json', 'config_sha256')]:
        require(digest(managed / name) == uploads[name]['sha256'] == worker[field], 'Frozen worker/config identity differs')
    require(cfg['run_id'] == worker['run_id'] and cfg['output_directory'] == control['remote']['output_directory'],
            'Worker config output identity differs')
    state = scheduler_state(load(managed / 'run/SCHEDULER_LATEST.json'), job_id, worker['run_id'])
    interrupted, reason = parent_outcome(worker, state)
    relative = directory.relative_to(vm).as_posix()
    required = ['WORKER_RECEIPT.json'] + [relative + '/' + name for name in INPUT_FILES]
    provenance = load(directory / 'run_manifest.json')
    source = load(directory / 'source_episode_manifest.json')
    require(load(directory / 'run_status.json').get('status') == 'PASS_SUBSET_COMPLETE' and
            load(directory / 'validation.json').get('status') == 'PASS_SUBSET_COMPLETE',
            'Child lacks its own successful run status and original validation')
    if (vm / 'parallel_plan.json').is_file():
        required += ['parallel_plan.json', 'parallel_status.json']
        plan, wrapper = load(vm / 'parallel_plan.json'), load(vm / 'parallel_status.json')
        require(plan.get('schema') == 'falcon_parallel_pair_v1' and len(plan['children']) == 2 and
                len(wrapper['children']) == 2, 'Invalid pair structure')
        require(digest(vm / 'parallel_plan.json') == uploads['parallel_pair_plan.json']['sha256'] == wrapper['plan_sha256'],
                'Executed pair plan differs from frozen upload')
        require(plan['run_id'] == worker['run_id'] == wrapper['slurm_job_name'] and
                wrapper['slurm_job_id'] == job_id and wrapper['slurmd_nodename'] in plan['node_list'] and
                plan['remote_output_directory'] == cfg['output_directory'] == wrapper['output_directory'],
                'Pair job/node/output identity differs')
        require(cfg['gpu_count'] == 2 and wrapper['wrapper_sha256'] == uploads['run_falcon_parallel.py']['sha256'],
                'Pair GPU count or wrapper source differs')
        for name, sha in plan['runtime_script_sha256'].items():
            require(uploads[name]['sha256'] == sha, 'Pair runtime script differs from frozen upload: ' + name)
        ids = [row['shard_id'] for row in plan['children']]
        require(len(set(ids)) == 2 and all(re.fullmatch(r'balanced4_\d{2}', name) for name in ids), 'Invalid pair shard IDs')
        require([row['shard_id'] for row in wrapper['children']] == ids and source['shard_id'] in ids,
                'Pair executed child ordering/identity differs')
        index = ids.index(source['shard_id'])
        selected = plan['children'][index]
        child = wrapper['children'][index]
        uuids = worker['gpu_binding']['selected_uuids']
        require(len(uuids) == len(set(uuids)) == 2 and worker['gpu_binding']['cuda_actual_uuids'] == uuids and
                wrapper['allocated_gpu_uuids'] == uuids, 'Pair allocated/actual GPU UUID mapping differs')
        require(type(child.get('exit_code')) is int and child['exit_code'] == 0 and
                child['gpu_uuid'] == child['cuda_visible_devices'] == uuids[index] and
                child['num_environments'] == 8 and child['process_group'] == wrapper['process_group'],
                'Retained child did not exit normally on its assigned GPU/process group')
        remote_child = str(PurePosixPath(plan['remote_output_directory']) / 'shards' / source['shard_id'])
        require(relative == 'shards/' + source['shard_id'] and child['output'] == remote_child and
                child['log'] == str(PurePosixPath(plan['remote_output_directory']) / 'logs' / (source['shard_id'] + '.log')),
                'Retained child local/remote output or log path differs from plan')
        required += ['logs/' + source['shard_id'] + '.log']
        runtime = plan['runtime']
        argv = [runtime['python'], str(PurePosixPath(control['remote']['source_directory']) / 'run_falcon_subset.py'),
                '--repo', runtime['repo'], '--manifest', str(PurePosixPath(control['remote']['source_directory']) / selected['manifest_file']),
                '--checkpoint', runtime['checkpoint'], '--pretrained', runtime['pretrained'],
                '--data-path', runtime['data_path'], '--scenes-dir', runtime['scenes_dir'],
                '--num-envs', '8', '--output', remote_child]
        require(child['argv'] == argv and provenance['argv'] == argv[1:], 'Child executed arguments differ from frozen plan')
        require(digest(directory / 'source_episode_manifest.json') == selected['manifest_sha256'] ==
                uploads[selected['manifest_file']]['sha256'] == provenance['inputs']['source_manifest']['sha256'] and
                source['episode_count'] == selected['episode_count'] == provenance['episode_count'] and
                source['scene_count'] == selected['scene_count'] == provenance['scene_count'], 'Child manifest/count differs from frozen plan')
        require(provenance['runner_sha256'] == plan['runtime_script_sha256']['run_falcon_subset.py'] and
                provenance['validator_sha256'] == plan['runtime_script_sha256']['validate_falcon_subset.py'],
                'Child scientific script provenance differs')
        for actual, expected in [('checkpoint', 'checkpoint'), ('pointnav_initialization', 'pretrained')]:
            require(provenance['inputs'][actual] == plan['runtime_inputs'][expected], 'Child model input differs from frozen pair plan')
        if interrupted:
            require(wrapper['status'] in ('RUNNING', 'CHILDREN_COMPLETED', 'FAILED_RETAINED'), 'Unexpected interrupted pair status')
            if wrapper['status'] == 'FAILED_RETAINED':
                require(wrapper.get('error_type') == 'RuntimeError' and wrapper.get('error') == 'WORKER_SIGNAL_15',
                        'Pair records a non-timeout child/environment/validation failure')
        else:
            require(wrapper['status'] == 'PARALLEL_COMPLETED' and all(row.get('exit_code') == 0 for row in wrapper['children']),
                    'Successful parent lacks normal pair completion')
    else:
        require(not interrupted, 'Timeout checkpoint exception applies only to a completed child of an explicit pair')
        require(relative == 'eval' and provenance['argv'] == cfg['argv'][1:], 'Single-shard executed entry/output differs from frozen config')
        require(digest(directory / 'source_episode_manifest.json') == uploads['episode_manifest.json']['sha256'],
                'Single-shard manifest differs from frozen upload')
    collection = check_collected(vm, managed, worker, required)
    return {'worker_receipt': record(worker_path), 'managed_plan': str(managed),
            'collected_shard_relative_path': relative,
            'original_collected_output_directory': str(managed / 'run/vm' / relative),
            'collected_manifest': collection, 'parent_scheduler_state': state,
            'interrupted_parent': interrupted, 'parent_batch_success': not interrupted,
            'acceptance_basis': reason,
            'scope': 'One completed shard only; an interrupted parent remains failed and incomplete'}
