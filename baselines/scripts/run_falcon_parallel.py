#!/usr/bin/env python3
"""Run two frozen Falcon shards on two already-verified allocated GPU UUIDs.

The managed worker starts this script as its scientific process-group leader.
Children deliberately stay in this process group so worker cancellation covers
the wrapper, evaluators and simulator subprocesses together. No Torch import.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

STOP_SIGNAL = None


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, data):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_file(source, name):
    path = Path(name)
    if path.is_absolute() or len(path.parts) != 1 or name in ('.', '..'):
        raise ValueError('UNSAFE_SOURCE_FILENAME')
    return source / path


def request_stop(signum, _frame):
    global STOP_SIGNAL
    STOP_SIGNAL = signum


def group_members(pgid):
    """Linux live members of our group, excluding self and harmless zombies."""
    members = []
    if not Path('/proc/self/stat').is_file():
        # Enables the same lifecycle check on the local macOS host. Production
        # TACC uses /proc; ps is only an observation, never a new process group.
        proc = subprocess.Popen(['ps', '-axo', 'pid=,pgid=,stat='], stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        stdout, _stderr = proc.communicate(timeout=3)
        if proc.returncode:
            raise RuntimeError('CANNOT_OBSERVE_SCIENTIFIC_PROCESS_GROUP')
        for line in stdout.splitlines():
            pid, group, state = line.split()[:3]
            if int(group) == pgid and int(pid) not in (os.getpid(), proc.pid) and not state.startswith('Z'):
                members.append(int(pid))
        return sorted(members)
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text()
            fields = text[text.rfind(')') + 2:].split()
            pid = int(path.parent.name)
            if int(fields[2]) == pgid and pid != os.getpid() and fields[0] != 'Z':
                members.append(pid)
        except (OSError, ValueError, IndexError):
            continue
    return sorted(members)


def stop_group(children, output, status):
    """Keep failure evidence, then clear this worker-owned scientific group."""
    pgid = os.getpgrp()
    if pgid != os.getpid():
        raise RuntimeError('REFUSING_TO_SIGNAL_GROUP_NOT_LED_BY_WRAPPER')
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    status['cleanup'] = {'started_utc': utc(), 'process_group': pgid}
    save(output / 'parallel_status.json', status)
    os.killpg(pgid, signal.SIGTERM)
    deadline = time.monotonic() + 6
    while time.monotonic() < deadline:
        for proc in children:
            proc.poll()
        remaining = group_members(pgid)
        if not remaining:
            break
        time.sleep(0.1)
    remaining = group_members(pgid)
    status['cleanup'].update(remaining_pids=remaining, finished_utc=utc(),
                             escalation='SIGKILL_GROUP' if remaining else 'NONE')
    save(output / 'parallel_status.json', status)
    if remaining:
        # This also kills the wrapper. The failure receipt above is durable,
        # and the parent worker records the signal as a failed scientific unit.
        os.killpg(pgid, signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    output = args.output.absolute()
    if os.getpgrp() != os.getpid():
        raise RuntimeError('WRAPPER_MUST_BE_WORKER_SCIENTIFIC_GROUP_LEADER')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'parallel_status.json').exists() or (output / 'shards').exists():
        raise ValueError('REFUSING_TO_OVERWRITE_PARALLEL_RESULTS')
    status = {'status': 'STARTING', 'started_utc': utc(), 'wrapper_pid': os.getpid(),
              'process_group': os.getpgrp(), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
              'slurm_job_name': os.environ.get('SLURM_JOB_NAME'),
              'slurmd_nodename': os.environ.get('SLURMD_NODENAME'),
              'output_directory': str(output), 'children': []}
    children, streams = [], []
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    try:
        plan = json.loads(args.plan.read_text())
        if plan.get('schema') != 'falcon_parallel_pair_v1' or len(plan.get('children', [])) != 2:
            raise ValueError('INVALID_TWO_SHARD_PLAN')
        if any(not re.fullmatch(r'balanced4_\d{2}', row.get('shard_id', '')) for row in plan['children']):
            raise ValueError('UNSAFE_SHARD_ID')
        if not Path(plan['remote_output_directory']).is_absolute() or output != Path(plan['remote_output_directory']):
            raise ValueError('OUTPUT_ROOT_DIFFERS_FROM_FROZEN_PAIR_PLAN')
        if os.environ.get('SLURM_JOB_NAME') != plan['run_id']:
            raise ValueError('JOB_NAME_DIFFERS_FROM_PAIR_PLAN')
        if os.environ.get('SLURMD_NODENAME') not in plan['node_list']:
            raise ValueError('NODE_DIFFERS_FROM_PAIR_PLAN')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if (len(visible) != 2 or len(set(visible)) != 2 or not all(
                re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', v)
                for v in visible)):
            raise RuntimeError('EXPECTED_TWO_WORKER_VERIFIED_GPU_UUIDS')
        status['allocated_gpu_uuids'] = visible
        status['plan_sha256'] = sha(args.plan)
        status['wrapper_sha256'] = sha(__file__)
        for name, digest in plan['runtime_script_sha256'].items():
            if sha(local_file(source, name)) != digest:
                raise ValueError('RUNTIME_SCRIPT_HASH_MISMATCH: ' + name)
        (output / 'parallel_plan.json').write_bytes(args.plan.read_bytes())
        (output / 'logs').mkdir(exist_ok=False)
        (output / 'shards').mkdir(exist_ok=False)
        sets = []
        for row in plan['children']:
            manifest = local_file(source, row['manifest_file'])
            if sha(manifest) != row['manifest_sha256']:
                raise ValueError('INPUT_MANIFEST_SHA_MISMATCH')
            doc = json.loads(manifest.read_text())
            keys = [tuple(k) for k in doc['episode_keys']]
            if len(keys) != len(set(keys)) or len(keys) != row['episode_count']:
                raise ValueError('MANIFEST_EPISODE_COUNT_OR_DUPLICATES')
            sets.append(set(keys))
        if sets[0] & sets[1]:
            raise ValueError('SHARD_EPISODE_OVERLAP')
        common = plan['runtime']
        for i, row in enumerate(plan['children']):
            child_output = output / 'shards' / row['shard_id']
            argv = [common['python'], str(source / 'run_falcon_subset.py'),
                    '--repo', common['repo'], '--manifest', str(source / row['manifest_file']),
                    '--checkpoint', common['checkpoint'], '--pretrained', common['pretrained'],
                    '--data-path', common['data_path'], '--scenes-dir', common['scenes_dir'],
                    '--num-envs', '8', '--output', str(child_output)]
            env = dict(os.environ)
            env['CUDA_VISIBLE_DEVICES'] = visible[i]
            env['VLN_RUN_OUTPUT'] = str(child_output)
            log_path = output / 'logs' / (row['shard_id'] + '.log')
            log = log_path.open('wb')
            streams.append(log)
            proc = subprocess.Popen(argv, cwd=common['repo'], env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
            children.append(proc)
            status['children'].append({'shard_id': row['shard_id'], 'pid': proc.pid,
                'gpu_uuid': visible[i], 'cuda_visible_devices': visible[i], 'num_environments': 8,
                'output': str(child_output), 'log': str(log_path), 'argv': argv,
                'started_utc': utc(), 'exit_code': None, 'process_group': os.getpgrp()})
            save(output / 'parallel_status.json', status)
        status['status'] = 'RUNNING'
        save(output / 'parallel_status.json', status)
        while True:
            if STOP_SIGNAL is not None:
                raise RuntimeError('WORKER_SIGNAL_' + str(STOP_SIGNAL))
            live = False
            for proc, row in zip(children, status['children']):
                code = proc.poll()
                if code is None:
                    live = True
                elif row['exit_code'] is None:
                    row.update(exit_code=code, finished_utc=utc())
                    save(output / 'parallel_status.json', status)
                if code is not None and code != 0:
                    raise RuntimeError('CHILD_FAILED: %s exit=%s' % (row['shard_id'], code))
            if not live:
                break
            time.sleep(0.25)
        cleanup_deadline = time.monotonic() + 5
        while group_members(os.getpgrp()) and time.monotonic() < cleanup_deadline:
            time.sleep(0.1)
        if group_members(os.getpgrp()):
            raise RuntimeError('CHILDREN_EXITED_WITH_LIVE_SIMULATOR_DESCENDANTS')
        status.update(status='CHILDREN_COMPLETED', finished_utc=utc())
        save(output / 'parallel_status.json', status)
        from validate_falcon_parallel import validate_pair
        report = validate_pair(output, args.plan, require_final_wrapper=False)
        if report['status'] != 'PASS_PAIR_COMPLETE':
            raise RuntimeError('PAIR_VALIDATION_FAILED')
        if STOP_SIGNAL is not None:
            raise RuntimeError('WORKER_SIGNAL_' + str(STOP_SIGNAL))
        status.update(status='PARALLEL_COMPLETED', finished_utc=utc())
        save(output / 'parallel_status.json', status)
        return 0
    except BaseException as error:
        for proc, row in zip(children, status['children']):
            row['exit_code'] = proc.poll()
        status.update(status='FAILED_RETAINED', error_type=type(error).__name__, error=str(error),
                      finished_utc=utc())
        save(output / 'parallel_status.json', status)
        if children:
            stop_group(children, output, status)
        return 2
    finally:
        for stream in streams:
            stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
