#!/usr/bin/env python3
"""One read-only Falcon queue progress snapshot; no polling or allocation.

Reads local queue/owner receipts and only the active run's remote result files
through the existing frozen strict-SSH transport. Counts are progress, never
scientific completion or independent validation. No GPU query is performed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


REMOTE_CODE = r'''
import datetime,json,pathlib,re
root=pathlib.Path(REQUEST['remote_output'])

def doc(path):
 if not path.is_file(): return None
 if path.stat().st_size>2*1024*1024: return {'read_error':'JSON_SIZE_LIMIT'}
 try: return json.loads(path.read_text())
 except (OSError,ValueError): return {'read_error':'MISSING_OR_PARTIAL_JSON'}

def selected(document,names):
 if document is None: return None
 return {name:document[name] for name in names if name in document}

def tail(path):
 if not path.is_file(): return None
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-REQUEST['tail_bytes']))
  text=f.read(REQUEST['tail_bytes']).decode('utf-8',errors='replace')
 return re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)

def progress(row):
 target=root/row['relative_output']; episodes=target/'episode_metrics.jsonl'
 count=0;keys=set();invalid=0;error=None
 if episodes.is_file():
  if episodes.stat().st_size>64*1024*1024:error='EPISODE_LOG_SIZE_LIMIT'
  else:
   with episodes.open() as f:
    for line in f:
     if count+invalid>=10000:error='EPISODE_LINE_LIMIT';break
     try:
      item=json.loads(line)
      key=(item['scene_id'],item['episode_id'])
      if not all(isinstance(x,str) for x in key):raise ValueError('Nonstring key')
      keys.add(key);count+=1
     except (ValueError,KeyError,TypeError):invalid+=1
 result={'shard_id':row['shard_id'],'expected_episodes':row['expected_episodes'],
  'episode_records':count,'unique_episode_keys':len(keys),'repeated_rows':count-len(keys),
  'partial_or_invalid_lines':invalid,'read_error':error,
  'last_episode_file_mtime':episodes.stat().st_mtime if episodes.exists() else None,
  'run_status':selected(doc(target/'run_status.json'),['status','started_utc','finished_utc','read_error']),
  'completion_marker':selected(doc(target/'evaluation_complete.json'),['completed_episode_records','requested_episode_count','evals_per_episode','read_error']),
  'log_tail':tail(root/row['relative_log'])}
 return result

worker=doc(root/'WORKER_RECEIPT.json')
parent=doc(root/'parallel_status.json')
identity_errors=[]
if worker and 'read_error' not in worker:
 if str(worker.get('slurm_job_id'))!=REQUEST['job_id'] or worker.get('run_id')!=REQUEST['run_id']:
  identity_errors.append('WORKER_IDENTITY_MISMATCH')
if parent and 'read_error' not in parent:
 if str(parent.get('slurm_job_id'))!=REQUEST['job_id'] or parent.get('slurm_job_name')!=REQUEST['run_id']:
  identity_errors.append('PAIR_IDENTITY_MISMATCH')
result={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'output_directory_exists':root.is_dir(),'identity_errors':identity_errors,
 'worker':selected(worker,['status','run_id','slurm_job_id','scientific_success','scientific_exit_code','validation_exit_code','started_unix','finished_unix','gpu_count','gpu_names','error_type','error','read_error']),
 'parent':selected(parent,['status','slurm_job_id','slurm_job_name','slurmd_nodename','started_utc','finished_utc','error_type','error','read_error']),
 'children':[],'scientific_log_tail':tail(root/'scientific.log')}
if not identity_errors:result['children']=[progress(row) for row in REQUEST['children']]
if parent and isinstance(parent.get('children'),list):
 result['parent_children']=[selected(x,['shard_id','pid','exit_code','started_utc','finished_utc','num_environments']) for x in parent['children']]
print(json.dumps(result,ensure_ascii=False))
'''


def snapshot(queue_directory, tail_bytes):
    checkpoint_path = queue_directory / 'CHECKPOINT.json'
    checkpoint = read(checkpoint_path)
    if checkpoint.get('schema') != 'continuous_reproduction_checkpoint_v1':
        raise ValueError('UNSUPPORTED_QUEUE_CHECKPOINT')
    result = {'schema': 'falcon_continuous_readonly_progress_v1',
              'observed_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'PROGRESS_ONLY_NOT_VALIDATED_RESULT',
              'queue_directory': str(queue_directory), 'queue_status': checkpoint['status'],
              'checkpoint_updated_unix': checkpoint.get('updated_unix'),
              'completed_shards': sorted(checkpoint.get('completed', {})),
              'units': {uid: {'status': row['status'], 'attempt_count': len(row['attempts'])}
                        for uid, row in checkpoint['units'].items()}}
    active = [(uid, row['attempts'][-1]) for uid, row in checkpoint['units'].items()
              if row['status'] != 'COMPLETE' and row.get('attempts')]
    if not active:
        result['observation_status'] = 'NO_ACTIVE_ATTEMPT_NO_REMOTE_QUERY'
        return result
    if len(active) != 1:
        result['observation_status'] = 'MULTIPLE_ACTIVE_ATTEMPTS_NO_REMOTE_QUERY'
        return result
    uid, attempt = active[0]
    plan = Path(attempt['plan'])
    current = {'unit_id': uid, 'run_id': attempt['run_id'], 'phase': attempt['phase'],
               'plan_directory': str(plan)}
    result['active_attempt'] = current
    driver_path = plan / 'run/DRIVER_RECEIPT.json'
    if not driver_path.is_file():
        result['observation_status'] = 'OWNER_RECEIPT_NOT_PRESENT_NO_REMOTE_QUERY'
        return result
    driver, control = read(driver_path), read(plan / 'CONTROL.json')
    if (driver.get('session') != attempt['run_id'] or driver.get('session') != control['run']['session_name']
            or driver.get('attempt_id') != control['run']['attempt_id']
            or driver.get('control_freeze_sha256') != sha(plan / 'CONTROL_FREEZE.json')):
        raise ValueError('LOCAL_ACTIVE_ATTEMPT_IDENTITY_MISMATCH')
    current.update({name: driver.get(name) for name in
                   ['job_id', 'status', 'released', 'local_outputs_verified', 'last_checked_unix']})
    sched = plan / 'run/SCHEDULER_LATEST.json'
    if sched.is_file():
        s = read(sched)
        current['scheduler'] = {name: s.get(name) for name in
                                ['state', 'exit_code', 'released', 'identity_verified']}
    if driver.get('released') is True or not str(driver.get('job_id', '')).isdigit():
        result['observation_status'] = 'NO_UNRELEASED_CONFIRMED_JOB_NO_REMOTE_QUERY'
        return result
    uploads = {row['remote_name']: row for row in control['uploads']}

    def frozen_document(name):
        entry = uploads[name]
        if sha(entry['file']) != entry['sha256']:
            raise ValueError('LOCAL_FROZEN_PROGRESS_INPUT_CHANGED: ' + name)
        return read(entry['file'])

    if 'parallel_pair_plan.json' in uploads:
        pair = frozen_document('parallel_pair_plan.json')
        if pair['run_id'] != driver['session'] or pair['remote_output_directory'] != control['remote']['output_directory']:
            raise ValueError('FROZEN_PAIR_IDENTITY_MISMATCH')
        children = [{'shard_id': row['shard_id'], 'expected_episodes': row['episode_count'],
                     'relative_output': 'shards/' + row['shard_id'],
                     'relative_log': 'logs/' + row['shard_id'] + '.log'} for row in pair['children']]
    else:
        manifest = frozen_document('episode_manifest.json')
        children = [{'shard_id': manifest.get('shard_id', 'eval'), 'expected_episodes': manifest['episode_count'],
                     'relative_output': 'eval', 'relative_log': 'scientific.log'}]
    # Use the already frozen run's transport and key checks; never print profile
    # contents, credentials, key paths or the SSH command.
    sys.path.insert(0, str(plan))
    from transport import Remote
    import preflight
    profile = read(plan / 'profile.json')
    preflight.verify_keys(profile)
    request = {'remote_output': control['remote']['output_directory'], 'run_id': driver['session'],
               'job_id': str(driver['job_id']), 'children': children, 'tail_bytes': tail_bytes}
    code = 'REQUEST = ' + repr(request) + '\n' + REMOTE_CODE
    try:
        result['remote'] = Remote(profile).json(code, timeout=35, retry=False)
        result['observation_status'] = ('REMOTE_IDENTITY_ERROR' if result['remote']['identity_errors']
                                        else 'REMOTE_PROGRESS_READ')
    except Exception as error:
        result['observation_status'] = 'REMOTE_OBSERVATION_UNAVAILABLE'
        result['observation_error_type'] = type(error).__name__
        # No raw SSH stderr is surfaced: it can contain account/key metadata.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tail-bytes', type=int, default=1200, choices=range(0, 8193), metavar='0..8192')
    args = parser.parse_args()
    report = snapshot(args.queue_directory.absolute(), args.tail_bytes)
    if args.output:
        save(args.output.absolute(), report)
    active = report.get('active_attempt', {})
    remote = report.get('remote', {})
    compact = {'status': report['observation_status'], 'queue_status': report['queue_status'],
               'job_id': active.get('job_id'), 'unit_id': active.get('unit_id'),
               'owner_status': active.get('status'), 'released': active.get('released'),
               'parent_status': (remote.get('parent') or {}).get('status'),
               'node': (remote.get('parent') or {}).get('slurmd_nodename'),
               'completed_shards': report['completed_shards'],
               'children': [{name: row.get(name) for name in
                             ['shard_id', 'episode_records', 'unique_episode_keys', 'expected_episodes',
                              'partial_or_invalid_lines', 'run_status']} for row in remote.get('children', [])],
               'output': str(args.output.absolute()) if args.output else None}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0 if report['observation_status'] not in ['REMOTE_IDENTITY_ERROR', 'MULTIPLE_ACTIVE_ATTEMPTS_NO_REMOTE_QUERY'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
