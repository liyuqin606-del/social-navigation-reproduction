#!/usr/bin/env python3
"""Single-submit Slurm owner. Invoked only by the pinned managed lifecycle."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import preflight
from transport import Remote, read, save, sha

class ObservationUnavailable(RuntimeError):
    """An authenticated read could not establish remote state; never means failure."""


def observed_json(remote, code):
    try:
        return remote.json(code, retry=True)
    except subprocess.TimeoutExpired as error:
        raise ObservationUnavailable('READ_TIMEOUT: ' + str(error)) from error
    except RuntimeError as error:
        if str(error).startswith('SSH_COMMAND_FAILED:'):
            raise ObservationUnavailable(str(error)) from error
        raise


TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


def parse_scheduler(value, job_id, run_id, user):
    queued = []
    for line in value['squeue'].splitlines():
        fields = line.split('|')
        if len(fields) < 4 or fields[:3] != [job_id, user, run_id]:
            raise RuntimeError('SQUEUE_IDENTITY_MISMATCH')
        queued.append(fields)
    accounts = []
    for line in value['sacct'].splitlines():
        fields = line.split('|')
        if fields[0] != job_id:
            continue
        if len(fields) < 5 or fields[1:3] != [run_id, user]:
            raise RuntimeError('SACCT_IDENTITY_MISMATCH')
        accounts.append(fields)
    if len(accounts) > 1:
        raise RuntimeError('MULTIPLE_ACCOUNTING_ROWS')
    state = accounts[0][3].split()[0].rstrip('+') if accounts else (queued[0][3] if queued else 'UNKNOWN')
    return {**value, 'state': state, 'exit_code': accounts[0][4] if accounts else None,
            'identity_verified': bool(queued or accounts), 'released': bool(accounts and state in TERMINAL and not queued)}


def relative(root, name):
    p = Path(name)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise RuntimeError('UNBOUNDED_RELATIVE_PATH')
    return root / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--freeze', type=Path, required=True)
    a = ap.parse_args()
    base = a.config.absolute().parent
    cfg, freeze = read(a.config), read(a.freeze)
    if freeze.get('status') != 'FROZEN_BEFORE_ALLOCATION':
        raise RuntimeError('FREEZE_MISSING')
    for name, digest in freeze['files'].items():
        p = Path(name) if Path(name).is_absolute() else base / name
        if sha(p) != digest:
            raise RuntimeError('FROZEN_INPUT_CHANGED: ' + p.name)
    run = relative(base, cfg['run']['local_run_directory'])
    run.mkdir(exist_ok=False)
    for name in ['vm', 'incremental_cache']:
        (run / name).mkdir()
    run_id = cfg['run']['session_name']
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,63}', run_id):
        raise RuntimeError('UNSAFE_RUN_ID')
    driver = {'session': run_id, 'attempt_id': cfg['run']['attempt_id'],
              'control_freeze_sha256': sha(a.freeze), 'entry_driver_sha256': sha(__file__),
              'status': 'ADMITTED', 'local_outputs_verified': False, 'released': False,
              'job_id': None, 'started_unix': time.time()}
    receipt = run / 'DRIVER_RECEIPT.json'
    save(receipt, driver)
    profile = read(base / cfg['profile_file'])
    remote = Remote(profile)
    source = Path(cfg['remote']['source_directory'])
    output = Path(cfg['remote']['output_directory'])
    user = profile['username']
    job_id = None
    submitted = False
    monitoring_interrupted = False

    def query():
        code = '''import json,subprocess
def call(a):
 p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout.strip()
print(json.dumps({'squeue':call(Q),'sacct':call(A)}))
'''.replace('call(Q)', 'call(' + repr(['squeue', '-h', '-j', job_id, '-o', '%i|%u|%j|%T|%R']) + ')').replace('call(A)', 'call(' + repr(['sacct', '-n', '-P', '-j', job_id, '--format=JobIDRaw,JobName%128,User,State,ExitCode,Start,End,Elapsed,NodeList']) + ')')
        return parse_scheduler(observed_json(remote, code), job_id, run_id, user)

    def cancel_owned():
        q = query()
        if q['released']:
            return q
        if not q['identity_verified'] or not q['squeue']:
            raise RuntimeError('CANCELLATION_IDENTITY_NOT_PROVEN')
        remote.code('import subprocess;subprocess.check_call(' + repr(['scancel', job_id]) + ')')
        for _ in range(20):
            time.sleep(3)
            q = query()
            if q['released']:
                return q
        raise RuntimeError('CANCELLATION_TERMINAL_NOT_CONFIRMED')

    def recover_submission():
        # Explicit two-day UTC window avoids this cluster's unsupported now-1day syntax.
        # Run names are globally unique; a wider date window cannot authorize a second submit.
        start = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=2)).strftime('%Y-%m-%d')
        code = '''import json,subprocess
def call(a):
 p=subprocess.run(a,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 if p.returncode:raise RuntimeError(p.stderr)
 return p.stdout
print(json.dumps({'queue':call(Q),'account':call(A)}))
'''.replace('call(Q)', 'call(' + repr(['squeue', '-h', '-u', user, '-n', run_id, '-o', '%i|%u|%j']) + ')').replace('call(A)', 'call(' + repr(['sacct', '-n', '-P', '-u', user, '--name=' + run_id, '-S', start, '--format=JobIDRaw,User,JobName%128']) + ')')
        rows = remote.json(code, retry=True)
        ids = set()
        for line in (rows['queue'] + '\n' + rows['account']).splitlines():
            f = line.split('|')
            if len(f) >= 3 and f[0].isdigit() and f[1:3] == [user, run_id]:
                ids.add(f[0])
        if len(ids) != 1:
            raise RuntimeError('SUBMISSION_UNKNOWN_NO_RESUBMIT')
        return next(iter(ids))

    def collect():
        manifest = remote.json('import json;print(open(' + repr(str(output / 'OUTPUT_MANIFEST.json')) + ').read())', retry=True)
        if manifest.get('run_id') != run_id or str(manifest.get('slurm_job_id')) != job_id:
            raise RuntimeError('OUTPUT_MANIFEST_IDENTITY_MISMATCH')
        allowed = set(read(base / 'worker.json')['required_outputs'] + ['scientific.log', 'validation.log', 'WORKER_RECEIPT.json'])
        if len(manifest['files']) != len(set(row['path'] for row in manifest['files'])):
            raise RuntimeError('DUPLICATE_OUTPUT_NAME')
        budget = cfg['storage_budget']['output_bytes']
        if sum(row['bytes'] for row in manifest['files']) > budget:
            raise RuntimeError('OUTPUT_BUDGET_EXCEEDED')
        save(run / 'REMOTE_OUTPUT_MANIFEST.json', manifest)
        collected = []
        for row in manifest['files']:
            if row['path'] not in allowed or row['bytes'] < 0:
                raise RuntimeError('UNEXPECTED_OUTPUT_FILE')
            target = relative(run / 'vm', row['path'])
            if shutil.disk_usage(run).free < cfg['storage_budget']['reserve_bytes'] + row['bytes']:
                raise RuntimeError('LOCAL_TRANSFER_RESERVE')
            remote.download(relative(output, row['path']), target, row)
            cached = relative(run / 'incremental_cache', row['path'])
            cached.parent.mkdir(parents=True, exist_ok=True)
            if not cached.exists():
                os.link(target, cached)
            if sha(cached) != row['sha256']:
                raise RuntimeError('CACHE_HASH_MISMATCH')
            collected.append(row)
            save(run / 'COLLECTED.json', {'files': collected})
        result = read(run / 'vm/WORKER_RECEIPT.json')
        if result['run_id'] != run_id or str(result['slurm_job_id']) != job_id or result['worker_sha256'] != sha(base / 'worker.py') or result['config_sha256'] != sha(base / 'worker.json'):
            raise RuntimeError('WORKER_IDENTITY_MISMATCH')
        return result

    def collect_scheduler_log():
        log = source / ('slurm-' + job_id + '.out')
        row = remote.json('import json,hashlib;from pathlib import Path;p=Path(' + repr(str(log)) + ');h=hashlib.sha256();f=p.open("rb");\nfor b in iter(lambda:f.read(4<<20),b""):h.update(b)\nprint(json.dumps({"bytes":p.stat().st_size,"sha256":h.hexdigest()}))', retry=True)
        if row['bytes'] > cfg['storage_budget']['output_bytes'] or shutil.disk_usage(run).free < cfg['storage_budget']['reserve_bytes'] + row['bytes']:
            raise RuntimeError('SCHEDULER_LOG_TRANSFER_BUDGET')
        target = run / 'vm' / log.name
        remote.download(log, target, row)
        cached = run / 'incremental_cache' / log.name
        if not cached.exists():
            os.link(target, cached)
        save(run / 'SCHEDULER_LOG_VERIFIED.json', {**row, 'file': str(target)})

    try:
        preflight.verify_keys(profile)
        required = cfg['remote']['peak_additional_bytes'] + cfg['storage_budget']['reserve_bytes']
        admission = remote.json('import json,shutil;f=shutil.disk_usage(' + repr(profile['remote_userdir']) + ').free;assert f>=' + str(required) + ";print(json.dumps({'free_bytes':f,'required_bytes':" + str(required) + '}))')
        save(run / 'REMOTE_ADMISSION.json', admission)
        # Files must already be present; no downloads or installs in this owner.
        for row in cfg.get('remote_inputs', []):
            actual = remote.json('import hashlib,json;from pathlib import Path;p=Path(' + repr(row['path']) + ');h=hashlib.sha256();f=p.open("rb");\nfor b in iter(lambda:f.read(4<<20),b""):h.update(b)\nprint(json.dumps({"bytes":p.stat().st_size,"sha256":h.hexdigest()}))', timeout=600, retry=True)
            if actual != {k: row[k] for k in ['bytes', 'sha256']}:
                raise RuntimeError('REMOTE_INPUT_CHANGED: ' + Path(row['path']).name)
        remote.code('from pathlib import Path;Path(' + repr(str(source)) + ').mkdir(parents=True,exist_ok=False);assert not Path(' + repr(str(output)) + ').exists()')
        for row in cfg['uploads']:
            remote.upload(row['file'], relative(source, row['remote_name']), row)
        s = cfg['slurm']
        argv = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--cpus-per-task=' + str(s['cpus']),
                '--partition=' + s['partition'], '--gres=gpu:' + str(s['gpus']), '--mem=0', '--time=' + s['wall_time'],
                '--deadline=' + s['deadline'], '--mail-type=NONE', '--job-name=' + run_id,
                '--output=' + str(source / 'slurm-%j.out'), str(source / 'job.sh')]
        if s.get('node_list'):
            argv.insert(-1, '--nodelist=' + ','.join(s['node_list']))
        driver.update(status='SUBMITTING', submission_argv=argv)
        save(receipt, driver)
        submitted = True
        try:
            response = remote.code('import subprocess;print(subprocess.check_output(' + repr(argv) + ',universal_newlines=True).strip())')
            job_id = response.split(';')[0]
            if not job_id.isdigit():
                raise RuntimeError('SUBMIT_RESPONSE_UNRECOGNIZED')
        except Exception as error:
            driver.update(status='SUBMISSION_UNCERTAIN', submission_error=str(error))
            save(receipt, driver)
            job_id = recover_submission()  # Queries only, never a second sbatch.
        driver.update(status='SUBMITTED', job_id=job_id)
        save(receipt, driver)
        print('SUBMITTED ' + job_id, flush=True)
        deadline = time.monotonic() + cfg['monitor_timeout_seconds']
        last = None
        while True:
            current = query()
            save(run / 'SCHEDULER_LATEST.json', current)
            driver.update(status=current['state'], last_checked_unix=time.time())
            save(receipt, driver)
            if current['state'] != last:
                print('JOB_STATE ' + current['state'], flush=True)
                last = current['state']
            if current['released']:
                driver['released'] = True
                save(receipt, driver)
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('OWNER_MONITOR_DEADLINE')
            if shutil.disk_usage(run).free < cfg['storage_budget']['reserve_bytes']:
                raise RuntimeError('LOCAL_RESERVE_CROSSED')
            time.sleep(10)
        result = collect()  # Retain failed scientific-unit outputs too.
        if current['state'] != 'COMPLETED' or current['exit_code'] != '0:0' or result.get('scientific_success') is not True:
            raise RuntimeError('SCIENTIFIC_UNIT_DID_NOT_PASS')
        driver.update(status='OUTPUTS_VERIFIED', local_outputs_verified=True)
        return 0
    except BaseException as error:
        monitoring_interrupted = isinstance(error, ObservationUnavailable)
        if monitoring_interrupted:
            driver.update(monitoring_interrupted=True, remote_state='UNKNOWN_NO_RESUBMIT',
                          recovery_required=True, cancellation_deferred_to_server_limits=True)
        driver.update(status='FAILED_RETAINED' if job_id or not submitted else 'SUBMISSION_UNKNOWN',
                      error_type=type(error).__name__, error=str(error))
        print(type(error).__name__ + ': ' + str(error), flush=True)
        return 5
    finally:
        if job_id and not driver['released'] and not monitoring_interrupted:
            try:
                release = cancel_owned()
                driver['released'] = release['released']
                save(run / 'SCHEDULER_RELEASE.json', release)
            except Exception as error:
                driver['release_error'] = str(error)
        if not submitted:
            driver['released'] = True  # No submission was attempted.
        if job_id and driver['released']:
            if not (run / 'vm/WORKER_RECEIPT.json').exists():
                try:
                    collect()
                except Exception as error:
                    driver['collection_error'] = str(error)
            try:
                collect_scheduler_log()
            except Exception as error:
                driver['scheduler_log_collection_error'] = str(error)
        driver.update(finished_unix=time.time(), final_free_bytes=shutil.disk_usage(run).free)
        save(receipt, driver)


if __name__ == '__main__':
    raise SystemExit(main())
