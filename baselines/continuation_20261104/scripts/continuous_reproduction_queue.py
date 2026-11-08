#!/usr/bin/env python3
"""Finite, restart-safe queue around prepare_run.py and its managed_run.py.

This controller never calls a scheduler or modifies frozen scientific code.
An allocation is owned exclusively by the existing managed lifecycle. Only
complete, independently checked units become scientific checkpoints.
"""
import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from zoneinfo import ZoneInfo

HERE = Path(__file__).absolute().parent
PREPARE = HERE.parent / 'lifecycle/prepare_run.py'
PROFILE = Path('/Users/yuqinli/.codex/skills/tacc-3090/references/profile.json')
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL',
            'OUT_OF_MEMORY', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}
MANAGED_FINAL = {'OWNER_COMPLETED', 'OWNER_FAILED_RETAINED', 'LIFECYCLE_FAILED_RETAINED'}


class Attention(RuntimeError):
    pass


def require(condition, code):
    if not condition:
        raise Attention(code)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class ControllerLock:
    """Never unlink the lock inode; child executables do not inherit its fd."""
    def __init__(self, directory):
        self.directory = Path(directory)
        self.handle = None

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.handle = (self.directory / 'CONTROLLER.lock').open('a+')
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise Attention('CONTROLLER_ALREADY_RUNNING')
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(json.dumps({'pid': os.getpid(), 'started_unix': time.time()}) + '\n')
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, *_):
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()


def absolute(path, base):
    p = Path(path)
    # Preserve the lexical venv interpreter path, not its resolved target.
    return Path(os.path.abspath(p if p.is_absolute() else base / p))


def safe_id(value):
    return isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,63}', value)


def expand(argv, values):
    return [substitute(str(arg), values) for arg in argv]


def substitute(arg, values):
    for key, value in values.items():
        arg = arg.replace('{' + key + '}', str(value))
    require(not re.search(r'\{(?:vm|plan|python)\}', arg), 'UNEXPANDED_VALIDATOR_PLACEHOLDER')
    return arg


def load_contract(path):
    """Read-only admission; never prepares a run or contacts TACC."""
    path = Path(path).absolute()
    cfg, base = read(path), path.parent
    require(cfg.get('schema') == 'continuous_reproduction_queue_v1', 'QUEUE_SCHEMA')
    require(isinstance(cfg.get('units'), list) and 0 < len(cfg['units']) <= 100, 'FINITE_NONEMPTY_QUEUE_REQUIRED')
    cfg['local_python'] = str(absolute(cfg.get('local_python', sys.executable), base))
    cfg['profile'] = str(absolute(cfg.get('profile', str(PROFILE)), base))
    cfg['poll_seconds'] = float(cfg.get('poll_seconds', 30))
    require(1 <= cfg['poll_seconds'] <= 60, 'POLL_INTERVAL_OUT_OF_RANGE')
    files = {str(path), str(PREPARE), cfg['local_python'], cfg['profile'], str(Path(__file__).absolute())}
    files.update(str(p) for p in PREPARE.parent.glob('*.py'))
    seen_units, seen_completions, seen_runs = set(), set(), set()
    for unit in cfg['units']:
        uid, ids = unit.get('unit_id'), unit.get('completion_ids')
        require(safe_id(uid) and uid not in seen_units, 'DUPLICATE_OR_INVALID_UNIT_ID')
        seen_units.add(uid)
        require(isinstance(ids, list) and ids and all(safe_id(i) for i in ids)
                and len(ids) == len(set(ids)) and not seen_completions.intersection(ids), 'COMPLETION_IDS_NOT_DISJOINT')
        seen_completions.update(ids)
        retries = unit.get('retry_specs', [])
        require(isinstance(retries, list) and len(retries) <= 10, 'BOUNDED_EXPLICIT_RETRIES_REQUIRED')
        deadline = unit.get('deadline_after_seconds')
        require(deadline is None or type(deadline) is int and 60 <= deadline <= 7 * 86400, 'INVALID_DEADLINE_WINDOW')
        variants = [{'spec': unit['spec'], 'completion_ids': ids}] + [
            {'spec': x, 'completion_ids': ids} if isinstance(x, str) else x for x in retries]
        normalized = []
        for variant in variants:
            spec_path = absolute(variant['spec'], base)
            spec = read(spec_path)
            run_id = spec.get('run_id')
            require(safe_id(run_id) and run_id not in seen_runs, 'RUN_IDS_MUST_BE_GLOBALLY_UNIQUE')
            seen_runs.add(run_id)
            cids = variant.get('completion_ids', ids)
            require(isinstance(cids, list) and cids and len(cids) == len(set(cids))
                    and set(cids) <= set(ids), 'INVALID_RETRY_COMPLETION_IDS')
            wall = spec['slurm']['wall_time'].split(':')
            require(len(wall) == 3 and all(s.isdigit() for s in wall), 'INVALID_WALL_TIME')
            seconds = sum(int(s) * scale for s, scale in zip(wall, (3600, 60, 1)))
            require(0 < seconds <= 3600 and spec['slurm']['gpus'] in (1, 2), 'TACC_RESOURCE_BOUND')
            require(spec.get('gpu_binding') == 'slurm_linux_minor_uuid', 'EXPLICIT_ALLOCATED_UUID_BINDING_REQUIRED')
            if deadline is not None:
                require(deadline > seconds, 'DEADLINE_WINDOW_MUST_EXCEED_WALLTIME')
            argv = variant.get('local_validator_argv', unit.get('local_validator_argv'))
            require(isinstance(argv, list) and argv and all(isinstance(x, str) for x in argv), 'LOCAL_INDEPENDENT_VALIDATOR_REQUIRED')
            checks = variant.get('completion_checks', [c for c in unit.get('completion_checks', [])
                                                       if c.get('completion_id') in cids])
            require(isinstance(checks, list), 'INVALID_COMPLETION_CHECKS')
            checked_ids = []
            for check in checks:
                require(check.get('completion_id') in cids and isinstance(check.get('argv'), list)
                        and check['argv'] and all(isinstance(x, str) for x in check['argv']), 'INVALID_COMPLETION_CHECK')
                checked_ids.append(check['completion_id'])
            require(len(checked_ids) == len(set(checked_ids)), 'DUPLICATE_COMPLETION_CHECK')
            validation_files = unit.get('validation_files', []) + variant.get('validation_files', [])
            for item in validation_files:
                files.add(str(absolute(item, base)))
            for args in [argv] + [c['argv'] for c in checks]:
                require(args[0] == '{python}' or absolute(args[0], base).is_file(), 'VALIDATOR_EXECUTABLE_REQUIRED')
                for token in args:
                    if '{' not in token and not token.startswith('-'):
                        candidate = absolute(token, base)
                        if candidate.is_file():
                            files.add(str(candidate))
            files.add(str(spec_path))
            for row in spec.get('uploads', []):
                require(Path(row['file']).is_absolute(), 'SPEC_UPLOAD_PATH_MUST_BE_ABSOLUTE')
                files.add(row['file'])
            for filename in spec.get('scientific_files', []):
                require(Path(filename).is_absolute(), 'SPEC_SCIENTIFIC_FILE_MUST_BE_ABSOLUTE')
                files.add(filename)
            normalized.append({'spec': str(spec_path), 'run_id': run_id, 'completion_ids': cids,
                               'local_validator_argv': argv, 'completion_checks': checks})
        unit['variants'] = normalized
        unit['validator_timeout_seconds'] = int(unit.get('validator_timeout_seconds', 120))
        require(0 < unit['validator_timeout_seconds'] <= 600, 'BOUNDED_LOCAL_VALIDATION_REQUIRED')
    return cfg, {name: sha(name) for name in sorted(files)}


def process_snapshot(pid):
    if not isinstance(pid, int) or pid <= 0:
        return None
    p = subprocess.run(['ps', '-p', str(pid), '-o', 'stat=', '-o', 'lstart=', '-o', 'command='],
                       capture_output=True, text=True, timeout=10)
    value = p.stdout.strip()
    if p.returncode or not value or value.split()[0].startswith('Z'):
        return None
    # R/S/D are transient scheduler states, not process identity. Bind the
    # process start time and command; use stat solely to reject zombies.
    return value.split(maxsplit=1)[1]


def process_matches(pid, script, expected_snapshot=None):
    snapshot = process_snapshot(pid)
    return bool(snapshot and str(script) in snapshot and
                (expected_snapshot is None or snapshot == expected_snapshot))


def verify_freeze(plan):
    for name, status in [('CONTROL_FREEZE.json', 'FROZEN_BEFORE_ALLOCATION'),
                         ('MANAGED_FREEZE.json', 'FROZEN_BEFORE_MANAGED_RUN')]:
        frozen = read(plan / name)
        require(frozen.get('status') == status, 'PLAN_NOT_FROZEN')
        for path, digest in frozen['files'].items():
            require(sha(absolute(path, plan)) == digest, 'FROZEN_PLAN_CHANGED:' + Path(path).name)


def scheduler_terminal(plan, driver):
    """Require actual main sacct row + empty squeue, never only a status flag."""
    control = read(plan / 'CONTROL.json')
    profile = read(plan / 'profile.json')
    found = []
    for name in ('SCHEDULER_LATEST.json', 'SCHEDULER_RELEASE.json'):
        path = plan / 'run' / name
        if not path.exists():
            continue
        evidence = read(path)
        rows = [line.split('|') for line in evidence.get('sacct', '').splitlines()
                if line.split('|')[0] == str(driver.get('job_id'))]
        if len(rows) == 1 and len(rows[0]) >= 5:
            row = rows[0]
            require(row[1:3] == [control['run']['session_name'], profile['username']], 'SCHEDULER_IDENTITY_MISMATCH')
            state = row[3].split()[0].rstrip('+')
            if state in TERMINAL and not evidence.get('squeue', '').strip():
                found.append((path, state, row[4]))
    require(found, 'TERMINAL_AND_RELEASE_NOT_PROVEN')
    require(len({(x[1], x[2]) for x in found}) == 1, 'CONFLICTING_TERMINAL_RECEIPTS')
    return found[-1]


def verify_outputs(plan, driver, complete=False):
    vm, run = plan / 'run/vm', plan / 'run'
    manifest_path = run / 'REMOTE_OUTPUT_MANIFEST.json'
    manifest = read(manifest_path)
    require(manifest.get('run_id') == driver['session'] and str(manifest.get('slurm_job_id')) == str(driver['job_id']),
            'OUTPUT_MANIFEST_IDENTITY_MISMATCH')
    worker_cfg = read(plan / 'worker.json')
    allowed = set(worker_cfg['required_outputs'] + ['scientific.log', 'validation.log', 'WORKER_RECEIPT.json'])
    seen, proof = set(), {str(manifest_path): sha(manifest_path)}
    for row in manifest['files']:
        name = row['path']
        relative = Path(name)
        require(name in allowed and name not in seen and not relative.is_absolute()
                and '..' not in relative.parts, 'UNEXPECTED_OR_DUPLICATE_MANIFEST_PATH')
        path = vm / name
        require(not path.is_symlink() and vm.resolve() in path.resolve().parents, 'OUTPUT_SYMLINK_OR_ESCAPE')
        require(path.stat().st_size == row['bytes'] and sha(path) == row['sha256'], 'COLLECTED_OUTPUT_HASH_MISMATCH:' + name)
        seen.add(name)
        proof[str(path)] = row['sha256']
    require('WORKER_RECEIPT.json' in seen, 'WORKER_RECEIPT_NOT_COLLECTED')
    worker = read(vm / 'WORKER_RECEIPT.json')
    require(worker.get('run_id') == driver['session'] and str(worker.get('slurm_job_id')) == str(driver['job_id'])
            and worker.get('worker_sha256') == sha(plan / 'worker.py')
            and worker.get('config_sha256') == sha(plan / 'worker.json'), 'WORKER_IDENTITY_MISMATCH')
    if complete:
        require(set(worker_cfg['required_outputs']) <= seen and worker.get('scientific_success') is True
                and worker.get('scientific_exit_code') == 0 and worker.get('validation_exit_code') == 0,
                'WORKER_SCIENTIFIC_UNIT_INCOMPLETE')
    return worker, proof


def inspect_attempt(attempt):
    """WAIT/COMPLETE/FAILED only; unknown submission raises and never launches."""
    plan = Path(attempt['plan'])
    verify_freeze(plan)
    control = read(plan / 'CONTROL.json')
    dpath, mpath = plan / 'run/DRIVER_RECEIPT.json', plan / 'managed_records/MANAGED_RECEIPT.json'
    managed = read(mpath) if mpath.exists() else {}
    driver = read(dpath) if dpath.exists() else {}
    if driver:
        expected = {'session': control['run']['session_name'], 'attempt_id': control['run']['attempt_id'],
                    'control_freeze_sha256': sha(plan / 'CONTROL_FREEZE.json'),
                    'entry_driver_sha256': sha(plan / 'owner.py')}
        require(all(driver.get(k) == v for k, v in expected.items()), 'DRIVER_IDENTITY_MISMATCH')
    if managed:
        expected = {'source_sha256': sha(plan / 'managed_run.py'), 'managed_config_sha256': sha(plan / 'MANAGED.json'),
                    'control_sha256': sha(plan / 'CONTROL.json'), 'control_freeze_sha256': sha(plan / 'CONTROL_FREEZE.json'),
                    'run': str(plan / 'run')}
        require(all(managed.get(k) == v for k, v in expected.items()), 'MANAGED_IDENTITY_MISMATCH')
    own = managed.get('children', {}).get('OWNER', {})
    active_manager = process_matches(attempt.get('pid'), plan / 'managed_run.py', attempt.get('process_snapshot'))
    active_owner = process_matches(own.get('pid'), plan / 'owner.py')
    if active_manager or active_owner:
        return {'kind': 'WAIT', 'job_id': driver.get('job_id'), 'owner_status': driver.get('status')}
    if not managed or not driver or managed.get('status') not in MANAGED_FINAL or 'finished_unix' not in driver:
        raise Attention('SUBMISSION_OR_OWNER_STATE_UNKNOWN_NO_RESUBMIT')
    require(driver.get('released') is True, 'OWNER_RELEASE_NOT_CONFIRMED')
    require(str(driver.get('job_id', '')).isdigit(), 'NO_CONFIRMED_JOB_ID_ATTENTION')
    spath, state, exit_code = scheduler_terminal(plan, driver)
    require(managed.get('status') != 'LIFECYCLE_FAILED_RETAINED' and not managed.get('lifecycle_error'), 'MANAGED_LIFECYCLE_ERROR')
    good = (managed.get('status') == 'OWNER_COMPLETED' and managed.get('owner_returncode') == 0
            and driver.get('status') == 'OUTPUTS_VERIFIED' and driver.get('local_outputs_verified') is True
            and state == 'COMPLETED' and exit_code == '0:0')
    proof = {str(p): sha(p) for p in (dpath, mpath, spath)}
    worker, output_error = None, None
    try:
        worker, output_proof = verify_outputs(plan, driver, complete=good)
        proof.update(output_proof)
    except (Attention, OSError, ValueError, KeyError, TypeError) as error:
        if good:
            raise
        output_error = str(error)
    if not good:
        # Retry is never permitted unless the failed run retained a verified scheduler log.
        log_receipt = plan / 'run/SCHEDULER_LOG_VERIFIED.json'
        require(log_receipt.is_file(), 'FAILED_RUN_LOG_NOT_VERIFIED')
        log = read(log_receipt)
        require(Path(log['file']).stat().st_size == log['bytes'] and sha(log['file']) == log['sha256'], 'FAILED_RUN_LOG_HASH_MISMATCH')
        proof.update({str(log_receipt): sha(log_receipt), log['file']: log['sha256']})
    controlled_worker_timeout = bool(worker and worker.get('error_type') == 'TimeoutError'
                                     and worker.get('error') == 'SCIENTIFIC_UNIT_TIMEOUT')
    timeout = state == 'TIMEOUT' or controlled_worker_timeout
    if worker and worker.get('scientific_success') is not True and worker.get('error') and not controlled_worker_timeout:
        # A scheduler timeout does not erase an already recorded environment or
        # validation error. Such failures still need attention.
        timeout = False
    return {'kind': 'COMPLETE' if good else 'FAILED', 'timeout': timeout, 'scheduler_state': state,
            'scheduler_exit_code': exit_code, 'job_id': driver['job_id'], 'worker': worker,
            'output_error': output_error, 'proof_files': proof}


class QueueController:
    def __init__(self, contract_path, directory):
        self.contract_path, self.directory = Path(contract_path).absolute(), Path(directory).absolute()
        self.cfg, self.inputs = load_contract(self.contract_path)
        self.state_path = self.directory / 'CHECKPOINT.json'
        self.children = {}
        run_ids = {v['run_id'] for u in self.cfg['units'] for v in u['variants']}
        # Existing project plans/receipts are not imported as new work. A queue
        # restart may of course find its own previously frozen plans.
        for existing in Path('/Users/yuqinli/Documents/ChatGPT/VLN/baselines/runs').glob('**/CONTROL.json'):
            if self.directory in existing.absolute().parents:
                continue
            require(read(existing).get('run', {}).get('session_name') not in run_ids,
                    'RUN_ID_ALREADY_HAS_PROJECT_PLAN:' + str(existing))
        if self.state_path.exists():
            self.state = read(self.state_path)
            require(self.state.get('schema') == 'continuous_reproduction_checkpoint_v1', 'CHECKPOINT_SCHEMA')
            require(self.state.get('input_files') == self.inputs, 'QUEUE_INPUTS_CHANGED_AFTER_ADMISSION')
        else:
            # A deleted checkpoint must not erase possible launch history.
            require(not (self.directory / 'attempts').exists(), 'ATTEMPT_HISTORY_WITHOUT_CHECKPOINT')
            self.state = {'schema': 'continuous_reproduction_checkpoint_v1', 'status': 'READY',
                          'contract': str(self.contract_path), 'input_files': self.inputs, 'created_unix': time.time(),
                          'units': {u['unit_id']: {'status': 'NEW', 'attempts': []} for u in self.cfg['units']},
                          'completed': {}, 'events': []}
            self.save('QUEUE_ADMITTED')

    def save(self, event=None):
        self.state['updated_unix'] = time.time()
        if event:
            self.state['events'].append({'event': event, 'unix': time.time()})
        atomic_json(self.state_path, self.state)

    def verify_inputs(self):
        for path, digest in self.inputs.items():
            require(sha(path) == digest, 'QUEUE_INPUT_CHANGED:' + Path(path).name)

    def verify_completions(self):
        for cid, completion in self.state['completed'].items():
            require(completion.get('independently_validated') is True, 'UNVERIFIED_COMPLETION:' + cid)
            for path, digest in completion['proof_files'].items():
                require(sha(path) == digest, 'COMPLETION_EVIDENCE_CHANGED:' + cid)

    def prepare(self, unit, variant_index):
        variant = unit['variants'][variant_index]
        require(not set(variant['completion_ids']) & set(self.state['completed']), 'RETRY_WOULD_REPEAT_COMPLETED_SHARD')
        self.verify_inputs()
        entry = self.state['units'][unit['unit_id']]
        attempt_dir = self.directory / 'attempts' / (unit['unit_id'] + '_' + str(variant_index) + '_' + uuid.uuid4().hex[:12])
        attempt = {'variant_index': variant_index, 'run_id': variant['run_id'], 'plan': str(attempt_dir / 'plan'),
                   'directory': str(attempt_dir), 'phase': 'PREPARING', 'created_unix': time.time()}
        entry['attempts'].append(attempt)
        entry['status'] = 'ACTIVE'
        self.save('PREPARE_INTENT:' + unit['unit_id'])
        attempt_dir.mkdir(parents=True, exist_ok=False)
        original = read(variant['spec'])
        derived = copy.deepcopy(original)
        deadline = unit.get('deadline_after_seconds')
        changes = []
        if deadline is not None:
            now = dt.datetime.now(ZoneInfo('Asia/Shanghai'))
            value = (now + dt.timedelta(seconds=deadline)).strftime('%Y-%m-%dT%H:%M:%S')
            changes.append({'field': 'slurm.deadline', 'before': derived['slurm'].get('deadline'), 'after': value})
            derived['slurm']['deadline'] = value
        atomic_json(attempt_dir / 'DERIVED_SPEC.json', derived)
        atomic_json(attempt_dir / 'SPEC_DERIVATION.json', {
            'original_spec': variant['spec'], 'original_sha256': sha(variant['spec']),
            'derived_sha256': sha(attempt_dir / 'DERIVED_SPEC.json'), 'changes': changes,
            'clock_timezone': 'Asia/Shanghai', 'run_id_preserved': derived['run_id'] == original['run_id'],
            'scientific_parameters_unchanged': derived.get('science') == original.get('science')})
        args = [self.cfg['local_python'], str(PREPARE), '--spec', str(attempt_dir / 'DERIVED_SPEC.json'),
                '--directory', attempt['plan'], '--python', self.cfg['local_python'], '--profile', self.cfg['profile']]
        with (attempt_dir / 'PREPARE.log').open('wb') as log:
            result = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        require(result.returncode == 0, 'PREPARE_FAILED_RETAINED')
        verify_freeze(Path(attempt['plan']))
        attempt['phase'] = 'PREPARED'
        self.save('PREPARED:' + unit['unit_id'])
        return attempt

    def launch(self, attempt):
        plan = Path(attempt['plan'])
        require(attempt['phase'] == 'PREPARED', 'FRESH_PREPARED_ATTEMPT_REQUIRED')
        require(not (plan / 'run').exists() and not (plan / 'managed_records').exists(), 'EXISTING_OWNER_HISTORY_NO_RESUBMIT')
        self.verify_inputs()
        verify_freeze(plan)
        args = [self.cfg['local_python'], str(plan / 'managed_run.py'), '--config', str(plan / 'MANAGED.json'),
                '--freeze', str(plan / 'MANAGED_FREEZE.json')]
        # Persist before Popen: a crash in the following gap is uncertain, never a reason to retry.
        attempt.update(phase='LAUNCH_INTENT', launch_argv=args, launch_intent_unix=time.time())
        self.save('LAUNCH_INTENT:' + attempt['run_id'])
        with (Path(attempt['directory']) / 'MANAGED_CONSOLE.log').open('ab') as log:
            child = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True)
        self.children[attempt['plan']] = child
        attempt.update(phase='WAITING', pid=child.pid, process_snapshot=process_snapshot(child.pid))
        self.save('MANAGED_STARTED:' + attempt['run_id'])

    def validator(self, unit, attempt, argv, label):
        """Validators may write reports, so they run on full copies, not evidence."""
        vm = Path(attempt['plan']) / 'run/vm'
        directory = Path(attempt['directory']) / ('local_validation_' + label)
        directory.mkdir(exist_ok=True)
        size = sum(p.stat().st_size for p in vm.rglob('*') if p.is_file())
        reserve = read(Path(attempt['plan']) / 'SPEC.json').get('reserve_bytes', 4 << 30)
        require(shutil.disk_usage(directory).free >= reserve + size, 'LOCAL_VALIDATION_COPY_RESERVE')
        require(not any(p.is_symlink() for p in vm.rglob('*')), 'COLLECTED_OUTPUT_CONTAINS_SYMLINK')
        with tempfile.TemporaryDirectory(prefix='verification_', dir=directory) as temporary:
            copied = Path(temporary) / 'vm'
            shutil.copytree(vm, copied)
            args = expand(argv, {'python': self.cfg['local_python'], 'vm': copied, 'plan': attempt['plan']})
            log_path = directory / 'VALIDATOR.log'
            timed_out = False
            with log_path.open('wb') as log:
                try:
                    result = subprocess.run(args, cwd=self.contract_path.parent, stdin=subprocess.DEVNULL,
                                            stdout=log, stderr=subprocess.STDOUT, timeout=unit['validator_timeout_seconds'])
                    returncode = result.returncode
                except subprocess.TimeoutExpired:
                    timed_out, returncode = True, None
            report = {'argv': args, 'exit_code': returncode, 'timed_out': timed_out, 'original_outputs_unchanged': True,
                      'validation_on_disposable_copy': True, 'finished_unix': time.time(), 'log_sha256': sha(log_path)}
            atomic_json(directory / 'VALIDATOR_RECEIPT.json', report)
            require(not timed_out, 'LOCAL_VALIDATOR_TIMEOUT_ATTENTION')
            return returncode == 0, {str(p): sha(p) for p in (log_path, directory / 'VALIDATOR_RECEIPT.json')}

    def record_completion(self, ids, unit, attempt, proof):
        for cid in ids:
            require(cid not in self.state['completed'], 'DUPLICATE_COMPLETION')
            self.state['completed'][cid] = {'unit_id': unit['unit_id'], 'plan': attempt['plan'],
                                          'run_id': attempt['run_id'], 'independently_validated': True,
                                          'proof_files': proof, 'completed_unix': time.time()}
        self.save('COMPLETIONS_VERIFIED:' + ','.join(ids))

    def next_retry(self, unit, entry):
        """Use each explicit candidate once, selecting only unfinished shards."""
        remaining = set(unit['completion_ids']) - set(self.state['completed'])
        require(entry.get('timeout_retry_authorized') is True, 'RETRY_NOT_AUTHORIZED_BY_TIMEOUT')
        used = {a['variant_index'] for a in entry['attempts']}
        skipped = entry.setdefault('skipped_retry_candidates', {})
        for index, candidate in enumerate(unit['variants']):
            if index == 0 or index in used:
                continue
            ids = set(candidate['completion_ids'])
            if not ids <= remaining:
                skipped[str(index)] = 'already_complete' if not ids & remaining else 'overlaps_completed_shards'
                continue
            self.save('BOUNDED_EXPLICIT_TIMEOUT_RETRY')
            fresh = self.prepare(unit, index)
            self.launch(fresh)
            return 'WAIT'
        raise Attention('TIMEOUT_RETRY_CANDIDATES_EXHAUSTED_WITH_UNFINISHED_SHARDS')

    def advance(self):
        """One state transition/poll; useful for deterministic offline tests."""
        self.verify_inputs()
        self.verify_completions()
        if self.state.get('status') == 'ATTENTION':
            return 'ATTENTION'
        for unit in self.cfg['units']:
            entry = self.state['units'][unit['unit_id']]
            if set(unit['completion_ids']) <= set(self.state['completed']):
                entry['status'] = 'COMPLETE'
                continue
            if not entry['attempts']:
                attempt = self.prepare(unit, 0)
                self.launch(attempt)
                return 'WAIT'
            attempt = entry['attempts'][-1]
            plan = Path(attempt['plan'])
            if attempt['phase'] == 'PREPARING':
                # The preparer is read-only with respect to TACC; only a fully frozen plan can resume.
                verify_freeze(plan)
                require(not (plan / 'run').exists() and not (plan / 'managed_records').exists(), 'PREPARATION_HISTORY_AMBIGUOUS')
                attempt['phase'] = 'PREPARED'
                self.save('RECOVERED_PREPARED_PLAN')
            if attempt['phase'] == 'PREPARED':
                if (plan / 'run').exists() or (plan / 'managed_records').exists():
                    attempt['phase'] = 'LAUNCH_INTENT'
                    self.save('EXISTING_RECEIPT_RECOVERY_ONLY')
                else:
                    self.launch(attempt)
                    return 'WAIT'
            child = self.children.get(attempt['plan'])
            if child is not None:
                child.poll()  # Reap our local child; a zombie is not an active owner.
            observed = inspect_attempt(attempt)
            attempt['last_observation'] = {k: v for k, v in observed.items() if k not in ('proof_files', 'worker')}
            self.save()
            if observed['kind'] == 'WAIT':
                return 'WAIT'
            variant = unit['variants'][attempt['variant_index']]
            attempt['phase'] = 'TERMINAL_RELEASED'
            self.save('TERMINAL_RELEASED:' + attempt['run_id'])
            if observed['kind'] == 'COMPLETE':
                if not set(variant['completion_ids']) <= set(self.state['completed']):
                    passed, proof = self.validator(unit, attempt, variant['local_validator_argv'], 'whole_unit')
                    require(passed, 'INDEPENDENT_LOCAL_VALIDATION_FAILED')
                    self.record_completion(variant['completion_ids'], unit, attempt, {**observed['proof_files'], **proof})
                if set(unit['completion_ids']) <= set(self.state['completed']):
                    entry['status'] = 'COMPLETE'
                    self.save('UNIT_COMPLETE:' + unit['unit_id'])
                    return 'PROGRESS'
                return self.next_retry(unit, entry)
            # A failed/timed-out pair may contain one complete shard. Only explicit,
            # independent per-shard checks can checkpoint that shard, never log counts.
            if observed['worker'] and not observed['output_error']:
                for check in variant['completion_checks']:
                    cid = check['completion_id']
                    if cid in self.state['completed']:
                        continue
                    passed, proof = self.validator(unit, attempt, check['argv'], cid)
                    if passed:
                        self.record_completion([cid], unit, attempt, {**observed['proof_files'], **proof})
            require(observed['timeout'], 'FAILED_SCIENTIFIC_OR_ENVIRONMENT_UNIT_ATTENTION')
            entry['timeout_retry_authorized'] = True
            self.save('CONTROLLED_TIMEOUT_RECORDED')
            remaining = set(unit['completion_ids']) - set(self.state['completed'])
            if not remaining:
                entry['status'] = 'COMPLETE'
                self.save('TIMED_OUT_BATCH_ALL_SHARDS_INDEPENDENTLY_COMPLETE')
                return 'PROGRESS'
            return self.next_retry(unit, entry)
        self.state['status'] = 'COMPLETE'
        self.save('FINITE_QUEUE_COMPLETE_NO_MORE_ALLOCATION')
        return 'COMPLETE'

    def run(self):
        while True:
            try:
                outcome = self.advance()
            except (Attention, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
                self.state.update(status='ATTENTION', error_type=type(error).__name__, error=str(error))
                self.save('ATTENTION_NO_NEW_SUBMISSION')
                print(json.dumps({'status': 'ATTENTION', 'error': str(error), 'checkpoint': str(self.state_path)}), flush=True)
                return 2
            if outcome in ('COMPLETE', 'ATTENTION'):
                print(json.dumps({'status': outcome, 'completed': sorted(self.state['completed']),
                                  'checkpoint': str(self.state_path)}), flush=True)
                return 0 if outcome == 'COMPLETE' else 2
            if outcome == 'WAIT':
                time.sleep(self.cfg['poll_seconds'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue', type=Path, required=True)
    p.add_argument('--directory', type=Path, help='Persistent controller directory; use the same directory after restart')
    p.add_argument('--check', action='store_true', help='Read/validate inputs only; no state, preparation, SSH or allocation')
    args = p.parse_args()
    if args.check:
        cfg, files = load_contract(args.queue)
        print(json.dumps({'status': 'VALID_QUEUE_NOT_STARTED', 'units': [u['unit_id'] for u in cfg['units']],
                          'bound_input_count': len(files), 'maximum_attempts': sum(len(u['variants']) for u in cfg['units'])}, indent=2))
        return 0
    require(args.directory is not None, 'PERSISTENT_CONTROLLER_DIRECTORY_REQUIRED')
    with ControllerLock(args.directory):
        return QueueController(args.queue, args.directory).run()


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (Attention, OSError, ValueError, KeyError, TypeError) as error:
        print(json.dumps({'status': 'ATTENTION', 'error': str(error)}), file=sys.stderr)
        raise SystemExit(2)
