"""Trusted-host bridge from a reserved call to the existing managed lifecycle.

The editor must not receive this API or arbitrary SPEC/credential paths. Prepare
is local only. Launch checks the whole account queue through the frozen strict
SSH transport, then invokes only the copied managed_run.py. The account check
is a snapshot, not a distributed lock against manual submissions from elsewhere.
Scientific correctness still depends on the operator's frozen forward runner
and validator. Offline receipt fixtures do not establish a real GPU experiment.
"""
from contextlib import contextmanager, ExitStack
import copy
import fcntl
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from .io import atomic_json, json_value, read_json, require, sha256
from .session import Session, episode_keys, reference, verify_reference

PROJECT = Path(__file__).absolute().parent.parent
PREPARER = PROJECT / 'baselines/lifecycle/prepare_run.py'
PROFILE = Path('/Users/yuqinli/.codex/skills/tacc-3090/references/profile.json')
FEEDBACK = 'forward/editlab_feedback.json'
FORWARD_OUTPUTS = ['forward/' + name for name in (
    'editlab_feedback.json', 'forward_context.json', 'navigation_runtime_receipt.json',
    'navigation_episode_metrics.jsonl', 'diagnostic_activations.npz', 'patch_application.json',
    'independent_validation.json', 'run_manifest.json', 'resolved_config.json')]
REQUEST = 'weightlab_request.json'
CANDIDATE = 'weightlab_candidate.npz'
CANDIDATE_MANIFEST = 'weightlab_candidate_manifest.json'
TERMINAL = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'NODE_FAIL', 'OUT_OF_MEMORY',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'}


@contextmanager
def _lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            require(False, 'MANAGED_BRIDGE_ALREADY_ACTIVE')
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _reservation(session, call_id):
    require(re.fullmatch(r'call_[0-9a-f]{32}', call_id or ''), 'INVALID_CALL_ID')
    state = session.status()  # verifies the Session's reservation and all inputs
    calls = [r for r in state['calls'] if r['call_id'] == call_id]
    require(len(calls) == 1 and calls[0]['status'] == 'reserved', 'RESERVED_CALL_REQUIRED')
    reservation = read_json(session.root / 'calls' / call_id / 'RESERVATION.json')
    require(reservation['kind'] in ('forward', 'activation_probe', 'confirmation'),
            'REPLAY_CALL_CANNOT_LAUNCH_NAVIGATION')
    candidate = session.candidate_files(reservation['candidate'])['effective_npz']
    require(candidate['sha256'] == reservation['candidate_tensor_sha256'], 'CANDIDATE_CHANGED')
    return reservation, candidate


def _flag(argv, flag, value):
    require(isinstance(argv, list) and argv.count(flag) == 1, 'EXPLICIT_RUNNER_FLAG_REQUIRED:' + flag)
    index = argv.index(flag)
    require(index + 1 < len(argv) and argv[index + 1] == value, 'RUNNER_FLAG_BINDING:' + flag)


def _verify_freezes(plan):
    refs = []
    for name, status in [('CONTROL_FREEZE.json', 'FROZEN_BEFORE_ALLOCATION'),
                         ('MANAGED_FREEZE.json', 'FROZEN_BEFORE_MANAGED_RUN')]:
        path = plan / name
        doc = read_json(path)
        require(doc.get('status') == status and doc.get('files'), 'PLAN_FREEZE_REQUIRED')
        for file, digest in doc['files'].items():
            target = Path(file) if Path(file).is_absolute() else plan / file
            require(sha256(target) == digest, 'FROZEN_PLAN_CHANGED:' + target.name)
        refs.append(reference(path))
    return refs


def prepare_managed_call(session_root, call_id, operator_spec, directory, *,
                         profile=PROFILE, python=None):
    """Freeze a trusted operator SPEC. No SSH, Slurm, model, or budget refund.

    The runner must accept --request {source}/weightlab_request.json and
    --candidate-tensors {source}/weightlab_candidate.npz and --candidate-manifest
    {source}/weightlab_candidate_manifest.json. Its validator must
    accept --request with the same path. Both are supplied by the operator.
    """
    session = Session(session_root)
    with _lock(session.root / 'MANAGED_BRIDGE.lock'):
        call, candidate = _reservation(session, call_id)
        marker = session.root / 'calls' / call_id / 'MANAGED_BINDING.json'
        require(not marker.exists(), 'CALL_ALREADY_PREPARED_NO_REUSE')
        directory = Path(directory).absolute()
        require(session.root.resolve() in directory.resolve().parents,
                'BRIDGE_DIRECTORY_MUST_BE_IN_SESSION')
        require(not directory.exists(), 'FRESH_BRIDGE_DIRECTORY_REQUIRED')
        spec_path = Path(operator_spec).absolute()
        spec = copy.deepcopy(read_json(spec_path))
        require(re.fullmatch(r'[A-Za-z0-9_-]{1,63}', spec.get('run_id', '')), 'UNIQUE_RUN_ID_REQUIRED')
        require(spec.get('gpu_binding') == 'slurm_linux_minor_uuid', 'ALLOCATED_GPU_BINDING_REQUIRED')
        require(spec.get('reserve_bytes', 4 << 30) >= 4 << 30, 'FOUR_GIB_RESERVE_REQUIRED')
        require(spec['slurm']['gpus'] in (1, 2), 'AT_MOST_TWO_GPUS')
        _flag(spec['science']['argv'], '--request', '{source}/' + REQUEST)
        _flag(spec['science']['argv'], '--candidate-tensors', '{source}/' + CANDIDATE)
        _flag(spec['science']['argv'], '--candidate-manifest', '{source}/' + CANDIDATE_MANIFEST)
        _flag(spec['science']['argv'], '--output', '{output}/forward')
        _flag(spec['science']['validator_argv'], '--request', '{source}/' + REQUEST)
        _flag(spec['science']['validator_argv'], '--validate-output', '{output}/forward')
        require(set(FORWARD_OUTPUTS) <= set(spec['science']['required_outputs']),
                'COMPLETE_SCIENTIFIC_EVIDENCE_REQUIRED')
        require(not {REQUEST, CANDIDATE, CANDIDATE_MANIFEST} & {r['remote_name'] for r in spec.get('uploads', [])},
                'BRIDGE_UPLOAD_COLLISION')
        required = spec.get('reserve_bytes', 4 << 30) + candidate['bytes'] + (2 << 20)
        require(shutil.disk_usage(session.root).free >= required, 'FREE_SPACE_RESERVE')
        directory.mkdir()
        # A durable marker precedes preparation; ambiguous partial preparation
        # needs operator reconciliation rather than another plan for this token.
        binding = {'schema': 'weightlab_managed_binding_v1', 'call_id': call_id,
                   'session_root': str(session.root), 'directory': str(directory),
                   'phase': 'PREPARING', 'operator_spec': reference(spec_path)}
        atomic_json(marker, binding)
        try:
            shutil.copyfile(spec_path, directory / 'OPERATOR_SPEC.json')
            require(sha256(directory / 'OPERATOR_SPEC.json') == binding['operator_spec']['sha256'], 'OPERATOR_SPEC_CHANGED')
            shutil.copyfile(candidate['path'], directory / CANDIDATE)
            require(sha256(directory / CANDIDATE) == candidate['sha256'], 'CANDIDATE_COPY_CHANGED')
            candidate_manifest = session.candidate_files(call['candidate'])['manifest']
            shutil.copyfile(candidate_manifest['path'], directory / CANDIDATE_MANIFEST)
            require(sha256(directory / CANDIDATE_MANIFEST) == candidate_manifest['sha256'], 'CANDIDATE_MANIFEST_COPY_CHANGED')
            request = {'schema': 'weightlab_falcon_forward_request_v1',
                       'phase': 'FROZEN' if call['kind'] == 'confirmation' else 'OPEN',
                       **{k: call[k] for k in ('call_id', 'call_token', 'kind', 'candidate', 'split',
                                             'candidate_tensor_sha256')},
                       'checkpoint_sha256': session.config['checkpoint_sha256'],
                       'episode_keys': call['request']['episode_keys'],
                       'transition_steps': call['request']['transition_steps'],
                       'reservation_sha256': sha256(session.root / 'calls' / call_id / 'RESERVATION.json')}
            atomic_json(directory / REQUEST, request)
            spec.setdefault('uploads', []).extend([
                {'file': str(directory / name), 'remote_name': name} for name in (REQUEST, CANDIDATE, CANDIDATE_MANIFEST)])
            atomic_json(directory / 'DERIVED_SPEC.json', spec)
            python = str(Path(python or sys.executable).absolute())
            argv = [python, str(PREPARER), '--spec', str(directory / 'DERIVED_SPEC.json'),
                    '--directory', str(directory / 'plan'), '--python', python,
                    '--profile', str(Path(profile).absolute())]
            with (directory / 'PREPARE.log').open('wb') as log:
                result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=120)
            require(result.returncode == 0, 'PREPARATION_FAILED_RETAINED')
            plan = directory / 'plan'
            freezes = _verify_freezes(plan)
            binding.update(phase='PREPARED', plan=str(plan), run_id=spec['run_id'],
                           refs=freezes + [reference(directory / n) for n in
                                          (REQUEST, CANDIDATE, CANDIDATE_MANIFEST, 'OPERATOR_SPEC.json', 'DERIVED_SPEC.json')],
                           launch_argv=[python, str(plan / 'managed_run.py'), '--config',
                                        str(plan / 'MANAGED.json'), '--freeze', str(plan / 'MANAGED_FREEZE.json')])
            atomic_json(marker, binding)
            return binding
        except BaseException as error:
            atomic_json(directory / 'PREPARE_ERROR.json', {'error_type': type(error).__name__,
                        'error': str(error), 'budget_refunded': False, 'no_submission_attempted': True})
            raise


def _binding(session, call_id):
    call, _ = _reservation(session, call_id)
    binding = read_json(session.root / 'calls' / call_id / 'MANAGED_BINDING.json')
    require(binding.get('schema') == 'weightlab_managed_binding_v1' and binding['call_id'] == call_id
            and binding['session_root'] == str(session.root) and binding['phase'] == 'PREPARED',
            'PREPARED_BOUND_CALL_REQUIRED')
    for row in binding['refs']:
        verify_reference(row)
    plan = Path(binding['plan'])
    require(plan == Path(binding['directory']) / 'plan' and session.root.resolve() in plan.resolve().parents,
            'PLAN_OUTSIDE_SESSION')
    _verify_freezes(plan)
    request = read_json(Path(binding['directory']) / REQUEST)
    require(all(request[k] == call[k] for k in ('call_id', 'call_token', 'kind', 'candidate', 'split',
                                               'candidate_tensor_sha256')), 'REQUEST_IDENTITY_CHANGED')
    require(request['episode_keys'] == call['request']['episode_keys']
            and request['transition_steps'] == call['request']['transition_steps'], 'REQUEST_LIMITS_CHANGED')
    require(request['checkpoint_sha256'] == session.config['checkpoint_sha256'], 'CHECKPOINT_CHANGED')
    return call, binding, plan


@contextmanager
def _baseline_guard():
    """Cooperate with all existing baseline queue locks; never alter checkpoints."""
    root = PROJECT / 'baselines/runs'
    with ExitStack() as stack:
        for checkpoint in sorted(root.glob('*/CHECKPOINT.json')):
            stack.enter_context(_lock(checkpoint.parent / 'CONTROLLER.lock'))
            state = read_json(checkpoint)
            require(state.get('status') == 'COMPLETE' and
                    all(u.get('status') == 'COMPLETE' for u in state.get('units', {}).values()),
                    'BASELINE_QUEUE_UNFINISHED_OR_UNCERTAIN')
        for path in root.glob('**/DRIVER_RECEIPT.json'):
            driver = read_json(path)
            require(driver.get('released') is True, 'BASELINE_OWNER_UNRELEASED_OR_UNCERTAIN')
        process = subprocess.run(['ps', '-axo', 'command='], capture_output=True, text=True, check=True)
        for line in process.stdout.splitlines():
            require(not (str(root) in line and any(s in line for s in ('/owner.py', '/managed_run.py'))),
                    'BASELINE_OWNER_STILL_ACTIVE')
        yield


def _account_guard(plan, python):
    """Only read-only SSH; stdout contains account name and queue status, no keys."""
    code = ("import sys,json;sys.path.insert(0,sys.argv[1]);import preflight;"
            "from transport import Remote,read;p=read(sys.argv[1]+'/profile.json');preflight.verify_keys(p);"
            "remote_code='import json,subprocess\\np=subprocess.run('+repr(['squeue','-h','-u',p['username'],"
            "'-o','%i|%u|%j|%T'])+',stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)"
            "\\nassert p.returncode==0,p.stderr\\nprint(json.dumps({\\\"squeue\\\":p.stdout.strip()}))';"
            "r=Remote(p).json(remote_code,timeout=35,retry=False);r['username']=p['username'];print(json.dumps(r))")
    result = subprocess.run([python, '-c', code, str(plan)], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=45)
    require(result.returncode == 0, 'ACCOUNT_GUARD_UNAVAILABLE_NO_LAUNCH')
    import json
    guard = json.loads(result.stdout)
    profile = read_json(plan / 'profile.json')
    require(guard.get('username') == profile['username'] and isinstance(guard.get('squeue'), str),
            'ACCOUNT_GUARD_IDENTITY')
    require(not guard['squeue'].strip(), 'ACCOUNT_HAS_PENDING_OR_RUNNING_JOB')
    return {'username': guard['username'], 'squeue': guard['squeue'], 'observed_unix': time.time(),
            'scope': 'snapshot; does not lock other hosts or manual submissions'}


def launch_managed_call(session_root, call_id):
    """Operator-only actual launch. Never call this from untrusted editor code."""
    session = Session(session_root)
    with _lock(PROJECT / '.work/weightlab_allocation.lock'), _lock(session.root / 'MANAGED_BRIDGE.lock'):
        call, binding, plan = _binding(session, call_id)
        directory = Path(binding['directory'])
        intent = directory / 'LAUNCH_INTENT.json'
        require(not intent.exists(), 'LAUNCH_ALREADY_ATTEMPTED_RECONCILE_NO_RESUBMIT')
        require(not (plan / 'run').exists() and not (plan / 'managed_records').exists(),
                'EXISTING_OWNER_HISTORY_NO_RESUBMIT')
        registry_path = PROJECT / '.work/weightlab_allocations.json'
        registry = read_json(registry_path) if registry_path.exists() else {'schema': 'weightlab_allocations_v1', 'launches': []}
        require(registry.get('schema') == 'weightlab_allocations_v1'
                and isinstance(registry.get('launches'), list), 'ALLOCATION_REGISTRY_SCHEMA')
        # Covers the gap between local Popen and the owner's sbatch. An empty
        # remote queue alone cannot prove another local launch is safe.
        for prior in registry['launches']:
            previous = Path(prior['plan'])
            require(sha256(previous / 'CONTROL_FREEZE.json') == prior['control_freeze_sha256'],
                    'PRIOR_ALLOCATION_FREEZE_CHANGED')
            previous_control = read_json(previous / 'CONTROL.json')
            driver_path = previous / 'run/DRIVER_RECEIPT.json'
            require(driver_path.is_file(), 'PRIOR_LAUNCH_UNKNOWN_NO_NEW_ALLOCATION')
            previous_driver = read_json(driver_path)
            require(previous_driver.get('session') == prior['run_id']
                    and previous_driver.get('attempt_id') == previous_control['run']['attempt_id']
                    and previous_driver.get('control_freeze_sha256') == prior['control_freeze_sha256']
                    and previous_driver.get('entry_driver_sha256') == sha256(previous / 'owner.py')
                    and previous_driver.get('released') is True,
                    'PRIOR_ALLOCATION_UNRELEASED_OR_UNKNOWN')
            _terminal(previous, previous_control, previous_driver, success=False)
        with _baseline_guard():
            guard = _account_guard(plan, binding['launch_argv'][0])
            atomic_json(directory / 'ACCOUNT_GUARD.json', guard)
            record = {'call_id': call_id, 'run_id': binding['run_id'], 'argv': binding['launch_argv'],
                      'intent_unix': time.time(), 'account_guard': reference(directory / 'ACCOUNT_GUARD.json'),
                      'reservation': reference(session.root / 'calls' / call_id / 'RESERVATION.json')}
            atomic_json(intent, record)  # durable before Popen; no retry on ambiguity
            registry['launches'].append({'call_id': call_id, 'plan': str(plan),
                'run_id': binding['run_id'], 'control_freeze_sha256': sha256(plan / 'CONTROL_FREEZE.json'),
                'intent': reference(intent)})
            atomic_json(registry_path, registry)
            with (directory / 'MANAGED.log').open('ab') as log:
                process = subprocess.Popen(binding['launch_argv'], stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
            receipt = {**record, 'pid': process.pid, 'status': 'MANAGED_STARTED_JOB_ID_NOT_YET_KNOWN'}
            atomic_json(directory / 'LAUNCH_RECEIPT.json', receipt)
            return receipt


def _safe_output(root, name):
    part = Path(name)
    require(not part.is_absolute() and part.parts and '..' not in part.parts, 'UNSAFE_COLLECTED_PATH')
    path = root / part
    require(path.is_file() and not path.is_symlink() and root.resolve() in path.resolve().parents,
            'COLLECTED_OUTPUT_OUTSIDE_ROOT')
    require(not any(p.is_symlink() for p in path.parents if root in p.parents), 'COLLECTED_PARENT_SYMLINK')
    return path


def _terminal(plan, control, driver, *, success=True):
    profile = read_json(plan / 'profile.json')
    scheduler = read_json(plan / 'run/SCHEDULER_LATEST.json')
    require(scheduler.get('released') is True and scheduler.get('identity_verified') is True
            and not scheduler.get('squeue', '').strip(), 'TERMINAL_RELEASE_REQUIRED')
    rows = [line.split('|') for line in scheduler.get('sacct', '').splitlines()
            if line.split('|')[0] == driver['job_id']]
    require(len(rows) == 1 and len(rows[0]) >= 5 and rows[0][1:3] ==
            [control['run']['session_name'], profile['username']], 'MAIN_SCHEDULER_IDENTITY')
    state = rows[0][3].split()[0].rstrip('+')
    require(state in TERMINAL and scheduler.get('state') == state
            and scheduler.get('exit_code') == rows[0][4], 'TERMINAL_SCHEDULER_EXIT_REQUIRED')
    if success:
        require(state == 'COMPLETED' and rows[0][4] == '0:0', 'SUCCESSFUL_SCHEDULER_EXIT_REQUIRED')


def validate_feedback(feedback, call, checkpoint_sha256):
    json_value(feedback)  # reject NaN/Infinity before writing a completion receipt
    require(feedback.get('schema') == 'weightlab_forward_feedback_v1', 'FEEDBACK_SCHEMA')
    require(all(feedback.get(k) == call[k] for k in ('call_id', 'candidate_tensor_sha256', 'split')),
            'FEEDBACK_CALL_OR_CANDIDATE_MISMATCH')
    require(feedback.get('checkpoint_sha256') == checkpoint_sha256, 'FEEDBACK_CHECKPOINT_MISMATCH')
    require(set(episode_keys(feedback.get('episode_keys'))) == set(episode_keys(call['request']['episode_keys'])),
            'FEEDBACK_EPISODE_COVERAGE')
    steps = feedback.get('transition_steps')
    require(type(steps) is int and 0 < steps <= call['request']['transition_steps'], 'FEEDBACK_TRANSITION_CAP')
    require(feedback.get('no_model_navigation_executed') is False, 'NAVIGATION_EXECUTION_EVIDENCE_REQUIRED')
    require(isinstance(feedback.get('metrics'), dict), 'FEEDBACK_METRICS_REQUIRED')


def collect_managed_call(session_root, call_id):
    """Validate already-collected terminal artifacts; never query or download."""
    session = Session(session_root)
    with _lock(session.root / 'MANAGED_BRIDGE.lock'):
        call, binding, plan = _binding(session, call_id)
        directory = Path(binding['directory'])
        intent = read_json(directory / 'LAUNCH_INTENT.json')
        require(intent['call_id'] == call_id and intent['run_id'] == binding['run_id']
                and intent['argv'] == binding['launch_argv'], 'LAUNCH_INTENT_IDENTITY')
        verify_reference(intent['reservation'])
        control, cfg = read_json(plan / 'CONTROL.json'), read_json(plan / 'worker.json')
        require(control['run']['session_name'] == binding['run_id'] == cfg['run_id']
                and control['remote']['output_directory'] == cfg['output_directory'], 'FROZEN_RUN_IDENTITY')
        driver = read_json(plan / 'run/DRIVER_RECEIPT.json')
        expected = {'session': control['run']['session_name'], 'attempt_id': control['run']['attempt_id'],
                    'control_freeze_sha256': sha256(plan / 'CONTROL_FREEZE.json'),
                    'entry_driver_sha256': sha256(plan / 'owner.py')}
        require(all(driver.get(k) == v for k, v in expected.items())
                and str(driver.get('job_id', '')).isdigit(), 'DRIVER_IDENTITY')
        require(driver.get('status') == 'OUTPUTS_VERIFIED' and driver.get('released') is True
                and driver.get('local_outputs_verified') is True, 'OWNER_COMPLETE_AND_RELEASED_REQUIRED')
        managed = read_json(plan / 'managed_records/MANAGED_RECEIPT.json')
        expected = {'source_sha256': sha256(plan / 'managed_run.py'),
                    'managed_config_sha256': sha256(plan / 'MANAGED.json'),
                    'control_sha256': sha256(plan / 'CONTROL.json'),
                    'control_freeze_sha256': sha256(plan / 'CONTROL_FREEZE.json'), 'run': str(plan / 'run')}
        require(all(managed.get(k) == v for k, v in expected.items())
                and managed.get('status') == 'OWNER_COMPLETED' and managed.get('owner_returncode') == 0
                and type(managed.get('owner_returncode')) is int
                and not managed.get('lifecycle_error'), 'MANAGED_SUCCESS_IDENTITY')
        _terminal(plan, control, driver)
        manifest = read_json(plan / 'run/REMOTE_OUTPUT_MANIFEST.json')
        require(manifest.get('run_id') == driver['session']
                and str(manifest.get('slurm_job_id')) == driver['job_id'], 'MANIFEST_IDENTITY')
        require(read_json(plan / 'run/COLLECTED.json').get('files') == manifest['files'], 'COLLECTION_RECEIPT_MISMATCH')
        allowed = set(cfg['required_outputs']) | {'WORKER_RECEIPT.json', 'scientific.log', 'validation.log'}
        names, refs, total = set(), [], 0
        for row in manifest['files']:
            name = row['path']
            require(name in allowed and name not in names, 'UNEXPECTED_OR_DUPLICATE_OUTPUT')
            path = _safe_output(plan / 'run/vm', name)
            require(type(row['bytes']) is int and row['bytes'] >= 0
                    and path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'], 'COLLECTED_HASH_MISMATCH')
            names.add(name); total += row['bytes']; refs.append(reference(path))
        require(total <= control['storage_budget']['output_bytes'], 'COLLECTED_OUTPUT_BUDGET')
        require(set(cfg['required_outputs']) | {'WORKER_RECEIPT.json'} <= names, 'REQUIRED_OUTPUTS_MISSING')
        worker = read_json(plan / 'run/vm/WORKER_RECEIPT.json')
        expected = {'run_id': driver['session'], 'slurm_job_id': driver['job_id'],
                    'worker_sha256': sha256(plan / 'worker.py'), 'config_sha256': sha256(plan / 'worker.json'),
                    'scientific_success': True, 'scientific_exit_code': 0, 'validation_exit_code': 0,
                    'status': 'SCIENTIFIC_OUTPUTS_VALIDATED', 'gpu_count': control['slurm']['gpus']}
        require(all(worker.get(k) == v for k, v in expected.items()), 'WORKER_SUCCESS_IDENTITY')
        require(worker.get('scientific_success') is True
                and all(type(worker.get(k)) is int for k in ('scientific_exit_code', 'validation_exit_code', 'gpu_count')),
                'WORKER_STATUS_TYPES')
        feedback = read_json(plan / 'run/vm' / FEEDBACK)
        validate_feedback(feedback, call, session.config['checkpoint_sha256'])
        from .falcon_forward import validate_output
        validation = validate_output(plan / 'run/vm/forward', directory / REQUEST)
        refs += [reference(p) for p in [plan / 'run/DRIVER_RECEIPT.json',
                 plan / 'managed_records/MANAGED_RECEIPT.json', plan / 'run/SCHEDULER_LATEST.json',
                 plan / 'run/REMOTE_OUTPUT_MANIFEST.json', plan / 'run/COLLECTED.json', directory / 'LAUNCH_INTENT.json']]
        # Retain the executable/config binding, not only hashes inside an
        # unchecked freeze file. Session rechecks these bytes on later reads.
        refs += binding['refs']
        for freeze_name in ('CONTROL_FREEZE.json', 'MANAGED_FREEZE.json'):
            for name in read_json(plan / freeze_name)['files']:
                refs.append(reference(Path(name) if Path(name).is_absolute() else plan / name))
        refs = list({r['path']: r for r in refs}.values())
        receipt = directory / 'COLLECTION_VERIFIED.json'
        require(not receipt.exists(), 'COLLECTION_ALREADY_VERIFIED_RECONCILE_NO_OVERWRITE')
        atomic_json(receipt, {'schema': 'weightlab_managed_collection_v1', 'call_id': call_id,
                    'run_id': driver['session'], 'job_id': driver['job_id'], 'released': True,
                    'artifacts': refs, 'independent_validation': validation,
                    'scope': 'Hash/identity/coverage and metric recomputation; no model rerun'})
        return session.complete_call(call_id, feedback, {'status': 'completed',
                    'files': refs + [reference(receipt)], 'backend': 'existing_managed_lifecycle'})
