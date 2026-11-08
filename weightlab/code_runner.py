"""Execute an editor's tensor-only Python in a macOS filesystem/network sandbox.

This is deliberately not a general Python tool and fails closed on unsupported
hosts. The caller remains the trusted experimenter; this does not constrain an
operator's separate shell, or prove that arbitrary NumPy code avoids all fitting.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import sysconfig
import time
import uuid


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic(path, obj):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def sandbox_profile(work, python):
    """Only the copied work directory is writable; no network or home access."""
    import numpy
    roots = ['/System', '/usr/lib', '/usr/share', '/Library/Apple',
             str(Path(sys.base_prefix).resolve()), str(Path(numpy.__file__).resolve().parents[1]),
             str(Path(sysconfig.get_paths()['stdlib']).resolve()), str(work)]
    read_rules = ' '.join('(subpath ' + json.dumps(p) + ')' for p in sorted(set(roots)))
    # Keep Darwin runtime/Mach initialization available; explicitly close the
    # relevant capabilities rather than denying undocumented loader operations.
    return ('(version 1)\n(allow default)\n(deny network*)\n'
            '(deny file-read*)\n(deny file-write*)\n(deny process-exec)\n(deny process-fork)\n'
            '(allow file-read* ' + read_rules + ' (literal "/dev/null") (literal "/dev/urandom"))\n'
            '(allow file-write* (subpath ' + json.dumps(str(work)) + ') (literal "/dev/null"))\n'
            '(allow process-exec (literal ' + json.dumps(str(python)) + '))\n'
            '(allow sysctl-read)\n(allow signal (target self))\n')


def run_tensor_code(session_root, base_npz, active_npz, code, rationale,
                    max_code_runs=300, timeout_seconds=30, probe_files=()):
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').is_file():
        raise RuntimeError('This build requires macOS sandbox-exec; no unsandboxed fallback')
    if not rationale.strip() or not code.strip():
        raise ValueError('Explain the hypothesis before supplying code')
    if len(code.encode()) > 100000 or not 1 <= timeout_seconds <= 60:
        raise ValueError('Code or time budget exceeded')
    root = Path(session_root).resolve()
    if shutil.disk_usage(root).free < (4 << 30) + (128 << 20):
        raise RuntimeError('4GiB free-space reserve plus code output budget required')
    ledger_path = root / 'CODE_LEDGER.json'
    with (root / 'CODE.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {'calls': []}
        if len(ledger['calls']) >= max_code_runs:
            raise RuntimeError('CODE_RUN_BUDGET_EXHAUSTED')
        ident = 'code_%03d_%s' % (len(ledger['calls']) + 1, uuid.uuid4().hex[:8])
        row = {'id': ident, 'status': 'RESERVED', 'rationale': rationale,
               'source_sha256': hashlib.sha256(code.encode()).hexdigest(), 'started_unix': time.time()}
        ledger['calls'].append(row)
        _atomic(ledger_path, ledger)
    work = root / 'code' / ident
    work.mkdir(parents=True)
    shutil.copyfile(base_npz, work / 'base.npz')
    shutil.copyfile(active_npz, work / 'active.npz')
    shutil.copyfile(Path(__file__).with_name('code_worker.py'), work / 'worker.py')
    (work / 'proposal.py').write_text(code)
    (work / 'hypothesis.txt').write_text(rationale + '\n')
    (work / 'probes').mkdir()
    names = []
    for path in probe_files:
        path = Path(path)
        if path.suffix != '.npz' or path.name in names or path.stat().st_size > 64 << 20:
            raise ValueError('Probe artifacts must be distinct, bounded NPZ files')
        shutil.copyfile(path, work / 'probes' / path.name)
        names.append(path.name)
    (work / 'probe_files.json').write_text(json.dumps(names))
    python = Path(sys.executable).resolve()
    profile = work / 'sandbox.sb'
    profile.write_text(sandbox_profile(work, python))
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(work), 'TMPDIR': str(work),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    with (work / 'process.log').open('wb') as log:
        child = subprocess.Popen(['/usr/bin/sandbox-exec', '-f', str(profile), str(python),
                                  '-I', str(work / 'worker.py'), str(work)], cwd=work, env=env,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        try:
            rc = child.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            rc = -9
    result_path = work / 'worker_result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {'status': 'CODE_FAILED'}
    result.update(id=ident, exit_code=rc, directory=str(work),
                  edits_npz=str(work / 'edits.npz') if (work / 'edits.npz').exists() else None,
                  stdout=(work / 'stdout.txt').read_text() if (work / 'stdout.txt').exists() else '',
                  process_log=(work / 'process.log').read_text(errors='replace')[-4000:],
                  finished_unix=time.time())
    if rc != 0:
        result['status'] = 'CODE_FAILED'
    _atomic(work / 'receipt.json', result)
    with (root / 'CODE.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(ledger_path.read_text())
        entry = next(x for x in ledger['calls'] if x['id'] == ident)
        entry.update(status=result['status'], receipt=str(work / 'receipt.json'),
                     receipt_sha256=sha(work / 'receipt.json'), finished_unix=result['finished_unix'])
        _atomic(ledger_path, ledger)
    return result
