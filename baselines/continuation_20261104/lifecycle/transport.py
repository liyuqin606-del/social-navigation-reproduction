"""Strict SSH transport; no resource allocation in this module."""
import hashlib
import json
import shlex
import subprocess
import time
from pathlib import Path
import preflight


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


class Remote:
    def __init__(self, profile):
        self.profile = profile
        self.ssh = preflight.ssh_argv(profile)

    def code(self, code, timeout=40, retry=False):
        # Only read-only queries may use retry. Mutation callers get one attempt.
        attempts = 3 if retry else 1
        for attempt in range(attempts):
            try:
                proc = subprocess.run(self.ssh + ['python3 -c ' + shlex.quote(code)],
                                      stdin=subprocess.DEVNULL, capture_output=True,
                                      text=True, timeout=timeout)
                if proc.returncode:
                    raise RuntimeError('SSH_COMMAND_FAILED: ' + proc.stderr[-1200:])
                return proc.stdout.strip()
            except (RuntimeError, subprocess.TimeoutExpired):
                if attempt + 1 == attempts:
                    raise
                time.sleep(2 ** attempt)

    def json(self, code, timeout=40, retry=False):
        return json.loads(self.code(code, timeout, retry))

    def upload(self, local, remote, expected):
        local = Path(local)
        if local.stat().st_size != expected['bytes'] or sha(local) != expected['sha256']:
            raise RuntimeError('LOCAL_UPLOAD_CHANGED: ' + local.name)
        code = '''import sys,hashlib,os
from pathlib import Path
p=Path(REMOTE);p.parent.mkdir(parents=True,exist_ok=True)
t=p.with_name(p.name+'.partial');h=hashlib.sha256();n=0
with t.open('xb') as f:
 while True:
  b=sys.stdin.buffer.read(4<<20)
  if not b:break
  f.write(b);h.update(b);n+=len(b)
assert n==SIZE and h.hexdigest()==SHA,'UPLOAD_HASH_MISMATCH'
assert not p.exists(),'UPLOAD_DEST_EXISTS'
t.replace(p)
print(h.hexdigest())
'''.replace('REMOTE', repr(str(remote))).replace('SIZE', str(expected['bytes'])).replace('SHA', repr(expected['sha256']))
        with local.open('rb') as stream:
            proc = subprocess.run(self.ssh + ['python3 -c ' + shlex.quote(code)], stdin=stream,
                                  capture_output=True, text=True, timeout=600)
        if proc.returncode or proc.stdout.strip() != expected['sha256']:
            raise RuntimeError('UPLOAD_NOT_VERIFIED: ' + proc.stderr[-1000:])

    def download(self, remote, target, expected):
        target = Path(target)
        if target.exists():
            if target.stat().st_size == expected['bytes'] and sha(target) == expected['sha256']:
                return
            raise RuntimeError('EXISTING_RESULT_DIFFERS: ' + target.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + '.partial')
        code = 'import sys,shutil;f=open(' + repr(str(remote)) + ",'rb');shutil.copyfileobj(f,sys.stdout.buffer)"
        with tmp.open('wb') as stream:
            proc = subprocess.run(self.ssh + ['python3 -c ' + shlex.quote(code)],
                                  stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.PIPE, timeout=600)
        if proc.returncode or tmp.stat().st_size != expected['bytes'] or sha(tmp) != expected['sha256']:
            raise RuntimeError('DOWNLOAD_NOT_VERIFIED: ' + target.name)
        tmp.replace(target)
