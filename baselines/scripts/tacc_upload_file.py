#!/usr/bin/env python3
"""Hash-verified bounded file transfer to the authorized baseline prep root. No jobs."""
import argparse, hashlib, importlib.util, json, shlex, subprocess, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('relative_target');a=p.parse_args()
assert a.source.is_file()
rel=Path(a.relative_target)
assert not rel.is_absolute() and '..' not in rel.parts
skill=Path('/Users/yuqinli/.codex/skills/tacc-3090')
spec=importlib.util.spec_from_file_location('pf',skill/'scripts/preflight.py')
pf=importlib.util.module_from_spec(spec);spec.loader.exec_module(pf)
profile=json.loads((skill/'references/profile.json').read_text());pf.verify_keys(profile)
size=a.source.stat().st_size
h=hashlib.sha256()
with a.source.open('rb') as f:
 for block in iter(lambda:f.read(4*2**20),b''):h.update(block)
sha=h.hexdigest();target='/mnt/home/liyuqin/USERDIR/vln_baselines_20261103/'+rel.as_posix()
code='''import hashlib,json,pathlib,shutil,sys
p=pathlib.Path(TARGET);p.parent.mkdir(parents=True,exist_ok=True)
assert shutil.disk_usage(p.parent).free>4*2**30+SIZE
partial=p.with_name(p.name+'.upload-partial')
h=hashlib.sha256();n=0
with partial.open('wb') as f:
 while True:
  b=sys.stdin.buffer.read(4*2**20)
  if not b:break
  n+=len(b);assert n<=SIZE
  f.write(b);h.update(b)
assert n==SIZE and h.hexdigest()==SHA
partial.replace(p)
print(json.dumps({'target':str(p),'bytes':n,'sha256':h.hexdigest(),'verified':True}))
'''.replace('TARGET',repr(target)).replace('SIZE',str(size)).replace('SHA',repr(sha))
with a.source.open('rb') as f:
 r=subprocess.run(pf.ssh_argv(profile)+[profile['remote_python']+' -c '+shlex.quote(code)],stdin=f,capture_output=True,timeout=1800)
print(r.stdout.decode(),end='');print(r.stderr.decode(),end='',file=sys.stderr)
raise SystemExit(r.returncode)
