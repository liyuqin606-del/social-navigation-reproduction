#!/usr/bin/env python3
"""Strict-SSH, bounded Falcon preparation transport; no Slurm submit operations."""
import importlib.util,json,subprocess,shlex,sys
from pathlib import Path
S=Path('/Users/yuqinli/.codex/skills/tacc-3090')
sp=importlib.util.spec_from_file_location('pf',S/'scripts/preflight.py')
pf=importlib.util.module_from_spec(sp);sp.loader.exec_module(pf)
profile=json.loads((S/'references/profile.json').read_text())
pf.verify_keys(profile)
command=Path(sys.argv[1]).read_text()
r=subprocess.run(pf.ssh_argv(profile)+[profile['remote_python']+' -c '+shlex.quote(command)],capture_output=True,text=True,timeout=int(sys.argv[2]) if len(sys.argv)>2 else 60)
print(r.stdout,end='');print(r.stderr,end='',file=sys.stderr)
sys.exit(r.returncode)
