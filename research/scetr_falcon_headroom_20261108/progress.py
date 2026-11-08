"""Read-only run observation, never submits, cancels or changes scientific files."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'lifecycle'))
from transport import Remote,read

cfg=read(HERE/'plan/CONTROL.json')
receipt=read(HERE/'plan/run/DRIVER_RECEIPT.json')
remote=Remote(read(HERE/'plan/profile.json'))
code='''import json,subprocess
from pathlib import Path
root=Path(ROOT)
def tail(p,n):
 return '\\n'.join(p.read_text(errors='replace').splitlines()[-n:]) if p.exists() else None
summary={'queue':subprocess.check_output(['squeue','-h','-j',JOB,'-o','%i|%u|%j|%T|%R']).decode(),'log':tail(root/'scientific.log',8),'worker':tail(root/'WORKER_RECEIPT.json',25),'cases':[]}
for p in sorted((root/'headroom').glob('case*')):
 if p.is_dir():
  result={'name':p.name}
  for name in ['CASE_STATUS.json','ERROR.txt']:
   if (p/name).exists():result[name]=tail(p/name,20)
  summary['cases'].append(result)
print(json.dumps(summary))
'''.replace('ROOT',repr(cfg['remote']['output_directory'])).replace('JOB',repr(str(receipt['job_id'])))
print(json.dumps(remote.json(code,timeout=45,retry=True),indent=2))
