"""Read-only observation of the unique bound job."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'lifecycle'))
from transport import Remote,read
control=read(HERE/'plan/CONTROL.json');receipt=read(HERE/'plan/run/DRIVER_RECEIPT.json')
remote=Remote(read(HERE/'plan/profile.json'))
code='''import json,subprocess
from pathlib import Path
p=Path(ROOT)
rows=[]
for d in sorted((p/'headroom').glob('case*')):
 if d.is_dir() and (d/'trace.json').exists():
  t=json.loads((d/'trace.json').read_text());m=t[-1]['metrics'] if t else {}
  rows.append({'name':d.name,'steps':len(t),'success':m.get('success'),'collision':m.get('human_collision'),'status':json.loads((d/'CASE_STATUS.json').read_text()) if (d/'CASE_STATUS.json').exists() else None})
log=(p/'scientific.log').read_text(errors='replace').splitlines()[-3:] if (p/'scientific.log').exists() else []
print(json.dumps({'queue':subprocess.check_output(['squeue','-h','-j',JOB,'-o','%i|%T|%R']).decode(),'log':log,'cases':rows}))
'''.replace('ROOT',repr(control['remote']['output_directory'])).replace('JOB',repr(receipt['job_id']))
x=remote.json(code,timeout=30,retry=True)
for r in x['cases']:r.pop('status',None)
if '--compact' in sys.argv:
    x['completed_cases']=len(x['cases'])
    x['outcomes']={r['name']:[r['steps'],r['success'],r['collision']] for r in x.pop('cases')}
print(json.dumps(x,indent=2))
