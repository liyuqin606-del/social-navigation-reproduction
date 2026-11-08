#!/usr/bin/env python3
"""Read one existing managed run; never submit, cancel, install or run a model."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'baselines/lifecycle'))
import preflight
from transport import Remote

p = argparse.ArgumentParser()
p.add_argument('--plan', type=Path, required=True)
p.add_argument('--output', type=Path)
a = p.parse_args()
spec = json.loads((a.plan / 'SPEC.json').read_text())
profile = json.loads((a.plan / 'profile.json').read_text())
preflight.verify_keys(profile)
remote_output = str(Path(spec['remote_root']) / ('result_' + spec['run_id']))
code = '''import json,pathlib,datetime
r=pathlib.Path(OUTPUT)
result={'run_id':RUN_ID,'observed_utc':datetime.datetime.utcnow().isoformat()+'Z'}
for name in ['WORKER_RECEIPT.json','eval/run_status.json','eval/validation.json','diag/diagnostics_summary.json']:
 p=r/name
 if p.is_file():
  try:result[name]=json.loads(p.read_text())
  except ValueError:result[name]={'read_status':'WRITE_IN_PROGRESS'}
p=r/'eval/episode_metrics.jsonl'
if p.is_file():
 rows=[]
 for line in p.read_text().splitlines():
  try:rows.append(json.loads(line))
  except ValueError:pass
 result['completed_episode_rows']=len(rows)
 result['unique_episode_keys']=len({(x['scene_id'],str(x['episode_id'])) for x in rows})
 result['completed_action_steps']=sum(x['action_steps'] for x in rows)
 result['completed_scenes']=len({x['scene_id'] for x in rows})
p=r/'scientific.log'
if p.is_file():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-8000));value=f.read().decode('utf-8',errors='replace')
 result['log_tail']=value[-2000:]
print(json.dumps(result,allow_nan=False))
'''.replace('OUTPUT', repr(remote_output)).replace('RUN_ID', repr(spec['run_id']))
result = Remote(profile).json(code, timeout=60, retry=True)
receipt = a.plan / 'run/DRIVER_RECEIPT.json'
if receipt.is_file():
    d = json.loads(receipt.read_text())
    result['local_owner'] = {k: d.get(k) for k in ['job_id', 'status', 'released', 'local_outputs_verified']}
if a.output:
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps(result, ensure_ascii=False, indent=2))
