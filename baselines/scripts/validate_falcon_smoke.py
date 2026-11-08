#!/usr/bin/env python3
import argparse, hashlib, json, math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=json.loads((a.output/'smoke_result.json').read_text())
assert r['status']=='PASS' and r['scope']=='CHECKPOINT_AND_RENDERER_SMOKE_ONLY'
assert r['benchmark_episodes']==0 and r['navigation_metrics'] is None
assert r['strict_checkpoint_load'] and r['missing_keys']==[] and r['unexpected_keys']==[]
assert r['checkpoint_layout']=='multi_agent_integer_0/state_dict'
repo=Path(r['repo']).resolve()
assert set(r['module_paths'])=={'falcon','habitat','habitat_baselines'}
assert all(repo in Path(path).parents for path in r['module_paths'].values())
assert r['checkpoint_sha256']=='ba7f2824604639d934be2359226c9e9fb35d5b1b2fb39ffbcbbbd833252db6ca'
assert r['seed']==100 and r['batch_size']==2 and r['steps']==8
assert len(r['actions'])==len(r['values'])==len(r['log_probabilities'])==8
for step in r['actions']:
 assert len(step)==2 and all(len(row)==1 and isinstance(row[0],int) and 0<=row[0]<4 for row in step)
for name in ['values','log_probabilities']:
 for step in r[name]:
  assert len(step)==2 and all(len(row)==1 and math.isfinite(row[0]) for row in step)
assert r['hidden_shapes']==[[2,4,512]]*8
assert 0<=r['reset_value_error']<1e-6
assert r['renderer']=={'scene':'NONE','depth_shape':[64,64],'finite':True,'physics_enabled':True}
assert '3090' in r['gpu'] and r['peak_cuda_bytes']>0
assert hashlib.sha256((a.output/'resolved_config.yaml').read_bytes()).hexdigest()==r['config_sha256']
print('Validated official-checkpoint CUDA and EGL smoke. No benchmark result asserted.')
