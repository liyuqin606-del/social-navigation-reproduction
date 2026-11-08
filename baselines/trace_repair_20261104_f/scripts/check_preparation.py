"""Local audit only. No SSH, GPU, scheduler, or changes to retained evidence."""
from pathlib import Path
import ast, copy, hashlib, json, subprocess, sys, tempfile
R=Path(__file__).resolve().parent.parent;B=R.parent
sys.path[:0]=[str(R/'validation'),str(R/'scripts')]
from falcon_full_val_integrity import collect, exact_keys, output_profile, protocol
from continuous_reproduction_queue import load_contract
from finalize_falcon_continuous_queue import retained_selection

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
checks=[]
def reject(name,fn):
 try:fn()
 except (ValueError,RuntimeError,KeyError):checks.append(name);return
 raise AssertionError('must reject '+name)
oldstate=read(B/'runs/falcon_repair6_20261104_e/CHECKPOINT.json')
for p,h in oldstate['input_files'].items():assert sha(p)==h,p
checks.append('all_1996_previous_frozen_inputs_unchanged')
negative=read(B/'runs/falcon_repair6_20261104_e/recovery_28160_failed_20261104_a/RECOVERY_RECEIPT.json')
for p,h in negative['verified_artifacts'].items():assert sha(p)==h,p
checks.append('negative_recovered_artifacts_unchanged')
full=B/'prep/falcon_full_val_20261103_manifest.json'
fullkeys=set(exact_keys(read(full)))
parts=[read(B/'continuation_20261104/prep'/f'balanced7_{i:02d}_manifest.json') for i in range(7)]
keys=[set(exact_keys(p)) for p in parts]
assert set.union(*keys)==fullkeys and sum(map(len,keys))==len(fullkeys)==1087
assert len(set.union(*keys[:3]))==471 and len(set.union(*keys[3:]))==616
assert len({k[0] for k in set.union(*keys[:3])})==24 and len({k[0] for k in set.union(*keys[3:])})==32
partition={'status':'PASS_EXACT_DISJOINT_471_PLUS_616_EQUALS_1087','retained_episodes':471,'retained_scenes':24,'remaining_episodes':616,'remaining_scenes':32,'full_episodes':1087,'full_scenes':56,'units':[{'id':p['shard_id'],'episodes':len(k),'scenes':p['scene_count'],'retained':i<3} for i,(p,k) in enumerate(zip(parts,keys))]}
save(R/'prep/PARTITION_CHECK.json',partition);checks.append('exact_coverage')
retained=read(R/'prep/RETAINED_471.json');protocols=[]
for entry in retained['completions']:
 for p,h in entry['proof_files'].items():assert sha(p)==h,p
 plan=Path(entry['plan']);data=collect(full,[plan/'run/vm/eval'],single_shard=True,managed_plan=plan)
 assert data['shards'][0]['shard_id']==entry['unit_id']
 protocols.append(data['protocol']);checks.append('independent_recheck_'+entry['unit_id'])
assert protocols[0]==protocols[1]==protocols[2]
checks.append('retained_scientific_protocols_equal')
proof={};selected=retained_selection({'input_files':{str(R/'prep/RETAINED_471.json'):sha(R/'prep/RETAINED_471.json')}},proof)
assert len(selected)==3 and len(proof)>60
checks.append('finalizer_retains_three_explicit_completions')
fixture=Path(retained['completions'][0]['plan'])/'run/vm/eval'
prov=read(fixture/'run_manifest.json');profiles=read(R/'validation/OUTPUT_PROFILES.json')
changed=copy.deepcopy(prov);changed['instrumentation']['derived_sha256']='0'*64
reject('unknown_evaluator_rejected',lambda:output_profile(changed))
changed=copy.deepcopy(prov);changed['instrumentation']['derived_sha256']=profiles['partition_v1']['evaluator']
reject('crossed_evaluator_validator_rejected',lambda:output_profile(changed))
changed['validator_sha256']=profiles['partition_v1']['validator']
assert output_profile(changed)=='partition_v1'
assert protocol(read(fixture/'resolved_config.json'),changed)==protocol(read(fixture/'resolved_config.json'),prov)
checks.append('only_output_version_normalized_scientific_protocol_equal')
plan=B/'runs/falcon_repair6_20261104_e/attempts/balanced7_03_0_2b14b955fb6e/plan'
reject('original_28160_still_rejected',lambda:collect(full,[plan/'run/vm/eval'],single_shard=True,managed_plan=plan))
cfg,bound=load_contract(R/'prep/QUEUE.json')
assert len(cfg['units'])==4 and sum(len(u['variants']) for u in cfg['units'])==4
for u in cfg['units']:
 spec=read(u['spec']);assert spec['slurm']['gpus']==1 and spec['slurm']['wall_time']=='01:00:00'
 assert spec['reserve_bytes']==4<<30
 with tempfile.TemporaryDirectory() as tmp:
  result=subprocess.run([sys.executable,'-B',str(R/'lifecycle/prepare_run.py'),'--spec',u['spec'],'--directory',str(Path(tmp)/'plan')],capture_output=True,text=True)
  assert result.returncode==0,result.stderr
 checks.append('managed_prepare_local_'+u['unit_id'])
# Missing approval fails before Remote is instantiated; it never submits.
with tempfile.TemporaryDirectory() as tmp:
 a=Path(tmp)/'approval.json';save(a,{'approved':False})
 p=subprocess.run([sys.executable,'-B',str(R/'scripts/launch_after_approval.py'),'--approval',str(a)],capture_output=True,text=True)
 assert p.returncode!=0 and 'EXPLICIT_QUEUE_BOUND_USER_APPROVAL_REQUIRED' in p.stderr
checks.append('unapproved_launch_rejected_before_network')
for d in ['candidate','scripts','validation','lifecycle']:
 for p in (R/d).glob('*.py'):ast.parse(p.read_text())
checks.append('all_python_sources_parse')
for p,h in oldstate['input_files'].items():assert sha(p)==h,p
for p,h in negative['verified_artifacts'].items():assert sha(p)==h,p
checks.append('original_inputs_and_negative_evidence_unchanged_after_tests')
result={'status':'PASS_LOCAL_PREPARATION_ONLY','checks':checks,'count':len(checks),'bound_inputs':len(bound),'retained_episodes':471,'new_scientific_results':0,'gpu_allocations':0,'network_calls':0,'remote_runtime_tested':False,'full_aggregate_tested':False}
save(R/'prep/PREPARATION_CHECK.json',result);print(json.dumps(result,indent=2))
