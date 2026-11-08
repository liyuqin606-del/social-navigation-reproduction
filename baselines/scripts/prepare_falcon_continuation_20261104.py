"""Local-only preparation of seven disjoint eight-scene units; no SSH/submit."""
from pathlib import Path
import json,copy,hashlib,shutil,datetime
BASE=Path(__file__).resolve().parents[1];ROOT=BASE/'continuation_20261104';ROOT.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text())
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
old=read(BASE/'runs/falcon_continuous_queue_20261103_a/CHECKPOINT.json')
assert all(sha(Path(p))==h for p,h in old['input_files'].items())
for sub in ['scripts','lifecycle','prep']:(ROOT/sub).mkdir()
for p in (BASE/'lifecycle').glob('*.py'):shutil.copyfile(p,ROOT/'lifecycle'/p.name)
shutil.copyfile(BASE/'lifecycle_recovery_20261104/worker.py',ROOT/'lifecycle/worker.py')
controller=(BASE/'scripts/continuous_reproduction_queue.py').read_text()
controller=controller.replace("(HERE.parent / 'runs').glob('**/CONTROL.json')",repr(str(BASE/'runs'))+"_PLACEHOLDER") if False else controller
needle="(HERE.parent / 'runs').glob('**/CONTROL.json')";assert controller.count(needle)==1
controller=controller.replace(needle,"Path("+repr(str(BASE/'runs'))+").glob('**/CONTROL.json')")
(ROOT/'scripts/continuous_reproduction_queue.py').write_text(controller)
fullpath=BASE/'prep/falcon_full_val_20261103_manifest.json';full=read(fullpath);groups=[[] for _ in range(7)]
for scene in sorted(full['scenes'],key=lambda x:(-x['episode_count'],x['scene_name'])):
 i=min((i for i,g in enumerate(groups) if len(g)<8),key=lambda i:(sum(s['episode_count'] for s in groups[i]),i));groups[i].append(scene)
expected={tuple(x) for x in full['episode_keys']};seen=set();specbase=read(BASE/'prep/falcon_retry_single_00_20261103.json');units=[];summary=[]
for i,group in enumerate(groups):
 sid=f'balanced7_{i:02d}';ids={s['scene_id'] for s in group};doc=copy.deepcopy(full);doc.update(shard_id=sid,parent_full_manifest=str(fullpath),partition_kind='DETERMINISTIC_LPT_SEVEN_EIGHT_SCENE_GROUPS',partition_plan=str(ROOT/'prep/PARTITION.json'),scenes=[s for s in full['scenes'] if s['scene_id'] in ids],episode_keys=[k for k in full['episode_keys'] if k[0] in ids],recommended_num_environments=8,generator_requires_explicit_num_envs=True,evaluation_scope='One of seven disjoint eight-scene units; restarted stochastic streams, one released checkpoint only.')
 doc['content_scenes']=[s['scene_name'] for s in doc['scenes']];doc['scene_count']=len(ids);doc['episode_count']=len(doc['episode_keys']);keys={tuple(k) for k in doc['episode_keys']};assert len(ids)==8 and not keys&seen;seen|=keys
 manifest=ROOT/'prep'/f'{sid}_manifest.json';save(manifest,doc)
 spec=copy.deepcopy(specbase);spec['run_id']=f'vln_cont7_{i:02d}_20261104_'+sha(manifest)[:8];spec['slurm']['deadline']='2026-11-05T23:00:00';spec['science']['timeout_seconds']=3300
 spec['uploads']=[dict(row,file=str(manifest)) if row['remote_name']=='episode_manifest.json' else row for row in spec['uploads']]
 spec['scientific_files']=[str(manifest) if 'balanced4_shard_00' in p else p for p in spec['scientific_files']]
 spec['evaluation_plan'].update(mode=sid,scene_count=8,episode_count=len(keys),retry_scope='New finite continuation after explicit retained cancellations; each candidate once.',status='PREPARED_NOT_SUBMITTED')
 sp=ROOT/'prep'/f'{sid}_spec.json';save(sp,spec)
 args=['{python}',str(BASE/'scripts/check_falcon_shard_completion.py'),'--eval-output','{vm}/eval','--expected-manifest',str(manifest),'--full-manifest',str(fullpath),'--managed-plan','{plan}','--output','{vm}/completion_check.json']
 units.append({'unit_id':sid,'spec':str(sp),'completion_ids':[sid],'deadline_after_seconds':7200,'validator_timeout_seconds':120,'local_validator_argv':args,'validation_files':[str(BASE/'scripts'/n) for n in ['check_falcon_shard_completion.py','falcon_full_val_integrity.py','falcon_shard_completion_evidence.py','validate_falcon_subset.py']]+[str(fullpath),str(manifest)],'completion_checks':[],'retry_specs':[]})
 summary.append({'shard_id':sid,'scene_count':8,'episode_count':len(keys),'manifest':str(manifest),'sha256':sha(manifest)})
assert seen==expected and len(seen)==1087
save(ROOT/'prep/PARTITION.json',{'schema':'falcon_continuation_partition_v1','shards':summary,'coverage':{'episodes':1087,'scenes':56,'duplicates':0,'missing':0},'num_environments':8,'old_queue_preserved':str(BASE/'runs/falcon_continuous_queue_20261103_a'),'old_inputs_unchanged':42,'parent_full_manifest_sha256':sha(fullpath),'limitations':['Different scene grouping changes stochastic streams; not paired with old shards.','One released checkpoint only; no SOTA or multiple-training-seed claim.','No cancelled-run results admitted.']})
save(ROOT/'prep/QUEUE.json',{'schema':'continuous_reproduction_queue_v1','local_python':str(BASE.parent/'.venv/bin/python'),'poll_seconds':30,'units':units})
save(ROOT/'BUILD_MANIFEST.json',{'status':'PREPARED_NOT_SUBMITTED','files':{str(p):sha(p) for p in ROOT.rglob('*') if p.is_file()},'original_frozen_inputs':old['input_files'],'worker_probe_evidence':str(BASE/'runs/cancellation_probe_20261104_b/INDEPENDENT_PROBE_VALIDATION.json'),'maximum_allocations':7,'gpus_per_allocation':1,'wall_time':'01:00:00'})
print(json.dumps(summary,indent=2))
