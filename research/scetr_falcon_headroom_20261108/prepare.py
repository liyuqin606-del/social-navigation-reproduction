"""Select failure development cases and read-only remote inventory. No sbatch."""
import collections
import datetime
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'research/falcon_migration_20261106'
sys.path.insert(0,str(OLD/'lifecycle'))
from transport import Remote,read,save,sha


def main():
    if (HERE/'SPEC.json').exists(): raise FileExistsError('Already prepared; do not overwrite frozen inputs')
    rows=[json.loads(x) for x in (ROOT/'baselines/runs/falcon_subset174_20261103_b/run/vm/eval/episode_metrics.jsonl').read_text().splitlines()]
    grouped=collections.defaultdict(list)
    for r in rows:
        if not r['metrics']['success'] and r['metrics']['human_collision'] and 40<=r['action_steps']<=300:
            name=Path(r['scene_id']).name.split('.')[0]
            grouped[name].append(r)
    scenes=sorted(grouped,key=lambda n:(-len(grouped[n]),n))[:4]
    selected=[]
    content=ROOT/'baselines/assets/falcon/episodes/social-hm3d/val/content'
    for scene in scenes:
        r=min(grouped[scene],key=lambda r:hashlib.sha256((scene+'|'+str(r['episode_id'])).encode()).hexdigest())
        shard=content/(scene+'.json.gz');doc=json.loads(gzip.decompress(shard.read_bytes()))
        matches=[e for e in doc['episodes'] if str(e['episode_id'])==str(r['episode_id'])]
        assert len(matches)==1
        selected.append(dict(scene_name=scene,scene_id=matches[0]['scene_id'],episode_id=str(r['episode_id']),
                             old_steps=r['action_steps'],old_metrics=r['metrics'],shard_sha256=sha(shard)))
    assert len(selected)==4
    spec=read(OLD/'SPEC.json')
    remote_base='/mnt/home/liyuqin/USERDIR/vln_baselines_20261103'
    data=remote_base+'/src/Falcon/data/datasets/pointnav/social-hm3d-fullval/val/val.json.gz'
    paths=[r['path'] for r in spec['remote_inputs'][:3]]
    for c in selected:
        paths.append(str(Path(data).parent/'content'/(c['scene_name']+'.json.gz')))
        mesh=remote_base+'/fullval_assets/scene_datasets/'+c['scene_id'].replace('data/scene_datasets/','',1)
        paths.extend([mesh,str(Path(mesh).with_suffix('.navmesh'))])
    remote=Remote(read(OLD/'plan/profile.json'))
    code='''import hashlib,json,datetime,subprocess,shutil
from pathlib import Path
rows=[]
for name in PATHS:
 p=Path(name);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 rows.append({'path':name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
print(json.dumps({'files':rows,'server_deadline':(datetime.datetime.now()+datetime.timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%S'),'nodes':subprocess.check_output(['sinfo','-N','-h','-o','%N|%t|%G']).decode(),'own_jobs':subprocess.check_output(['squeue','-h','-u','liyuqin','-o','%i|%j|%T|%b']).decode(),'remote_free_bytes':shutil.disk_usage('/mnt/home/liyuqin/USERDIR').free}))
'''.replace('PATHS',repr(paths))
    inventory=remote.json(code,timeout=90,retry=True)
    save(HERE/'REMOTE_INVENTORY.json',inventory)
    if inventory['own_jobs'].strip(): raise RuntimeError('Existing jobs require reconciliation before preparing new allocation')
    for old,new in zip(spec['remote_inputs'][:3],inventory['files'][:3]): assert old==new
    bypath={r['path']:r for r in inventory['files']}
    for c in selected:
        shard=str(Path(data).parent/'content'/(c['scene_name']+'.json.gz'))
        assert bypath[shard]['sha256']==c['shard_sha256']
        mesh=remote_base+'/fullval_assets/scene_datasets/'+c['scene_id'].replace('data/scene_datasets/','',1)
        c.update(mesh_sha256=bypath[mesh]['sha256'],navmesh_sha256=bypath[str(Path(mesh).with_suffix('.navmesh'))]['sha256'])
    save(HERE/'episodes.json',dict(source_split='val',role='PREVIOUS_FAILURES_NOW_DEVELOPMENT_EXCLUDED_FROM_CONFIRMATION',cases=selected))
    shutil.copyfile(OLD/'derive.py',HERE/'derive.py')
    shutil.copytree(OLD/'lifecycle',HERE/'lifecycle',ignore=shutil.ignore_patterns('__pycache__','CHECK.json'))
    spec['run_id']='vln_scetr_headroom_20261108_a'
    spec['slurm'].update(wall_time='01:00:00',deadline=inventory['server_deadline'])
    # Known successful rendering node; retain audited node restriction.
    spec['remote_inputs']=inventory['files']
    spec['science']['argv']=[spec['remote_python'],'{source}/run.py','--output','{output}/headroom','--protocol','{source}/PROTOCOL.json',
                            '--repo',remote_base+'/src/Falcon','--manifest','{source}/episodes.json','--checkpoint',paths[0],'--pretrained',paths[1],
                            '--data-path',data,'--scenes-dir',remote_base+'/fullval_assets/scene_datasets']
    spec['science']['timeout_seconds']=3200
    spec['science']['validator_argv']=[spec['remote_python'],'{source}/validate.py','--output','{output}/headroom']
    spec['science']['validation_timeout_seconds']=120
    spec['science']['required_outputs']=['headroom/'+n for n in ['SCHEDULE.json','VALIDATION.json','RUN_COMPLETE.json','EVIDENCE.tar.gz']]
    spec['uploads']=[dict(file=str(HERE/n),remote_name=n) for n in ['bridge.py','derive.py','run_case.py','run.py','validate.py','episodes.json','PROTOCOL.json']]
    spec['scientific_files']=[str(HERE/'PROTOCOL.json'),str(HERE/'prepare.py'),str(HERE/'test_headroom.py')]
    spec['output_budget_bytes']=128<<20;spec['temporary_transfer_bytes']=128<<20;spec['remote_peak_additional_bytes']=256<<20
    spec['monitor_timeout_seconds']=18000
    save(HERE/'SPEC.json',spec)
    print(json.dumps(dict(status='PREPARED_NOT_SUBMITTED',cases=selected,nodes=inventory['nodes']),indent=2))


if __name__=='__main__': main()
