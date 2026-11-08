"""Independent collection/identity audit around replay of the frozen controller."""
import argparse,gzip,hashlib,json,shutil,tarfile
from pathlib import Path
from validate import validate,physical

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(output):
    manifest=read(HERE/'episodes.json')
    summary=validate(output,write=False)
    assert summary==read(output/'VALIDATION.json')
    actor='7a2106b161fff34e77364812b9fedefe597e146fcc8d6df0711ee6cb8bf542a2'
    assert summary['actor_sha256']==actor
    repeated_old=0
    for d in output.glob('case*'):
        if not d.is_dir():continue
        inputs=read(d/'INPUTS.json');cfg=read(d/'CONFIG.json');tr=read(d/'trace.json')
        case=manifest['cases'][read(d/'CASE_STATUS.json')['case']]
        assert inputs['case']==case and inputs['manifest_sha256']==sha(HERE/'episodes.json')
        scene=str(Path(cfg['habitat']['dataset']['scenes_dir'])/case['scene_id'])
        assert all(r['episode']==[scene,case['episode_id']] for r in tr)
        assert cfg['habitat']['seed']==100 and cfg['habitat_baselines']['num_environments']==1
        assert cfg['habitat']['environment']['max_episode_steps']==500
        assert tr[-1]['metrics']['num_steps']==len(tr)
        shard=list((d/'dataset/val/content').glob('*.gz'));assert len(shard)==1
        episodes=json.loads(gzip.decompress(shard[0].read_bytes()))['episodes'];assert len(episodes)==1
        assert str(episodes[0]['episode_id'])==case['episode_id'] and episodes[0]['scene_id']==case['scene_id']
        if d.name.endswith('_reference'):
            old=HERE.parent/'scetr_falcon_headroom_20261108/collected_a'/d.name/'trace.json'
            repeated_old+=int([physical(r) for r in tr]==read(old))
    freeze=read(HERE/'plan/MANAGED_FREEZE.json')['files']
    assert all(sha(p)==h for p,h in freeze.items())
    collection=read(HERE/'plan/run/COLLECTED.json')['files']
    assert all(sha(HERE/'plan/run/vm'/r['path'])==r['sha256'] for r in collection)
    driver=read(HERE/'plan/run/DRIVER_RECEIPT.json')
    assert driver['released'] and driver['local_outputs_verified']
    result={**summary,'independent_identity_and_manifest_audit':'PASS','frozen_files':len(freeze),
            'collected_files':len(collection),'previous_round_reference_matches':repeated_old,
            'files':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='LOCAL_AUDIT.json'}}
    (output/'LOCAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['cases','files']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path);a=p.parse_args()
    if a.archive:
        assert shutil.disk_usage(a.output.parent).free>4*(1<<30)+512*(1<<20)
        a.output.mkdir(exist_ok=False)
        with tarfile.open(a.archive,'r:gz') as t:
            members=t.getmembers();names=set()
            for m in members:
                q=Path(m.name);assert not q.is_absolute() and '..' not in q.parts and (m.isfile() or m.isdir())
                assert m.name not in names;names.add(m.name)
            assert sum(m.size for m in members)<=512*(1<<20)
            t.extractall(a.output,members=members,filter='data')
    main(a.output)
