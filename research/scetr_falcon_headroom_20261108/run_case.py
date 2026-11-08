"""One complete episode in a fresh seeded Habitat/Falcon process."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from derive import derive


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    for name in ['repo','manifest','checkpoint','pretrained','data-path','scenes-dir','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case',type=int,required=True)
    p.add_argument('--start',type=int,default=-1)
    p.add_argument('--actions',default='[]')
    a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    os.environ.update(FALCON_PROBE_OUTPUT=str(out),SCET_BRANCH_START=str(a.start),SCET_BRANCH_ACTIONS=a.actions)
    status=dict(status='PREPARING',case=a.case,start=a.start,actions=json.loads(a.actions))
    try:
        if sha(a.checkpoint)!='ba7f2824604639d934be2359226c9e9fb35d5b1b2fb39ffbcbbbd833252db6ca': raise ValueError('checkpoint SHA')
        if sha(a.pretrained)!='b73ca263e986d584b5c89073c8046e2b2cded757a31a2000a18f2774838fa83f': raise ValueError('pretrained SHA')
        manifest=json.loads(a.manifest.read_text());case=manifest['cases'][a.case]
        data=out/'dataset/val';(data/'content').mkdir(parents=True)
        root=json.loads(gzip.decompress(a.data_path.read_bytes()))
        if root['episodes']!=[]: raise ValueError('expected empty root')
        (data/'val.json.gz').write_bytes(a.data_path.read_bytes())
        shard=a.data_path.parent/'content'/(case['scene_name']+'.json.gz')
        if sha(shard)!=case['shard_sha256']: raise ValueError('shard SHA')
        document=json.loads(gzip.decompress(shard.read_bytes()))
        document['episodes']=[e for e in document['episodes'] if (e['scene_id'],str(e['episode_id']))==(case['scene_id'],case['episode_id'])]
        if len(document['episodes'])!=1: raise ValueError('single selected episode not found')
        (data/'content'/shard.name).write_bytes(gzip.compress(json.dumps(document).encode(),mtime=0))
        mesh=a.scenes_dir/case['scene_id'].replace('data/scene_datasets/','',1)
        if sha(mesh)!=case['mesh_sha256'] or sha(mesh.with_suffix('.navmesh'))!=case['navmesh_sha256']: raise ValueError('scene assets SHA')
        (out/'INPUTS.json').write_text(json.dumps(dict(manifest_sha256=sha(a.manifest),case=case),indent=2)+'\n')
        derive(a.repo,out/'instrumentation')
        os.chdir(a.repo)
        sys.path[:0]=[str(out/'instrumentation'),str(a.repo),str(a.repo/'habitat-lab'),str(a.repo/'habitat-baselines')]
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('managed CUDA required')
        import habitat_sim
        from habitat.config import read_write
        from habitat_baselines.config.default import get_config
        from habitat_baselines.run import execute_exp
        from omegaconf import OmegaConf
        from bridge import ProbeComplete
        cfg=get_config('social_nav_v2/falcon_hm3d.yaml')
        with read_write(cfg):
            cfg.habitat.seed=100
            cfg.habitat.environment.max_episode_steps=500
            cfg.habitat.environment.iterator_options.shuffle=False
            cfg.habitat.dataset.split='val';cfg.habitat.dataset.data_path=str(data/'val.json.gz')
            cfg.habitat.dataset.scenes_dir=str(a.scenes_dir);cfg.habitat.dataset.content_scenes=[case['scene_name']]
            cfg.habitat_baselines.evaluate=True;cfg.habitat_baselines.num_environments=1
            cfg.habitat_baselines.test_episode_count=-1;cfg.habitat_baselines.eval.split='val'
            cfg.habitat_baselines.eval.evals_per_ep=1;cfg.habitat_baselines.eval.video_option=[]
            cfg.habitat_baselines.eval.use_ckpt_config=False;cfg.habitat_baselines.eval.should_load_ckpt=True
            cfg.habitat_baselines.load_resume_state_config=False
            cfg.habitat_baselines.eval_ckpt_path_dir=str(a.checkpoint)
            cfg.habitat_baselines.rl.ddppo.pretrained_weights=str(a.pretrained)
            cfg.habitat_baselines.evaluator._target_='probe_evaluator.FALCONEvaluator'
            cfg.habitat_baselines.tensorboard_dir=str(out/'tensorboard');cfg.habitat_baselines.checkpoint_folder=str(out/'checkpoints')
            cfg.habitat_baselines.log_file=str(out/'habitat.log')
        (out/'CONFIG.json').write_text(json.dumps(OmegaConf.to_container(cfg,resolve=True),indent=2)+'\n')
        OmegaConf.set_readonly(cfg.habitat.dataset.content_scenes,False)
        try: execute_exp(cfg,'eval')
        except ProbeComplete: status['bounded_terminal']=True
        result=json.loads((out/'bridge_result.json').read_text())
        if not result['done'] or result['pending'] or not result['weights_unchanged']: raise RuntimeError('invalid terminal case')
        status.update(status='TERMINAL_COLLECTED',gpu=torch.cuda.get_device_name(0),job_id=os.environ.get('SLURM_JOB_ID'))
    except Exception:
        status['status']='FAILED_RETAINED';(out/'ERROR.txt').write_text(traceback.format_exc());raise
    finally:
        (out/'CASE_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')


if __name__=='__main__': main()
