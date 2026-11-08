"""Real Habitat/Falcon, bounded engineering trace on existing demo episodes."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from derive import derive


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--scenes-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['reference','shadow','override'], required=True)
    args = parser.parse_args()
    output = args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ.update(FALCON_PROBE_OUTPUT=str(output), FALCON_PROBE_MODE=args.mode)
    result = dict(status='PREPARING', scope='ENGINEERING_ONLY_ON_EXISTING_DEMOS_NO_FITTING_OR_MODEL_SELECTION')
    try:
        if sha(args.checkpoint) != 'ba7f2824604639d934be2359226c9e9fb35d5b1b2fb39ffbcbbbd833252db6ca':
            raise ValueError('checkpoint hash')
        if sha(args.pretrained) != 'b73ca263e986d584b5c89073c8046e2b2cded757a31a2000a18f2774838fa83f':
            raise ValueError('pretrained hash')
        manifest = json.loads(args.manifest.read_text())
        expected = set(map(tuple,manifest['episode_keys']))
        if len(expected)!=3 or manifest['source_split']!='val':
            raise ValueError('exact previous demo selection required')
        data = output/'dataset/val';(data/'content').mkdir(parents=True)
        root = json.loads(gzip.decompress(args.data_path.read_bytes()))
        if root['episodes'] != []:
            raise ValueError('unexpected dataset root')
        (data/'val.json.gz').write_bytes(args.data_path.read_bytes())
        loaded, provenance = set(), []
        for scene in manifest['scenes']:
            shard = args.data_path.parent/'content'/(scene['scene_name']+'.json.gz')
            if sha(shard) != scene['sha256']:
                raise ValueError('original shard hash')
            document = json.loads(gzip.decompress(shard.read_bytes()))
            document['episodes'] = [e for e in document['episodes'] if (e['scene_id'],str(e['episode_id'])) in expected]
            loaded.update((e['scene_id'],str(e['episode_id'])) for e in document['episodes'])
            (data/'content'/shard.name).write_bytes(gzip.compress(json.dumps(document).encode(),mtime=0))
            mesh = args.scenes_dir/scene['scene_id'].replace('data/scene_datasets/','',1)
            provenance.append(dict(mesh=str(mesh), mesh_sha=sha(mesh), navmesh_sha=sha(mesh.with_suffix('.navmesh'))))
        if loaded != expected:
            raise ValueError('demo subset mismatch')
        (output/'INPUTS.json').write_text(json.dumps(dict(manifest_sha=sha(args.manifest),meshes=provenance),indent=2)+'\n')
        derive(args.repo,output/'instrumentation')
        os.chdir(args.repo)
        sys.path[:0] = [str(output/'instrumentation'),str(args.repo),str(args.repo/'habitat-lab'),str(args.repo/'habitat-baselines')]
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('requires managed CUDA allocation')
        import habitat_sim  # bundled Magnum plugins must initialize first
        from habitat.config import read_write
        from habitat_baselines.config.default import get_config
        from habitat_baselines.run import execute_exp
        from omegaconf import OmegaConf
        from bridge import ProbeComplete
        cfg = get_config(manifest['base_config'])
        with read_write(cfg):
            cfg.habitat.seed = 100
            cfg.habitat.environment.max_episode_steps = 500
            cfg.habitat.environment.iterator_options.shuffle = False
            cfg.habitat.dataset.split = 'val'
            cfg.habitat.dataset.data_path = str(data/'val.json.gz')
            cfg.habitat.dataset.scenes_dir = str(args.scenes_dir)
            cfg.habitat.dataset.content_scenes = manifest['content_scenes']
            cfg.habitat_baselines.evaluate = True
            cfg.habitat_baselines.num_environments = 1
            cfg.habitat_baselines.test_episode_count = -1
            cfg.habitat_baselines.eval.split = 'val'
            cfg.habitat_baselines.eval.evals_per_ep = 1
            cfg.habitat_baselines.eval.video_option = []
            cfg.habitat_baselines.eval.use_ckpt_config = False
            cfg.habitat_baselines.eval.should_load_ckpt = True
            cfg.habitat_baselines.load_resume_state_config = False
            cfg.habitat_baselines.eval_ckpt_path_dir = str(args.checkpoint)
            cfg.habitat_baselines.rl.ddppo.pretrained_weights = str(args.pretrained)
            cfg.habitat_baselines.evaluator._target_ = 'probe_evaluator.FALCONEvaluator'
            cfg.habitat_baselines.tensorboard_dir = str(output/'tensorboard')
            cfg.habitat_baselines.checkpoint_folder = str(output/'checkpoints')
            cfg.habitat_baselines.log_file = str(output/'habitat.log')
        (output/'CONFIG.json').write_text(json.dumps(OmegaConf.to_container(cfg,resolve=True),indent=2)+'\n')
        OmegaConf.set_readonly(cfg.habitat.dataset.content_scenes,False)
        result.update(gpu_name=torch.cuda.get_device_name(0), torch_version=torch.__version__,
                      slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        try:
            execute_exp(cfg,'eval')
        except ProbeComplete:
            result['bounded_stop'] = True
        bridge = json.loads((output/'bridge_result.json').read_text())
        if bridge['steps'] != 16 or bridge['pending_transition'] or not bridge['weights_unchanged']:
            raise RuntimeError('incomplete bridge probe')
        result['status'] = 'PROBE_EXECUTED_PENDING_CROSS_CASE_VALIDATION'
        print(json.dumps(result),flush=True)
    except Exception:
        result['status'] = 'FAILED_RETAINED'
        (output/'ERROR.txt').write_text(traceback.format_exc())
        raise
    finally:
        (output/'CASE_STATUS.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
