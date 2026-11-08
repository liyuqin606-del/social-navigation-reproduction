#!/usr/bin/env python3
"""Run a manifest-defined official Falcon val subset with auditable logging.

Use inside the authorized Slurm job and prepared Habitat environment. By default
requires CUDA. --check-inputs-only performs filesystem checks without importing
Habitat/PyTorch. Output directory must be absent or empty; no result is overwritten.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

from prepare_falcon_observed import derive, OFFICIAL_COMMIT
from validate_falcon_subset import validate


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def file_record(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError('Required input absent or empty: ' + str(path))
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, required=True)
    parser.add_argument('--data-path', type=Path, required=True, help='Unmodified val.json.gz')
    parser.add_argument('--scenes-dir', type=Path, required=True, help='Scene root containing hm3d/val/...')
    parser.add_argument('--scene-dataset-config', type=Path, help='Optional official Habitat scene dataset configuration')
    parser.add_argument('--num-envs', type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument('--output', type=Path, default=os.environ.get('VLN_RUN_OUTPUT'))
    parser.add_argument('--check-inputs-only', action='store_true')
    args = parser.parse_args()
    if args.output is None:
        parser.error('--output or VLN_RUN_OUTPUT is required')
    for name in ('repo', 'manifest', 'checkpoint', 'pretrained', 'data_path', 'scenes_dir', 'output'):
        setattr(args, name, getattr(args, name).absolute())
    if args.scene_dataset_config:
        args.scene_dataset_config = args.scene_dataset_config.absolute()
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError('Output must be absent or empty: ' + str(output))
    output.mkdir(parents=True, exist_ok=True)
    os.environ['VLN_RUN_OUTPUT'] = str(output)
    status = {'status': 'PREPARING', 'started_utc': datetime.now(timezone.utc).isoformat()}
    try:
        manifest = json.loads(args.manifest.read_text())
        if manifest.get('source_split') != 'val':
            raise ValueError('This runner supports official val subsets only')
        expected_original = [tuple(k) for k in manifest['episode_keys']]
        if len(expected_original) != len(set(expected_original)) or len(expected_original) != manifest['episode_count']:
            raise ValueError('Source manifest has duplicate or inconsistent episode keys')
        if args.num_envs > manifest['scene_count']:
            raise ValueError('More environments than scenes would repeat the dataset')
        root_data = json.loads(gzip.decompress(args.data_path.read_bytes()))
        if root_data.get('episodes') != []:
            raise ValueError('Expected original empty val root; nonempty root can defeat scene filtering')
        shards, loaded_keys, mesh_records = [], [], []
        for scene in manifest['scenes']:
            path = args.data_path.parent / 'content' / (scene['scene_name'] + '.json.gz')
            rec = file_record(path)
            if rec['sha256'] != scene['sha256']:
                raise ValueError('Original episode shard SHA mismatch: ' + str(path))
            data = json.loads(gzip.decompress(path.read_bytes()))
            loaded_keys.extend((ep['scene_id'], str(ep['episode_id'])) for ep in data['episodes'])
            shards.append(rec)
            relative = scene['scene_id'].removeprefix('data/scene_datasets/')
            mesh_path = args.scenes_dir / relative
            # Retain alias spelling for runtime keys; resolved target is provenance only.
            rec = file_record(mesh_path)
            rec['resolved_target'] = str(mesh_path.resolve())
            rec['navmesh'] = file_record(mesh_path.with_suffix('.navmesh'))
            mesh_records.append(rec)
        if sorted(loaded_keys) != sorted(expected_original):
            raise ValueError('Actual unmodified episode shards differ from the exact manifest')
        expected_runtime = [[str(args.scenes_dir / scene.removeprefix('data/scene_datasets/')), ep]
                            for scene, ep in expected_original]
        inputs = {'checkpoint': file_record(args.checkpoint), 'pointnav_initialization': file_record(args.pretrained),
                  'episode_root': file_record(args.data_path), 'shards': shards, 'meshes': mesh_records,
                  'source_manifest': file_record(args.manifest)}
        if args.scene_dataset_config:
            inputs['scene_dataset_config'] = file_record(args.scene_dataset_config)
        instrumentation = derive(args.repo, output / 'instrumentation')
        (output / 'source_episode_manifest.json').write_bytes(args.manifest.read_bytes())
        (output / 'expected_episodes.json').write_text(json.dumps({
            'episode_keys': expected_runtime, 'original_episode_keys': expected_original,
            'episode_count': len(expected_runtime), 'source_manifest_sha256': digest(args.manifest),
            'mapping': 'Official PointNav loader: join(scenes_dir, scene_id without optional data/scene_datasets/ prefix)'
        }, indent=2) + '\n')
        provenance = {'scope': 'Official Falcon checkpoint on an exact official val subset, not full-paper reproduction',
                      'official_commit': OFFICIAL_COMMIT, 'base_config': manifest['base_config'],
                      'seed': 100, 'policy_deterministic': False, 'max_episode_steps': 500,
                      'num_environments': args.num_envs, 'official_default_num_environments': 8,
                      'parallelism_difference': ('Fewer environments change scene assignment and random-number streams'
                                                 if args.num_envs != 8 else 'Official default environment count retained'),
                      'episode_count': len(expected_runtime), 'scene_count': manifest['scene_count'],
                      'inputs': inputs, 'instrumentation': instrumentation,
                      'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                      'python_executable': sys.executable, 'argv': sys.argv,
                      'runner_sha256': digest(__file__),
                      'validator_sha256': digest(Path(__file__).with_name('validate_falcon_subset.py'))}
        (output / 'run_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
        if args.check_inputs_only:
            status['status'] = 'INPUTS_CHECKED_NO_MODEL_EXECUTED'
            print(json.dumps(status))
            return 0
        os.chdir(args.repo)
        sys.path[:0] = [str(output / 'instrumentation'), str(args.repo),
                       str(args.repo / 'habitat-lab'), str(args.repo / 'habitat-baselines')]
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; run in the managed GPU allocation')
        # Habitat-Sim must initialize its bundled Magnum plugins before Habitat
        # imports magnum. The successful checkpoint/render diagnostic uses this
        # order; importing Habitat first produced duplicate plugins and SIGABRT.
        import habitat_sim
        from habitat.config import read_write
        from habitat_baselines.config.default import get_config
        from habitat_baselines.run import execute_exp
        from omegaconf import OmegaConf
        cfg = get_config(manifest['base_config'])
        with read_write(cfg):
            cfg.habitat.seed = 100
            cfg.habitat.environment.max_episode_steps = 500
            cfg.habitat.environment.iterator_options.shuffle = False
            cfg.habitat.dataset.split = 'val'
            cfg.habitat.dataset.data_path = str(args.data_path)
            cfg.habitat.dataset.scenes_dir = str(args.scenes_dir)
            cfg.habitat.dataset.content_scenes = manifest['content_scenes']
            if args.scene_dataset_config:
                cfg.habitat.simulator.scene_dataset = str(args.scene_dataset_config)
            cfg.habitat_baselines.evaluate = True
            cfg.habitat_baselines.num_environments = args.num_envs
            cfg.habitat_baselines.test_episode_count = -1
            cfg.habitat_baselines.eval.split = 'val'
            cfg.habitat_baselines.eval.evals_per_ep = 1
            cfg.habitat_baselines.eval.video_option = []
            cfg.habitat_baselines.eval.use_ckpt_config = False
            cfg.habitat_baselines.eval.should_load_ckpt = True
            cfg.habitat_baselines.load_resume_state_config = False
            cfg.habitat_baselines.eval_ckpt_path_dir = str(args.checkpoint)
            cfg.habitat_baselines.rl.ddppo.pretrained_weights = str(args.pretrained)
            cfg.habitat_baselines.evaluator._target_ = 'vln_falcon_evaluator.FALCONEvaluator'
            cfg.habitat_baselines.tensorboard_dir = str(output / 'tensorboard')
            cfg.habitat_baselines.video_dir = str(output / 'video')
            cfg.habitat_baselines.checkpoint_folder = str(output / 'checkpoints')
            cfg.habitat_baselines.log_file = str(output / 'habitat.log')
        (output / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
        (output / 'resolved_config.json').write_text(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2) + '\n')
        provenance['torch_version'] = torch.__version__
        provenance['cuda_version'] = torch.version.cuda
        provenance['gpu_name'] = torch.cuda.get_device_name(0)
        provenance['runtime_compatibility'] = [
            'Import habitat_sim before habitat/standalone magnum; no upstream source changed',
            'Allow only explicit content_scenes ListConfig mutation for the official env factory random.shuffle; values and seed unchanged']
        (output / 'run_manifest.json').write_text(json.dumps(provenance, indent=2) + '\n')
        status['status'] = 'EVALUATING'
        (output / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')
        # The upstream default '*' is converted by its factory to a mutable
        # Python list. An explicit subset remains a read-only ListConfig unless
        # this one node is unlocked; the same official shuffle still runs.
        OmegaConf.set_readonly(cfg.habitat.dataset.content_scenes, False)
        execute_exp(cfg, 'eval')
        report = validate(output)
        status['status'] = report['status']
        print(json.dumps(report, indent=2))
        return 0 if report['status'] == 'PASS_SUBSET_COMPLETE' else 2
    except Exception:
        status['status'] = 'FAILED'
        (output / 'failure_traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        (output / 'run_status.json').write_text(json.dumps(status, indent=2) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
