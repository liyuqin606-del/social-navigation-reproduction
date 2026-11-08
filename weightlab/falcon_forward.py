"""Bounded Falcon action-head editing on held-out training/development scenes.

Scientific entry for the existing managed Slurm lifecycle. It is deliberately
separate from the frozen official validation baseline. No optimizer or training
is used. --check-inputs-only imports neither Torch nor Habitat and runs no model.
Real Habitat execution must still be integration-tested in the prepared GPU
environment; the local tests validate contracts and generated instrumentation.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import difflib
import gzip
import json
import math
import os
from pathlib import Path
import re
import sys
import traceback

import numpy as np

from .falcon_checkpoint import CHECKPOINT_SHA256, HEAD_KEYS
from .falcon_runtime import FalconCapture, apply_patch
from .io import atomic_json, atomic_npz, load_npz, read_json, require, sha256, tensor_record

EVALUATOR_PATH = 'habitat-baselines/habitat_baselines/rl/ppo/falcon_evaluator.py'
EVALUATOR_SHA256 = 'd8bde7a4ed64917571f9125862e9446220a6576d3467e48159f78cfea47375bd'
OFFICIAL_COMMIT = 'd15ef512c4b1451f9daeaf6d4d7ecc702ed57430'
BASE_CONFIG = 'social_nav_v2/falcon_hm3d.yaml'
POINTNAV_SHA256 = 'b73ca263e986d584b5c89073c8046e2b2cded757a31a2000a18f2774838fa83f'
MAX_EPISODES, MAX_TRANSITIONS, MAX_EPISODE_STEPS = 8, 4096, 500
ACTIVE_RECORDERS = []


def keys(value):
    require(isinstance(value, list) and value, 'NONEMPTY_EPISODE_KEYS_REQUIRED')
    require(all(isinstance(k, (list, tuple)) and len(k) == 2
                and all(isinstance(x, str) and x for x in k) for k in value), 'EXACT_STRING_KEYS_REQUIRED')
    result = [tuple(k) for k in value]
    require(len(set(result)) == len(result), 'DUPLICATE_EPISODE_KEYS')
    return result


def record(path):
    p = Path(path)
    require(p.is_file() and p.stat().st_size > 0, 'INPUT_FILE_MISSING:' + str(p))
    return {'path': str(p.absolute()), 'bytes': p.stat().st_size, 'sha256': sha256(p)}


def gzip_json(path):
    # A source scene shard is bounded before and after decompression.
    require(Path(path).stat().st_size <= 64 << 20, 'EPISODE_ARCHIVE_TOO_LARGE')
    with gzip.open(path, 'rb') as stream:
        data = stream.read((256 << 20) + 1)
    require(len(data) <= 256 << 20, 'EPISODE_JSON_TOO_LARGE')
    return json.loads(data)


def validate_request(request, manifest):
    require(request.get('schema') == 'weightlab_falcon_forward_request_v1', 'REQUEST_SCHEMA')
    require(isinstance(request.get('call_id'), str)
            and re.fullmatch(r'call_[0-9a-f]{32}', request['call_id']), 'BOUND_CALL_ID_REQUIRED')
    require(request.get('call_token', request['call_id']) == request['call_id'], 'CALL_TOKEN_MISMATCH')
    require(request.get('checkpoint_sha256') == CHECKPOINT_SHA256, 'OFFICIAL_CHECKPOINT_REQUIRED')
    require(isinstance(request.get('candidate'), str) and request['candidate'], 'CANDIDATE_REQUIRED')
    require(isinstance(request.get('candidate_tensor_sha256'), str)
            and re.fullmatch('[0-9a-f]{64}', request['candidate_tensor_sha256']), 'CANDIDATE_SHA_REQUIRED')
    split, phase = request.get('split'), request.get('phase')
    require(split in ('diagnosis', 'selection', 'confirmation'), 'NO_OFFICIAL_TEST_SPLIT_IN_WORKBENCH')
    if split == 'confirmation':
        require(phase == 'FROZEN' and request.get('kind') == 'confirmation', 'FROZEN_CONFIRMATION_REQUIRED')
    else:
        require(phase == 'OPEN' and request.get('kind') in ('forward', 'activation_probe'), 'OPEN_ANALYSIS_REQUIRED')
    selected = keys(request.get('episode_keys'))
    require(len(selected) <= MAX_EPISODES, 'EPISODE_BATCH_LIMIT')
    cap = request.get('transition_steps')
    require(type(cap) is int and len(selected) * MAX_EPISODE_STEPS <= cap <= MAX_TRANSITIONS,
            'COMPLETE_BATCH_TRANSITION_BUDGET_REQUIRED')
    require(manifest.get('schema') == 'weightlab_falcon_development_manifest_v1'
            and manifest.get('purpose') == 'weight_editing_development', 'EXPLICIT_DEVELOPMENT_MANIFEST_REQUIRED')
    require(manifest.get('source_split') == 'train', 'ONLY_ORIGINAL_TRAIN_SCENES_ALLOWED')
    require(manifest.get('role') == split, 'DEVELOPMENT_ROLE_MISMATCH')
    require(set(keys(manifest.get('episode_keys'))) == set(selected), 'MANIFEST_REQUEST_EPISODES_DIFFER')
    for scene, _ in selected:
        p = Path(scene.removeprefix('data/scene_datasets/'))
        require(not p.is_absolute() and '..' not in p.parts and p.parts[:2] == ('hm3d', 'train'),
                'OFFICIAL_VAL_TEST_OR_UNKNOWN_SCENE_FORBIDDEN')
    return selected


def validate_candidate(request, candidate_npz, candidate_manifest):
    require(sha256(candidate_npz) == request['candidate_tensor_sha256'], 'CANDIDATE_BYTES_MISMATCH')
    document = read_json(candidate_manifest)
    require(document.get('checkpoint_sha256') == CHECKPOINT_SHA256, 'CANDIDATE_CHECKPOINT_MISMATCH')
    arrays = load_npz(candidate_npz)
    require(set(arrays) == set(HEAD_KEYS), 'INITIAL_FALCON_BACKEND_REQUIRES_BOTH_HEAD_TENSORS_ONLY')
    require(arrays[HEAD_KEYS[0]].shape == (4, 512) and arrays[HEAD_KEYS[1]].shape == (4,), 'HEAD_SHAPE_MISMATCH')
    base = document.get('base_tensors', {})
    require(set(base) == set(HEAD_KEYS), 'HEAD_BASE_METADATA_REQUIRED')
    if request['candidate'] == 'base':
        require(document.get('schema') in ('weightlab_input_binding_v1', 'weightlab_falcon_base_v1'),
                'BASE_BINDING_MANIFEST_REQUIRED')
        expected = base
    else:
        require(document.get('schema') == 'weightlab_patch_v1'
                and document.get('candidate_id') == request['candidate'], 'PATCH_MANIFEST_CANDIDATE_MISMATCH')
        require(set(document.get('allowed_keys', [])) <= set(HEAD_KEYS), 'ONLY_ACTION_HEAD_EDITS_SUPPORTED')
        require(document.get('effective_npz', {}).get('sha256') == request['candidate_tensor_sha256'],
                'MATERIALIZED_EFFECTIVE_STATE_REQUIRED')
        expected = document.get('effective_tensors', {})
    require(set(expected) == set(HEAD_KEYS), 'EFFECTIVE_HEAD_METADATA_REQUIRED')
    for name, array in arrays.items():
        require(all(expected[name].get(k) == v for k, v in tensor_record(array).items()), 'TENSOR_METADATA_MISMATCH')
    return document


def prepare_inputs(args):
    request, manifest = read_json(args.request), read_json(args.manifest)
    selected = validate_request(request, manifest)
    candidate_manifest = validate_candidate(request, args.candidate_tensors, args.candidate_manifest)
    require(sha256(args.checkpoint) == CHECKPOINT_SHA256, 'OFFICIAL_CHECKPOINT_BYTES_MISMATCH')
    require(sha256(args.pretrained) == POINTNAV_SHA256, 'POINTNAV_INITIALIZATION_BYTES_MISMATCH')
    require(sha256(args.repo / EVALUATOR_PATH) == EVALUATOR_SHA256, 'OFFICIAL_EVALUATOR_CHANGED')
    require(args.data_path.name == 'train.json.gz', 'UNMODIFIED_TRAIN_ROOT_REQUIRED')
    require(sha256(args.data_path) == manifest.get('source_root_sha256'), 'ORIGINAL_TRAIN_ROOT_SHA_MISMATCH')
    root = gzip_json(args.data_path)
    require(root.get('episodes') == [], 'EXPECTED_ORIGINAL_EMPTY_TRAIN_ROOT')
    scenes = manifest.get('scenes')
    require(isinstance(scenes, list) and scenes
            and len({s['scene_id'] for s in scenes}) == len(scenes)
            and {s['scene_id'] for s in scenes} == {s for s, _ in selected}, 'EXACT_DEVELOPMENT_SCENE_LIST_REQUIRED')
    episodes, source_records, geometries = {}, [], []
    for scene in scenes:
        name = scene['scene_name']
        require(isinstance(name, str) and re.fullmatch('[A-Za-z0-9_-]+', name), 'SAFE_SCENE_NAME_REQUIRED')
        shard = args.data_path.parent / 'content' / (name + '.json.gz')
        source_record = record(shard)
        require(source_record['sha256'] == scene.get('sha256'), 'SOURCE_TRAIN_SCENE_HASH_MISMATCH')
        source_records.append(source_record)
        for ep in gzip_json(shard)['episodes']:
            require(ep['scene_id'] == scene['scene_id'], 'SOURCE_SCENE_IDENTIFIER_MISMATCH')
            key = (ep['scene_id'], str(ep['episode_id']))
            require(key not in episodes, 'SOURCE_DUPLICATE_EPISODE')
            episodes[key] = ep
        mesh = args.scenes_dir / scene['scene_id'].removeprefix('data/scene_datasets/')
        require(args.scenes_dir.resolve() in mesh.resolve().parents, 'GEOMETRY_ESCAPES_SCENES_ROOT')
        geometries.extend([record(mesh), record(mesh.with_suffix('.navmesh'))])
    require(set(selected) <= set(episodes), 'REQUESTED_EPISODE_NOT_IN_SOURCE_TRAIN_SHARD')
    filtered = {'episodes': [episodes[k] for k in selected]}
    inputs = {'request': record(args.request), 'development_manifest': record(args.manifest),
              'candidate_tensors': record(args.candidate_tensors), 'candidate_manifest': record(args.candidate_manifest),
              'checkpoint': record(args.checkpoint), 'pointnav': record(args.pretrained),
              'source_root': record(args.data_path), 'source_shards': source_records, 'geometry': geometries,
              'base_config': record(args.repo / 'habitat-baselines/habitat_baselines/config' / BASE_CONFIG)}
    if args.scene_dataset_config:
        inputs['scene_dataset_config'] = record(args.scene_dataset_config)
    mapping = {str(args.scenes_dir / scene.removeprefix('data/scene_datasets/')): scene for scene, _ in selected}
    return {'request': request, 'candidate_manifest': candidate_manifest, 'runtime_scene_mapping': mapping,
            'candidate_tensors': str(args.candidate_tensors), 'output': str(args.output), 'inputs': inputs}, filtered


def derive_evaluator(repo, output, context_path):
    source = Path(repo) / EVALUATOR_PATH
    require(sha256(source) == EVALUATOR_SHA256, 'OFFICIAL_EVALUATOR_CHANGED')
    original = source.read_text()
    derived = original
    replacements = [
        ('            with inference_mode():\n',
         '            wl.before_act(current_episodes_info, prev_actions, not_done_masks)\n'
         '            with inference_mode():\n'),
        ('            outputs = envs.step(step_data)\n',
         '            outputs = envs.step(step_data)\n            wl.after_step()\n'),
        ('                    stats_episodes[(k, ep_eval_count[k])] = episode_stats\n',
         '                    stats_episodes[(k, ep_eval_count[k])] = episode_stats\n'
         '                    wl.episode_completed(k, episode_stats)\n'),
        ('os.path.join("output/", "result.json")', 'os.path.join(wl.output, "result.json")'),
        ('os.path.join("output/", "actions.json")', 'os.path.join(wl.output, "actions.json")'),
    ]
    for before, after in replacements:
        require(derived.count(before) == 1, 'OFFICIAL_SOURCE_ANCHOR_CHANGED')
        derived = derived.replace(before, after, 1)
    marker = '        success_cal = 0 ## my added\n'
    require(derived.count(marker) == 1, 'EVALUATOR_BODY_ANCHOR_CHANGED')
    header, body = derived.split(marker, 1)
    body = marker + body.rstrip() + '\n        wl.finish()\n'
    derived = (header + '        from weightlab.falcon_forward import NavigationRecorder\n'
               + '        wl = NavigationRecorder(agent, ' + repr(str(context_path)) + ')\n'
               + '        try:\n' + ''.join('    ' + line if line.strip() else line for line in body.splitlines(True))
               + '        finally:\n            wl.close()\n')
    ast.parse(derived)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'weightlab_falcon_evaluator.py').write_text(derived)
    patch = ''.join(difflib.unified_diff(original.splitlines(True), derived.splitlines(True),
                                       fromfile=EVALUATOR_PATH, tofile='weightlab_falcon_evaluator.py'))
    (out / 'instrumentation.patch').write_text(patch)
    result = {'official_commit': OFFICIAL_COMMIT, 'source_sha256': EVALUATOR_SHA256,
              'derived_sha256': sha256(out / 'weightlab_falcon_evaluator.py'),
              'patch_sha256': sha256(out / 'instrumentation.patch'), 'original_checkout_modified': False,
              'robot_actor_path': 'agent._agents[0]._actor_critic', 'one_environment': True,
              'changes': ['verified action-head replacement before first act', 'bounded hidden/logit capture',
                          'per-episode scalar records', 'output directory', 'exact coverage feedback']}
    atomic_json(out / 'instrumentation_manifest.json', result)
    return result


def numpy_cpu(value):
    return value.detach().cpu().numpy().copy()


class NavigationRecorder:
    """Used only from the generated official evaluator, after checkpoint load."""
    def __init__(self, agent, context_path):
        self.context = read_json(context_path)
        self.request = self.context['request']
        self.output = Path(self.context['output'])
        self.mapping = self.context['runtime_scene_mapping']
        self.expected = set(keys(self.request['episode_keys']))
        self.steps, self.pending, self.rows, self.completed = 0, None, [], {}
        self.per_episode = defaultdict(int)
        require(hasattr(agent, '_agents') and len(agent._agents) > 0, 'EXPLICIT_FALCON_ROBOT_MANAGER_REQUIRED')
        actor = agent._agents[0]._actor_critic
        params = dict(actor.named_parameters())
        self.base = {k: numpy_cpu(params[k]) for k in HEAD_KEYS}
        for key, array in self.base.items():
            require(all(self.context['candidate_manifest']['base_tensors'][key].get(k) == v
                        for k, v in tensor_record(array).items()), 'LOADED_ACTOR_IS_NOT_OFFICIAL_BASE')
        if self.request['candidate'] != 'base':
            self.patch_receipt = apply_patch(actor, self.context['candidate_tensors'], self.context['candidate_manifest'])
        else:
            import torch
            with torch.no_grad():
                for param in actor.parameters():
                    param.requires_grad_(False)
            actor.eval()
            self.patch_receipt = {'status': 'ORIGINAL_BASE_UNCHANGED', 'model_forward_executed': False}
        self.capture = FalconCapture(actor, recorder=self.capture_row,
                                     policy_id=self.request['candidate'],
                                     base_manifest=self.context['candidate_manifest'], max_records=MAX_TRANSITIONS)
        self.capture.__enter__()
        self.finished = False
        ACTIVE_RECORDERS.append(self)
        atomic_json(self.output / 'patch_application.json', self.patch_receipt)

    def before_act(self, episodes, prev_actions, masks):
        require(len(episodes) == 1, 'ONE_ENVIRONMENT_REQUIRED_FOR_EXACT_CALL_BOUND')
        require(self.pending is None and self.steps < self.request['transition_steps'], 'TRANSITION_BUDGET_EXHAUSTED')
        ep = episodes[0]
        require(ep.scene_id in self.mapping, 'UNREQUESTED_RUNTIME_SCENE')
        key = (self.mapping[ep.scene_id], str(ep.episode_id))
        require(key in self.expected and key not in self.completed, 'UNREQUESTED_OR_REPEATED_EPISODE')
        self.pending = key
        self.capture.on_step(scene_ids=[key[0]], episode_ids=[key[1]],
                             prev_actions=numpy_cpu(prev_actions)[:, 0].astype(np.int64),
                             reset_masks=~numpy_cpu(masks)[:, 0].astype(bool),
                             step_ids=np.array([self.per_episode[key]], dtype=np.int64), split=self.request['split'])

    def capture_row(self, row):
        require(self.pending is not None and len(self.rows) == self.steps, 'CAPTURE_TRANSITION_ORDER_MISMATCH')
        hidden = row['hidden']
        base_logits = hidden @ self.base[HEAD_KEYS[0]].T + self.base[HEAD_KEYS[1]]
        candidate_logits = row.get('logits', row.get('base_logits'))
        if self.request['candidate'] == 'base':
            require(np.allclose(base_logits, candidate_logits, rtol=1e-5, atol=1e-5), 'BASE_HEAD_LOGIT_MISMATCH')
        self.rows.append({**row, 'base_logits': base_logits, 'candidate_logits': candidate_logits})

    def after_step(self):
        require(self.pending is not None and len(self.rows) == self.steps + 1, 'ENV_STEP_WITHOUT_POLICY_CAPTURE')
        self.steps += 1
        self.per_episode[self.pending] += 1
        self.pending = None

    def episode_completed(self, runtime_key, metrics):
        require(runtime_key[0] in self.mapping, 'UNEXPECTED_COMPLETED_SCENE')
        key = (self.mapping[runtime_key[0]], str(runtime_key[1]))
        require(key in self.expected and key not in self.completed and self.per_episode[key] > 0,
                'MISSING_OR_DUPLICATE_COMPLETED_EPISODE')
        require(all(isinstance(v, (int, float, np.number)) and np.isfinite(v) for v in metrics.values()),
                'FINITE_SCALAR_METRICS_REQUIRED')
        clean = {str(k): float(v) for k, v in metrics.items()}
        self.completed[key] = clean
        with (self.output / 'navigation_episode_metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'scene_id': key[0], 'episode_id': key[1],
                                    'action_steps': self.per_episode[key], 'metrics': clean}, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def finish(self):
        require(self.pending is None and self.steps > 0 and len(self.rows) == self.steps, 'NO_COMPLETE_NAVIGATION_CAPTURE')
        require(set(self.completed) == self.expected, 'EXACT_REQUESTED_EPISODE_COVERAGE_REQUIRED')
        require(self.steps <= self.request['transition_steps'], 'TRANSITION_CAP_EXCEEDED')
        arrays = {key: np.concatenate([row[key] for row in self.rows], axis=0) for key in
                  ('hidden', 'base_logits', 'candidate_logits', 'scene_ids', 'episode_ids',
                   'prev_actions', 'reset_masks', 'step_ids')}
        arrays['split'] = np.array(self.request['split'])
        atomic_npz(self.output / 'diagnostic_activations.npz', arrays)
        metric_keys = set.intersection(*(set(x) for x in self.completed.values()))
        require({'success', 'spl'} <= metric_keys, 'NAVIGATION_METRICS_MISSING')
        averages = {key: float(np.mean([row[key] for row in self.completed.values()])) for key in sorted(metric_keys)}
        feedback = {'schema': 'weightlab_forward_feedback_v1', 'status': 'COMPLETE_TRAIN_DEVELOPMENT_BATCH',
                    'call_id': self.request['call_id'], 'call_token': self.request['call_id'],
                    'candidate': self.request['candidate'], 'candidate_tensor_sha256': self.request['candidate_tensor_sha256'],
                    'checkpoint_sha256': CHECKPOINT_SHA256, 'split': self.request['split'],
                    'episode_keys': self.request['episode_keys'], 'completed_episode_count': len(self.completed),
                    'transition_steps': self.steps, 'transition_step_cap': self.request['transition_steps'],
                    'metrics': averages, 'no_model_navigation_executed': False,
                    'closed_loop_navigation_executed': True, 'num_environments': 1,
                    'source_split': 'train', 'policy_deterministic': False, 'seed': 100,
                    'paired_logits_scope': 'Base and edited heads on the edited policy trajectory hidden states; not two paired closed-loop rollouts',
                    'official_benchmark_result': False, 'scientific_improvement_demonstrated': False,
                    'artifacts': {name: record(self.output / name) for name in
                                  ('diagnostic_activations.npz', 'navigation_episode_metrics.jsonl', 'patch_application.json')}}
        atomic_json(self.output / 'editlab_feedback.json', feedback)
        self.finished = True

    def close(self):
        self.capture.close()
        atomic_json(self.output / 'navigation_runtime_receipt.json', {
            'status': 'COMPLETED' if self.finished else 'FAILED_RETAINED',
            'completed_env_transitions': self.steps, 'captured_forwards': len(self.rows),
            'complete_episodes': len(self.completed), 'call_id': self.request['call_id'],
            'patch_applied': self.request['candidate'] != 'base'})
        if self in ACTIVE_RECORDERS:
            ACTIVE_RECORDERS.remove(self)


def run_habitat(args, context):
    require(str(os.environ.get('SLURM_JOB_ID', '')).isdigit() and os.environ.get('SLURM_JOB_NAME'),
            'MANAGED_SLURM_ALLOCATION_REQUIRED')
    os.chdir(args.repo)
    sys.path[:0] = [str(args.output / 'instrumentation'), str(args.repo),
                   str(args.repo / 'habitat-lab'), str(args.repo / 'habitat-baselines')]
    import torch
    require(torch.cuda.is_available(), 'CUDA_ALLOCATION_REQUIRED')
    import habitat_sim  # Must precede standalone Magnum/Habitat imports.
    from habitat.config import read_write
    from habitat_baselines.config.default import get_config
    from habitat_baselines.run import execute_exp
    from omegaconf import OmegaConf
    cfg = get_config(BASE_CONFIG)
    with read_write(cfg):
        cfg.habitat.seed = 100
        cfg.habitat.environment.max_episode_steps = MAX_EPISODE_STEPS
        cfg.habitat.environment.iterator_options.shuffle = False
        cfg.habitat.dataset.split = 'train'
        cfg.habitat.dataset.data_path = str(args.output / 'filtered/train.json.gz')
        cfg.habitat.dataset.scenes_dir = str(args.scenes_dir)
        cfg.habitat.dataset.content_scenes = ['*']
        if args.scene_dataset_config:
            cfg.habitat.simulator.scene_dataset = str(args.scene_dataset_config)
        cfg.habitat_baselines.evaluate = True
        cfg.habitat_baselines.num_environments = 1
        cfg.habitat_baselines.test_episode_count = len(context['request']['episode_keys'])
        cfg.habitat_baselines.eval.split = 'train'
        cfg.habitat_baselines.eval.evals_per_ep = 1
        cfg.habitat_baselines.eval.video_option = []
        cfg.habitat_baselines.eval.use_ckpt_config = False
        cfg.habitat_baselines.eval.should_load_ckpt = True
        cfg.habitat_baselines.load_resume_state_config = False
        cfg.habitat_baselines.eval_ckpt_path_dir = str(args.checkpoint)
        cfg.habitat_baselines.rl.ddppo.pretrained_weights = str(args.pretrained)
        cfg.habitat_baselines.evaluator._target_ = 'weightlab_falcon_evaluator.FALCONEvaluator'
        cfg.habitat_baselines.tensorboard_dir = str(args.output / 'tensorboard')
        cfg.habitat_baselines.video_dir = str(args.output / 'video')
        cfg.habitat_baselines.checkpoint_folder = str(args.output / 'checkpoints')
        cfg.habitat_baselines.log_file = str(args.output / 'habitat.log')
    (args.output / 'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg, resolve=True))
    atomic_json(args.output / 'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.set_readonly(cfg.habitat.dataset.content_scenes, False)
    execute_exp(cfg, 'eval')
    require((args.output / 'editlab_feedback.json').is_file(), 'NAVIGATION_DID_NOT_EMIT_VERIFIED_FEEDBACK')


def validate_output(output, request_path=None):
    """Recompute coverage and metrics from retained files; no model execution."""
    output = Path(output)
    feedback = read_json(output / 'editlab_feedback.json')
    context = read_json(output / 'forward_context.json')
    request = context['request']
    if request_path is not None:
        require(read_json(request_path) == request
                and sha256(request_path) == context['inputs']['request']['sha256'],
                'VALIDATOR_REQUEST_BINDING_MISMATCH')
    require(feedback.get('schema') == 'weightlab_forward_feedback_v1'
            and feedback.get('status') == 'COMPLETE_TRAIN_DEVELOPMENT_BATCH', 'NAVIGATION_FEEDBACK_SCHEMA')
    for name in ('call_id', 'candidate', 'candidate_tensor_sha256', 'split', 'episode_keys'):
        require(feedback.get(name) == request.get(name), 'FEEDBACK_REQUEST_BINDING_MISMATCH:' + name)
    require(feedback.get('checkpoint_sha256') == CHECKPOINT_SHA256
            and feedback.get('source_split') == 'train'
            and feedback.get('no_model_navigation_executed') is False
            and feedback.get('official_benchmark_result') is False, 'FEEDBACK_SCOPE_MISMATCH')
    receipt = read_json(output / 'navigation_runtime_receipt.json')
    require(receipt.get('status') == 'COMPLETED' and receipt.get('call_id') == request['call_id'], 'RUNTIME_NOT_COMPLETE')
    expected = set(keys(request['episode_keys']))
    for name in ('diagnostic_activations.npz', 'navigation_episode_metrics.jsonl', 'patch_application.json'):
        bound = feedback['artifacts'][name]
        path = output / name
        require(path.stat().st_size == bound['bytes'] and sha256(path) == bound['sha256'], 'RESULT_ARTIFACT_HASH_MISMATCH')
    rows = [json.loads(line) for line in (output / 'navigation_episode_metrics.jsonl').read_text().splitlines()]
    observed = [(row['scene_id'], row['episode_id']) for row in rows]
    require(len(observed) == len(set(observed)) and set(observed) == expected, 'RESULT_EPISODE_COVERAGE_MISMATCH')
    require(all(type(row['action_steps']) is int and 0 < row['action_steps'] <= MAX_EPISODE_STEPS for row in rows),
            'RESULT_EPISODE_STEP_BOUND')
    steps = sum(row['action_steps'] for row in rows)
    require(steps == feedback['transition_steps'] == receipt['completed_env_transitions']
            == receipt['captured_forwards'] and steps <= request['transition_steps'], 'RESULT_STEP_ACCOUNTING_MISMATCH')
    require(len(rows) == feedback['completed_episode_count'] == receipt['complete_episodes'], 'RESULT_EPISODE_COUNT_MISMATCH')
    metric_keys = set.intersection(*(set(row['metrics']) for row in rows))
    require(set(feedback['metrics']) == metric_keys and {'success', 'spl'} <= metric_keys, 'RESULT_METRIC_KEYS_MISMATCH')
    for key in metric_keys:
        values = [float(row['metrics'][key]) for row in rows]
        require(all(math.isfinite(v) for v in values), 'RESULT_NONFINITE_METRIC')
        recomputed = math.fsum(values) / len(rows)
        require(math.isclose(recomputed, feedback['metrics'][key], rel_tol=1e-12, abs_tol=1e-12), 'RESULT_METRIC_MEAN_MISMATCH')
    with np.load(output / 'diagnostic_activations.npz', allow_pickle=False) as arrays:
        require(arrays['hidden'].shape == (steps, 512) and arrays['base_logits'].shape == (steps, 4)
                and arrays['candidate_logits'].shape == (steps, 4), 'ACTIVATION_SHAPE_MISMATCH')
        for name in ('hidden', 'base_logits', 'candidate_logits'):
            require(arrays[name].dtype == np.float32 and np.isfinite(arrays[name]).all(), 'INVALID_ACTIVATION_DTYPE_OR_VALUES')
        require(arrays['split'].item() == request['split'], 'ACTIVATION_SPLIT_MISMATCH')
        observed_steps = list(zip(arrays['scene_ids'].tolist(), arrays['episode_ids'].tolist()))
        require(len(observed_steps) == steps and set(observed_steps) == expected, 'ACTIVATION_KEY_COVERAGE_MISMATCH')
        for row in rows:
            key = (row['scene_id'], row['episode_id'])
            idx = [i for i, k in enumerate(observed_steps) if k == key]
            require(len(idx) == row['action_steps'] and arrays['step_ids'][idx].tolist() == list(range(len(idx))),
                    'ACTIVATION_EPISODE_STEPS_MISMATCH')
    return {'status': 'PASS_WEIGHTLAB_NAVIGATION_ARTIFACTS', 'call_id': request['call_id'],
            'completed_episode_count': len(rows), 'transition_steps': steps,
            'scope': 'Artifact integrity and recomputed development metrics, not a SOTA claim'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validate-output', type=Path)
    for name in ('request', 'repo', 'manifest', 'checkpoint', 'pretrained', 'candidate-tensors',
                 'candidate-manifest', 'data-path', 'scenes-dir', 'output'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--scene-dataset-config', type=Path)
    parser.add_argument('--check-inputs-only', action='store_true')
    args = parser.parse_args(argv)
    if args.validate_output:
        report = validate_output(args.validate_output.absolute(), args.request)
        atomic_json(args.validate_output / 'independent_validation.json', report)
        print(json.dumps(report))
        return 0
    for name in ('request', 'repo', 'manifest', 'checkpoint', 'pretrained', 'candidate_tensors',
                 'candidate_manifest', 'data_path', 'scenes_dir', 'output'):
        if getattr(args, name) is None:
            parser.error('--' + name.replace('_', '-') + ' is required')
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.absolute())
    require(not args.output.exists() or not any(args.output.iterdir()), 'OUTPUT_DIRECTORY_MUST_BE_EMPTY')
    context, filtered = prepare_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'input_validation.json', {
        'status': 'INPUTS_CHECKED_NO_MODEL_EXECUTED', 'inputs': context['inputs'],
        'episode_count': len(filtered['episodes']), 'no_model_navigation_executed': True})
    if args.check_inputs_only:
        print(json.dumps({'status': 'INPUTS_CHECKED_NO_MODEL_EXECUTED', 'no_model_navigation_executed': True}))
        return 0
    context_path = args.output / 'forward_context.json'
    atomic_json(context_path, context)
    (args.output / 'filtered').mkdir()
    with gzip.GzipFile(filename=str(args.output / 'filtered/train.json.gz'), mode='wb', mtime=0) as stream:
        stream.write(json.dumps(filtered).encode())
    instrumentation = derive_evaluator(args.repo, args.output / 'instrumentation', context_path)
    atomic_json(args.output / 'run_manifest.json', {'schema': 'weightlab_falcon_navigation_run_v1',
                'inputs': context['inputs'], 'instrumentation': instrumentation,
                'entry_sha256': sha256(__file__), 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                'scope': 'Train/development episodes; single released checkpoint; no official test score'})
    try:
        run_habitat(args, context)
    except BaseException:
        (args.output / 'failure_traceback.txt').write_text(traceback.format_exc())
        raise
    print(json.dumps(read_json(args.output / 'editlab_feedback.json')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
