"""CPU/static contracts only. These tests do not establish Habitat execution."""
import argparse
import ast
from collections import defaultdict
import contextlib
import copy
import gzip
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from weightlab import falcon_forward as f
from weightlab.io import atomic_json, atomic_npz, sha256, tensor_record


class ForwardContracts(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scene = 'data/scene_datasets/hm3d/train/fake_scene/fake_scene.basis.glb'
        self.request = {'schema': 'weightlab_falcon_forward_request_v1', 'call_id': 'call_' + 'a' * 32,
                        'checkpoint_sha256': f.CHECKPOINT_SHA256, 'candidate': 'base',
                        'candidate_tensor_sha256': 'b' * 64, 'split': 'diagnosis', 'phase': 'OPEN',
                        'kind': 'activation_probe', 'episode_keys': [[self.scene, '1']], 'transition_steps': 500}
        self.manifest = {'schema': 'weightlab_falcon_development_manifest_v1',
                         'source_split': 'train', 'purpose': 'weight_editing_development',
                         'role': 'diagnosis', 'episode_keys': self.request['episode_keys']}

    def tearDown(self):
        self.tmp.cleanup()

    def test_open_train_diagnosis_and_frozen_train_confirmation_are_distinct(self):
        self.assertEqual(f.validate_request(self.request, self.manifest), [(self.scene, '1')])
        request, manifest = copy.deepcopy(self.request), copy.deepcopy(self.manifest)
        request.update(split='confirmation', phase='FROZEN', kind='confirmation')
        manifest['role'] = 'confirmation'
        f.validate_request(request, manifest)
        request['phase'] = 'OPEN'
        with self.assertRaisesRegex(ValueError, 'FROZEN_CONFIRMATION'):
            f.validate_request(request, manifest)

    def test_official_val_and_test_cannot_be_relabelled_as_development(self):
        for source in ('val', 'test', 'development'):
            manifest = copy.deepcopy(self.manifest)
            manifest['source_split'] = source
            with self.subTest(source=source), self.assertRaisesRegex(ValueError, 'ONLY_ORIGINAL_TRAIN'):
                f.validate_request(self.request, manifest)
        for scene in ('data/scene_datasets/hm3d/val/x/a.glb', '/absolute/hm3d/train/a.glb',
                      'hm3d/train/../../val/a.glb', 'hm3d/test/x/a.glb'):
            request, manifest = copy.deepcopy(self.request), copy.deepcopy(self.manifest)
            request['episode_keys'] = manifest['episode_keys'] = [[scene, '1']]
            with self.subTest(scene=scene), self.assertRaisesRegex(ValueError, 'SCENE_FORBIDDEN'):
                f.validate_request(request, manifest)

    def test_transition_caps_phase_and_call_identity_rejected(self):
        for edits in ({'transition_steps': 499}, {'transition_steps': 4097}, {'transition_steps': True},
                      {'phase': 'FROZEN'}, {'kind': 'replay_forward'}, {'call_id': 'wrong'},
                      {'call_token': 'another'}, {'split': 'official_test'}):
            request = {**self.request, **edits}
            with self.subTest(edits=edits), self.assertRaises(ValueError):
                f.validate_request(request, self.manifest)

    def test_duplicate_and_mismatching_episode_keys_refused(self):
        request = copy.deepcopy(self.request)
        request['episode_keys'] *= 2
        with self.assertRaisesRegex(ValueError, 'DUPLICATE'):
            f.validate_request(request, self.manifest)
        manifest = copy.deepcopy(self.manifest)
        manifest['episode_keys'] = [[self.scene, '2']]
        with self.assertRaisesRegex(ValueError, 'EPISODES_DIFFER'):
            f.validate_request(self.request, manifest)

    def candidate_files(self):
        arrays = {f.HEAD_KEYS[0]: np.zeros((4, 512), dtype=np.float32),
                  f.HEAD_KEYS[1]: np.zeros(4, dtype=np.float32)}
        npz = self.root / 'candidate.npz'
        atomic_npz(npz, arrays)
        meta = {'schema': 'weightlab_input_binding_v1', 'checkpoint_sha256': f.CHECKPOINT_SHA256,
                'base_tensors': {k: tensor_record(v) for k, v in arrays.items()}}
        path = self.root / 'candidate.json'
        atomic_json(path, meta)
        self.request['candidate_tensor_sha256'] = sha256(npz)
        return npz, path, meta, arrays

    def test_base_and_effective_head_candidate_contracts(self):
        npz, path, meta, arrays = self.candidate_files()
        f.validate_candidate(self.request, npz, path)
        self.request['candidate'] = 'edited'
        meta.update(schema='weightlab_patch_v1', candidate_id='edited', allowed_keys=list(f.HEAD_KEYS),
                    effective_npz={'sha256': sha256(npz)}, effective_tensors=meta['base_tensors'])
        atomic_json(path, meta)
        f.validate_candidate(self.request, npz, path)
        arrays['net.visual_fc.1.bias'] = np.zeros(512, dtype=np.float32)
        atomic_npz(npz, arrays)
        self.request['candidate_tensor_sha256'] = sha256(npz)
        with self.assertRaisesRegex(ValueError, 'BOTH_HEAD_TENSORS_ONLY'):
            f.validate_candidate(self.request, npz, path)

    def test_wrong_candidate_bytes_or_tensor_metadata_refused(self):
        npz, path, meta, _ = self.candidate_files()
        self.request['candidate_tensor_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'CANDIDATE_BYTES'):
            f.validate_candidate(self.request, npz, path)
        self.request['candidate_tensor_sha256'] = sha256(npz)
        meta['base_tensors'][f.HEAD_KEYS[0]]['sha256'] = '0' * 64
        atomic_json(path, meta)
        with self.assertRaisesRegex(ValueError, 'TENSOR_METADATA'):
            f.validate_candidate(self.request, npz, path)

    def test_derivation_preserves_official_source_and_policy_sampling(self):
        repo = Path(__file__).resolve().parents[1] / 'baselines/repos/Falcon'
        source = repo / f.EVALUATOR_PATH
        original = source.read_bytes()
        out = self.root / 'instrumentation'
        metadata = f.derive_evaluator(repo, out, self.root / 'context.json')
        module = (out / 'weightlab_falcon_evaluator.py').read_text()
        ast.parse(module)
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(module.count('deterministic=False,'), original.decode().count('deterministic=False,'))
        self.assertEqual(module.count('wl.before_act('), 1)
        self.assertEqual(module.count('wl.after_step()'), 1)
        self.assertIn('finally:\n            wl.close()', module)
        self.assertEqual(metadata['source_sha256'], f.EVALUATOR_SHA256)

    def test_changed_official_source_cannot_be_instrumented(self):
        path = self.root / f.EVALUATOR_PATH
        path.parent.mkdir(parents=True)
        path.write_text('unknown evaluator')
        with self.assertRaisesRegex(ValueError, 'OFFICIAL_EVALUATOR_CHANGED'):
            f.derive_evaluator(self.root, self.root / 'derived', self.root / 'context.json')

    def test_compute_entry_requires_slurm_before_importing_torch(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'SLURM_ALLOCATION_REQUIRED'):
                f.run_habitat(argparse.Namespace(), {})

    def test_input_only_route_never_calls_runtime_or_creates_feedback(self):
        npz, path, meta, arrays = self.candidate_files()
        args = ['--request', str(self.root / 'request.json'), '--repo', str(self.root / 'repo'),
                '--manifest', str(self.root / 'manifest.json'), '--checkpoint', str(self.root / 'ckpt'),
                '--pretrained', str(self.root / 'pointnav'), '--candidate-tensors', str(npz),
                '--candidate-manifest', str(path), '--data-path', str(self.root / 'train.json.gz'),
                '--scenes-dir', str(self.root / 'scenes'), '--output', str(self.root / 'out'), '--check-inputs-only']
        context = {'inputs': {}, 'request': self.request}
        with patch.object(f, 'prepare_inputs', return_value=(context, {'episodes': [{'synthetic': True}]})), \
             patch.object(f, 'run_habitat', side_effect=AssertionError('Runtime invoked')), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(f.main(args), 0)
        report = json.loads((self.root / 'out/input_validation.json').read_text())
        self.assertTrue(report['no_model_navigation_executed'])
        self.assertFalse((self.root / 'out/editlab_feedback.json').exists())

    def test_gzip_reader_has_no_pickle_path(self):
        path = self.root / 'episodes.json.gz'
        with gzip.open(path, 'wt') as stream:
            json.dump({'episodes': []}, stream)
        self.assertEqual(f.gzip_json(path), {'episodes': []})

    def test_static_training_filter_binds_original_shard_and_exact_requested_ids(self):
        # Synthetic files exercise admission only, including explicit replacement
        # of known checkpoint hashes. They are not Habitat scene/model fixtures.
        npz, meta_path, meta, _ = self.candidate_files()
        repo = self.root / 'repo'
        official = Path(__file__).resolve().parents[1] / 'baselines/repos/Falcon'
        source = repo / f.EVALUATOR_PATH
        source.parent.mkdir(parents=True)
        source.write_bytes((official / f.EVALUATOR_PATH).read_bytes())
        config = repo / 'habitat-baselines/habitat_baselines/config' / f.BASE_CONFIG
        config.parent.mkdir(parents=True)
        config.write_text('# Synthetic static input fixture\n')
        train = self.root / 'data/train.json.gz'
        train.parent.mkdir()
        with gzip.open(train, 'wt') as stream:
            json.dump({'episodes': []}, stream)
        shard = train.parent / 'content/fake_scene.json.gz'
        shard.parent.mkdir()
        with gzip.open(shard, 'wt') as stream:
            json.dump({'episodes': [{'scene_id': self.scene, 'episode_id': '1', 'fake': True},
                                   {'scene_id': self.scene, 'episode_id': '2', 'fake': True}]}, stream)
        scenes = self.root / 'scenes'
        mesh = scenes / self.scene.removeprefix('data/scene_datasets/')
        mesh.parent.mkdir(parents=True)
        mesh.write_bytes(b'SYNTHETIC NOT A REAL MESH')
        mesh.with_suffix('.navmesh').write_bytes(b'SYNTHETIC NOT A REAL NAVMESH')
        ckpt, pretrained = self.root / 'ckpt', self.root / 'pointnav'
        ckpt.write_bytes(b'SYNTHETIC NOT A MODEL')
        pretrained.write_bytes(b'SYNTHETIC NOT AN INITIALIZATION')
        self.request['checkpoint_sha256'] = meta['checkpoint_sha256'] = sha256(ckpt)
        atomic_json(meta_path, meta)
        self.manifest.update(source_root_sha256=sha256(train), scenes=[
            {'scene_id': self.scene, 'scene_name': 'fake_scene', 'sha256': sha256(shard)}])
        request_path, manifest_path = self.root / 'request.json', self.root / 'manifest.json'
        atomic_json(request_path, self.request)
        atomic_json(manifest_path, self.manifest)
        args = argparse.Namespace(request=request_path, manifest=manifest_path,
                                  candidate_tensors=npz, candidate_manifest=meta_path,
                                  checkpoint=ckpt, pretrained=pretrained, repo=repo, data_path=train,
                                  scenes_dir=scenes, scene_dataset_config=None, output=self.root / 'out')
        with patch.object(f, 'CHECKPOINT_SHA256', sha256(ckpt)), patch.object(f, 'POINTNAV_SHA256', sha256(pretrained)):
            context, filtered = f.prepare_inputs(args)
            self.assertEqual([ep['episode_id'] for ep in filtered['episodes']], ['1'])
            self.assertEqual(context['inputs']['source_shards'][0]['sha256'], sha256(shard))
            self.assertFalse(args.output.exists())
            with gzip.open(shard, 'wt') as stream:
                json.dump({'episodes': []}, stream)
            with self.assertRaisesRegex(ValueError, 'SOURCE_TRAIN_SCENE_HASH_MISMATCH'):
                f.prepare_inputs(args)

    def test_capture_transition_guard_requires_exact_requested_episode(self):
        class Tensor:
            def __init__(self, array): self.value = np.asarray(array)
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.value
        class Capture:
            def on_step(self, **kwargs): self.metadata = kwargs
        recorder = object.__new__(f.NavigationRecorder)
        recorder.request = self.request
        recorder.mapping = {'/runtime/scene.glb': self.scene}
        recorder.expected = {(self.scene, '1')}
        recorder.completed, recorder.steps, recorder.pending = {}, 0, None
        recorder.per_episode, recorder.capture = defaultdict(int), Capture()
        episode = type('Episode', (), {'scene_id': '/runtime/scene.glb', 'episode_id': '1'})()
        recorder.before_act([episode], Tensor([[0, 17]]), Tensor([[False, False]]))
        self.assertEqual(recorder.capture.metadata['prev_actions'].tolist(), [0])
        self.assertEqual(recorder.capture.metadata['reset_masks'].tolist(), [True])
        recorder.pending = None
        recorder.steps = 500
        with self.assertRaisesRegex(ValueError, 'TRANSITION_BUDGET_EXHAUSTED'):
            recorder.before_act([episode], Tensor([[0]]), Tensor([[True]]))

    def test_validator_rejects_fabricated_completion_without_real_artifacts(self):
        out = self.root / 'out'
        out.mkdir()
        atomic_json(out / 'forward_context.json', {'request': self.request})
        atomic_json(out / 'editlab_feedback.json', {'schema': 'weightlab_forward_feedback_v1',
                    'status': 'COMPLETE_TRAIN_DEVELOPMENT_BATCH', 'call_id': 'wrong'})
        with self.assertRaisesRegex(ValueError, 'FEEDBACK_REQUEST_BINDING'):
            f.validate_output(out)


if __name__ == '__main__':
    unittest.main()
