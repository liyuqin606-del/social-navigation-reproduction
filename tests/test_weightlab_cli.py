"""CPU end-to-end operator CLI checks; no LLM, SSH, Slurm or Habitat calls."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from weightlab import __main__ as cli
from weightlab.io import sha256
from weightlab.replay import WEIGHT_KEY, BIAS_KEY
from weightlab.session import Session

PROJECT = Path(__file__).resolve().parents[1]


class CliTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.session = self.directory / 'session'
        self.base = self.directory / 'base.npz'
        np.savez(self.base, **{WEIGHT_KEY: np.full((4, 512), 0.1, np.float32),
                              BIAS_KEY: np.ones(4, np.float32)})
        self.cfg = {'base_tensors': 'base.npz', 'checkpoint_sha256': 'a' * 64,
                    'editable_keys': [WEIGHT_KEY, BIAS_KEY], 'split_manifests': {},
                    'reserve_free_bytes': 4 << 30, 'max_forward_calls': 5,
                    'max_batch_episodes': 1, 'max_transition_steps': 8}
        for role in ('diagnosis', 'selection', 'confirmation'):
            (self.directory / (role + '.json')).write_text(json.dumps({'role': role, 'records': [
                {'scene_id': role + '_SECRET_SCENE', 'episode_id': 'e'}]}))
            self.cfg['split_manifests'][role] = role + '.json'
            np.savez(self.directory / (role + '.npz'), hidden=np.ones((3, 512), np.float32),
                     scene_ids=np.asarray([role + '_SECRET_SCENE'] * 3),
                     episode_ids=np.asarray(['e'] * 3), split=np.asarray(role), step_ids=np.arange(3))
        self.config = self.directory / 'config.json'

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *args, success=True):
        result = subprocess.run([sys.executable, '-m', 'weightlab', *map(str, args),
                                 '--session', str(self.session)], cwd=PROJECT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0 if success else 2, result.stderr + result.stdout)
        return json.loads(result.stdout if success else result.stderr)

    def create(self):
        self.config.write_text(json.dumps(self.cfg))
        return self.invoke('init', '--config', self.config)

    def forward(self, role='diagnosis', candidate='base', success=True):
        return self.invoke('replay', '--split', role, '--candidate', candidate,
                           '--batch', self.directory / (role + '.npz'), success=success)

    def test_operator_pipeline_freezes_exact_candidate_and_confirms_without_retuning(self):
        base_sha = sha256(self.base)
        self.create()
        prompt = self.invoke('agent-prompt')
        self.assertNotIn('confirmation_SECRET_SCENE', json.dumps(prompt))
        self.assertNotIn(str(self.directory), json.dumps(prompt))
        self.assertFalse(prompt['automated_llm_invocation'])
        held = self.invoke('replay', '--split', 'confirmation', '--candidate', 'base',
                           '--batch', '/path/that/must/not/be/opened', success=False)
        self.assertIn('WITHHELD', held['error'])
        self.assertEqual(self.invoke('status')['forward_calls_used'], 0)
        self.forward()
        patch = self.directory / 'patch.npz'
        bias = np.ones(4, np.float32); bias[1] += np.float32(.01)
        np.savez(patch, **{BIAS_KEY: bias})
        self.invoke('submit', '--name', 'probe', '--npz', patch, '--rationale', 'Synthetic control probe')
        selected_feedback = self.forward('selection', 'probe')
        self.assertTrue(selected_feedback['feedback']['no_model_navigation_executed'])
        self.assertEqual(selected_feedback['feedback']['transition_steps'], 3)
        self.invoke('select', '--candidate', 'probe')
        frozen = self.invoke('freeze', '--candidate', 'probe')
        frozen_sha = sha256(frozen['effective_npz']['path'])
        error = self.invoke('submit', '--name', 'late', '--npz', patch, '--rationale', 'retune', success=False)
        self.assertIn('FROZEN', error['error'])
        self.forward('confirmation', 'probe')
        state = self.invoke('status')
        self.assertEqual(state['phase'], 'CONFIRMED')
        self.assertEqual(state['forward_calls_used'], 3)
        self.assertEqual(sha256(frozen['effective_npz']['path']), frozen_sha)
        self.assertEqual(sha256(self.base), base_sha)
        self.assertIn('EXHAUSTED', self.forward('confirmation', 'probe', success=False)['error'])
        self.invoke('reject', '--candidate', 'probe', '--reason', 'Confirmation execution is not evidence of benefit')
        self.assertEqual(self.invoke('status')['phase'], 'DECLINED')

    def test_backend_failure_keeps_debit_and_negative_evidence(self):
        self.create()
        path = self.directory / 'diagnosis.npz'
        with np.load(path, allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
        arrays['base_logits'] = np.full((3, 4), 999, np.float32)
        np.savez(path, **arrays)
        error = self.forward(success=False)
        self.assertIn('Recorded base_logits', error['error'])
        state = self.invoke('status')
        self.assertEqual(state['forward_calls_used'], 1)
        self.assertEqual(state['calls'][0]['status'], 'failed')
        call = self.session / 'calls' / state['calls'][0]['call_id']
        self.assertTrue((call / 'EXECUTION_ERROR.json').is_file())
        self.assertTrue((call / 'BATCH.npz').is_file())

    def test_reservation_is_durable_before_backend_and_batch_is_a_snapshot(self):
        self.create()
        source = self.directory / 'diagnosis.npz'
        real_backend = cli.evaluate_head_replay
        def backend(*args, **kwargs):
            state = Session(self.session).status()
            self.assertEqual(state['forward_calls_used'], 1)
            self.assertEqual(len(state['pending_calls']), 1)
            self.assertNotEqual(Path(args[2]), source)
            source.write_bytes(b'replaced original batch after admission')
            return real_backend(*args, **kwargs)
        with mock.patch.object(cli, 'evaluate_head_replay', backend):
            result = cli.replay(Session(self.session), 'base', 'diagnosis', source)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(Session(self.session).status()['forward_calls_used'], 1)

    def test_wrong_split_and_caps_refuse_before_reservation(self):
        self.cfg['max_transition_steps'] = 2
        self.create()
        self.assertIn('STEP_CAP', self.forward(success=False)['error'])
        wrong = self.invoke('replay', '--split', 'selection', '--candidate', 'base',
                            '--batch', self.directory / 'diagnosis.npz', success=False)
        self.assertIn('SPLIT', wrong['error'])
        self.assertEqual(self.invoke('status')['forward_calls_used'], 0)

    def test_budget_is_not_reset_by_cli_restart(self):
        self.cfg['max_forward_calls'] = 2
        self.create(); self.forward(); self.forward()
        self.assertIn('BUDGET_EXHAUSTED', self.forward(success=False)['error'])
        self.assertEqual(self.invoke('status')['forward_calls_used'], 2)

    def test_code_is_fail_closed_and_unknown_call_has_no_refund(self):
        self.create()
        self.assertIn('CODE_EXECUTION_DISABLED', self.invoke('code', success=False)['error'])
        session = Session(self.session)
        reserved = session.reserve_call('replay_forward', 'base', 'diagnosis',
            {'episode_keys': [['diagnosis_SECRET_SCENE', 'e']], 'transition_steps': 3})
        result = self.invoke('resolve-call', '--call-id', reserved['call_id'], '--outcome', 'unknown',
                             '--reason', 'Synthetic crash reconciliation; no rerun')
        self.assertFalse(result['budget_refunded'])
        self.assertEqual(self.invoke('status')['forward_calls_used'], 1)
        self.assertEqual(self.invoke('status')['pending_calls'], [])

    def test_evidence_tampering_is_detected(self):
        self.create(); result = self.forward()
        batch = self.session / 'calls' / result['call_id'] / 'BATCH.npz'
        batch.write_bytes(b'tampered')
        self.assertIn('BOUND_FILE_CHANGED', self.invoke('status', success=False)['error'])

    def test_cli_requires_four_gib_reserve(self):
        self.cfg['reserve_free_bytes'] = 0
        self.config.write_text(json.dumps(self.cfg))
        self.assertIn('FOUR_GIB', self.invoke('init', '--config', self.config, success=False)['error'])
        self.assertFalse(self.session.exists())

    @unittest.skipUnless((PROJECT / 'baselines/prep/weightlab_falcon_base.npz').is_file(), 'Official export unavailable')
    def test_actual_export_synthetic_demo_is_four_calls_and_zero_navigation(self):
        output = self.directory / 'actual_head_synthetic_demo'
        run = subprocess.run([sys.executable, str(PROJECT / 'examples/weightlab/synthetic_demo.py'),
                              '--output', str(output)], cwd=PROJECT, text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        report = json.loads(run.stdout)
        self.assertTrue(report['real_exported_falcon_head'])
        self.assertEqual(report['forward_calls_used'], 4)
        self.assertEqual(report['closed_loop_episodes'], 0)
        self.assertEqual(report['llm_invocations'], 0)
        self.assertFalse(report['scientific_improvement_claim'])
        self.assertEqual(report['phase'], 'CONFIRMED')


if __name__ == '__main__':
    unittest.main()
