"""Offline NumPy/accounting tests; no GPU, model inference, network or training."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from weightlab.io import ContractError, load_npz, read_json, sha256, tensor_sha256
from weightlab.patches import bias_shift, rank_one, remove_direction
from weightlab.session import Session


class WeightlabSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.root = self.directory / 'session'
        self.base = self.directory / 'base.npz'
        self.weight = np.ones((2, 3), dtype=np.float32)
        self.bias = np.ones(2, dtype=np.float32)
        np.savez(self.base, weight=self.weight, bias=self.bias, fixed=np.ones(3, dtype=np.float32))
        paths = {}
        for role in ('diagnosis', 'selection', 'confirmation'):
            path = self.directory / (role + '_PRIVATE_MANIFEST.json')
            rows = [{'scene_id': role + '_PRIVATE_SCENE', 'episode_id': str(i)} for i in range(2)]
            path.write_text(json.dumps({'role': role, 'records': rows}))
            paths[role] = str(path)
        self.config = {'base_tensors': str(self.base), 'checkpoint_sha256': 'a' * 64,
                       'editable_keys': ['weight', 'bias'], 'split_manifests': paths,
                       'reserve_free_bytes': 0, 'max_forward_calls': 5, 'max_batch_episodes': 2,
                       'max_transition_steps': 10, 'max_candidates': 3, 'max_relative_frobenius': 0.5}
        self.index = 0

    def tearDown(self):
        self.temporary.cleanup()

    def create(self):
        Session.create(self.root, self.config)
        return Session(self.root)

    def request(self, role='diagnosis'):
        return {'episode_keys': [[role + '_PRIVATE_SCENE', '0']], 'transition_steps': 4}

    def patch(self, **arrays):
        self.index += 1
        path = self.directory / ('patch' + str(self.index) + '.npz')
        np.savez(path, **arrays)
        return path

    def complete(self, session, call, status='completed'):
        path = self.directory / (call['call_id'] + '.json')
        path.write_text(json.dumps({'synthetic_offline_test': True, 'call_id': call['call_id']}))
        return session.complete_call(call['call_id'], {'metric': 0.25, 'synthetic': True},
                                     {'status': status, 'files': [str(path)]})

    def select_base(self, session):
        call = session.reserve_call('replay_forward', 'base', 'selection', self.request('selection'))
        self.complete(session, call)
        session.select('base')

    def test_confirmation_hidden_and_official_test_unavailable(self):
        session = self.create()
        status = json.dumps(session.status())
        self.assertNotIn('confirmation_PRIVATE_SCENE', status)
        self.assertNotIn('confirmation_PRIVATE_MANIFEST', status)
        self.assertTrue(session.status()['split_summary']['confirmation']['withheld'])
        with self.assertRaisesRegex(ContractError, 'WITHHELD'):
            session.split_episode_keys('confirmation')
        with self.assertRaisesRegex(ContractError, 'UNAVAILABLE'):
            session.split_episode_keys('official_test')
        with self.assertRaisesRegex(ContractError, 'FREEZE_REQUIRED'):
            session.reserve_call('confirmation', 'base', 'confirmation', self.request('confirmation'))
        self.assertEqual(session.status()['forward_calls_used'], 0)

    def test_scene_overlap_is_rejected_even_with_different_episode_ids(self):
        path = Path(self.config['split_manifests']['selection'])
        path.write_text(json.dumps([{'scene_id': 'diagnosis_PRIVATE_SCENE', 'episode_id': 'different'}]))
        with self.assertRaisesRegex(ContractError, 'SCENE_DISJOINT'):
            self.create()
        self.assertFalse(self.root.exists())

    def test_absolute_parent_inheritance_and_manifest_actual_changes(self):
        session = self.create()
        original_sha = sha256(self.base)
        changed_w = self.weight + np.float32(0.1)
        session.submit_patch('first', self.patch(weight=changed_w), 'Synthetic rank-one shift')
        changed_b = self.bias + np.float32(0.1)
        second = session.submit_patch('second', self.patch(bias=changed_b), 'Inherit W, replace b', parent='first')
        effective = load_npz(second['effective_npz']['path'])
        np.testing.assert_array_equal(effective['weight'], changed_w)
        np.testing.assert_array_equal(effective['bias'], changed_b)
        self.assertEqual(set(effective), {'weight', 'bias'})
        self.assertEqual(sha256(self.base), original_sha)
        manifest = read_json(second['manifest']['path'])
        self.assertEqual(manifest['changes']['weight']['from_parent']['changed_entries'], 0)
        self.assertEqual(manifest['changes']['weight']['from_base']['changed_entries'], 6)
        self.assertEqual(manifest['changes']['weight']['from_base']['rank'], 1)
        self.assertEqual(manifest['patch_tensors']['bias']['sha256'], tensor_sha256(changed_b))
        self.assertEqual(session.status()['forward_calls_used'], 0)

    def test_allowlist_shape_finite_float32_and_noop_guards(self):
        session = self.create()
        invalid = [('fixed', {'fixed': np.ones(3, dtype=np.float32)}, 'NOT_EDITABLE'),
                   ('shape', {'weight': np.ones((3, 2), dtype=np.float32)}, 'SHAPE'),
                   ('dtype', {'weight': self.weight.astype(np.float64)}, 'FLOAT32'),
                   ('nan', {'weight': np.full((2, 3), np.nan, dtype=np.float32)}, 'NONFINITE'),
                   ('noop', {'weight': self.weight}, 'NO_ACTUAL_CHANGE')]
        for name, arrays, error in invalid:
            with self.subTest(name=name), self.assertRaisesRegex(ContractError, error):
                session.submit_patch(name, self.patch(**arrays), 'Invalid synthetic candidate')
        self.assertEqual(session.status()['candidates'], [])

    def test_cumulative_base_norm_prevents_parent_chain_budget_bypass(self):
        self.config['max_relative_frobenius'] = 0.1
        session = self.create()
        session.submit_patch('first', self.patch(weight=self.weight * np.float32(1.06)), 'Six percent')
        with self.assertRaisesRegex(ContractError, 'RELATIVE_NORM_LIMIT'):
            session.submit_patch('second', self.patch(weight=self.weight * np.float32(1.12)), 'Another six percent', parent='first')

    def test_reservation_survives_restart_failed_unknown_are_not_refunded(self):
        self.config['max_forward_calls'] = 2
        session = self.create()
        first = session.reserve_call('replay_forward', 'base', 'diagnosis', self.request())
        restarted = Session(self.root)
        self.assertEqual(restarted.status()['forward_calls_used'], 1)
        self.assertEqual(restarted.status()['pending_calls'], [first['call_id']])
        restarted.complete_call(first['call_id'], {'reason': 'crash'}, {'status': 'unknown'})
        second = restarted.reserve_call('activation_probe', 'base', 'diagnosis', self.request())
        restarted.complete_call(second['call_id'], {'reason': 'runtime failure'}, {'status': 'failed'})
        self.assertEqual(restarted.status()['remaining_forward_calls'], 0)
        self.assertEqual([c['status'] for c in restarted.status()['calls']], ['unknown', 'failed'])
        with self.assertRaisesRegex(ContractError, 'BUDGET_EXHAUSTED'):
            restarted.reserve_call('forward', 'base', 'diagnosis', self.request())
        with self.assertRaisesRegex(ContractError, 'NO_OVERWRITE'):
            restarted.complete_call(first['call_id'], {}, {'status': 'failed'})

    def test_unique_episode_and_transition_caps_and_split_membership(self):
        session = self.create()
        with self.assertRaisesRegex(ContractError, 'NOT_IN_REQUESTED_SPLIT'):
            session.reserve_call('forward', 'base', 'selection', self.request('diagnosis'))
        bad = self.request()
        bad['transition_steps'] = 11
        with self.assertRaisesRegex(ContractError, 'TRANSITION_STEP_CAP'):
            session.reserve_call('forward', 'base', 'diagnosis', bad)
        bad = self.request()
        bad['episode_keys'] *= 2
        with self.assertRaisesRegex(ContractError, 'DUPLICATE_EPISODE'):
            session.reserve_call('forward', 'base', 'diagnosis', bad)
        self.assertEqual(session.status()['forward_calls_used'], 0)

    def test_success_requires_hash_bound_artifact_and_cannot_change_evidence(self):
        session = self.create()
        call = session.reserve_call('forward', 'base', 'diagnosis', self.request())
        with self.assertRaisesRegex(ContractError, 'ARTIFACT_EVIDENCE'):
            session.complete_call(call['call_id'], {}, {'status': 'completed'})
        self.complete(session, call)
        (self.directory / (call['call_id'] + '.json')).write_text('tampered')
        with self.assertRaisesRegex(ContractError, 'BOUND_FILE_CHANGED'):
            session.status()

    def test_selection_and_freeze_require_real_completed_selection_call(self):
        session = self.create()
        with self.assertRaisesRegex(ContractError, 'SELECTION_FORWARD'):
            session.select('base')
        probe = session.reserve_call('activation_probe', 'base', 'selection', self.request('selection'))
        self.complete(session, probe)
        with self.assertRaisesRegex(ContractError, 'SELECTION_FORWARD'):
            session.select('base')
        self.select_base(session)
        pending = session.reserve_call('forward', 'base', 'diagnosis', self.request())
        with self.assertRaisesRegex(ContractError, 'RESOLVE_RESERVED'):
            session.freeze('base')
        session.complete_call(pending['call_id'], {}, {'status': 'unknown'})
        session.freeze('base')
        self.assertEqual(session.status()['phase'], 'FROZEN')

    def test_freeze_confirmation_and_decline_never_reopen_retuning(self):
        session = self.create()
        self.select_base(session)
        frozen = session.freeze('base')
        self.assertEqual(frozen['candidate'], 'base')
        self.assertEqual(len(session.split_episode_keys('confirmation')), 2)
        with self.assertRaisesRegex(ContractError, 'FROZEN'):
            session.submit_patch('late', self.patch(weight=self.weight + np.float32(0.1)), 'Forbidden')
        with self.assertRaisesRegex(ContractError, 'FROZEN'):
            session.select('base')
        with self.assertRaisesRegex(ContractError, 'AFTER_FREEZE'):
            session.reserve_call('activation_probe', 'base', 'diagnosis', self.request())
        call = session.reserve_call('confirmation', 'base', 'confirmation', self.request('confirmation'))
        self.complete(session, call)
        self.assertEqual(session.status()['phase'], 'CONFIRMED')
        with self.assertRaisesRegex(ContractError, 'CONFIRMATION_CALL_BUDGET'):
            session.reserve_call('confirmation', 'base', 'confirmation', self.request('confirmation'))
        session.reject('base', 'Decline after withheld confirmation')
        self.assertEqual(session.status()['phase'], 'DECLINED')
        with self.assertRaisesRegex(ContractError, 'FROZEN'):
            session.select('base')

    def test_no_overwrite_candidate_session_or_call_and_candidate_cap(self):
        self.config['max_candidates'] = 1
        session = self.create()
        path = self.patch(weight=self.weight + np.float32(0.1))
        session.submit_patch('one', path, 'First')
        with self.assertRaisesRegex(ContractError, 'NO_OVERWRITE'):
            session.submit_patch('one', path, 'Same name')
        with self.assertRaisesRegex(ContractError, 'CANDIDATE_BUDGET'):
            session.submit_patch('two', path, 'Extra candidate')
        with self.assertRaisesRegex(ContractError, 'ROOT_ALREADY_EXISTS'):
            Session.create(self.root, self.config)

    def test_external_base_mutation_is_detected_before_new_reservation(self):
        session = self.create()
        np.savez(self.base, weight=self.weight * 2, bias=self.bias)
        with self.assertRaisesRegex(ContractError, 'BOUND_FILE_CHANGED'):
            session.reserve_call('forward', 'base', 'diagnosis', self.request())

    def test_two_processes_cannot_overspend_last_forward_call(self):
        self.config['max_forward_calls'] = 1
        session = self.create()
        code = ('import sys,json;from weightlab.session import Session;'
                's=Session(sys.argv[1]);s.reserve_call("replay_forward","base","diagnosis",'
                '{"episode_keys":[["diagnosis_PRIVATE_SCENE","0"]],"transition_steps":1})')
        children = [subprocess.Popen([sys.executable, '-c', code, str(self.root)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    for _ in range(2)]
        results = [p.communicate(timeout=20) for p in children]
        self.assertEqual(sorted(p.returncode == 0 for p in children), [False, True])
        self.assertTrue(any(b'BUDGET_EXHAUSTED' in stderr for _, stderr in results))
        self.assertEqual(session.status()['forward_calls_used'], 1)


class RecipeTests(unittest.TestCase):
    def test_rank_one_exact_formula_and_no_input_mutation(self):
        w = np.ones((2, 3), dtype=np.float32)
        u = np.array([1, 2], dtype=np.float32)
        v = np.array([1, 0, -1], dtype=np.float32)
        result = rank_one(w, u, v, strength=0.5)
        np.testing.assert_array_equal(result, w + 0.5 * np.outer(u, v))
        np.testing.assert_array_equal(w, np.ones_like(w))
        self.assertEqual(np.linalg.matrix_rank(result - w), 1)

    def test_direction_removal_affects_only_selected_columns(self):
        w = np.array([[2, 4, 6], [1, 3, 5]], dtype=np.float32)
        d = np.array([1, 0], dtype=np.float32)
        full = remove_direction(w, d, selected_columns=[0, 2], strength=1)
        np.testing.assert_array_equal(full, [[0, 4, 0], [1, 3, 5]])
        half = remove_direction(w, d, selected_columns=[0], strength=0.5)
        self.assertEqual(half[0, 0], 1)
        np.testing.assert_array_equal(w, [[2, 4, 6], [1, 3, 5]])
        with self.assertRaisesRegex(ContractError, 'ZERO_DIRECTION'):
            remove_direction(w, np.zeros(2, dtype=np.float32))
        with self.assertRaisesRegex(ContractError, 'INVALID_SELECTED'):
            remove_direction(w, d, [0, 0])

    def test_bias_shift_scalar_or_vector_only(self):
        b = np.array([1, 2], dtype=np.float32)
        np.testing.assert_array_equal(bias_shift(b, 0.5), [1.5, 2.5])
        np.testing.assert_array_equal(bias_shift(b, [0.25, -0.25], strength=2), [1.5, 1.5])
        with self.assertRaisesRegex(ContractError, 'SHAPE'):
            bias_shift(b, [[1, 2]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
