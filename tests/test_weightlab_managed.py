"""Managed bridge contract fixtures. No SSH, Slurm, GPU, or model execution."""
from contextlib import nullcontext
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from weightlab import managed as m
from weightlab.io import ContractError, atomic_json, read_json, sha256
from weightlab.session import Session
from weightlab.falcon_checkpoint import CHECKPOINT_SHA256


class ManagedBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        base = self.root / 'base.npz'
        np.savez(base, weight=np.ones((2, 3), dtype=np.float32))
        split = {}
        for role in ('diagnosis', 'selection', 'confirmation'):
            path = self.root / (role + '.json')
            atomic_json(path, [{'scene_id': role + '_scene', 'episode_id': '0'}])
            split[role] = str(path)
        self.session_root = self.root / 'session'
        Session.create(self.session_root, {'base_tensors': str(base), 'checkpoint_sha256': CHECKPOINT_SHA256,
                       'editable_keys': ['weight'], 'split_manifests': split, 'reserve_free_bytes': 0})
        self.session = Session(self.session_root)
        self.call = self.session.reserve_call('forward', 'base', 'diagnosis',
                    {'episode_keys': [['diagnosis_scene', '0']], 'transition_steps': 500})
        self.call_id = self.call['call_id']
        self.directory = self.session_root / 'managed' / self.call_id
        self.directory.parent.mkdir()
        self.profile = self.root / 'profile.json'
        atomic_json(self.profile, {'remote_userdir': '/remote/fixture', 'username': 'fixture'})
        script = self.root / 'unused_runner.py'
        script.write_text('raise RuntimeError("Fixture runner must never execute")\n')
        self.spec = {'run_id': 'weightlab_fixture_' + self.call_id[-12:],
            'remote_root': '/remote/fixture/runs', 'remote_python': '/remote/python',
            'gpu_binding': 'slurm_linux_minor_uuid',
            'slurm': {'partition': 'tacc', 'gpus': 1, 'cpus': 2, 'wall_time': '00:03:00',
                      'deadline': '2099-01-01T00:00:00'},
            'science': {'cwd': '{source}',
                'argv': ['/remote/python', '{source}/unused_runner.py', '--request', '{source}/' + m.REQUEST,
                         '--candidate-tensors', '{source}/' + m.CANDIDATE,
                         '--candidate-manifest', '{source}/' + m.CANDIDATE_MANIFEST, '--output', '{output}/forward'],
                'validator_argv': ['/remote/python', '{source}/unused_runner.py', '--request', '{source}/' + m.REQUEST,
                                   '--validate-output', '{output}/forward'],
                'timeout_seconds': 60, 'validation_timeout_seconds': 30,
                'required_outputs': m.FORWARD_OUTPUTS},
            'uploads': [{'file': str(script), 'remote_name': script.name}],
            'reserve_bytes': 4 << 30, 'output_budget_bytes': 32768,
            'remote_peak_additional_bytes': 65536, 'monitor_timeout_seconds': 600}
        self.spec_path = self.root / 'spec.json'
        atomic_json(self.spec_path, self.spec)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self):
        return m.prepare_managed_call(self.session_root, self.call_id, self.spec_path,
                                     self.directory, profile=self.profile, python=sys.executable)

    def launch_fixture(self):
        with mock.patch.object(m, 'PROJECT', self.root / 'fake_project'), \
             mock.patch.object(m, '_baseline_guard', return_value=nullcontext()), \
             mock.patch.object(m, '_account_guard', return_value={'squeue': '', 'username': 'fixture'}), \
             mock.patch.object(m.subprocess, 'Popen', return_value=SimpleNamespace(pid=999999)) as popen:
            result = m.launch_managed_call(self.session_root, self.call_id)
        self.assertEqual(popen.call_count, 1)
        return result

    def artifacts(self):
        """Synthetic successful owner output, never an actual GPU result."""
        self.prepare(); self.launch_fixture()
        plan = self.directory / 'plan'
        (plan / 'run/vm/forward').mkdir(parents=True)
        (plan / 'managed_records').mkdir()
        control = read_json(plan / 'CONTROL.json')
        run_id = control['run']['session_name']
        driver = {'session': run_id, 'attempt_id': control['run']['attempt_id'],
            'control_freeze_sha256': sha256(plan / 'CONTROL_FREEZE.json'),
            'entry_driver_sha256': sha256(plan / 'owner.py'), 'job_id': '12345',
            'status': 'OUTPUTS_VERIFIED', 'released': True, 'local_outputs_verified': True}
        atomic_json(plan / 'run/DRIVER_RECEIPT.json', driver)
        manager = {'source_sha256': sha256(plan / 'managed_run.py'),
            'managed_config_sha256': sha256(plan / 'MANAGED.json'),
            'control_sha256': sha256(plan / 'CONTROL.json'),
            'control_freeze_sha256': sha256(plan / 'CONTROL_FREEZE.json'),
            'run': str(plan / 'run'), 'status': 'OWNER_COMPLETED', 'owner_returncode': 0}
        atomic_json(plan / 'managed_records/MANAGED_RECEIPT.json', manager)
        atomic_json(plan / 'run/SCHEDULER_LATEST.json', {
            'squeue': '', 'sacct': f'12345|{run_id}|fixture|COMPLETED|0:0\n12345.batch|batch||COMPLETED|0:0',
            'state': 'COMPLETED', 'exit_code': '0:0', 'released': True, 'identity_verified': True})
        worker = {'run_id': run_id, 'slurm_job_id': '12345',
            'worker_sha256': sha256(plan / 'worker.py'), 'config_sha256': sha256(plan / 'worker.json'),
            'scientific_success': True, 'scientific_exit_code': 0, 'validation_exit_code': 0,
            'status': 'SCIENTIFIC_OUTPUTS_VALIDATED', 'gpu_count': 1}
        atomic_json(plan / 'run/vm/WORKER_RECEIPT.json', worker)
        feedback = {'schema': 'weightlab_forward_feedback_v1', 'call_id': self.call_id,
            'candidate_tensor_sha256': self.call['candidate_tensor_sha256'], 'split': 'diagnosis',
            'checkpoint_sha256': CHECKPOINT_SHA256, 'episode_keys': [['diagnosis_scene', '0']],
            'transition_steps': 1, 'no_model_navigation_executed': False,
            'metrics': {'success': 1.0, 'spl': 0.5}, 'synthetic_fixture': True,
            'status': 'COMPLETE_TRAIN_DEVELOPMENT_BATCH', 'source_split': 'train',
            'official_benchmark_result': False, 'candidate': 'base', 'completed_episode_count': 1}
        forward = plan / 'run/vm/forward'
        request = read_json(self.directory / m.REQUEST)
        atomic_json(forward / 'forward_context.json', {'request': request,
                    'inputs': {'request': {'sha256': sha256(self.directory / m.REQUEST)}}})
        atomic_json(forward / 'navigation_runtime_receipt.json', {'status': 'COMPLETED',
            'call_id': self.call_id, 'completed_env_transitions': 1, 'captured_forwards': 1, 'complete_episodes': 1})
        (forward / 'navigation_episode_metrics.jsonl').write_text(json.dumps({
            'scene_id': 'diagnosis_scene', 'episode_id': '0', 'action_steps': 1,
            'metrics': feedback['metrics']}) + '\n')
        np.savez(forward / 'diagnostic_activations.npz', hidden=np.zeros((1, 512), dtype=np.float32),
                 base_logits=np.zeros((1, 4), dtype=np.float32), candidate_logits=np.zeros((1, 4), dtype=np.float32),
                 scene_ids=np.array(['diagnosis_scene']), episode_ids=np.array(['0']), step_ids=np.array([0]),
                 split=np.array('diagnosis'))
        for name in ('patch_application.json', 'independent_validation.json', 'run_manifest.json', 'resolved_config.json'):
            atomic_json(forward / name, {'synthetic_fixture': True})
        feedback['artifacts'] = {name: {'bytes': (forward / name).stat().st_size,
                                      'sha256': sha256(forward / name)}
            for name in ('diagnostic_activations.npz', 'navigation_episode_metrics.jsonl', 'patch_application.json')}
        atomic_json(plan / 'run/vm' / m.FEEDBACK, feedback)
        self.remanifest(plan)
        return plan

    def remanifest(self, plan):
        files = [{'path': name, 'bytes': (plan / 'run/vm' / name).stat().st_size,
                  'sha256': sha256(plan / 'run/vm' / name)}
                 for name in [*m.FORWARD_OUTPUTS, 'WORKER_RECEIPT.json']]
        atomic_json(plan / 'run/REMOTE_OUTPUT_MANIFEST.json', {'run_id': self.spec['run_id'],
                    'slurm_job_id': '12345', 'files': files})
        atomic_json(plan / 'run/COLLECTED.json', {'files': files})

    def test_real_local_preparer_freezes_call_and_candidate(self):
        original = sha256(self.spec_path)
        result = self.prepare()
        self.assertEqual(result['phase'], 'PREPARED')
        self.assertEqual(sha256(self.spec_path), original)
        request = read_json(self.directory / m.REQUEST)
        self.assertEqual(request['call_id'], self.call_id)
        self.assertEqual(request['schema'], 'weightlab_falcon_forward_request_v1')
        self.assertEqual(request['transition_steps'], 500)
        self.assertEqual(sha256(self.directory / m.CANDIDATE), self.call['candidate_tensor_sha256'])
        self.assertFalse((self.directory / 'plan/run').exists())
        self.assertEqual(self.session.status()['forward_calls_used'], 1)
        with self.assertRaisesRegex(ContractError, 'ALREADY_PREPARED'):
            self.prepare()

    def test_missing_runner_binding_and_oversized_gpu_request_refused(self):
        for change, error in [(lambda s: s['science']['argv'].remove('--request'), 'RUNNER_FLAG'),
                              (lambda s: s['slurm'].update(gpus=3), 'TWO_GPUS')]:
            spec = copy.deepcopy(self.spec); change(spec); atomic_json(self.spec_path, spec)
            with self.assertRaisesRegex(ContractError, error): self.prepare()
        self.assertFalse(self.directory.exists())

    def test_launch_persists_intent_and_never_repeats(self):
        self.prepare(); result = self.launch_fixture()
        self.assertEqual(result['pid'], 999999)
        self.assertTrue((self.directory / 'LAUNCH_INTENT.json').exists())
        with self.assertRaisesRegex(ContractError, 'RECONCILE_NO_RESUBMIT'):
            m.launch_managed_call(self.session_root, self.call_id)

    def test_launch_exception_preserves_ambiguous_intent_and_budget(self):
        self.prepare()
        with mock.patch.object(m, 'PROJECT', self.root / 'fake_project'), \
             mock.patch.object(m, '_baseline_guard', return_value=nullcontext()), \
             mock.patch.object(m, '_account_guard', return_value={'squeue': '', 'username': 'fixture'}), \
             mock.patch.object(m.subprocess, 'Popen', side_effect=OSError('fixture launch failure')):
            with self.assertRaises(OSError): m.launch_managed_call(self.session_root, self.call_id)
        with self.assertRaisesRegex(ContractError, 'RECONCILE_NO_RESUBMIT'):
            m.launch_managed_call(self.session_root, self.call_id)
        self.assertEqual(self.session.status()['forward_calls_used'], 1)

    def test_account_guard_checks_whole_user_queue_and_fails_closed(self):
        self.prepare(); plan = self.directory / 'plan'
        for payload in [{'username': 'fixture', 'squeue': '42|fixture|other|RUNNING'},
                        {'username': 'foreign', 'squeue': ''}]:
            with mock.patch.object(m.subprocess, 'run', return_value=SimpleNamespace(
                    returncode=0, stdout=json.dumps(payload))):
                with self.assertRaises(ContractError): m._account_guard(plan, sys.executable)
        def inspected_run(argv, **kwargs):
            compile(argv[2], '<account-guard>', 'exec')
            self.assertIn("'squeue','-h','-u'", argv[2])
            self.assertNotIn('sbatch', argv[2])
            return SimpleNamespace(returncode=0, stdout=json.dumps({'username': 'fixture', 'squeue': ''}))
        with mock.patch.object(m.subprocess, 'run', side_effect=inspected_run):
            self.assertEqual(m._account_guard(plan, sys.executable)['squeue'], '')

    def test_account_guard_inner_remote_code_compiles_without_remote_execution(self):
        plan = self.root / 'fake_guard_plan'; plan.mkdir()
        atomic_json(plan / 'profile.json', {'username': 'fixture'})
        (plan / 'preflight.py').write_text('def verify_keys(p): pass\n')
        (plan / 'transport.py').write_text('import json\n'
            'def read(p): return json.load(open(p))\n'
            'class Remote:\n'
            ' def __init__(self,p): pass\n'
            ' def json(self,code,**kw):\n'
            '  compile(code,"<remote-fixture>","exec")\n'
            '  assert "squeue" in code and "sbatch" not in code\n'
            '  return {"squeue":""}\n')
        self.assertEqual(m._account_guard(plan, sys.executable)['username'], 'fixture')

    def test_account_empty_does_not_allow_second_unresolved_local_launch(self):
        self.prepare(); self.launch_fixture()
        second = self.session.reserve_call('forward', 'base', 'diagnosis',
                    {'episode_keys': [['diagnosis_scene', '0']], 'transition_steps': 500})
        spec = {**self.spec, 'run_id': self.spec['run_id'] + '_b'}; atomic_json(self.spec_path, spec)
        m.prepare_managed_call(self.session_root, second['call_id'], self.spec_path,
                              self.directory.parent / 'second', profile=self.profile)
        with mock.patch.object(m, 'PROJECT', self.root / 'fake_project'), \
             mock.patch.object(m, '_account_guard', return_value={'squeue': '', 'username': 'fixture'}) as guard:
            with self.assertRaisesRegex(ContractError, 'PRIOR_LAUNCH_UNKNOWN'):
                m.launch_managed_call(self.session_root, second['call_id'])
            guard.assert_not_called()

    def test_baseline_guard_rejects_unfinished_queue_even_without_live_owner(self):
        fixture = self.root / 'fake_project'; queue = fixture / 'baselines/runs/q'
        queue.mkdir(parents=True)
        atomic_json(queue / 'CHECKPOINT.json', {'status': 'READY', 'units': {'x': {'status': 'ACTIVE'}}})
        with mock.patch.object(m, 'PROJECT', fixture):
            with self.assertRaisesRegex(ContractError, 'BASELINE_QUEUE_UNFINISHED'):
                with m._baseline_guard(): pass

    def test_collected_fixture_completes_exactly_one_reserved_call(self):
        self.artifacts()
        result = m.collect_managed_call(self.session_root, self.call_id)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['feedback']['synthetic_fixture'])
        self.assertEqual(self.session.status()['forward_calls_used'], 1)
        with self.assertRaisesRegex(ContractError, 'RESERVED_CALL_REQUIRED'):
            m.collect_managed_call(self.session_root, self.call_id)

    def test_registry_allows_new_call_only_after_exact_terminal_release(self):
        self.artifacts()
        second = self.session.reserve_call('forward', 'base', 'diagnosis',
                    {'episode_keys': [['diagnosis_scene', '0']], 'transition_steps': 500})
        spec = {**self.spec, 'run_id': self.spec['run_id'] + '_b'}; atomic_json(self.spec_path, spec)
        m.prepare_managed_call(self.session_root, second['call_id'], self.spec_path,
                              self.directory.parent / 'second', profile=self.profile)
        with mock.patch.object(m, 'PROJECT', self.root / 'fake_project'), \
             mock.patch.object(m, '_baseline_guard', return_value=nullcontext()), \
             mock.patch.object(m, '_account_guard', return_value={'squeue': '', 'username': 'fixture'}), \
             mock.patch.object(m.subprocess, 'Popen', return_value=SimpleNamespace(pid=999998)):
            result = m.launch_managed_call(self.session_root, second['call_id'])
        self.assertEqual(result['pid'], 999998)

    def test_completed_call_rechecks_frozen_execution_inputs(self):
        plan = self.artifacts()
        m.collect_managed_call(self.session_root, self.call_id)
        with (plan / 'worker.py').open('a') as stream: stream.write('\n# later tamper\n')
        with self.assertRaisesRegex(ContractError, 'BOUND_FILE_CHANGED'):
            self.session.status()

    def test_unreleased_or_foreign_terminal_receipt_never_completes(self):
        plan = self.artifacts()
        path = plan / 'run/SCHEDULER_LATEST.json'; original = read_json(path)
        for changed in [{**original, 'squeue': '12345|fixture|run|RUNNING'},
                        {**original, 'sacct': original['sacct'].replace('|fixture|', '|foreign|')},
                        {**original, 'sacct': '12345.batch|batch||COMPLETED|0:0'}]:
            atomic_json(path, changed)
            with self.assertRaises(ContractError): m.collect_managed_call(self.session_root, self.call_id)
        self.assertEqual(self.session.status()['pending_calls'], [self.call_id])

    def test_feedback_identity_coverage_budget_and_execution_rejections(self):
        plan = self.artifacts(); path = plan / 'run/vm' / m.FEEDBACK; original = read_json(path)
        changes = [{'call_id': 'call_' + '0' * 32}, {'candidate_tensor_sha256': 'b' * 64},
                   {'checkpoint_sha256': 'b' * 64}, {'split': 'selection'},
                   {'episode_keys': [['other_scene', '0']]},
                   {'episode_keys': [['diagnosis_scene', '0']] * 2},
                   {'transition_steps': 501}, {'transition_steps': True},
                   {'no_model_navigation_executed': True}]
        for change in changes:
            atomic_json(path, {**original, **change}); self.remanifest(plan)
            with self.subTest(change=change), self.assertRaises(ContractError):
                m.collect_managed_call(self.session_root, self.call_id)
        self.assertEqual(self.session.status()['pending_calls'], [self.call_id])

    def test_artifact_tamper_and_frozen_worker_tamper_refused(self):
        plan = self.artifacts()
        feedback = plan / 'run/vm' / m.FEEDBACK
        original = feedback.read_bytes(); feedback.write_bytes(original + b' ')
        with self.assertRaisesRegex(ContractError, 'HASH_MISMATCH'):
            m.collect_managed_call(self.session_root, self.call_id)
        feedback.write_bytes(original)
        with (plan / 'worker.py').open('a') as stream: stream.write('\n# tampered\n')
        with self.assertRaisesRegex(ContractError, 'FROZEN_PLAN_CHANGED'):
            m.collect_managed_call(self.session_root, self.call_id)

    def test_failure_or_bool_exit_status_is_not_success(self):
        plan = self.artifacts(); path = plan / 'run/vm/WORKER_RECEIPT.json'; original = read_json(path)
        for changed in [{**original, 'scientific_success': False}, {**original, 'validation_exit_code': False}]:
            atomic_json(path, changed); self.remanifest(plan)
            with self.assertRaises(ContractError): m.collect_managed_call(self.session_root, self.call_id)

    def test_independent_metric_recomputation_rejects_hash_consistent_wrong_mean(self):
        plan = self.artifacts(); path = plan / 'run/vm' / m.FEEDBACK
        feedback = read_json(path); feedback['metrics']['spl'] = 0.75
        atomic_json(path, feedback); self.remanifest(plan)
        with self.assertRaisesRegex(ContractError, 'RESULT_METRIC_MEAN_MISMATCH'):
            m.collect_managed_call(self.session_root, self.call_id)


if __name__ == '__main__':
    unittest.main()
