#!/usr/bin/env python3
"""Offline controller state tests. Never contacts TACC or starts managed_run."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).with_name('continuous_reproduction_queue.py').absolute()
spec = importlib.util.spec_from_file_location('queue_under_test', MODULE)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def controller(self, units):
        obj = object.__new__(q.QueueController)
        obj.directory = self.root / 'controller'
        obj.directory.mkdir(exist_ok=True)
        obj.contract_path = self.root / 'queue.json'
        obj.state_path = obj.directory / 'CHECKPOINT.json'
        obj.children = {}
        obj.cfg = {'units': units, 'poll_seconds': 1, 'local_python': sys.executable, 'profile': str(self.root / 'profile.json')}
        obj.inputs = {}
        obj.state = {'schema': 'continuous_reproduction_checkpoint_v1', 'status': 'READY',
                     'input_files': {}, 'events': [], 'completed': {},
                     'units': {u['unit_id']: {'status': 'NEW', 'attempts': []} for u in units}}
        obj.save()
        return obj

    def unit(self, ids=('A', 'B')):
        return {'unit_id': 'batch01', 'completion_ids': list(ids), 'validator_timeout_seconds': 10,
                'variants': [{'completion_ids': list(ids), 'local_validator_argv': ['unused'], 'completion_checks': []}]}

    def complete(self, obj, cid):
        proof = self.root / (cid + '.proof')
        proof.write_text('synthetic offline completion proof')
        obj.state['completed'][cid] = {'independently_validated': True, 'proof_files': {str(proof): q.sha(proof)}}

    def frozen_plan(self, state='TIMEOUT', worker_error='SCIENTIFIC_UNIT_TIMEOUT', final=True):
        plan = self.root / 'plan'
        plan.mkdir(exist_ok=True)
        for name in ('owner.py', 'worker.py', 'managed_run.py'):
            (plan / name).write_text('# synthetic offline fixture\n')
        put(plan / 'CONTROL.json', {'run': {'session_name': 'offline_unique', 'attempt_id': 'attempt01'}})
        put(plan / 'profile.json', {'username': 'offline_user'})
        put(plan / 'worker.json', {'required_outputs': ['result.json']})
        put(plan / 'MANAGED.json', {})
        put(plan / 'SPEC.json', {'reserve_bytes': 0})
        files = {str(p): q.sha(p) for p in plan.iterdir() if p.is_file()}
        put(plan / 'CONTROL_FREEZE.json', {'status': 'FROZEN_BEFORE_ALLOCATION', 'files': files})
        put(plan / 'MANAGED_FREEZE.json', {'status': 'FROZEN_BEFORE_MANAGED_RUN', 'files': files})
        if not final:
            return plan
        worker = {'run_id': 'offline_unique', 'slurm_job_id': '123', 'worker_sha256': q.sha(plan / 'worker.py'),
                  'config_sha256': q.sha(plan / 'worker.json'), 'scientific_success': False,
                  'error_type': 'TimeoutError' if worker_error == 'SCIENTIFIC_UNIT_TIMEOUT' else 'RuntimeError',
                  'error': worker_error}
        put(plan / 'run/vm/WORKER_RECEIPT.json', worker)
        put(plan / 'run/vm/result.json', {'synthetic': True})
        log = plan / 'run/vm/slurm-123.out'
        log.write_text('synthetic retained failure log\n')
        rows = [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': q.sha(p)} for p in
                (plan / 'run/vm/WORKER_RECEIPT.json', plan / 'run/vm/result.json')]
        put(plan / 'run/REMOTE_OUTPUT_MANIFEST.json', {'run_id': 'offline_unique', 'slurm_job_id': '123', 'files': rows})
        put(plan / 'run/SCHEDULER_LOG_VERIFIED.json', {'file': str(log), 'bytes': log.stat().st_size, 'sha256': q.sha(log)})
        driver = {'session': 'offline_unique', 'attempt_id': 'attempt01', 'job_id': '123',
                  'control_freeze_sha256': q.sha(plan / 'CONTROL_FREEZE.json'), 'entry_driver_sha256': q.sha(plan / 'owner.py'),
                  'released': True, 'finished_unix': 1, 'status': 'FAILED_RETAINED', 'local_outputs_verified': False}
        put(plan / 'run/DRIVER_RECEIPT.json', driver)
        put(plan / 'run/SCHEDULER_LATEST.json', {'squeue': '', 'sacct': '123|offline_unique|offline_user|' + state + '|5:0',
                                               'state': state, 'released': True})
        managed = {'status': 'OWNER_FAILED_RETAINED', 'owner_returncode': 5, 'children': {},
                   'source_sha256': q.sha(plan / 'managed_run.py'), 'managed_config_sha256': q.sha(plan / 'MANAGED.json'),
                   'control_sha256': q.sha(plan / 'CONTROL.json'), 'control_freeze_sha256': q.sha(plan / 'CONTROL_FREEZE.json'),
                   'run': str(plan / 'run')}
        put(plan / 'managed_records/MANAGED_RECEIPT.json', managed)
        return plan

    def test_duplicate_controller_lock_is_rejected_by_separate_process(self):
        lockdir = self.root / 'locks'
        code = ('import importlib.util,sys;from pathlib import Path;'
                's=importlib.util.spec_from_file_location("q",sys.argv[1]);'
                'q=importlib.util.module_from_spec(s);s.loader.exec_module(q);'
                'q.ControllerLock(Path(sys.argv[2])).__enter__()')
        with q.ControllerLock(lockdir):
            result = subprocess.run([sys.executable, '-c', code, str(MODULE), str(lockdir)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('CONTROLLER_ALREADY_RUNNING', result.stderr)
        with q.ControllerLock(lockdir):
            pass  # Lock is released when the original controller exits.

    def test_process_identity_survives_running_sleeping_state_changes(self):
        from types import SimpleNamespace
        stable = 'Tue Nov  3 12:00:00 2026 /absolute/plan/managed_run.py --config exact'
        with patch.object(q.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='S ' + stable)):
            saved = q.process_snapshot(123)
        with patch.object(q.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='R ' + stable)):
            self.assertTrue(q.process_matches(123, '/absolute/plan/managed_run.py', saved))
        with patch.object(q.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='Z ' + stable)):
            self.assertFalse(q.process_matches(123, '/absolute/plan/managed_run.py', saved))

    def test_restart_skips_verified_completed_shards_without_prepare_or_launch(self):
        obj = self.controller([self.unit()])
        self.complete(obj, 'A')
        self.complete(obj, 'B')
        obj.save()
        obj.state = q.read(obj.state_path)  # Actual serialization/restart boundary.
        with patch.object(obj, 'prepare') as prepare, patch.object(obj, 'launch') as launch:
            self.assertEqual(obj.advance(), 'COMPLETE')
            prepare.assert_not_called()
            launch.assert_not_called()

    def test_completed_evidence_tamper_stops_before_any_new_prepare(self):
        obj = self.controller([self.unit()])
        self.complete(obj, 'A')
        (self.root / 'A.proof').write_text('changed after checkpoint')
        with patch.object(obj, 'prepare') as prepare, patch.object(obj, 'launch') as launch:
            self.assertEqual(obj.run(), 2)
            self.assertIn('COMPLETION_EVIDENCE_CHANGED', obj.state['error'])
            prepare.assert_not_called()
            launch.assert_not_called()

    def test_successful_released_and_validated_unit_advances_to_next_unit(self):
        first, second = self.unit(('A',)), self.unit(('B',))
        second['unit_id'] = 'batch02'
        obj = self.controller([first, second])
        plan = self.frozen_plan()
        obj.state['units']['batch01']['attempts'] = [{'phase': 'WAITING', 'plan': str(plan), 'run_id': 'offline_unique', 'variant_index': 0}]
        observed = {'kind': 'COMPLETE', 'proof_files': {}, 'worker': {'scientific_success': True}}
        with patch.object(q, 'inspect_attempt', return_value=observed), patch.object(obj, 'validator', return_value=(True, {})):
            self.assertEqual(obj.advance(), 'PROGRESS')
        self.assertEqual(set(obj.state['completed']), {'A'})
        with patch.object(obj, 'prepare', return_value={}) as prepare, patch.object(obj, 'launch') as launch:
            self.assertEqual(obj.advance(), 'WAIT')
            self.assertEqual(prepare.call_args.args[0]['unit_id'], 'batch02')
            launch.assert_called_once()

    def test_launch_intent_without_receipt_never_resubmits(self):
        obj = self.controller([self.unit()])
        plan = self.frozen_plan(final=False)
        obj.state['units']['batch01']['attempts'] = [{'phase': 'LAUNCH_INTENT', 'plan': str(plan), 'run_id': 'offline_unique', 'variant_index': 0}]
        with patch.object(obj, 'prepare') as prepare, patch.object(obj, 'launch') as launch:
            self.assertEqual(obj.run(), 2)
            self.assertIn('UNKNOWN_NO_RESUBMIT', obj.state['error'])
            prepare.assert_not_called()
            launch.assert_not_called()
        self.assertEqual(q.read(obj.state_path)['status'], 'ATTENTION')

    def test_active_owner_receipt_recovers_wait_without_a_second_launch(self):
        obj = self.controller([self.unit()])
        plan = self.frozen_plan()
        managed_path = plan / 'managed_records/MANAGED_RECEIPT.json'
        managed = q.read(managed_path)
        managed.update(status='OWNER_STARTED', children={'OWNER': {'pid': 444}})
        put(managed_path, managed)
        obj.state['units']['batch01']['attempts'] = [{'phase': 'LAUNCH_INTENT', 'plan': str(plan), 'run_id': 'offline_unique', 'variant_index': 0}]
        with patch.object(q, 'process_matches', side_effect=lambda pid, *a: pid == 444), patch.object(obj, 'launch') as launch:
            self.assertEqual(obj.advance(), 'WAIT')
            launch.assert_not_called()

    def test_timeout_is_distinguished_from_environment_failure_and_logs_verified(self):
        plan = self.frozen_plan()
        got = q.inspect_attempt({'plan': str(plan)})
        self.assertEqual(got['kind'], 'FAILED')
        self.assertTrue(got['timeout'])
        (plan / 'run/vm/slurm-123.out').write_text('tampered')
        with self.assertRaisesRegex(q.Attention, 'LOG_HASH_MISMATCH'):
            q.inspect_attempt({'plan': str(plan)})

    def test_terminal_flag_without_empty_squeue_is_not_release(self):
        plan = self.frozen_plan()
        path = plan / 'run/SCHEDULER_LATEST.json'
        row = q.read(path)
        row['squeue'] = '123|offline_user|offline_unique|COMPLETING|node'
        put(path, row)
        with self.assertRaisesRegex(q.Attention, 'TERMINAL_AND_RELEASE_NOT_PROVEN'):
            q.inspect_attempt({'plan': str(plan)})

    def test_environment_failure_does_not_select_available_retry(self):
        unit = self.unit()
        unit['variants'].append({'completion_ids': ['A'], 'local_validator_argv': ['unused'], 'completion_checks': []})
        obj = self.controller([unit])
        plan = self.frozen_plan(state='FAILED', worker_error='ALLOCATED_GPU_DIFFERS')
        obj.state['units']['batch01']['attempts'] = [{'phase': 'WAITING', 'plan': str(plan), 'run_id': 'offline_unique', 'variant_index': 0}]
        with patch.object(obj, 'prepare') as prepare, patch.object(obj, 'launch') as launch:
            self.assertEqual(obj.run(), 2)
            self.assertIn('ENVIRONMENT_UNIT_ATTENTION', obj.state['error'])
            prepare.assert_not_called()
            launch.assert_not_called()

    def test_retry_candidates_skip_completed_and_never_repeat_a_candidate(self):
        unit = self.unit()
        unit['variants'] += [{'completion_ids': ['A']}, {'completion_ids': ['B']}]
        obj = self.controller([unit])
        self.complete(obj, 'A')
        entry = obj.state['units']['batch01']
        entry.update(timeout_retry_authorized=True, attempts=[{'variant_index': 0}])
        with patch.object(obj, 'prepare', return_value={'phase': 'PREPARED'}) as prepare, patch.object(obj, 'launch') as launch:
            self.assertEqual(obj.next_retry(unit, entry), 'WAIT')
            self.assertEqual(prepare.call_args.args[1], 2)
            launch.assert_called_once()
        self.assertEqual(entry['skipped_retry_candidates']['1'], 'already_complete')
        entry['attempts'].append({'variant_index': 2})
        with patch.object(obj, 'prepare') as prepare:
            with self.assertRaisesRegex(q.Attention, 'CANDIDATES_EXHAUSTED'):
                obj.next_retry(unit, entry)
            prepare.assert_not_called()

    def test_two_unfinished_single_retry_candidates_run_in_order(self):
        unit = self.unit()
        unit['variants'] += [{'completion_ids': ['A']}, {'completion_ids': ['B']}]
        obj = self.controller([unit])
        entry = obj.state['units']['batch01']
        entry.update(timeout_retry_authorized=True, attempts=[{'variant_index': 0}])
        with patch.object(obj, 'prepare', return_value={}) as prepare, patch.object(obj, 'launch'):
            obj.next_retry(unit, entry)
            self.assertEqual(prepare.call_args.args[1], 1)
        entry['attempts'].append({'variant_index': 1})
        self.complete(obj, 'A')
        with patch.object(obj, 'prepare', return_value={}) as prepare, patch.object(obj, 'launch'):
            obj.next_retry(unit, entry)
            self.assertEqual(prepare.call_args.args[1], 2)

    def test_local_validator_writes_only_copy_and_original_proof_stays_intact(self):
        unit = self.unit(('A',))
        obj = self.controller([unit])
        plan = self.frozen_plan()
        evidence = plan / 'run/vm/result.json'
        before = q.sha(evidence)
        program = self.root / 'validator.py'
        program.write_text('import pathlib,sys\np=pathlib.Path(sys.argv[1]);(p/"result.json").write_text("changed copy")\n')
        attempt = {'plan': str(plan), 'directory': str(self.root)}
        passed, proof = obj.validator(unit, attempt, ['{python}', str(program), '{vm}'], 'test')
        self.assertTrue(passed)
        self.assertEqual(q.sha(evidence), before)
        self.assertEqual(len(proof), 2)

    def test_real_prepare_freezes_deadline_only_and_preserves_embedded_run_id(self):
        # Actual prepare_run invocation is CPU-only/no SSH; managed_run is never launched.
        profile = put(self.root / 'profile.json', {'remote_userdir': '/synthetic/USERDIR'})
        original = {'run_id': 'offline_queue_unique_' + os.urandom(4).hex(),
                    'slurm': {'gpus': 1, 'cpus': 1, 'wall_time': '00:05:00', 'deadline': '2000-01-01T00:00:00', 'partition': 'tacc'},
                    'remote_root': '/synthetic/USERDIR/queue', 'remote_python': '/synthetic/python',
                    'remote_peak_additional_bytes': 1024, 'output_budget_bytes': 1024, 'monitor_timeout_seconds': 600,
                    'gpu_binding': 'slurm_linux_minor_uuid',
                    'science': {'argv': ['/synthetic/python', '-c', 'pass'], 'validator_argv': ['/synthetic/python', '-c', 'pass'],
                                'required_outputs': ['synthetic.json'], 'timeout_seconds': 200, 'validation_timeout_seconds': 20,
                                'cwd': '{source}', 'embedded_run_id': 'must_not_be_renamed'}}
        spec_path = put(self.root / 'source_spec.json', original)
        unit = self.unit(('A',))
        unit['deadline_after_seconds'] = 7200
        unit['variants'][0].update(spec=str(spec_path), run_id=original['run_id'])
        obj = self.controller([unit])
        obj.cfg['profile'] = str(profile)
        attempt = obj.prepare(unit, 0)
        actual = q.read(Path(attempt['plan']) / 'SPEC.json')
        self.assertNotEqual(actual['slurm']['deadline'], original['slurm']['deadline'])
        actual['slurm']['deadline'] = original['slurm']['deadline']
        self.assertEqual(actual, original)
        self.assertEqual(q.read(spec_path), original)
        self.assertEqual(attempt['phase'], 'PREPARED')
        self.assertFalse((Path(attempt['plan']) / 'run').exists())
        q.verify_freeze(Path(attempt['plan']))


if __name__ == '__main__':
    unittest.main(verbosity=2)
