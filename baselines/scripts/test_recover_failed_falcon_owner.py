#!/usr/bin/env python3
"""Offline recovery tests: synthetic files, no SSH, Slurm, GPU, or submission."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import recover_failed_falcon_owner as r


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


class LocalFakeRemote:
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.calls = []

    def json(self, code, **kwargs):
        self.calls.append(code)
        if 'import json,subprocess' in code:
            return copy.deepcopy(self.scheduler)
        # Execute only the helper's read-only metadata code on temporary files.
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            exec(compile(code, '<offline-metadata>', 'exec'), {})
        return json.loads(capture.getvalue())

    def download(self, source, target, expected):
        if target.exists():
            if target.stat().st_size == expected['bytes'] and r.sha(target) == expected['sha256']:
                return
            raise RuntimeError('EXISTING_RESULT_DIFFERS')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.qdir = self.root / 'queue'
        self.plan = self.qdir / 'attempts/pair_0/plan'
        self.plan.mkdir(parents=True)
        self.run = self.plan / 'run'
        (self.run / 'vm').mkdir(parents=True)
        (self.run / 'incremental_cache').mkdir()
        self.output, self.source = self.root / 'remote/output', self.root / 'remote/source'
        self.output.mkdir(parents=True)
        self.source.mkdir()
        for name in ('owner.py', 'managed_run.py', 'worker.py'):
            (self.plan / name).write_text('# Synthetic no-execution fixture\n')
        put(self.plan / 'MANAGED.json', {})
        self.cfg = {'run': {'session_name': 'synthetic_unique', 'attempt_id': 'attempt01'},
                    'storage_budget': {'reserve_bytes': 4 << 30, 'output_bytes': 1 << 20},
                    'remote': {'output_directory': str(self.output), 'source_directory': str(self.source)}}
        put(self.plan / 'CONTROL.json', self.cfg)
        put(self.plan / 'profile.json', {'username': 'offline'})
        put(self.plan / 'worker.json', {'required_outputs': ['partial.json']})
        frozen = {str(p): r.sha(p) for p in self.plan.iterdir() if p.is_file()}
        put(self.plan / 'CONTROL_FREEZE.json', {'status': 'FROZEN_BEFORE_ALLOCATION', 'files': frozen})
        put(self.plan / 'MANAGED_FREEZE.json', {'status': 'FROZEN_BEFORE_MANAGED_RUN', 'files': frozen})
        self.driver = {'session': 'synthetic_unique', 'attempt_id': 'attempt01', 'job_id': '123',
                       'status': 'FAILED_RETAINED', 'released': False, 'local_outputs_verified': False,
                       'started_unix': 1, 'finished_unix': 2, 'error_type': 'RuntimeError',
                       'error': 'SSH_DNS_FAILURE', 'release_error': 'SSH_DNS_FAILURE',
                       'control_freeze_sha256': r.sha(self.plan / 'CONTROL_FREEZE.json'),
                       'entry_driver_sha256': r.sha(self.plan / 'owner.py')}
        self.managed = {'status': 'OWNER_FAILED_RETAINED', 'owner_returncode': 5, 'children': {},
                        'source_sha256': r.sha(self.plan / 'managed_run.py'),
                        'managed_config_sha256': r.sha(self.plan / 'MANAGED.json'),
                        'control_sha256': r.sha(self.plan / 'CONTROL.json'),
                        'control_freeze_sha256': r.sha(self.plan / 'CONTROL_FREEZE.json'),
                        'run': str(self.run), 'final_owner_flags': {'released': False}}
        put(self.run / 'DRIVER_RECEIPT.json', self.driver)
        put(self.plan / 'managed_records/MANAGED_RECEIPT.json', self.managed)
        put(self.run / 'SCHEDULER_LATEST.json', {'state': 'RUNNING', 'released': False})
        self.worker = {'run_id': 'synthetic_unique', 'slurm_job_id': '123',
                       'worker_sha256': r.sha(self.plan / 'worker.py'),
                       'config_sha256': r.sha(self.plan / 'worker.json'), 'scientific_success': False,
                       'error_type': 'TimeoutError', 'error': 'SCIENTIFIC_UNIT_TIMEOUT'}
        put(self.output / 'WORKER_RECEIPT.json', self.worker)
        put(self.output / 'partial.json', {'episodes': 1, 'synthetic': True})
        self.write_manifest()
        (self.source / 'slurm-123.out').write_text('offline fixture controlled timeout\n')
        self.scheduler = {'squeue': '', 'sacct': '123|synthetic_unique|offline|FAILED|5:0|start|end|55:11|node'}
        self.remote = LocalFakeRemote(self.scheduler)
        self.attempt = {'plan': str(self.plan), 'run_id': self.driver['session'], 'pid': None}
        self.contract = self.qdir / 'contract.json'
        put(self.contract, {'synthetic': True})
        self.checkpoint = {'schema': 'continuous_reproduction_checkpoint_v1', 'contract': str(self.contract),
                           'input_files': {str(self.contract): r.sha(self.contract)},
                           'units': {'pair': {'status': 'ACTIVE', 'attempts': [self.attempt]}}}
        put(self.qdir / 'CHECKPOINT.json', self.checkpoint)
        self.ctx = {'queue_directory': self.qdir, 'plan': self.plan, 'checkpoint': self.checkpoint,
                    'unit_id': 'pair', 'attempt': self.attempt, 'driver': self.driver,
                    'managed': self.managed, 'control': self.cfg, 'profile': {'username': 'offline'}}

    def tearDown(self):
        self.tmp.cleanup()

    def write_manifest(self):
        self.manifest = {'run_id': 'synthetic_unique', 'slurm_job_id': '123', 'files': [
            {'path': p.name, 'bytes': p.stat().st_size, 'sha256': r.sha(p)} for p in
            [self.output / 'WORKER_RECEIPT.json', self.output / 'partial.json']]}
        put(self.output / 'OUTPUT_MANIFEST.json', self.manifest)

    def recover(self):
        with r.queue.ControllerLock(self.qdir):
            return r.recover(self.ctx, self.remote, self.qdir / 'recoveries/one')

    def test_success_preserves_failure_and_independent_retry_evidence(self):
        original = {str(p): p.read_bytes() for p in [self.run / 'DRIVER_RECEIPT.json',
                    self.run / 'SCHEDULER_LATEST.json', self.plan / 'managed_records/MANAGED_RECEIPT.json',
                    self.qdir / 'CHECKPOINT.json']}
        result = self.recover()
        self.assertEqual(result['queue_observation']['kind'], 'FAILED')
        self.assertTrue(result['queue_observation']['timeout'])
        self.assertTrue(result['managed_receipt_unchanged'])
        updated = r.read(self.run / 'DRIVER_RECEIPT.json')
        self.assertTrue(updated['released'])
        for key, val in self.driver.items():
            if key != 'released':
                self.assertEqual(updated[key], val)
        ref = updated['recovery_evidence']
        self.assertEqual(ref['sha256'], r.sha(ref['file']))
        receipt = r.read(ref['file'])
        for name, row in receipt['originals'].items():
            self.assertEqual(Path(row['archive']).read_bytes(), original[name])
        self.assertEqual((self.plan / 'managed_records/MANAGED_RECEIPT.json').read_bytes(),
                         original[str(self.plan / 'managed_records/MANAGED_RECEIPT.json')])
        self.assertEqual((self.qdir / 'CHECKPOINT.json').read_bytes(), original[str(self.qdir / 'CHECKPOINT.json')])
        self.assertFalse(receipt['scientific_success_claimed'])
        self.assertEqual(receipt['submissions'], 0)
        self.assertTrue(all('sbatch' not in call and 'scancel' not in call for call in self.remote.calls))
        self.assertEqual(sum('import json,subprocess' in call for call in self.remote.calls), 2)

    def test_environment_failure_remains_nonretryable(self):
        self.worker.update(error_type='RuntimeError', error='EGL_BAD_PARAMETER')
        put(self.output / 'WORKER_RECEIPT.json', self.worker)
        self.write_manifest()
        result = self.recover()
        self.assertFalse(result['queue_observation']['timeout'])

    def test_wrong_hash_never_changes_owner_release(self):
        self.manifest['files'][1]['sha256'] = 'a' * 64
        put(self.output / 'OUTPUT_MANIFEST.json', self.manifest)
        with self.assertRaisesRegex(RuntimeError, 'TRANSFER_HASH_MISMATCH'):
            self.recover()
        self.assertEqual(r.read(self.run / 'DRIVER_RECEIPT.json'), self.driver)
        self.assertTrue((self.qdir / 'recoveries/one/RECOVERY_ERROR.json').is_file())

    def test_already_downloaded_valid_outputs_are_reused(self):
        shutil.copyfile(self.output / 'partial.json', self.run / 'vm/partial.json')
        before = (self.run / 'vm/partial.json').stat().st_ino
        self.recover()
        self.assertEqual((self.run / 'vm/partial.json').stat().st_ino, before)

    def test_conflicting_existing_output_retained_and_refused(self):
        (self.run / 'vm/partial.json').write_text('conflict')
        with self.assertRaisesRegex(RuntimeError, 'EXISTING_RESULT_DIFFERS'):
            self.recover()
        self.assertEqual((self.run / 'vm/partial.json').read_text(), 'conflict')
        self.assertEqual(r.read(self.run / 'DRIVER_RECEIPT.json'), self.driver)

    def test_empty_queue_and_exact_main_row_required(self):
        cases = [
            {'squeue': '999|offline|synthetic_unique|RUNNING|node', 'sacct': self.scheduler['sacct']},
            {'squeue': '888|offline|other|PENDING|node', 'sacct': self.scheduler['sacct']},
            {'squeue': '', 'sacct': '123.batch|synthetic_unique|offline|COMPLETED|0:0'},
            {'squeue': '', 'sacct': '123|foreign_name|offline|FAILED|5:0'},
            {'squeue': '', 'sacct': '123|synthetic_unique|foreign_user|FAILED|5:0'},
            {'squeue': '', 'sacct': '123|synthetic_unique|offline|RUNNING|0:0'},
            {'squeue': '', 'sacct': self.scheduler['sacct'] + '\n' + self.scheduler['sacct']},
        ]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                r.parse_scheduler(data, '123', 'synthetic_unique', 'offline')

    def test_manifest_budget_duplicates_and_foreign_paths_rejected(self):
        cases = []
        d = copy.deepcopy(self.manifest); d['files'].append(d['files'][0]); cases.append(d)
        d = copy.deepcopy(self.manifest); d['files'][0]['path'] = '../escape'; cases.append(d)
        d = copy.deepcopy(self.manifest); d['files'][0]['bytes'] = 2 << 20; cases.append(d)
        d = copy.deepcopy(self.manifest); d['files'][0]['bytes'] = -1; cases.append(d)
        d = copy.deepcopy(self.manifest); d['slurm_job_id'] = '456'; cases.append(d)
        for data in cases:
            with self.subTest(data=data), self.assertRaises(RuntimeError):
                r.validate_manifest(data, self.ctx)

    def test_symlink_cannot_replace_result(self):
        (self.run / 'vm/partial.json').symlink_to(self.root / 'outside')
        with self.assertRaisesRegex(RuntimeError, 'SYMLINK'):
            self.recover()
        self.assertFalse((self.root / 'outside').exists())

    def test_disk_reserve_refuses_before_archive_or_download(self):
        with patch.object(r.shutil, 'disk_usage', return_value=type('Disk', (), {'free': (4 << 30)-1})()):
            with self.assertRaisesRegex(RuntimeError, 'LOCAL_RESERVE_CROSSED'):
                self.recover()
        self.assertFalse((self.qdir / 'recoveries').exists())
        self.assertEqual(self.remote.calls, [])

    def test_admission_checks_bound_inputs_and_failed_owner_identity(self):
        inputs = self.checkpoint['input_files']
        with patch.object(r.queue, 'load_contract', return_value=({}, inputs)):
            ctx = r.admit(self.qdir, self.plan)
            self.assertEqual(ctx['driver']['job_id'], '123')
            self.contract.write_text('changed')
            with self.assertRaisesRegex(RuntimeError, 'QUEUE_INPUT_CHANGED'):
                r.admit(self.qdir, self.plan)

    def test_frozen_plan_tamper_refused(self):
        with patch.object(r.queue, 'load_contract', return_value=({}, self.checkpoint['input_files'])):
            (self.plan / 'worker.py').write_text('# tamper')
            with self.assertRaisesRegex(RuntimeError, 'FROZEN_PLAN_CHANGED'):
                r.admit(self.qdir, self.plan)

    def test_live_owner_refused(self):
        with patch.object(r.queue, 'process_matches', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'MANAGER_STILL_ACTIVE'):
                r.no_live_plan_processes(self.plan, self.attempt, self.managed)

    def test_controller_lock_cannot_be_shared(self):
        with r.queue.ControllerLock(self.qdir):
            with self.assertRaisesRegex(RuntimeError, 'CONTROLLER_ALREADY_RUNNING'):
                with r.queue.ControllerLock(self.qdir):
                    self.fail('Second controller lock accepted')


if __name__ == '__main__':
    unittest.main()
