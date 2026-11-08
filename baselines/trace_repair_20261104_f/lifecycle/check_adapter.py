#!/usr/bin/env python3
"""Offline identity/path/freeze/worker tests. No SSH, Slurm or real CUDA."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from owner import parse_scheduler, relative
from transport import read, save, sha
from worker import output_path

HERE = Path(__file__).resolve().parent


def denied(fn):
    try:
        fn()
    except (RuntimeError, ValueError):
        return
    raise AssertionError('Expected refusal')


def main():
    checks = []
    q = {'squeue': '', 'sacct': '123|unit-test|alice|COMPLETED|0:0|x|x|1|node'}
    assert parse_scheduler(q, '123', 'unit-test', 'alice')['released'] is True
    q['squeue'] = '123|alice|unit-test|COMPLETING|node'
    assert parse_scheduler(q, '123', 'unit-test', 'alice')['released'] is False
    checks.append('terminal_accounting_and_queue_exit_both_required')
    denied(lambda: parse_scheduler({'squeue': '123|bob|unit-test|RUNNING|node', 'sacct': ''}, '123', 'unit-test', 'alice'))
    denied(lambda: parse_scheduler({'squeue': '', 'sacct': '123|other|alice|COMPLETED|0:0'}, '123', 'unit-test', 'alice'))
    assert parse_scheduler({'squeue': '', 'sacct': '123.batch|unit-test|alice|COMPLETED|0:0'}, '123', 'unit-test', 'alice')['released'] is False
    checks.append('foreign_identity_and_batch_only_rejected')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        denied(lambda: relative(root, '../escape'))
        denied(lambda: output_path(root, '/absolute'))
        (root / 'escape').symlink_to('/tmp')
        denied(lambda: output_path(root, 'escape/file'))
        checks.append('output_and_upload_paths_bounded')
        spec = {'run_id': 'offline-fake-only', 'remote_root': '/mnt/home/liyuqin/USERDIR/vln_checks',
                'remote_python': '/mnt/home/liyuqin/.Miniconda3/bin/python',
                'slurm': {'partition': 'tacc', 'gpus': 1, 'cpus': 2, 'wall_time': '00:05:00', 'deadline': '2099-01-01T00:00:00'},
                'science': {'argv': ['python', '-c', 'pass'], 'cwd': '{source}', 'timeout_seconds': 120,
                            'validator_argv': ['python', '-c', 'pass'], 'required_outputs': ['result.json']},
                'output_budget_bytes': 1024*1024, 'remote_peak_additional_bytes': 2*1024*1024, 'monitor_timeout_seconds': 600}
        save(root / 'spec.json', spec)
        subprocess.run([sys.executable, str(HERE / 'prepare_run.py'), '--spec', str(root / 'spec.json'), '--directory', str(root / 'plan')], check=True, capture_output=True)
        assert not (root / 'plan/run').exists() and not (root / 'plan/managed_records').exists()
        cfg = read(root / 'plan/CONTROL.json')
        assert cfg['slurm']['gpus'] == 1 and cfg['run']['local_run_directory'] == 'run'
        m = importlib.util.spec_from_file_location('frozen_test_manager', root / 'plan/managed_run.py')
        module = importlib.util.module_from_spec(m);m.loader.exec_module(module)
        module.load_plan(root / 'plan/MANAGED.json', root / 'plan/MANAGED_FREEZE.json')
        (root / 'plan/worker.py').write_text('changed')
        denied(lambda: module.load_plan(root / 'plan/MANAGED.json', root / 'plan/MANAGED_FREEZE.json'))
        checks.append('freeze_generation_is_offline_and_tamper_rejected')
        # Fake CUDA module; does not import PyTorch or invoke any GPU API.
        (root / 'torch.py').write_text("class C:\n def device_count(self): return 1\n def get_device_name(self, i): return 'FAKE RTX 3090'\ncuda=C()\n")
        shutil.copyfile(HERE / 'worker.py', root / 'worker.py')
        for should_pass in (True, False):
            out = root / ('good' if should_pass else 'failed')
            command = [sys.executable, '-c', "import os,pathlib;pathlib.Path(os.environ['VLN_RUN_OUTPUT'],'metric.json').write_text('{\"n\": 2}')"]
            validate = [sys.executable, '-c', 'raise SystemExit(' + ('0' if should_pass else '1') + ')']
            worker = {'run_id': 'fake-worker', 'output_directory': str(out), 'gpu_count': 1,
                      'reserve_bytes': 0, 'output_budget_bytes': 1024*1024,
                      'argv': command, 'cwd': str(root), 'timeout_seconds': 10,
                      'validator_argv': validate, 'required_outputs': ['metric.json']}
            save(root / 'worker.json', worker)
            env = dict(os.environ, SLURM_JOB_ID='123', SLURM_JOB_NAME='fake-worker', CUDA_VISIBLE_DEVICES='0')
            result = subprocess.run([sys.executable, str(root / 'worker.py'), '--config', str(root / 'worker.json')], env=env, capture_output=True)
            assert result.returncode == (0 if should_pass else 5), result.stderr
            receipt = read(out / 'WORKER_RECEIPT.json')
            assert receipt['scientific_success'] is should_pass
            for row in read(out / 'OUTPUT_MANIFEST.json')['files']:
                f = out / row['path']
                assert f.stat().st_size == row['bytes'] and sha(f) == row['sha256']
        checks.append('fake_worker_requires_scientific_validator_and_retains_failed_outputs')
    save(HERE / 'ADAPTER_CHECK.json', {'status': 'PASS', 'checks': checks, 'network_calls': 0, 'slurm_submissions': 0, 'real_cuda_calls': 0,
                                    'evidence_scope': 'OFFLINE_SYNTHETIC_IDENTITY_FREEZE_AND_WORKER_ONLY'})
    print(json.dumps({'status': 'PASS', 'checks': len(checks), 'resource_calls': 0}))


if __name__ == '__main__':
    main()
