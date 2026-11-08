#!/usr/bin/env python3
"""One bounded scientific unit inside Slurm. No installing or downloading."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import shutil
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def save(path, value):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def output_path(root, relative):
    p = Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:
        raise ValueError('UNSAFE_OUTPUT_PATH')
    target = root / p
    if target.is_symlink() or root.resolve() not in target.resolve().parents:
        raise ValueError('OUTPUT_OUTSIDE_ROOT')
    return target


def stop(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    root = Path(cfg['output_directory'])
    root.mkdir(parents=True, exist_ok=False)
    report = {'status': 'STARTING', 'run_id': cfg['run_id'], 'started_unix': time.time(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'worker_sha256': sha(__file__),
              'config_sha256': sha(a.config), 'scientific_success': False}
    child = None
    exit_code = 5
    try:
        if not str(report['slurm_job_id'] or '').isdigit():
            raise RuntimeError('NOT_INSIDE_SLURM')
        if os.environ.get('SLURM_JOB_NAME') != cfg['run_id']:
            raise RuntimeError('JOB_NAME_MISMATCH')
        if cfg.get('expected_nodes') and os.environ.get('SLURMD_NODENAME') not in cfg['expected_nodes']:
            raise RuntimeError('ALLOCATED_NODE_DIFFERS')
        memory = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            parts = line.split()
            if parts[0] in ['MemTotal:', 'MemAvailable:']:
                memory[parts[0][:-1] + '_bytes'] = int(parts[1]) * 1024
        for limit_name, usage_name in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                                       ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
            limit_file, usage_file = Path(limit_name), Path(usage_name)
            if limit_file.is_file() and limit_file.read_text().strip().isdigit():
                limit = int(limit_file.read_text())
                memory['cgroup_limit_bytes'] = limit
                if usage_file.is_file():
                    memory['cgroup_usage_bytes'] = int(usage_file.read_text())
                    memory['cgroup_available_bytes'] = max(0, limit - memory['cgroup_usage_bytes'])
                break
        report['host_memory_before_science'] = memory
        required_memory = cfg.get('min_available_memory_bytes', 0)
        effective_available = min(memory.get('MemAvailable_bytes', 0), memory.get('cgroup_available_bytes', memory.get('MemTotal_bytes', 0)))
        if required_memory and effective_available < required_memory:
            raise RuntimeError('INSUFFICIENT_MEMORY_FOR_FROZEN_SCIENCE_PLAN')
        report['runtime_environment'] = {k: os.environ.get(k) for k in [
            'CUDA_VISIBLE_DEVICES', 'NVIDIA_VISIBLE_DEVICES', 'LD_LIBRARY_PATH',
            'SLURM_JOB_GPUS', 'SLURM_STEP_GPUS', 'SLURMD_NODENAME']}
        report['nvidia_device_files'] = [str(p) for p in Path('/dev').glob('nvidia*')]
        try:
            probe = subprocess.run(['nvidia-smi', '-q', '-x'],
                                   capture_output=True, text=True, timeout=15)
            report['nvidia_smi'] = {'exit_code': probe.returncode, 'stderr': probe.stderr,
                                    'xml_sha256': hashlib.sha256(probe.stdout.encode()).hexdigest()}
            if probe.returncode == 0:
                tree = ET.fromstring(probe.stdout)
                report['nvidia_smi']['driver_version'] = tree.findtext('driver_version')
                report['nvidia_smi']['devices'] = [
                    {'uuid': gpu.findtext('uuid'), 'name': gpu.findtext('product_name'),
                     'minor_number': gpu.findtext('minor_number'), 'pci_bus_id': gpu.findtext('pci/pci_bus_id')}
                    for gpu in tree.findall('gpu')]
        except (OSError, subprocess.TimeoutExpired, ET.ParseError) as error:
            report['nvidia_smi'] = {'error': str(error)}
        if cfg.get('gpu_binding') == 'slurm_linux_minor_uuid':
            allocated = os.environ.get('SLURM_JOB_GPUS', '').split(',')
            original = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
            if allocated != original or len(allocated) != cfg['gpu_count'] or not all(x.isdigit() for x in allocated):
                raise RuntimeError('GPU_MINOR_BINDING_ALLOCATION_AMBIGUOUS')
            query = report['nvidia_smi']
            if query.get('exit_code') != 0:
                raise RuntimeError('GPU_MINOR_BINDING_QUERY_FAILED')
            rows = query.get('devices', [])
            if not rows or any(not str(row.get('minor_number')).isdigit() or not str(row.get('uuid')).startswith('GPU-') for row in rows):
                raise RuntimeError('GPU_MINOR_BINDING_INVALID_QUERY')
            selected = []
            for minor in allocated:
                matches = [row for row in rows if row['minor_number'] == minor]
                device = Path('/dev/nvidia' + minor)
                if len(matches) != 1 or not device.exists() or not stat.S_ISCHR(device.stat().st_mode) or os.minor(device.stat().st_rdev) != int(minor):
                    raise RuntimeError('GPU_MINOR_BINDING_DEVICE_NOT_VERIFIED')
                selected.append(matches[0]['uuid'])
            if len(set(selected)) != cfg['gpu_count']:
                raise RuntimeError('GPU_MINOR_BINDING_NOT_UNIQUE')
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected)
            report['gpu_binding'] = {'mode': 'slurm_linux_minor_uuid', 'slurm_gpu_ids': allocated,
                                     'original_cuda_visible_devices': original, 'selected_uuids': selected}
        import torch
        count = torch.cuda.device_count()
        report['torch_version'] = torch.__version__
        report['torch_cuda_build'] = torch.version.cuda
        report['torch_file'] = torch.__file__
        if count == 0:
            try:
                torch.cuda.init()
            except Exception as error:
                report['cuda_initialization_error'] = str(error)
        visible = [x for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x]
        names = [torch.cuda.get_device_name(i) for i in range(count)]
        report.update(gpu_names=names, cuda_visible_devices=visible, gpu_count=count)
        if count != cfg['gpu_count'] or len(visible) != cfg['gpu_count'] or not all('3090' in n for n in names):
            raise RuntimeError('ALLOCATED_GPU_DIFFERS')
        if cfg.get('gpu_binding') == 'slurm_linux_minor_uuid':
            driver = ctypes.CDLL('libcuda.so.1')
            class CUuuid(ctypes.Structure):
                _fields_ = [('bytes', ctypes.c_ubyte * 16)]
            driver.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
            driver.cuDeviceGetUuid.argtypes = [ctypes.POINTER(CUuuid), ctypes.c_int]
            actual_uuids = []
            for index in range(count):
                device, ident = ctypes.c_int(), CUuuid()
                if driver.cuDeviceGet(ctypes.byref(device), index) != 0 or driver.cuDeviceGetUuid(ctypes.byref(ident), device) != 0:
                    raise RuntimeError('CUDA_DEVICE_UUID_QUERY_FAILED')
                actual_uuids.append('GPU-' + str(uuid.UUID(bytes=bytes(ident.bytes))))
            report['gpu_binding']['cuda_actual_uuids'] = actual_uuids
            if actual_uuids != selected:
                raise RuntimeError('CUDA_DEVICE_UUID_DIFFERS_FROM_ALLOCATION')
        if shutil.disk_usage(root).free < cfg['reserve_bytes'] + cfg['output_budget_bytes']:
            raise RuntimeError('REMOTE_STORAGE_ADMISSION_FAILED')
        env = dict(os.environ)
        env.update(cfg.get('env', {}))
        env['VLN_RUN_OUTPUT'] = str(root)
        with (root / 'scientific.log').open('wb') as log:
            child = subprocess.Popen(cfg['argv'], cwd=cfg['cwd'], env=env, stdout=log,
                                     stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + cfg['timeout_seconds']
            while child.poll() is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError('SCIENTIFIC_UNIT_TIMEOUT')
                if shutil.disk_usage(root).free < cfg['reserve_bytes']:
                    raise RuntimeError('REMOTE_STORAGE_RESERVE_CROSSED')
                size = sum(f.stat().st_size for f in root.rglob('*') if f.is_file() and not f.is_symlink())
                if size > cfg['output_budget_bytes']:
                    raise RuntimeError('REMOTE_OUTPUT_BUDGET_EXCEEDED')
                time.sleep(2)
            report['scientific_exit_code'] = child.returncode
            if child.returncode:
                raise RuntimeError('SCIENTIFIC_COMMAND_FAILED')
        for relative in cfg['required_outputs']:
            target = output_path(root, relative)
            if not target.is_file() or target.stat().st_size == 0:
                raise RuntimeError('MISSING_REQUIRED_OUTPUT: ' + relative)
        if sum(f.stat().st_size for f in root.rglob('*') if f.is_file() and not f.is_symlink()) > cfg['output_budget_bytes']:
            raise RuntimeError('REMOTE_OUTPUT_BUDGET_EXCEEDED')
        # Caller supplies a real scientific validator, not just process success.
        with (root / 'validation.log').open('wb') as log:
            valid = subprocess.run(cfg['validator_argv'], cwd=cfg['cwd'], env=env,
                                   stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=cfg.get('validation_timeout_seconds', 90))
        report['validation_exit_code'] = valid.returncode
        if valid.returncode:
            raise RuntimeError('SCIENTIFIC_VALIDATION_FAILED')
        report.update(status='SCIENTIFIC_OUTPUTS_VALIDATED', scientific_success=True)
        exit_code = 0
    except BaseException as error:
        if child is not None:
            stop(child)
        report.update(status='FAILED_RETAINED', error_type=type(error).__name__, error=str(error))
    finally:
        report.update(finished_unix=time.time(), free_bytes=shutil.disk_usage(root).free)
        save(root / 'WORKER_RECEIPT.json', report)
        rows = []
        # Manifest only explicit required outputs and our small audit logs.
        for name in dict.fromkeys(cfg['required_outputs'] + ['scientific.log', 'validation.log', 'WORKER_RECEIPT.json']):
            target = output_path(root, name)
            if target.is_file():
                rows.append({'path': name, 'bytes': target.stat().st_size, 'sha256': sha(target)})
        save(root / 'OUTPUT_MANIFEST.json', {'run_id': cfg['run_id'], 'slurm_job_id': report['slurm_job_id'], 'files': rows})
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
