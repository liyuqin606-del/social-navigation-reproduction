#!/usr/bin/env python3
"""Prepare a Nav-only import-order compatibility diagnostic; never launch a job."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--deadline', required=True, help='Future Slurm server-local YYYY-MM-DDTHH:MM:SS')
    p.add_argument('--remote-root', default='/mnt/home/liyuqin/USERDIR/vln_baselines_20261103')
    p.add_argument('--readiness-file', type=Path, action='append', default=[])
    a = p.parse_args()
    datetime.strptime(a.deadline, '%Y-%m-%dT%H:%M:%S')
    local = Path(__file__).resolve().parents[2]
    remote = Path(a.remote_root)
    if not remote.is_absolute():
        p.error('remote-root must be absolute')
    if a.output.exists():
        p.error('Refusing to overwrite any existing specification')
    py = str(remote / 'env/falcon/bin/python')
    scripts = ['run_reproduction_diagnostics.py', 'validate_reproduction_diagnostics.py',
               'smoke_navthinker_official.py', 'validate_navthinker_smoke.py']
    labels = ['nav_original', 'nav_import_overlay']
    required = ['diag/diagnostics_summary.json']
    for name in labels:
        required += ['diag/' + name + suffix for suffix in (
            '.json', '_classification.json', '.log', '_validation.log')]
    inputs = []
    scientific = [Path(__file__).resolve(), local / 'baselines/prep/navthinker_source_manifest.json',
                  local / 'baselines/prep/navthinker_il_import_overlay/compatibility_overlay_manifest.json',
                  local / 'baselines/prep/environment_evidence.json',
                  local / 'baselines/prep/egl_runtime_packages_20261103.json']

    def bind(source, target):
        inputs.append({'path': str(target), 'bytes': source.stat().st_size,
                       'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})

    bind(local / 'baselines/prep/navthinker_source.tar.gz', remote / 'inputs/navthinker_source.tar.gz')
    for name, source in [('NavThinker', local / 'baselines/repos/NavThinker'),
                         ('NavThinker_import_overlay', local / 'baselines/prep/navthinker_il_import_overlay')]:
        for rel in ['habitat-baselines/habitat_baselines/__init__.py', 'configs/navthinker_hm3d_eval.yaml']:
            bind(source / rel, remote / 'src' / name / rel)
    for name in ['models.py', 'networks.py', 'navthinker_policy.py']:
        rel = Path('habitat-baselines/habitat_baselines/rl/world_model') / name
        bind(local / 'baselines/prep/navthinker_il_import_overlay' / rel,
             remote / 'src/NavThinker_import_overlay' / rel)
    for path in a.readiness_file:
        if not path.is_file():
            p.error('Readiness evidence does not exist: ' + str(path))
        scientific.append(path.resolve())
    rid = 'vln_nav_diag_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    spec = {
        'run_id': rid, 'remote_root': str(remote / 'runs'), 'remote_python': py,
        'gpu_binding': 'slurm_linux_minor_uuid',
        'slurm': {'partition': 'tacc', 'gpus': 1, 'cpus': 4, 'wall_time': '00:15:00', 'deadline': a.deadline},
        'science': {
            'cwd': str(remote / 'src/NavThinker'),
            'argv': [py, '{source}/run_reproduction_diagnostics.py', '--output', '{output}/diag',
                     '--root', str(remote), '--mode', 'nav-only', '--habitat-sim-first'],
            'env': {'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MAGNUM_LOG': 'quiet',
                    'HABITAT_SIM_LOG': 'quiet'},
            'timeout_seconds': 720, 'validation_timeout_seconds': 120,
            'validator_argv': [py, '{source}/validate_reproduction_diagnostics.py', '--output', '{output}/diag',
                               '--mode', 'nav-only', '--require-habitat-sim-first'],
            'required_outputs': required},
        'uploads': [{'file': str(local / 'baselines/scripts' / name), 'remote_name': name} for name in scripts],
        'remote_inputs': inputs, 'scientific_files': list(dict.fromkeys(str(path) for path in scientific)),
        'reserve_bytes': 4 << 30, 'output_budget_bytes': 16 << 20,
        'temporary_transfer_bytes': 16 << 20, 'remote_peak_additional_bytes': 64 << 20,
        'monitor_timeout_seconds': 7200,
        'diagnostic_plan': {
            'status': 'SPEC_ONLY_NOT_FROZEN_NOT_SUBMITTED', 'paper_reproduction': False,
            'benchmark_episode_count': 0, 'weights_loaded': False,
            'runtime_environment_compatibility': 'Explicit habitat_sim import before Habitat-Lab/Magnum',
            'source_variants': ['Official source unchanged', 'Existing documented unrelated IL import overlay'],
            'expected_source_blockers': ['Official missing habitat_baselines.il.data', 'Default RSSM lacks policy feat_size'],
            'exit_contract': 'Only normal Nav exit 0 or 1 matching the report is classifiable; signals, timeouts, launch errors and other codes fail.',
            'causal_limit': 'An import-order hypothesis probe, not an established crash fix or benchmark result.',
            'preflight_note': 'Use the already prepared pinned environment and sources. No downloads, installs, simulator episodes or checkpoint loads.'}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps({'path': str(a.output.resolve()), 'run_id': rid, 'status': spec['diagnostic_plan']['status'],
                      'required_outputs': len(required), 'remote_inputs': len(inputs)}, indent=2))


if __name__ == '__main__':
    main()
