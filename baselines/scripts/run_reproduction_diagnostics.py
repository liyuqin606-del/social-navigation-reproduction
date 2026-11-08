#!/usr/bin/env python3
"""Run bounded baseline diagnostics; expected upstream failures remain explicit."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--root', type=Path, required=True)
p.add_argument('--mode', choices=('combined', 'nav-only'), default='combined')
p.add_argument('--habitat-sim-first', action='store_true', help='Explicit Nav runtime import-order compatibility probe')
a = p.parse_args()
assert os.environ.get('SLURM_JOB_ID')
a.output = a.output.resolve()
a.root = a.root.resolve()
if a.output.exists() and any(a.output.iterdir()):
    raise RuntimeError('Diagnostic output must be absent or empty; stale results are not accepted')
a.output.mkdir(parents=True, exist_ok=True)
here = Path(__file__).resolve().parent
results = {}
required_stages = (('falcon_smoke', 'falcon_validation') if a.mode == 'combined' else ()) + (
    'nav_original', 'nav_original_validation', 'nav_import_overlay', 'nav_import_overlay_validation')


def save_summary(complete=False):
    classified = (complete and set(results) == set(required_stages)
                  and (a.mode == 'nav-only' or results['falcon_smoke'].get('exit_code') == 0)
                  and all(results[name].get('exit_code') == 0 for name in required_stages if name.endswith('validation'))
                  and all(results[name].get('exit_code') in (0, 1) for name in ('nav_original', 'nav_import_overlay'))
                  and not any(row.get('timed_out', False) for row in results.values()))
    summary = {'scope': 'ENGINEERING_DIAGNOSTICS_NO_BENCHMARK', 'paper_reproduction': False,
               'diagnostic_mode': a.mode, 'habitat_sim_first_requested': a.habitat_sim_first,
               'benchmark_episode_count': 0, 'required_stages': required_stages,
               'all_stages_attempted': complete, 'diagnostics_complete': classified, 'stages': results,
               'all_diagnostics_classified': classified}
    target = a.output / 'diagnostics_summary.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(summary, indent=2) + '\n')
    temporary.replace(target)
    return summary


def run(label, argv):
    start = time.monotonic()
    results[label] = {'argv': argv, 'status': 'RUNNING', 'exit_code': None, 'timed_out': False}
    save_summary()
    with (a.output / (label + '.log')).open('wb') as f:
        try:
            completed = subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT, timeout=240)
            exit_code = completed.returncode
        except subprocess.TimeoutExpired as error:
            exit_code = 124
            results[label]['timed_out'] = True
            results[label]['error'] = str(error)
            f.write(b'\nDIAGNOSTIC_STAGE_TIMEOUT_240_SECONDS\n')
        except OSError as error:
            exit_code = 127
            results[label]['error'] = str(error)
            f.write(('\nDIAGNOSTIC_STAGE_LAUNCH_FAILURE: ' + str(error) + '\n').encode())
    results[label].update(status='FINISHED', exit_code=exit_code,
                          termination_signal=-exit_code if exit_code < 0 else None,
                          seconds=time.monotonic() - start)
    save_summary()
    print(label, exit_code, flush=True)
    return exit_code


if a.mode == 'combined':
    run('falcon_smoke', [sys.executable, str(here / 'falcon_gpu_smoke.py'), '--output', str(a.output / 'falcon'),
                        '--checkpoint', str(a.root / 'inputs/falcon_noaux_25.pth'), '--repo', str(a.root / 'src/Falcon')])
    run('falcon_validation', [sys.executable, str(here / 'validate_falcon_smoke.py'), '--output', str(a.output / 'falcon')])
for label, name in [('nav_original', 'NavThinker'), ('nav_import_overlay', 'NavThinker_import_overlay')]:
    report = a.output / (label + '.json')
    run(label, [sys.executable, str(here / 'smoke_navthinker_official.py'), '--repo', str(a.root / 'src' / name),
                '--output', str(report)] + (['--habitat-sim-first'] if a.habitat_sim_first else []))
    run(label + '_validation', [sys.executable, str(here / 'validate_navthinker_smoke.py'), str(report),
                               '--output', str(a.output / (label + '_classification.json'))])
summary = save_summary(complete=True)
raise SystemExit(0 if summary['all_diagnostics_classified'] else 1)
