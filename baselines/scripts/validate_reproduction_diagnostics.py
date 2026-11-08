#!/usr/bin/env python3
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--mode',choices=('combined','nav-only'),default='combined')
p.add_argument('--require-habitat-sim-first',action='store_true');a=p.parse_args()
here=Path(__file__).resolve().parent
r=json.loads((a.output/'diagnostics_summary.json').read_text())
assert r['scope']=='ENGINEERING_DIAGNOSTICS_NO_BENCHMARK' and r['paper_reproduction'] is False
assert r['benchmark_episode_count']==0 and r['all_diagnostics_classified']
expected={'falcon_smoke','falcon_validation','nav_original','nav_original_validation','nav_import_overlay','nav_import_overlay_validation'}
assert r.get('diagnostic_mode','combined')==a.mode
if a.mode=='nav-only': expected-={'falcon_smoke','falcon_validation'}
assert r['diagnostics_complete'] is True and set(r['stages'])==expected
assert all(row['status']=='FINISHED' and row['timed_out'] is False and type(row['exit_code']) is int and row['exit_code'] >= 0 and row['exit_code'] not in (124,127) for row in r['stages'].values()), 'A diagnostic process timed out, failed to launch, or terminated by signal'
if a.mode=='combined': assert r['stages']['falcon_smoke']['exit_code']==0
assert all(r['stages'][name]['exit_code']==0 for name in expected if name.endswith('validation'))
for name in ['nav_original','nav_import_overlay']:
 assert r['stages'][name]['exit_code'] in (0,1), 'Unexpected diagnostic process exit'
 report=json.loads((a.output/(name+'.json')).read_text())
 assert report['exit_code']==r['stages'][name]['exit_code'], 'Report exit code differs from observed process exit'
 if a.require_habitat_sim_first:
  assert r['habitat_sim_first_requested'] is True
  compat=report['runtime_environment_compatibility']
  assert compat['habitat_sim_first_requested'] is True and compat['habitat_sim_first_applied'] is True
  assert compat['native_modules_before_runtime']=={'habitat_sim':False,'magnum':False,'corrade':False}
  assert compat['habitat_sim_version']=='0.3.1' and Path(compat['habitat_sim_file']).is_absolute()
if a.mode=='combined': subprocess.run([sys.executable,str(here/'validate_falcon_smoke.py'),'--output',str(a.output/'falcon')],check=True)
for n in ['nav_original','nav_import_overlay']:
 subprocess.run([sys.executable,str(here/'validate_navthinker_smoke.py'),str(a.output/(n+'.json')),'--output',str(a.output/(n+'_classification.json'))],check=True)
print('Diagnostic outputs independently classified; no benchmark or paper metric asserted.')
