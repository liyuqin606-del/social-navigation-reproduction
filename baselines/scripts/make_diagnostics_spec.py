#!/usr/bin/env python3
import argparse,datetime,hashlib,json,uuid
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
local=Path(__file__).resolve().parents[2]
remote='/mnt/home/liyuqin/USERDIR/vln_baselines_20261103'
py=remote+'/env/falcon/bin/python'
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8)))
rid='vln_baseline_diag_'+now.strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
names=['run_reproduction_diagnostics.py','validate_reproduction_diagnostics.py','falcon_gpu_smoke.py',
       'validate_falcon_smoke.py','smoke_navthinker_official.py','validate_navthinker_smoke.py']
inputs=[]
for name in ['falcon_noaux_25.pth','pretrained_habitat3.pth']:
 f=local/'baselines/assets/falcon/pretrained_model'/name
 inputs.append({'path':remote+'/inputs/'+name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
for name in ['falcon_source.tar.gz','navthinker_source.tar.gz']:
 f=local/'baselines/prep'/name
 inputs.append({'path':remote+'/inputs/'+name,'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
required=['diag/diagnostics_summary.json','diag/falcon/smoke_result.json','diag/falcon/resolved_config.yaml','diag/falcon/runtime_environment.json',
          'diag/nav_original.json','diag/nav_original_classification.json','diag/nav_import_overlay.json','diag/nav_import_overlay_classification.json']
required+=['diag/'+n+'.log' for n in ['falcon_smoke','falcon_validation','nav_original','nav_original_validation','nav_import_overlay','nav_import_overlay_validation']]
spec={'run_id':rid,'remote_root':remote+'/runs','remote_python':py,'gpu_binding':'slurm_linux_minor_uuid',
      'slurm':{'partition':'tacc','gpus':1,'cpus':4,'wall_time':'00:20:00','deadline':(now+datetime.timedelta(minutes=100)).strftime('%Y-%m-%dT%H:%M:%S')},
      'science':{'cwd':remote+'/src/Falcon','argv':[py,'{source}/run_reproduction_diagnostics.py','--output','{output}/diag','--root',remote],
                 'env':{'PYTHONNOUSERSITE':'1','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MAGNUM_LOG':'quiet','HABITAT_SIM_LOG':'quiet'},
                 'timeout_seconds':900,'validator_argv':[py,'{source}/validate_reproduction_diagnostics.py','--output','{output}/diag'],
                 'validation_timeout_seconds':120,'required_outputs':required},
      'uploads':[{'file':str(local/'baselines/scripts'/n),'remote_name':n} for n in names],
      'remote_inputs':inputs,'scientific_files':[str(local/'baselines/audits/falcon_diagnostics_review_20261103.json'),str(local/'baselines/prep/navthinker_source_manifest.json'),str(local/'baselines/prep/environment_evidence.json'),str(local/'baselines/prep/egl_runtime_packages_20261103.json')],
      'output_budget_bytes':16*2**20,'temporary_transfer_bytes':16*2**20,'remote_peak_additional_bytes':64*2**20,'monitor_timeout_seconds':7200}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(spec,indent=2)+'\n');print(rid)
