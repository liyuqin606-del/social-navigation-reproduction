import json, os, pathlib, shutil, subprocess
root=pathlib.Path('/mnt/home/liyuqin/USERDIR/vln_baselines_20261103')
out={'root':str(root),'free_bytes':shutil.disk_usage(root).free,'logs':{}}
for name in ['conda_process.json','conda_create.log','falcon_prepare_status.json','falcon_prepare_process.json']:
 p=root/'prep_logs'/name
 out['logs'][name]=p.read_text(errors='replace')[-5000:] if p.exists() else None
env=root/'env/falcon'
out['python_exists']=(env/'bin/python').is_file()
out['original_conda_process_exists']=pathlib.Path('/proc/74744').exists()
for key,argv in [('owned_usage',['du','-sk',str(root)]),('quota',['quota','-s']),('jobs',['squeue','-h','-u','liyuqin','-o','%i|%j|%T'])]:
 try:
  r=subprocess.run(argv,capture_output=True,text=True,timeout=20)
  out[key]={'returncode':r.returncode,'stdout':r.stdout[-4000:],'stderr':r.stderr[-1000:]}
 except Exception as e: out[key]={'error':str(e)}
print(json.dumps(out,indent=2))
