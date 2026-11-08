from pathlib import Path
import json,os,shutil,subprocess
r=Path('/mnt/home/liyuqin/USERDIR/vln_baselines_20261103')
print('DISK',shutil.disk_usage(r)._asdict())
f=r/'prep_logs/conda_process.json';j=json.loads(f.read_text());print(j)
print(subprocess.run(['ps','-p',str(j['pid']),'-o','pid,etime,args'],stdout=subprocess.PIPE,text=True).stdout)
print((r/'prep_logs/conda_create.log').read_text()[-5000:])
print('ENV', (r/'env/falcon/bin/python').exists())
