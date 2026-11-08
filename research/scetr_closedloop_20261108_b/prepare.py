"""Read-only admission and new managed specification; does not submit."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'lifecycle'))
from transport import Remote,read,save

def main():
    if (HERE/'SPEC.json').exists():raise RuntimeError('already prepared')
    old=HERE.parent/'scetr_falcon_headroom_20261108'
    spec=read(old/'SPEC.json');remote=Remote(read(old/'plan/profile.json'))
    code="""import datetime,json,subprocess,shutil
print(json.dumps({'deadline':(datetime.datetime.now()+datetime.timedelta(hours=4)).strftime('%Y-%m-%dT%H:%M:%S'),'jobs':subprocess.check_output(['squeue','-h','-u','liyuqin','-o','%i|%j|%T|%b']).decode(),'remote_free_bytes':shutil.disk_usage('/mnt/home/liyuqin/USERDIR').free}))"""
    admission=remote.json(code,timeout=30,retry=True)
    assert not admission['jobs'].strip() and admission['remote_free_bytes']>5*(1<<30)
    save(HERE/'REMOTE_ADMISSION.json',admission)
    spec['run_id']='vln_scetr_closedloop_20261108_b'
    spec['slurm']['deadline']=admission['deadline']
    files=['bridge.py','derive.py','run_case.py','run.py','validate.py','controller.py','episodes.json','PROTOCOL.json']
    spec['uploads']=[{'file':str(HERE/n),'remote_name':n} for n in files]
    spec['scientific_files']=[str(HERE/n) for n in ['PROTOCOL.json','prepare.py','test_controller.py']]
    save(HERE/'SPEC.json',spec)
    print(json.dumps({'status':'PREPARED_NOT_SUBMITTED',**admission}))

if __name__=='__main__':main()
