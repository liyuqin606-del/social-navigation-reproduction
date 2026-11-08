import argparse,json,subprocess,sys,time,tarfile
from pathlib import Path
from validate import validate


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True);a,rest=p.parse_known_args()
    protocol=json.loads(a.protocol.read_text());a.output.mkdir(exist_ok=False,parents=True)
    begin=time.monotonic();schedule={'episodes':[]}
    try:
        for case in range(4):
            entry={'case':case,'runs':{}};schedule['episodes'].append(entry)
            for label in ['reference','repeat']+protocol['policies']:
                policy='reference' if label=='repeat' else label
                name=f'case{case}_{label}';print('START '+name,flush=True)
                if time.monotonic()-begin>2800:raise RuntimeError('BOUNDED_BUDGET_EXHAUSTED')
                argv=[sys.executable,str(Path(__file__).with_name('run_case.py')),*rest,'--output',str(a.output/name),'--case',str(case),'--policy',policy]
                with (a.output/(name+'.log')).open('x') as log:
                    result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=260)
                if result.returncode:raise RuntimeError('CASE_FAILED '+name)
                entry['runs'][label]=name
                (a.output/'SCHEDULE.json').write_text(json.dumps(schedule,indent=2)+'\n')
                tr=json.loads((a.output/name/'trace.json').read_text())
                print(json.dumps(dict(case=name,steps=len(tr),metrics=tr[-1]['metrics'])),flush=True)
        result=validate(a.output)
        (a.output/'RUN_COMPLETE.json').write_text(json.dumps(dict(rollouts=result['rollouts'],elapsed_seconds=time.monotonic()-begin))+'\n')
        print(json.dumps(result),flush=True)
    finally:
        # Retain completed cases even if a later scientific unit fails.
        with tarfile.open(a.output/'EVIDENCE.tar.gz','w:gz') as bundle:
            for item in sorted(a.output.iterdir()):
                if item.name!='EVIDENCE.tar.gz':bundle.add(item,arcname=item.name)

if __name__=='__main__':main()
