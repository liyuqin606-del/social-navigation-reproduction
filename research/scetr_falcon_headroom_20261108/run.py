import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from validate import validate,read,compare_prefix


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    a,remaining=p.parse_known_args()
    a.output.mkdir(parents=True,exist_ok=False)
    protocol=read(a.protocol);start_time=time.monotonic();schedule=dict(episodes=[])
    def run_case(case,name,start=-1,seq=()):
        if time.monotonic()-start_time>2850: raise RuntimeError('BOUNDED_BUDGET_EXHAUSTED_NO_RETRY')
        argv=[sys.executable,str(Path(__file__).with_name('run_case.py')),*remaining,'--output',str(a.output/name),'--case',str(case),'--start',str(start),'--actions',json.dumps(seq)]
        print('START '+name,flush=True)
        with (a.output/(name+'.log')).open('x') as log:
            proc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=210)
        if proc.returncode: raise RuntimeError('CASE_FAILED '+name)
        trace=read(a.output/name/'trace.json')
        print(json.dumps(dict(case=name,steps=len(trace),metrics=trace[-1]['metrics'])),flush=True)
        return trace
    for case in range(4):
        ref_name=f'case{case}_reference';rep_name=f'case{case}_repeat'
        ref=run_case(case,ref_name);rep=run_case(case,rep_name)
        if len(ref)!=len(rep): raise RuntimeError('reference repeat length mismatch')
        compare_prefix(ref,rep,len(ref))
        hit=[r['step'] for r in ref if r['metrics'].get('human_collision',r['metrics'].get('did_multi_agents_collide',0))>0]
        entry=dict(case=case,reference=ref_name,repeat=rep_name,first_collision=hit[0] if hit else None,branches=[])
        schedule['episodes'].append(entry)
        seen=set()
        if hit:
            for j,intervention in enumerate(protocol['interventions']):
                start=max(0,hit[0]-intervention['lead_steps']);seq=intervention['sequence']
                if (start,tuple(seq)) in seen: continue
                seen.add((start,tuple(seq)))
                name=f'case{case}_branch{j}';tr=run_case(case,name,start,seq)
                compare_prefix(ref,tr,start,True)
                entry['branches'].append(dict(directory=name,start=start,sequence=seq,lead_steps=intervention['lead_steps']))
        (a.output/'SCHEDULE.json').write_text(json.dumps(schedule,indent=2)+'\n')
    result=validate(a.output)
    # All evidence, including logs and derived source, packed for one SHA-verified transfer.
    (a.output/'RUN_COMPLETE.json').write_text(json.dumps(dict(rollouts=result['terminal_rollouts'],elapsed_seconds=time.monotonic()-start_time))+'\n')
    import tarfile
    archive=a.output/'EVIDENCE.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for item in sorted(a.output.iterdir()):
            if item!=archive: t.add(item,arcname=item.name)
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)


if __name__=='__main__': main()
