"""Terminal and exact prefix validation; summarize witnessed recovery only."""
import argparse
import json
from pathlib import Path
import numpy as np


def read(p): return json.loads(Path(p).read_text())


def check_case(path):
    trace=read(path/'trace.json');meta=read(path/'bridge_result.json')
    assert 0<len(trace)<=500 and meta['steps']==len(trace)
    assert trace[-1]['done'] and not any(r['done'] for r in trace[:-1])
    assert not meta['pending'] and meta['weights_unchanged'] and meta['actor_before_sha']==meta['actor_after_sha']
    assert read(path/'CASE_STATUS.json')['status']=='TERMINAL_COLLECTED'
    arrays=np.load(path/'features.npz')
    assert arrays['features'].shape==(len(trace),512) and arrays['logits'].shape==(len(trace),4)
    for i,row in enumerate(trace):
        assert row['step']==i and row['reset']==(i==0)
        if i: assert row['previous_action']==trace[i-1]['executed']
        offset=i-meta['intervention_start']
        expected=meta['sequence'][offset] if meta['intervention_start']>=0 and 0<=offset<len(meta['sequence']) else row['proposed']
        assert row['executed']==expected
    return trace,meta,arrays


def compare_prefix(reference,candidate,count,include_current=False):
    assert len(candidate)>=count and len(reference)>=count
    for i in range(count):
        assert reference[i]==candidate[i],f'prefix mismatch at {i}'
    if include_current and count<len(reference) and count<len(candidate):
        # At intervention step compare everything before env.step / override.
        for k in ['episode','observations_sha','goal','reset','previous_action','recurrent_sha','rng_sha','proposed','human_actions_sha','feature_sha','logit_sha']:
            assert reference[count][k]==candidate[count][k],f'intervention-state mismatch {k}:{count}'


def validate(output,write=True):
    output=Path(output);schedule=read(output/'SCHEDULE.json')
    rows=[];prefix_checks=0
    for item in schedule['episodes']:
        base=output/item['reference'];repeat=output/item['repeat']
        ref,meta,arr=check_case(base);rep,_,ar=check_case(repeat)
        assert len(ref)==len(rep);compare_prefix(ref,rep,len(ref));prefix_checks+=len(ref)
        np.testing.assert_array_equal(arr['features'],ar['features']);np.testing.assert_array_equal(arr['logits'],ar['logits'])
        final=ref[-1]['metrics'];collision=[r['step'] for r in ref if r['metrics'].get('human_collision',r['metrics'].get('did_multi_agents_collide',0))>0]
        assert item['first_collision']==(collision[0] if collision else None)
        recovered=[];candidates=[]
        for branch in item['branches']:
            tr,me,aa=check_case(output/branch['directory'])
            assert me['intervention_start']==branch['start'] and me['sequence']==branch['sequence']
            assert me['actor_before_sha']==meta['actor_before_sha']
            compare_prefix(ref,tr,branch['start'],True);prefix_checks+=branch['start']
            np.testing.assert_array_equal(aa['features'][:branch['start']+1],arr['features'][:branch['start']+1])
            m=tr[-1]['metrics'];success=m.get('success',0)==1 and m.get('human_collision',m.get('did_multi_agents_collide',0))==0
            if success: recovered.append(branch['directory'])
            candidates.append(dict(directory=branch['directory'],start=branch['start'],steps=len(tr),metrics=m,recovered=success))
        rows.append(dict(case=item['case'],reference_steps=len(ref),reference_metrics=final,
                         new_reference_collided=bool(collision),recovered=bool(recovered),recovered_by=recovered,candidates=candidates))
    assert len(rows)==4
    result=dict(status='PASS',scope='DEVELOPMENT_ORACLE_TIMING_FINITE_CANDIDATE_RECOVERABILITY_NOT_DEPLOYED_POLICY',
                episodes=4,colliding_references=sum(r['new_reference_collided'] for r in rows),
                witnessed_recoveries=sum(r['recovered'] for r in rows),
                terminal_rollouts=sum(2+len(r['candidates']) for r in rows),prefix_transitions_checked=prefix_checks,
                results=rows,no_population_gain_claim=True,no_active_perception_causal_claim=True)
    if write: (output/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=validate(a.output);print(json.dumps({k:v for k,v in r.items() if k!='results'}))
