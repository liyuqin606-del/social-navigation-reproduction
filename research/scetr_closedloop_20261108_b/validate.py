import argparse,hashlib,json
from pathlib import Path
import numpy as np
from controller import Controller,POLICIES


def read(p):return json.loads(Path(p).read_text())

def physical(row):return {k:v for k,v in row.items() if k!='control'}

def check_case(p,policy):
    tr=read(p/'trace.json');m=read(p/'bridge_result.json');a=np.load(p/'features.npz',allow_pickle=False)
    assert 0<len(tr)<=500 and m['steps']==len(tr)
    assert tr[-1]['done'] and not any(r['done'] for r in tr[:-1])
    assert m['done'] and not m['pending'] and m['weights_unchanged'] and m['actor_before_sha']==m['actor_after_sha']
    assert m['policy']==policy and m['sequence']==[] and m['intervention_start']==-1
    assert read(p/'CASE_STATUS.json')['status']=='TERMINAL_COLLECTED'
    assert a['features'].shape==(len(tr),512) and a['logits'].shape==(len(tr),4)
    assert a['depth_thumbnails'].shape==(len(tr),32,32)
    c=Controller(policy)
    for i,r in enumerate(tr):
        assert r['step']==i and r['reset']==(i==0)
        if i:assert r['previous_action']==tr[i-1]['executed']
        action,log=c.act(a['depth_thumbnails'][i],r['proposed'])
        assert action==r['executed']
        for k,v in log.items():
            if k=='signals':
                for sk,sv in v.items():assert np.isclose(sv,r['control'][k][sk],atol=1e-12,rtol=0)
            else:assert v==r['control'][k]
        for key,h in [('features','feature_sha'),('logits','logit_sha')]:
            assert hashlib.sha256(a[key][i].tobytes()).hexdigest()==r[h]
        assert np.isfinite(list(r['metrics'].values())).all()
    return tr,m,a


def validate(output,write=True):
    output=Path(output);schedule=read(output/'SCHEDULE.json')
    assert [x['case'] for x in schedule['episodes']]==list(range(4))
    summaries=[];nsteps=0;prefix_count=0;actor=set();names=[]
    for e in schedule['episodes']:
        assert list(e['runs'])==['reference','repeat']+list(POLICIES[1:])
        ref,meta,arr=check_case(output/e['runs']['reference'],'reference')
        rep,_,aa=check_case(output/e['runs']['repeat'],'reference')
        assert ref==rep
        np.testing.assert_array_equal(arr['features'],aa['features'])
        row={'case':e['case'],'reference':ref[-1]['metrics'],'policies':{}}
        for name,directory in e['runs'].items():
            assert directory==f"case{e['case']}_{name}";names.append(directory)
            tr,m,a=check_case(output/directory,'reference' if name=='repeat' else name)
            actor.add(m['actor_before_sha']);nsteps+=len(tr)
            changes=[i for i,r in enumerate(tr) if r['executed']!=r['proposed']]
            if name not in ['reference','repeat']:
                first=changes[0] if changes else len(tr)
                assert len(ref)>=first
                for i in range(first):assert physical(ref[i])==physical(tr[i]),f'prefix mismatch {name}:{i}'
                prefix_count+=first
                if changes:
                    for k in ['episode','observations_sha','goal','reset','previous_action','recurrent_sha','rng_sha','proposed','human_actions_sha','feature_sha','logit_sha']:
                        assert ref[first][k]==tr[first][k],f'first-change state mismatch {name}:{k}'
                    np.testing.assert_array_equal(a['features'][:first+1],arr['features'][:first+1])
                else:assert len(ref)==len(tr)
                row['policies'][name]={'metrics':tr[-1]['metrics'],'steps':len(tr),'changed_actions':len(changes),
                                      'events':max(r['control']['events'] for r in tr),'first_change':changes[0] if changes else None}
        summaries.append(row)
    assert len(actor)==1 and len(names)==20 and set(names)=={p.name for p in output.glob('case*') if p.is_dir()}
    aggregate={}
    for policy in POLICIES[1:]:
        wins=losses=successes=collisions=0
        for r in summaries:
            b=r['reference'];m=r['policies'][policy]['metrics']
            base=bool(b['success']==1 and b['human_collision']==0);new=bool(m['success']==1 and m['human_collision']==0)
            wins+=int(new and not base);losses+=int(base and not new);successes+=int(new);collisions+=int(m['human_collision']==1)
        aggregate[policy]=dict(wins=wins,regressions=losses,successes=successes,collisions=collisions)
    result=dict(status='PASS',scope='FOUR_DEVELOPMENT_CASES_ONLINE_OBSERVATION_ONLY_HEURISTIC_OPTIONS_NOT_TRAINED_SCET',
                episodes=4,rollouts=20,steps=nsteps,prefix_transitions=prefix_count,actor_sha256=next(iter(actor)),
                baseline_successes=sum(r['reference']['success']==1 and r['reference']['human_collision']==0 for r in summaries),
                aggregate=aggregate,cases=summaries)
    if write:(output/'VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();print(json.dumps(validate(a.output)))
