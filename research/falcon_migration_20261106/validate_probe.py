import argparse
import json
from pathlib import Path
import numpy as np


def require(ok,message):
    if not ok: raise RuntimeError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    traces={mode:json.loads((a.output/mode/'trace.json').read_text()) for mode in ['reference','shadow','override']}
    checks={}
    for mode,rows in traces.items():
        bridge=json.loads((a.output/mode/'bridge_result.json').read_text())
        require(len(rows)==16 and bridge['weights_unchanged'] and not bridge['pending_transition'],'case completeness/weights')
        for i,row in enumerate(rows):
            require(row['step']==i and row['executed'] in (0,1,2,3),'action validity')
            if i and not row['reset']:
                require(row['previous_action']==rows[i-1]['executed'],'previous action')
            if i: require(row['reset']==rows[i-1]['done'],'mask reset')
            if row['executed']==0: require(row['done'],'STOP did not terminate')
            if mode!='override': require(row['proposed']==row['executed'],'identity adapter changed action')
        checks[mode]=bridge
    same_fields=['episode','reset','previous_action','proposed','executed','done','depth_sha','goal','recurrent_in_sha']
    require(all(x[k]==y[k] for x,y in zip(traces['reference'],traces['shadow']) for k in same_fields),'reference/shadow trace mismatch')
    maximum={}
    with np.load(a.output/'reference/features.npz',allow_pickle=False) as base, np.load(a.output/'shadow/features.npz',allow_pickle=False) as shadow:
        for key,shape in [('features',(16,512)),('logits',(16,4))]:
            require(base[key].shape==shadow[key].shape==shape,'feature tensor shape')
            require(np.isfinite(base[key]).all() and np.isfinite(shadow[key]).all(),'nonfinite features')
            maximum[key]=float(np.max(np.abs(base[key]-shadow[key])))
            require(np.allclose(base[key],shadow[key],rtol=1e-5,atol=1e-6),'forward equivalence')
    changed=traces['override']
    require(changed[3]['executed'] in (2,3) and changed[3]['executed']!=changed[3]['proposed'],'forced turn absent')
    require(changed[4]['previous_action']==changed[3]['executed'] or changed[4]['reset'],'turn not fed back')
    require(changed[6]['executed']==0 and changed[6]['done'] and changed[7]['reset'],'forced STOP reset absent')
    report=dict(status='PASS_REAL_FALCON_INTERFACE_ONLY',total_steps=48,cases=checks,
                reference_shadow_max_abs=maximum,forced_turn_and_stop_verified=True,
                deployed_world_model=False,learned_policy_improvement=False,training_examples_collected=0,
                source_scope='Existing demo episodes; engineering only, excluded from fitting/calibration/selection')
    (a.output/'VALIDATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
