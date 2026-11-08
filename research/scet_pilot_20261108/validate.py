"""Replay saved weights and independently revalue selected policies."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from core import (H,N,Net,components,mixture,conditional_truth,event_predictions,
                  step_predictions,evaluate)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_cost(cues,modes,cutoffs):
    # Sum primitive event probability * realized cost, without cutoff_cost().
    weights=mixture(cues)
    hidden=components()
    values=[]
    risks=[]
    for i,(mode,stop) in enumerate(zip(modes,cutoffs)):
        total=0.
        risk=0.
        for latent in range(4):
            p=hidden[latent,mode]
            for event in range(N):
                t=event//2+1
                collision=(event%2==1)
                if event==N-1 or t>stop:
                    cost=stop+17.
                else:
                    cost=t+(60. if collision else 1.)
                    risk+=weights[i,latent]*p[event]*collision
                total+=weights[i,latent]*p[event]*cost
        values.append(total)
        risks.append(risk)
    return np.asarray(values),np.asarray(risks)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    frozen=json.loads((args.run/'FROZEN_INPUTS.json').read_text())
    root=Path(__file__).resolve().parent
    for name,sha in frozen['sources'].items():
        assert digest(root/name)==sha,(name,'source mismatch')
    hashes=json.loads((args.run/'OUTPUT_HASHES.json').read_text())
    for name,sha in hashes.items():
        assert digest(args.run/name)==sha,(name,'output mismatch')
    protocol=frozen['protocol']
    cues=np.load(args.run/'evaluation_cues.npy')
    truth=conditional_truth(cues)
    validated=[]
    expected_count=0
    for seed in protocol['training_seeds']:
        train=np.load(args.run/f'training_{seed}.npy')
        assert set(map(tuple,train[:,:2])).isdisjoint(set(map(tuple,cues)))
        for count in protocol['training_contexts']:
            for arm in protocol['arms']:
                name=f'{count}_{seed}_{arm}'
                weights=np.load(args.run/f'{name}_weights.npz')
                net=Net(3 if arm=='step' else N,protocol['hidden_width'][arm],0)
                net.p=[weights[f'p{i}'] for i in range(4)]
                assert sum(p.size for p in net.p)==1313
                kernel=step_predictions(net,cues) if arm=='step' else event_predictions(net,cues)
                np.testing.assert_allclose(kernel.sum(axis=-1),1,atol=1e-12)
                for age in range(H+1):
                    assert np.all(kernel[:,age,:,:2*age]==0)
                result=evaluate(kernel,truth)
                saved=np.load(args.run/f'{name}_evaluation.npz')
                for k in result:
                    np.testing.assert_allclose(result[k],saved[k],rtol=1e-11,atol=1e-11,err_msg=f'{name}:{k}')
                cost,risk=independent_cost(cues,saved['mode'],saved['cutoff'])
                np.testing.assert_allclose(cost,saved['cost'],atol=1e-11)
                np.testing.assert_allclose(risk,saved['collision'],atol=1e-11)
                assert np.all(saved['cost']>=np.load(args.run/'oracle.npz')['cost']-1e-10)
                validated.append(name)
                expected_count+=len(cues)
    report=dict(status='PASS',fits=len(validated),expected_policy_evaluations=expected_count,
                source_hashes=len(frozen['sources']),output_hashes=len(hashes),
                checks=['saved weight replay','1313 parameters','disjoint training/evaluation contexts',
                        'normalization and original deadline','independent primitive-outcome expected cost and collision sum',
                        'oracle lower bound'],
                scope='Exact finite-model policy expectations; no physical navigation trajectories',runs=validated)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='runs'},indent=2))


if __name__=='__main__':
    main()
