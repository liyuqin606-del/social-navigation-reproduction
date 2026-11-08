"""Post-run mechanism diagnosis; no parameter updates or new candidate selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from core import N,H,Net,prefixes,features,prefix_loss,consistency_loss,softmax,mask_for


def norm(gs):
    return float(np.sqrt(sum(np.square(g).sum() for g in gs)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report=json.loads((args.run/'SUMMARY.json').read_text())
    frozen=json.loads((args.run/'FROZEN_INPUTS.json').read_text())['protocol']
    oracle=float(np.load(args.run/'oracle.npz')['cost'].mean())
    gradients=[]
    for seed in frozen['training_seeds']:
        rows=prefixes(np.load(args.run/f'training_{seed}.npy'))
        rng=np.random.default_rng(62301+seed)
        idx=rng.integers(len(rows),size=1024)
        ages=rows[idx,3].astype(int)
        x=features(rows[idx,:2],ages,rows[idx,2].astype(int))
        net=Net(N,32,0)
        weights=np.load(args.run/f'512_{seed}_scet_weights.npz')
        net.p=[weights[f'p{i}'] for i in range(4)]
        nll,gn=prefix_loss(net,x,ages,rows[idx,4].astype(int))
        legal=np.flatnonzero((rows[:,4]>=2*(rows[:,3]+1))&(rows[:,3]<H-1))
        j=rng.choice(legal,size=1024)
        xx=features(rows[j,:2],rows[j,3],rows[j,2].astype(int))
        sc,gs=consistency_loss(net,xx,rows[j,3].astype(int))
        gradients.append(dict(seed=seed,nll=nll,consistency=sc,nll_grad_norm=norm(gn),
                              weighted_sc_grad_norm=10*norm(gs),weighted_gradient_ratio=10*norm(gs)/norm(gn)))
    headroom={}
    for budget,entry in report['results'].items():
        baseline=entry['arms']['event']['cost']
        headroom[budget]=dict(event_cost=baseline,oracle_cost=oracle,maximum_relative_reduction=1-oracle/baseline)
    result=dict(status='POSTHOC_DIAGNOSIS_NOT_SELECTION',headroom=headroom,gradients_on_training_only=gradients,
                interpretation='Absolute event-time classes plus survival masking already implement much of conditional time updating. Adjacent-prefix regularizer is small in this family. Oracle headroom is below the preregistered5% gate even for the ordinary event model. This family cannot demonstrate the desired large improvement at its present difficulty.',
                next='Use observation-dependent future branches and assess oracle headroom before further training. Keep a belief-aware primitive planner; do not use a deliberately memoryless baseline.')
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
