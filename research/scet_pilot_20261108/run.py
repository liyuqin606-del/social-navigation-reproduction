"""Single frozen cohort, 24 fits. Execute with --output NEW_DIRECTORY."""
import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import time
import numpy as np
from core import (H,N,Net,features,training_data,prefixes,prefix_loss,consistency_loss,
                  conditional_truth,event_predictions,step_predictions,evaluate)

ROOT=Path(__file__).resolve().parent


def save_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(records,arm,seed,protocol):
    rows=prefixes(records)
    x=features(rows[:,:2],rows[:,3],rows[:,2].astype(int))
    ages=rows[:,3].astype(int)
    target=rows[:,5 if arm=='step' else 4].astype(int)
    # Actual adjacent surviving prefixes only; no fixed deadline restart.
    paired=np.flatnonzero((rows[:,4]>=2*(ages+1))&(ages<H-1))
    net=Net(3 if arm=='step' else N,protocol['hidden_width'][arm],seed+100000)
    rng=np.random.default_rng(seed+200000)
    # Keep supervised minibatch sequence equal for B/C independent of SC draws.
    pair_rng=np.random.default_rng(seed+300000)
    log=[]
    started=time.perf_counter()
    for tick in range(protocol['updates'][arm]):
        idx=rng.integers(len(rows),size=protocol['batch_size'])
        nll,g=prefix_loss(net,x[idx],ages[idx],target[idx],arm=='step')
        sc=0.
        if arm=='scet':
            j=pair_rng.choice(paired,size=protocol['batch_size'])
            sc,gs=consistency_loss(net,x[j],ages[j])
            g=[a+protocol['consistency_weight']*b for a,b in zip(g,gs)]
        if not np.isfinite(nll+sc) or not all(np.isfinite(a).all() for a in g):
            raise ValueError('Non-finite training state')
        net.update(g,protocol['learning_rate'])
        if tick%100==0 or tick+1==protocol['updates'][arm]:
            log.append(dict(update=tick+1,nll=float(nll),consistency=float(sc)))
    return net,dict(seconds=time.perf_counter()-started,rows=len(rows),paired_rows=len(paired),
                    parameters=sum(a.size for a in net.p),history=log)


def summarize(out,protocol,all_results,training_times,prediction_times):
    summaries={}
    rng=np.random.default_rng(protocol['bootstrap']['seed'])
    count=protocol['evaluation']['contexts']
    indices=rng.integers(count,size=(protocol['bootstrap']['resamples'],count))
    for budget in protocol['training_contexts']:
        group={}
        for arm in protocol['arms']:
            selected=[all_results[(budget,seed,arm)] for seed in protocol['training_seeds']]
            group[arm]={k:float(np.mean([r[k] for r in selected])) for k in ['cost','collision','option_ticks','kernel_kl','consistency']}
            group[arm]['training_seconds_mean']=float(np.mean([training_times[(budget,s,arm)] for s in protocol['training_seeds']]))
            group[arm]['prediction_seconds_mean']=float(np.mean([prediction_times[(budget,s,arm)] for s in protocol['training_seeds']]))
        contrasts={}
        sc=np.mean([all_results[(budget,s,'scet')]['cost'] for s in protocol['training_seeds']],axis=0)
        sc_collision=np.mean([all_results[(budget,s,'scet')]['collision'] for s in protocol['training_seeds']],axis=0)
        for arm in ['step','event','event_long']:
            base=np.mean([all_results[(budget,s,arm)]['cost'] for s in protocol['training_seeds']],axis=0)
            base_c=np.mean([all_results[(budget,s,arm)]['collision'] for s in protocol['training_seeds']],axis=0)
            delta=sc-base
            interval=np.quantile(delta[indices].mean(axis=1),protocol['bootstrap']['interval'])
            reduction=float(1-sc.mean()/base.mean())
            collision_delta=float((sc_collision-base_c).mean())
            contrasts[arm]=dict(cost_difference=float(delta.mean()),ci99=interval.tolist(),relative_cost_reduction=reduction,
                                collision_difference=collision_delta,passes=bool(reduction>=.05 and interval[1]<0 and collision_delta<=.005))
        summaries[str(budget)]=dict(arms=group,contrasts=contrasts)
    gate=all(v['passes'] for v in summaries['512']['contrasts'].values())
    report=dict(scope=protocol['scope'],gate='GO_TO_NEXT_ENVIRONMENT' if gate else 'NO_GO',
                results=summaries,uncertainty='Paired context bootstrap after averaging three training seeds; conditional on these fitted models, not a full training-population confidence interval.',
                next='Do not claim Falcon improvement. Preserve cohort; no hyperparameter selection from it.')
    save_json(out/'SUMMARY.json',report)
    lines=['# SCET finite-model learning pilot','',f"Decision: **{report['gate']}**",'',protocol['scope'],'',
           '| Training contexts | Arm | Expected cost | Collision probability | Kernel KL | Train seconds |',
           '|---:|---|---:|---:|---:|---:|']
    for budget,entry in summaries.items():
        for arm,r in entry['arms'].items():
            lines.append(f"| {budget} | {arm} | {r['cost']:.5f} | {r['collision']:.5f} | {r['kernel_kl']:.5f} | {r['training_seconds_mean']:.2f} |")
    lines+=['','All methods have 1,313 parameters and receive the same observed history. The step model receives explicit age, so it does not reset the waiting-time distribution.',
            'SCET uses extra consistency passes; event_long receives 3x supervised updates. Recorded training time, not update count alone, determines actual compute comparisons.',
            'Expected costs are evaluated exactly under the simulator distribution, including a closed-loop decision at every survived prefix. These are not robot trajectories or benchmark success rates.',
            '',report['uncertainty'],'', '## SCET contrasts at 512 contexts','']
    for arm,r in summaries['512']['contrasts'].items():
        lines.append(f"- Versus {arm}: cost reduction {100*r['relative_cost_reduction']:.3f}%; cost delta {r['cost_difference']:.5f}, 99% paired-context interval {r['ci99']}; collision delta {r['collision_difference']:.5f}; gate {r['passes']}.")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    out=args.output
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    source={p.name:sha(p) for p in ROOT.iterdir() if p.suffix in ['.py','.json']}
    save_json(out/'FROZEN_INPUTS.json',dict(sources=source,python=platform.python_version(),numpy=np.__version__,pid=os.getpid(),protocol=protocol))
    save_json(out/'STATUS.json',dict(state='RUNNING',fits_complete=0))
    rng=np.random.default_rng(protocol['evaluation']['seed'])
    cues=rng.uniform(-1,1,(protocol['evaluation']['contexts'],2))
    np.save(out/'evaluation_cues.npy',cues)
    truth=conditional_truth(cues)
    oracle=evaluate(truth,truth)
    np.savez_compressed(out/'oracle.npz',**oracle)
    results={}
    times={}
    prediction_times={}
    for seed in protocol['training_seeds']:
        master=training_data(max(protocol['training_contexts']),seed)
        np.save(out/f'training_{seed}.npy',master)
        for count in protocol['training_contexts']:
            records=master[:count*3]
            for arm in protocol['arms']:
                name=f'{count}_{seed}_{arm}'
                net,log=train(records,arm,seed,protocol)
                np.savez_compressed(out/f'{name}_weights.npz',**{f'p{i}':p for i,p in enumerate(net.p)})
                save_json(out/f'{name}_training.json',log)
                started=time.perf_counter()
                kernel=step_predictions(net,cues) if arm=='step' else event_predictions(net,cues)
                elapsed=time.perf_counter()-started
                result=evaluate(kernel,truth)
                np.savez_compressed(out/f'{name}_evaluation.npz',**result)
                results[(count,seed,arm)]=result
                times[(count,seed,arm)]=log['seconds']
                prediction_times[(count,seed,arm)]=elapsed
                print(json.dumps(dict(fit=name,seconds=round(log['seconds'],2),cost=float(result['cost'].mean()),collision=float(result['collision'].mean()),kl=result['kernel_kl'])),flush=True)
                save_json(out/'STATUS.json',dict(state='RUNNING',fits_complete=len(results),last=name))
    report=summarize(out,protocol,results,times,prediction_times)
    for filename,digest in source.items():
        if sha(ROOT/filename)!=digest:
            raise ValueError(f'Source changed during run: {filename}')
    save_json(out/'STATUS.json',dict(state='COMPLETE_UNVALIDATED',fits_complete=len(results),gate=report['gate']))
    save_json(out/'OUTPUT_HASHES.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    print(report['gate'],flush=True)


if __name__=='__main__':
    main()
