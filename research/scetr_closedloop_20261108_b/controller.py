"""Temporal probe and event-terminated options from legal robot observations."""
import numpy as np
POLICIES=('reference','unknown_yield','probe_yield','probe_escape')

def signals(depth):
    d=np.asarray(depth,dtype=np.float64)/65535
    if d.shape!=(32,32):raise ValueError('depth contract')
    roi=d[6:25,8:24];zero=float(np.mean(roi<.001));near=float(np.mean((roi>=.001)&(roi<.09)))
    front=d[6:20,12:20]
    return dict(zero=zero,near=near,left=float(np.quantile(d[6:25,2:12],.25)),
                right=float(np.quantile(d[6:25,20:30],.25)),
                risk=bool(zero>.05 or near>.20),clear=bool(zero<.025 and near<.10),
                corridor=bool(np.mean(front<.001)<.10 and np.quantile(front,.20)>.06))

def temporal_change(a,b):
    a=np.asarray(a,dtype=float)[6:25,8:24]/65535
    b=np.asarray(b,dtype=float)[6:25,8:24]/65535
    return float(np.mean((np.abs(a-b)>.005)&((a<.2)|(b<.2))))

class Controller:
    def __init__(self,policy):
        if policy not in POLICIES:raise ValueError(policy)
        self.policy=policy;self.step=0;self.phase='baseline';self.age=0
        self.events=0;self.cooldown=0;self.entry=None;self.previous_pair=None
        self.motion=0.;self.stable=0;self.turn=2;self.total_overrides=0

    def release(self,cooldown):
        self.phase='baseline';self.age=0;self.cooldown=cooldown

    def act(self,depth,proposed):
        proposed=int(proposed);assert proposed in range(4)
        s=signals(depth);chosen=proposed;event=None
        if self.policy!='reference' and proposed!=0:
            if self.cooldown:self.cooldown-=1
            trigger=s['zero']>.05 if self.policy=='unknown_yield' else s['risk']
            if self.phase=='baseline' and self.cooldown==0 and self.step>=8 and proposed==1 and trigger and self.total_overrides<192:
                self.phase='yield' if self.policy=='unknown_yield' else 'probe'
                self.age=0;self.entry=np.asarray(depth).copy();self.previous_pair=np.asarray(depth).copy()
                self.motion=0.;self.stable=0;self.events+=1;event='trigger'
            if self.phase=='probe':
                if self.age and self.age%2==0:
                    self.motion=max(self.motion,temporal_change(self.entry,depth))
                if self.age>=4:
                    if self.motion<.02:
                        self.release(16);event='static_dismiss'
                    else:
                        self.phase='yield';self.age=0;event='dynamic_evidence'
                else:
                    chosen=2 if self.age%2==0 else 3;self.age+=1
            if self.phase=='yield':
                if self.age%2==0:
                    clear=s['zero']<.025 if self.policy=='unknown_yield' else s['clear']
                    self.stable=self.stable+1 if clear else 0
                if self.age>=8 and self.age%2==0 and self.stable>=3:
                    self.release(8);event='clear_handoff'
                elif self.age>= (16 if self.policy=='probe_escape' else 48):
                    if self.policy=='probe_escape':
                        self.phase='scan';self.age=0;self.turn=2 if s['left']>=s['right'] else 3;event='escape'
                    else:
                        self.release(8);event='bounded_handoff'
                else:
                    chosen=2 if self.age%2==0 else 3;self.age+=1
            if self.phase=='scan':
                if self.age>=6 and s['corridor']:
                    self.phase='advance';self.age=0;event='corridor_found'
                elif self.age>=24:
                    self.release(8);event='scan_bound'
                else:chosen=self.turn;self.age+=1
            if self.phase=='advance':
                if self.age>=8 or not s['corridor']:
                    self.release(8);event='escape_handoff'
                else:chosen=1;self.age+=1
        if chosen!=proposed:self.total_overrides+=1
        log=dict(policy=self.policy,signals=s,phase=self.phase,age=self.age,events=self.events,
                 cooldown=self.cooldown,event=event,motion=self.motion,stable=self.stable,
                 total_overrides=self.total_overrides,executed=chosen)
        self.step+=1
        return chosen,log
