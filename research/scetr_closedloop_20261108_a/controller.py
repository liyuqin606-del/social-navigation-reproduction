"""Observable-history options. No simulator state, future labels or random draws."""
import numpy as np

POLICIES = ('reference', 'yield24', 'yield_clear', 'sidestep')


def signals(depth):
    d = np.asarray(depth, dtype=np.float64) / 65535.0
    if d.shape != (32, 32):
        raise ValueError('expected recorded 32x32 uint16 depth thumbnail')
    roi = d[6:25, 8:24]
    zero = float(np.mean(roi < .001))
    near = float(np.mean((roi >= .001) & (roi < .09)))
    left = float(np.quantile(d[6:25, 2:12], .25))
    right = float(np.quantile(d[6:25, 20:30], .25))
    return dict(zero=zero, near=near, left=left, right=right,
                risk=bool(zero > .05 or near > .20),
                clear=bool(zero < .025 and near < .10))


class Controller:
    def __init__(self, policy):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.policy = policy
        self.step = 0
        self.age = -1
        self.events = 0
        self.cooldown = 0
        self.clear_count = 0
        self.turn = 2
        self.phase = 'baseline'
        self.phase_age = 0
        self.turns = 12

    def act(self, depth, proposed):
        proposed = int(proposed)
        if proposed not in range(4):
            raise ValueError('illegal baseline action')
        sig = signals(depth)
        event = None
        chosen = proposed
        if self.policy != 'reference' and proposed != 0:
            if self.cooldown:
                self.cooldown -= 1
            if self.age < 0 and self.cooldown == 0 and self.events < 3 and self.step >= 8 and proposed == 1 and sig['risk']:
                self.age = 0
                self.events += 1
                self.clear_count = 0
                self.turn = 2 if sig['left'] >= sig['right'] else 3
                self.phase = 'scan' if self.policy == 'sidestep' else 'yield'
                self.phase_age = 0
                event = 'trigger'
            if self.age >= 0:
                if self.policy in ('yield24', 'yield_clear'):
                    # Compare clearance only at restored headings after turn pairs.
                    if self.age % 2 == 0:
                        self.clear_count = self.clear_count + 1 if sig['clear'] else 0
                    end = self.age >= (24 if self.policy == 'yield24' else 64)
                    if self.policy == 'yield_clear' and self.age >= 8 and self.age % 2 == 0 and self.clear_count >= 3:
                        end = True
                    if end:
                        event = 'handoff'
                        self.age = -1
                        self.cooldown = 12
                        self.phase = 'baseline'
                    else:
                        chosen = 2 if self.age % 2 == 0 else 3
                        self.age += 1
                else:
                    if self.phase == 'scan' and self.phase_age == self.turns:
                        self.phase = 'advance'
                        self.phase_age = 0
                    if self.phase == 'advance' and (self.phase_age >= 6 or not sig['clear']):
                        self.phase = 'restore'
                        self.phase_age = 0
                    if self.phase == 'restore' and self.phase_age == self.turns:
                        self.age = -1
                        self.cooldown = 12
                        self.phase = 'baseline'
                        event = 'handoff'
                    else:
                        chosen = self.turn if self.phase == 'scan' else (1 if self.phase == 'advance' else 5-self.turn)
                        self.phase_age += 1
                        self.age += 1
        log = dict(policy=self.policy, signals=sig, age=self.age, events=self.events,
                   phase=self.phase, phase_age=self.phase_age, cooldown=self.cooldown,
                   clear_count=self.clear_count, event=event, executed=int(chosen))
        self.step += 1
        return int(chosen), log
