"""CPU reference math. Estimated risks are NOT certified safety."""
import math


def hazards(logits):
    if not logits or any(len(row) != 5 for row in logits):
        raise ValueError('Expected H x 5: survive, collision, success, failed_STOP, timeout')
    out = []
    for row in logits:
        if not all(math.isfinite(x) for x in row):
            raise ValueError('Nonfinite logits')
        m = max(row); e = [math.exp(x-m) for x in row]; z = sum(e)
        out.append([x/z for x in e])
    return out


def cumulative(logits):
    survival = 1.; incidence = [0.]*4; rows = []
    for q in hazards(logits):
        incidence = [v + survival*p for v,p in zip(incidence,q[1:])]
        survival *= q[0]
        rows.append({'survival':survival, 'events':incidence[:]})
    return rows


def nll(logits, observed_steps, event=None):
    """event 1..4 at observed_steps (1-based); None = right censoring.
    A zero-length censored window has zero likelihood contribution.
    """
    hazards(logits)  # validate schema/finite values
    if type(observed_steps) is not int or not 0 <= observed_steps <= len(logits):
        raise ValueError('Invalid exposure')
    if event is not None and (type(event) is not int or event not in range(1,5) or observed_steps == 0):
        raise ValueError('Invalid terminal event')
    loss = 0.
    for k,row in enumerate(logits[:observed_steps]):
        target = event if event is not None and k == observed_steps-1 else 0
        m = max(row)
        loss += m + math.log(sum(math.exp(v-m) for v in row)) - row[target]
    return loss


def interval_check(clearances, times, error_bound, lipschitz_bound, *, assumptions_verified=False):
    """Conditional grid certificate only. No sensor/envelope estimator is supplied.
    Both interval endpoints must be present. Empty/invalid envelopes return UNKNOWN.
    Failure of a sufficient condition does NOT prove a collision.
    """
    values = list(clearances)+list(times)+[error_bound,lipschitz_bound]
    if (not assumptions_verified or len(times)<2 or len(clearances)!=len(times)
        or not all(math.isfinite(v) for v in values) or error_bound<0 or lipschitz_bound<0
        or any(b<=a for a,b in zip(times,times[1:]))):
        return {'status':'UNKNOWN', 'reason':'missing_or_invalid_envelope_assumptions'}
    gap = max(b-a for a,b in zip(times,times[1:]))
    margin = min(clearances)-error_bound-lipschitz_bound*gap/2
    return {'status':'CONDITIONALLY_CERTIFIED' if margin>0 else 'UNKNOWN',
            'margin':margin, 'interval':[times[0],times[-1]],
            'recursive_safety':False}


def choose(falcon_action, estimates, risk_limit, supported_actions):
    """Empirical gate; STOP remains a terminal candidate, never WAIT.
    Each candidate has action, predicted collision risk, predicted success utility.
    No candidate passes -> original policy with explicit uncertified fallback.
    """
    if type(falcon_action) is not int or falcon_action not in range(4) or not 0<=risk_limit<=1:
        raise ValueError('Invalid action/threshold')
    seen=set(); feasible=[]
    for row in estimates:
        a,r,u=row['action'],row['collision'],row['success']
        if type(a) is not int or a not in range(4) or a in seen or not all(math.isfinite(x) and 0<=x<=1 for x in (r,u)):
            raise ValueError('Invalid/duplicate candidate')
        seen.add(a)
        if a in supported_actions and r<=risk_limit:feasible.append(row)
    if not feasible:return {'action':falcon_action,'status':'UNCERTIFIED_FALLBACK','empty_set':True}
    best=max(feasible,key=lambda x:(x['success'],x['action']==falcon_action,-x['collision'],-x['action']))
    return {'action':best['action'],'status':'RISK_ESTIMATED','empty_set':False}
