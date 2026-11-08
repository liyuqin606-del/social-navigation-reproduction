"""Independent tree evaluator: uses committed policy and model, not cached values."""
from fractions import Fraction as F


def require(condition):
    if not condition:
        raise ValueError('Policy tree violates its model, budget, or Bellman certificate')

def audit(plan, model, intervention_cost=F(0)):
    if plan.time_left==0:
        require(plan.option is None and not plan.children)
        require(plan.risk_bound==plan.utility==0)
        return {'collision_upper':F(0),'success_lower':F(0),'utility_lower':F(0),'max_interventions':0}
    require(plan.option in model.rows[(plan.state,plan.time_left)])
    opt=plan.option; children=dict(plan.children)
    expected={o for k in opt.kernels for o,p in k if p and o.event=='R'}
    require(len(children)==len(plan.children) and set(children)==expected)
    results={}
    for o,child in children.items():
        require(child.state==o.next_state and child.time_left==plan.time_left-o.duration)
        require(child.interventions_left==plan.interventions_left-int(opt.intervention))
        require(0<=child.budget<=1)
        results[o]=audit(child,model,intervention_cost)
    risks=[];successes=[];utilities=[];budgets=[]
    for k in opt.kernels:
        c=g=v=b=F(0)
        for o,p in k:
            if not p:continue
            if o.event=='C':c+=p;b+=p
            elif o.event=='G':g+=p;v+=p
            elif o.event=='R':
                c+=p*results[o]['collision_upper'];g+=p*results[o]['success_lower']
                v+=p*results[o]['utility_lower'];b+=p*children[o].budget
        risks.append(c);successes.append(g);utilities.append(v-intervention_cost*int(opt.intervention));budgets.append(b)
    require(max(risks)<=max(budgets)==plan.risk_bound<=plan.budget)
    require(min(utilities)==plan.utility)
    count=int(opt.intervention)+max([r['max_interventions'] for r in results.values()]+[0])
    require(count<=plan.interventions_left)
    return {'collision_upper':max(risks),'success_lower':min(successes),'utility_lower':min(utilities),'max_interventions':count}
