"""Exact rational, finite-grid robust semi-Markov dynamic programming.

Nature is history/action rectangular. Deterministic option choices only;
successor conditional risk budgets are chosen on a declared finite grid.
A feasible result is conditional on the supplied kernels, NOT a sensor or
learned-model coverage certificate. No unsafe default policy is provided.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from types import MappingProxyType


def rational(x):
    # Floats deliberately rejected: the caller must declare decimal intent.
    if isinstance(x, bool) or not isinstance(x, (int, str, F)):
        raise ValueError('Use integer, decimal string or Fraction, not float')
    return F(x)


@dataclass(frozen=True, order=True)
class Outcome:
    duration: int
    event: str
    next_state: str = ''

    def __post_init__(self):
        if type(self.duration) is not int or self.duration < 1:
            raise ValueError('Each option outcome must advance time')
        if self.event not in ('C', 'G', 'F', 'R'):
            raise ValueError('Unknown/disallowed terminal category')
        if not isinstance(self.next_state, str) or (self.event == 'R') != bool(self.next_state):
            raise ValueError('Only a nonterminal handoff has a successor state')


@dataclass(frozen=True)
class Option:
    name: str
    intervention: bool
    # Vertices of a convex polytope of probability distributions.
    kernels: tuple

    def __post_init__(self):
        if not self.name or type(self.intervention) is not bool or not self.kernels:
            raise ValueError('Invalid option')
        normalized = []
        for kernel in self.kernels:
            rows = tuple((o, rational(p)) for o, p in kernel)
            if not rows or len({o for o,p in rows}) != len(rows):
                raise ValueError('Empty kernel or duplicate outcome')
            if any(not isinstance(o, Outcome) or p < 0 for o,p in rows) or sum(p for o,p in rows) != 1:
                raise ValueError('Kernel must be a probability distribution')
            normalized.append(rows)
        object.__setattr__(self, 'kernels', tuple(normalized))


@dataclass(frozen=True)
class Model:
    # Explicit horizon-conditioned kernels; no truncating long options silently.
    rows: dict
    rectangular: bool
    provenance: str

    def __post_init__(self):
        if self.rectangular is not True:
            raise ValueError('Nonrectangular models need augmented information state or an explicit outer approximation')
        if not self.provenance:
            raise ValueError('Model provenance/assumptions required')
        object.__setattr__(self, 'rows', MappingProxyType({k: tuple(v) for k,v in self.rows.items()}))
        for (state, t), options in self.rows.items():
            if not isinstance(state,str) or not state or type(t) is not int or t<1:
                raise ValueError('Invalid model information-state/horizon key')
            if len({o.name for o in options}) != len(options):
                raise ValueError('Duplicate option name')
            if any(out.duration>t for opt in options for kernel in opt.kernels for out,p in kernel if p):
                raise ValueError('Outcome exceeds remaining time; provide a correct horizon-specific kernel')


@dataclass(frozen=True)
class Plan:
    state: str
    budget: F
    interventions_left: int
    time_left: int
    option: Option | None
    # Observed (duration, R, state) determines the committed child policy/budget.
    children: tuple
    risk_bound: F
    utility: F
    scope: str = 'CONDITIONAL_ON_SUPPLIED_RECTANGULAR_MODEL'


class Infeasible(Exception):
    """No feasible deterministic policy on the supplied budget grid."""


class SearchLimit(Exception):
    """Search not completed; this is NOT evidence of infeasibility."""


class Planner:
    def __init__(self, model, budgets, intervention_cost='0', max_allocations=100000):
        self.model = model
        self.grid = tuple(sorted(set(rational(x) for x in budgets)))
        if not self.grid or self.grid[0]<0 or self.grid[-1]>1:
            raise ValueError('Risk budget grid outside [0,1]')
        self.cost = rational(intervention_cost)
        if self.cost<0 or type(max_allocations) is not int or max_allocations<1:
            raise ValueError('Invalid cost/search bound')
        self.limit=max_allocations; self.cache={}

    def solve(self, state, budget, interventions_left, time_left):
        budget=rational(budget)
        if not 0<=budget<=1 or type(interventions_left) is not int or interventions_left<0 or type(time_left) is not int or time_left<0:
            raise ValueError('Invalid augmented state')
        key=(state,budget,interventions_left,time_left)
        if key in self.cache:
            result=self.cache[key]
            if result is None:raise Infeasible(str(key))
            return result
        if time_left==0:
            result=Plan(state,budget,interventions_left,0,None,(),F(0),F(0))
            self.cache[key]=result;return result
        if (state,time_left) not in self.model.rows:
            raise ValueError('Missing model row; missing data is not infeasibility')
        best=None; checked=0
        for option in self.model.rows[(state,time_left)]:
            n=interventions_left-int(option.intervention)
            if n<0:continue
            handoffs=sorted({o for k in option.kernels for o,p in k if p and o.event=='R'})
            choices=[]
            for o in handoffs:
                allowed=[]
                for b in self.grid:
                    try:allowed.append(self.solve(o.next_state,b,n,time_left-o.duration))
                    except Infeasible:pass
                choices.append(allowed)
            for allocation in product(*choices):
                checked+=1
                if checked>self.limit:raise SearchLimit('Budget allocation enumeration limit reached')
                children=dict(zip(handoffs,allocation))
                risk=max(sum(p*(1 if o.event=='C' else children[o].budget if o.event=='R' else 0)
                             for o,p in k if p) for k in option.kernels)
                if risk>budget:continue
                value=min(sum(p*(1 if o.event=='G' else children[o].utility if o.event=='R' else 0)
                              for o,p in k if p) for k in option.kernels)-self.cost*int(option.intervention)
                candidate=Plan(state,budget,interventions_left,time_left,option,tuple(children.items()),risk,value)
                if best is None or (value,-risk,not option.intervention)>(best.utility,-best.risk_bound,not best.option.intervention):
                    best=candidate
        self.cache[key]=best
        if best is None:raise Infeasible(str(key))
        return best
