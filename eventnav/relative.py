"""Relative robust improvement over an explicit FINITE distribution library.

Exact rational box bounds; not a continuous-simplex LP solver. Success and
collision concern the same full-horizon continuation convention for every
candidate. No absolute collision probability certificate is implied.
"""
from .planner import rational
from fractions import Fraction as F
from itertools import product


def distribution(p):
    p=tuple(rational(x) for x in p)
    if not p or min(p)<0 or sum(p)!=1:raise ValueError('Invalid distribution')
    return p


def intervals(rows,n):
    rows=tuple((rational(a),rational(b)) for a,b in rows)
    if len(rows)!=n or any(not 0<=a<=b<=1 for a,b in rows):raise ValueError('Invalid probability intervals')
    return rows


def advantage_bounds(p,p0,bounds):
    """One shared value vector: baseline uncertainty cancels exactly at p=p0."""
    d=[a-b for a,b in zip(p,p0)]
    return (sum(v*(lo if v>=0 else hi) for v,(lo,hi) in zip(d,bounds)),
            sum(v*(hi if v>=0 else lo) for v,(lo,hi) in zip(d,bounds)))


def simplex_library(actions,denominator=4):
    if type(actions) is not int or actions<1 or type(denominator) is not int or denominator<1:
        raise ValueError('Invalid finite library dimensions')
    return tuple(tuple(F(i,denominator) for i in row) for row in product(range(denominator+1),repeat=actions) if sum(row)==denominator)


def improve(baseline,success_intervals,collision_intervals,candidates,supported=None):
    p0=distribution(baseline);n=len(p0)
    g=intervals(success_intervals,n);c=intervals(collision_intervals,n)
    support=set(range(n)) if supported is None else set(supported)
    if not support<=set(range(n)):raise ValueError('Invalid support indices')
    library={p0}
    for p in candidates:
        p=distribution(p)
        if len(p)!=n:raise ValueError('Candidate dimension mismatch')
        library.add(p)
    best=p0;best_gain=F(0);best_risk=F(0);best_change=F(0)
    for p in sorted(library):
        if any(p[i]!=p0[i] for i in range(n) if i not in support):continue
        gain=advantage_bounds(p,p0,g)[0];risk=advantage_bounds(p,p0,c)[1]
        if gain<0 or risk>0:continue
        change=sum(abs(a-b) for a,b in zip(p,p0))
        if (gain,-risk,-change)>(best_gain,-best_risk,-best_change):
            best,best_gain,best_risk,best_change=p,gain,risk,change
    return {'distribution':best,'success_advantage_lower':best_gain,'collision_advantage_upper':best_risk,
            'changed':best!=p0,'scope':'RELATIVE_IMPROVEMENT_CONDITIONAL_ON_JOINT_VALUE_INTERVALS_FINITE_LIBRARY',
            'absolute_risk_guarantee':False}
