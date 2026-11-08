"""Finite numerical sanity checks, not proofs or navigation experiments."""
import itertools, math, random, json
from pathlib import Path
rng=random.Random(606)

def simplex(n):
 x=[rng.random() for _ in range(n)];return [v/sum(x) for v in x]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
# Box uncertainty: exhaust all 2^4 corners.
for _ in range(100):
 p,p0=simplex(4),simplex(4);d=[x-y for x,y in zip(p,p0)]
 q=[rng.random() for _ in range(4)];e=[.05*rng.random() for _ in range(4)]
 corners=[dot(d,[x+s*y for x,y,s in zip(q,e,signs)]) for signs in itertools.product([-1,1],repeat=4)]
 center=dot(d,q);radius=dot(list(map(abs,d)),e)
 assert math.isclose(min(corners),center-radius,abs_tol=1e-12)
 assert math.isclose(max(corners),center+radius,abs_tol=1e-12)
# Summation-by-parts bound on subdistributions with residual survival mass.
for _ in range(100):
 a,b=simplex(9),simplex(9);u=[rng.uniform(-2,2) for _ in range(9)]
 fa=[sum(a[:k]) for k in range(1,9)];fb=[sum(b[:k]) for k in range(1,9)]
 v=[x-u[-1] for x in u[:-1]]
 eps=max(abs(x-y) for x,y in zip(fa,fb));variation=abs(v[-1])+sum(abs(y-x) for x,y in zip(v,v[1:]))
 assert abs(dot(a,u)-dot(b,u))<=eps*variation+1e-12
# Two-stage tree: choose actions with nonnegative success advantage and
# nonpositive collision advantage, evaluated under the fixed baseline.
for _ in range(100):
 terminal=[ [simplex(3) for a in range(4)] for s in range(4)] # G,C,F
 base=[simplex(4) for s in range(5)]
 vg=[sum(base[s+1][a]*terminal[s][a][0] for a in range(4)) for s in range(4)]
 vc=[sum(base[s+1][a]*terminal[s][a][1] for a in range(4)) for s in range(4)]
 def pick(p0,g,c):
  options=[p0]+[[float(i==a) for i in range(4)] for a in range(4)]
  feasible=[p for p in options if dot(p,g)>=dot(p0,g)-1e-12 and dot(p,c)<=dot(p0,c)+1e-12]
  return max(feasible,key=lambda p:dot(p,g))
 newroot=pick(base[0],vg,vc)
 leaf=[pick(base[s+1],[t[0] for t in terminal[s]],[t[1] for t in terminal[s]]) for s in range(4)]
 ng=[sum(leaf[s][a]*terminal[s][a][0] for a in range(4)) for s in range(4)]
 nc=[sum(leaf[s][a]*terminal[s][a][1] for a in range(4)) for s in range(4)]
 assert dot(newroot,ng)>=dot(base[0],vg)-1e-12
 assert dot(newroot,nc)<=dot(base[0],vc)+1e-12
r={'status':'PASS_FINITE_SANITY_CHECKS_NOT_PROOF_OR_NAVIGATION','seed':606,'cases':300,'groups':['box_support_exact_at_vertices','cdf_variation_error_bound','two_stage_componentwise_policy_improvement']}
Path(__file__).with_name('THEORY_CHECK.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
