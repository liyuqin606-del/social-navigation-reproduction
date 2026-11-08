from fractions import Fraction as F
from .planner import Outcome as O, Option, Model, Planner
from .audit import audit

def fixture():
    root=Option('falcon_continue',False,(((O(1,'C'),'0.02'),(O(1,'R','hall'),'0.98')),))
    leaf=Option('falcon_continue',False,(((O(1,'C'),'0.08'),(O(1,'G'),'0.92')),))
    stop=Option('failed_STOP',True,(((O(1,'F'),'1'),),))
    detour=Option('short_detour',True,(((O(1,'C'),'0.05'),(O(1,'G'),'0.80'),(O(1,'F'),'0.15')),))
    return Model({('start',2):(root,detour),('hall',1):(leaf,stop)},True,'synthetic exact finite event kernels; no Habitat observations')

def main():
    import json
    m=fixture();p=Planner(m,['0','.05','.08','.1','1'])
    results=[]
    for budget in ['.10','.09']:
        plan=p.solve('start',budget,1,2);r=audit(plan,m)
        results.append({'budget':budget,'option':plan.option.name,**{k:str(v) for k,v in r.items()},'scope':plan.scope})
    print(json.dumps({'synthetic_only':True,'results':results},indent=2))
if __name__=='__main__':main()
