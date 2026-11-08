from fractions import Fraction as F
from itertools import product
import unittest
from eventnav import Outcome as O, Option, Model, Planner, Infeasible, SearchLimit
from eventnav.audit import audit
from eventnav.demo import fixture
from eventnav.execution import PolicyCursor

class TheoryTests(unittest.TestCase):
 def test_chain_risk_not_reset(self):
  m=fixture();p=Planner(m,['0','.05','.08','.1','1']).solve('start','.1',1,2)
  self.assertEqual(audit(p,m)['collision_upper'],F('0.0984'))
  self.assertEqual(p.option.name,'falcon_continue')
 def test_risk_budget_changes_switch(self):
  m=fixture();p=Planner(m,['0','.05','.08','.1','1']).solve('start','.09',1,2)
  self.assertEqual(p.option.name,'short_detour');self.assertEqual(p.utility,F('.8'))
 def test_no_intervention_budget(self):
  with self.assertRaises(Infeasible):Planner(fixture(),['0','.08','.1']).solve('start','.09',0,2)
 def test_stop_is_failure_terminal(self):
  s=Option('STOP',True,(((O(1,'F'),1),),));m=Model({('s',1):(s,)},True,'test')
  p=Planner(m,[0]).solve('s',0,1,1)
  self.assertEqual(audit(p,m)['success_lower'],0);self.assertFalse(p.children)
 def test_option_duration(self):
  a=Option('turn_then_go',True,(((O(2,'G'),1),),));m=Model({('s',2):(a,)},True,'test')
  self.assertEqual(Planner(m,[0]).solve('s',0,1,2).utility,1)
  with self.assertRaises(ValueError):Model({('s',1):(a,)},True,'test')
 def test_no_zero_time_or_postterminal_state(self):
  for args in [(0,'R','s'),(1,'G','s'),(1,'R','')]:
   with self.assertRaises(ValueError):O(*args)
 def test_nonrectangular_rejected(self):
  with self.assertRaises(ValueError):Model({},False,'shared neural parameters')
 def test_bad_probability_and_float(self):
  for weights in [('0.5','0.6'),('-0.1','1.1'),(.5,.5)]:
   with self.assertRaises(ValueError):Option('bad',False,(((O(1,'G'),weights[0]),(O(1,'F'),weights[1])),))
 def test_robust_max_and_min_over_vertices(self):
  a=Option('uncertain',False,(((O(1,'C'),'.1'),(O(1,'G'),'.9')),((O(1,'C'),'.3'),(O(1,'G'),'.7'))))
  m=Model({('s',1):(a,)},True,'two-vertex uncertainty')
  with self.assertRaises(Infeasible):Planner(m,['0','.3']).solve('s','.2',0,1)
  p=Planner(m,['0','.3']).solve('s','.3',0,1)
  self.assertEqual(audit(p,m)['collision_upper'],F('.3'));self.assertEqual(p.utility,F('.7'))
 def test_missing_kernel_is_not_infeasible(self):
  with self.assertRaises(ValueError):Planner(Model({},True,'test'),[0]).solve('missing',0,0,1)
 def test_cost_paid_once_per_intervention(self):
  m=fixture();p=Planner(m,[0,'.05','.08','.1'],intervention_cost='.07').solve('start','.09',1,2)
  self.assertEqual(audit(p,m,F('.07'))['utility_lower'],F('.73'))
 def test_committed_successor_budget(self):
  m=fixture();p=Planner(m,[0,'.08','.1']).solve('start','.1',1,2);c=PolicyCursor(p)
  c.observe(O(1,'R','hall'));self.assertEqual(c.plan.budget,F('.08'))
  c.observe(O(1,'G'));self.assertTrue(c.closed)
  with self.assertRaises(RuntimeError):c.option()
 def test_out_of_support_halts(self):
  p=Planner(fixture(),[0,'.08','.1']).solve('start','.1',1,2);c=PolicyCursor(p)
  with self.assertRaises(ValueError):c.observe(O(1,'R','unseen'))
  with self.assertRaises(RuntimeError):c.option()
 def test_no_risk_tolerance_slack(self):
  a=Option('tiny',False,(((O(1,'C'),'0.100000000000001'),(O(1,'G'),'0.899999999999999')),))
  with self.assertRaises(Infeasible):Planner(Model({('s',1):(a,)},True,'test'),[0]).solve('s','.1',0,1)
 def test_grid_infeasible_is_not_global_infeasible(self):
  m=fixture()
  with self.assertRaises(Infeasible):Planner(m,[0,1]).solve('start','.1',0,2)
  self.assertEqual(Planner(m,[0,'.08',1]).solve('start','.1',0,2).option.name,'falcon_continue')
 def test_search_limit_is_distinct(self):
  with self.assertRaises(SearchLimit):Planner(fixture(),[0,'.08','.1',1],max_allocations=1).solve('start','.1',0,2)
 def test_zero_horizon(self):
  p=Planner(Model({},True,'deadline'),[0]).solve('s',0,0,0)
  self.assertEqual(audit(p,Model({},True,'deadline'))['success_lower'],0)
 def test_tampered_certificate_rejected(self):
  from dataclasses import replace
  m=fixture();p=Planner(m,[0,'.08','.1']).solve('start','.1',1,2)
  with self.assertRaises(ValueError):audit(replace(p,risk_bound=F(0)),m)
 def test_independent_exhaustive_policies(self):
  # All three deterministic policies of this two-stage tree, computed directly.
  policies=[('continue_then_continue',F('.02')+F('.98')*F('.08'),F('.98')*F('.92')),
            ('continue_then_stop',F('.02'),F(0)),('detour',F('.05'),F('.8'))]
  for b in ['.02','.05','.09','.1','1']:
   expected=max(v for _,risk,v in policies if risk<=F(b))
   m=fixture();p=Planner(m,[0,'.05','.08','.1',1]).solve('start',b,1,2)
   self.assertEqual(p.utility,expected);audit(p,m)
 def test_all_handoff_branches_need_feasible_continuation(self):
  root=Option('branch',False,(((O(1,'R','a'),'.5'),(O(1,'R','b'),'.5')),))
  safe=Option('safe',False,(((O(1,'G'),1),),))
  m=Model({('s',2):(root,),('a',1):(safe,),('b',1):()},True,'test')
  with self.assertRaises(Infeasible):Planner(m,[0,1]).solve('s',1,0,2)
 def test_rectangular_branch_adversary(self):
  root=Option('branch',False,(((O(1,'R','a'),'.25'),(O(1,'R','b'),'.75')),((O(1,'R','a'),'.75'),(O(1,'R','b'),'.25'))))
  a=Option('a',False,(((O(1,'C'),'.1'),(O(1,'G'),'.9')),))
  b=Option('b',False,(((O(1,'C'),'.2'),(O(1,'G'),'.8')),))
  m=Model({('s',2):(root,),('a',1):(a,),('b',1):(b,)},True,'test')
  p=Planner(m,[0,'.1','.2',1]).solve('s','.18',0,2)
  result=audit(p,m)
  self.assertEqual(result['collision_upper'],F('.175'))
  self.assertEqual(result['success_lower'],F('.825'))
 def test_kernel_mapping_immutable(self):
  m=fixture()
  with self.assertRaises(TypeError):m.rows[('start',2)]=()
if __name__=='__main__':unittest.main()
