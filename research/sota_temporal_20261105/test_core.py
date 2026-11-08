import math, unittest
from core import cumulative,nll,interval_check,choose
class ReferenceTests(unittest.TestCase):
 def test_mass_monotonicity(self):
  rows=cumulative([[0,1,-2,3,-4]]*32)
  for i,r in enumerate(rows):
   self.assertAlmostEqual(r['survival']+sum(r['events']),1)
   if i:self.assertTrue(all(x>=y for x,y in zip(r['events'],rows[i-1]['events'])))
 def test_censoring_and_event(self):
  self.assertAlmostEqual(nll([[0]*5]*4,2),2*math.log(5))
  self.assertAlmostEqual(nll([[0]*5]*4,3,2),3*math.log(5))
  self.assertEqual(nll([[0]*5],0),0)
 def test_stability(self):self.assertTrue(math.isfinite(nll([[1000,-1000,0,0,0]],1,1)))
 def test_invalid_labels(self):
  for steps,event in [(0,1),(-1,None),(2,None),(1,5)]:
   with self.assertRaises(ValueError):nll([[0]*5],steps,event)
 def test_endpoint_only_cannot_certify_crossing(self):
  # d(t)=abs(2t-1)-0.1: safe endpoints, unsafe midpoint; L=2.
  self.assertEqual(interval_check([.9,.9],[0,1],0,2,assumptions_verified=True)['status'],'UNKNOWN')
 def test_conditional_only(self):
  self.assertEqual(interval_check([2,2],[0,1],0,2)['status'],'UNKNOWN')
  self.assertEqual(interval_check([2,2],[0,1],0,2,assumptions_verified=True)['status'],'CONDITIONALLY_CERTIFIED')
 def test_fallback_not_stop_or_certificate(self):
  self.assertEqual(choose(2,[{'action':1,'collision':.9,'success':.9}],.1,{1})['action'],2)
  self.assertEqual(choose(2,[],.1,set())['status'],'UNCERTIFIED_FALLBACK')
 def test_failed_stop_cannot_win_with_zero_success(self):
  a=choose(1,[{'action':0,'collision':0.,'success':0.},{'action':1,'collision':.05,'success':.7}],.1,{0,1})
  self.assertEqual(a['action'],1)
 def test_duplicate_candidates(self):
  r={'action':1,'collision':0.,'success':.5}
  with self.assertRaises(ValueError):choose(1,[r,r],.1,{1})
if __name__=='__main__':unittest.main()
