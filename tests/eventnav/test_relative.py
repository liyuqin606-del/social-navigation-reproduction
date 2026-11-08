import unittest,itertools
from fractions import Fraction as F
from eventnav.relative import improve,advantage_bounds,simplex_library
class RelativeTests(unittest.TestCase):
 def test_uncertainty_cancels(self):
  p=(F(1,2),F(1,2));self.assertEqual(advantage_bounds(p,p,[(0,1),(0,1)]),(0,0))
 def test_baseline_always_feasible(self):
  r=improve([1,0],[(0,1)]*2,[(0,1)]*2,[[0,1]])
  self.assertEqual(r['distribution'],(1,0));self.assertFalse(r['absolute_risk_guarantee'])
 def test_dominant_candidate(self):
  r=improve([1,0],[('.5','.6'),('.8','.9')],[('.2','.3'),('.05','.1')],[[0,1]])
  self.assertEqual(r['distribution'],(0,1));self.assertEqual(r['success_advantage_lower'],F('.2'))
 def test_stop_does_not_win(self):
  r=improve([1,0],[('.7','.9'),(0,0)],[('.1','.3'),(0,0)],[[0,1]])
  self.assertFalse(r['changed'])
 def test_support_gate(self):
  r=improve([1,0],[(0,0),(1,1)],[(1,1),(0,0)],[[0,1]],supported={0})
  self.assertFalse(r['changed'])
 def test_mixture_can_be_feasible_when_pure_choices_are_not(self):
  # A improves success but raises risk; B reduces risk but lowers success.
  r=improve([1,0,0],[('.5','.5'),('.8','.8'),('.4','.4')],[('.2','.2'),('.3','.3'),('.1','.1')],simplex_library(3,2))
  self.assertEqual(r['distribution'],(0,F(1,2),F(1,2)))
 def test_bounds_match_all_box_vertices(self):
  p=(F(1,4),F(3,4));p0=(F(2,3),F(1,3));bounds=[(F('.1'),F('.4')),(F('.2'),F('.9'))]
  values=[sum((a-b)*q for a,b,q in zip(p,p0,v)) for v in itertools.product(*bounds)]
  self.assertEqual(advantage_bounds(p,p0,bounds),(min(values),max(values)))
 def test_input_validation(self):
  with self.assertRaises(ValueError):improve([1,0],[(0,1)],[(0,1)]*2,[[0,1]])
if __name__=='__main__':unittest.main()
