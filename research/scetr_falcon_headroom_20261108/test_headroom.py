import ast
from pathlib import Path
import unittest
from bridge import selected_action
from validate import compare_prefix


class Checks(unittest.TestCase):
    def test_override_and_release(self):
        seq=[2,2,2,1,1,1,1]
        self.assertEqual([selected_action(t,3,5,seq) for t in range(14)],[3]*5+seq+[3]*2)
        self.assertEqual(selected_action(5,0,-1,[]),0)

    def test_prefix_detects_future_mismatch(self):
        a=[dict(step=0,x=1),dict(step=1,x=2)]
        compare_prefix(a,a,2)
        with self.assertRaises(AssertionError): compare_prefix(a,[a[0],dict(step=1,x=3)],2)

    def test_current_state_matched_before_override(self):
        keys=['episode','observations_sha','goal','reset','previous_action','recurrent_sha','rng_sha','proposed','human_actions_sha','feature_sha','logit_sha']
        a=[dict.fromkeys(keys,0)];b=[dict.fromkeys(keys,0)]
        b[0]['executed']=2
        compare_prefix(a,b,0,True)
        b[0]['rng_sha']=1
        with self.assertRaises(AssertionError): compare_prefix(a,b,0,True)

    def test_syntax_and_exact_source_derivation(self):
        here=Path(__file__).resolve().parent
        for p in here.glob('*.py'): ast.parse(p.read_text())
        from derive import derive
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            derive(here.parents[1]/'baselines/repos/Falcon',Path(d)/'derived')
            text=(Path(d)/'derived/probe_evaluator.py').read_text()
            self.assertIn('probe.before(',text);self.assertIn('probe.commit(',text);self.assertIn('probe.after(',text)


if __name__=='__main__': unittest.main()
