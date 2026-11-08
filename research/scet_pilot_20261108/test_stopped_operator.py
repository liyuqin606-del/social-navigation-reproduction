from fractions import Fraction as F
import unittest
from stopped_operator import Boundary as B,StoppedKernel as K


class Checks(unittest.TestCase):
    def test_resume_transitivity(self):
        k=K(10,0,(B(2,'goal','',F(1,3)),B(7,'goal','',F(1,3)),B(10,'timeout','',F(1,3))))
        self.assertEqual(k.resume(3).resume(2),k.resume(5))
        self.assertEqual(k.resume(5).remaining,5)

    def test_event_at_elapsed_is_not_survival(self):
        k=K(4,0,(B(3,'collision','',F(1,2)),B(4,'timeout','',F(1,2))))
        r=k.resume(3)
        self.assertEqual(r.boundaries,(B(1,'timeout','',F(1)),))
        with self.assertRaises(ValueError):
            k.resume(4)

    def test_zero_survival_rejected(self):
        k=K(4,0,(B(2,'goal','',F(1)),))
        with self.assertRaises(ValueError):
            k.resume(2)

    def test_future_information_value(self):
        k=K(5,0,(B(1,'observe','left-clear',F(1,2)),B(1,'observe','right-clear',F(1,2))))
        cost={(1,'left-clear'):{'left':F(0),'right':F(10)},
              (1,'right-clear'):{'left':F(10),'right':F(0)}}
        self.assertEqual(k.observation_value(cost,{}),1)
        # Same observation cost, but commit to one route before seeing signal.
        collapsed=1+min(sum(b.probability*cost[(b.ticks,b.information)][a] for b in k.boundaries)
                        for a in ['left','right'])
        self.assertEqual(collapsed,6)

    def test_composition_preserves_risk_and_deadline(self):
        first=K(5,0,(B(2,'observe','signal',F(3,4)),B(1,'collision','',F(1,4))))
        nxt=K(5,2,(B(1,'goal','',F(2,3)),B(3,'timeout','',F(1,3))))
        result=first.compose({(2,'signal'):nxt})
        self.assertEqual(sum(b.probability for b in result.boundaries if b.kind=='goal'),F(1,2))
        self.assertEqual(sum(b.probability for b in result.boundaries if b.kind=='collision'),F(1,4))
        self.assertEqual(sum(b.probability for b in result.boundaries if b.kind=='timeout'),F(1,4))
        with self.assertRaises(ValueError):
            first.compose({(2,'signal'):K(5,0,(B(1,'goal','',F(1)),))})

    def test_hidden_duplicates_share_observable_continuation(self):
        k=K(5,0,(B(1,'observe','same',F(1,3)),B(1,'observe','same',F(2,3))))
        nxt=K(5,1,(B(1,'collision','',F(1)),))
        result=k.compose({(1,'same'):nxt})
        self.assertEqual(sum(b.probability for b in result.boundaries if b.kind=='collision'),1)


if __name__=='__main__':
    unittest.main()
