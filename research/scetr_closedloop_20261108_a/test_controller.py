import unittest
import numpy as np
from controller import Controller


class Checks(unittest.TestCase):
    def test_reference_and_stop_are_preserved(self):
        d=np.zeros((32,32),dtype=np.uint16)
        c=Controller('reference')
        self.assertEqual([c.act(d,a)[0] for a in [1,3,2,0]*5],[1,3,2,0]*5)
        for name in ['yield24','yield_clear','sidestep']:
            c=Controller(name)
            for _ in range(10):c.act(d,1)
            self.assertEqual(c.act(d,0)[0],0)

    def test_hold_pairs_and_finite_handoff(self):
        d=np.zeros((32,32),dtype=np.uint16)
        c=Controller('yield24')
        actions=[c.act(d,1)[0] for _ in range(33)]
        self.assertEqual(actions[:8],[1]*8)
        self.assertEqual(actions[8:32],[2,3]*12)
        self.assertEqual(actions[32],1)

    def test_clearance_release_uses_observations(self):
        blocked=np.zeros((32,32),dtype=np.uint16)
        clear=np.full((32,32),65535,dtype=np.uint16)
        c=Controller('yield_clear')
        for _ in range(9):c.act(blocked,1)
        events=[c.act(clear,1)[1]['event'] for _ in range(10)]
        self.assertIn('handoff',events)

    def test_blocked_yield_has_budget(self):
        d=np.zeros((32,32),dtype=np.uint16)
        c=Controller('yield_clear');actions=[c.act(d,1)[0] for _ in range(500)]
        self.assertEqual(c.events,3)
        self.assertEqual(sum(a!=1 for a in actions),192)
        self.assertEqual(actions[-1],1)

    def test_sidestep_does_not_advance_into_blocked_depth(self):
        d=np.zeros((32,32),dtype=np.uint16)
        c=Controller('sidestep');a=[c.act(d,1)[0] for _ in range(33)]
        self.assertEqual(a[8:20],[2]*12)
        self.assertEqual(a[20:32],[3]*12)
        self.assertEqual(a[-1],1)

if __name__=='__main__':unittest.main()
