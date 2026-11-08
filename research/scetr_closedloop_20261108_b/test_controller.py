import unittest
import numpy as np
from controller import Controller,temporal_change

class Checks(unittest.TestCase):
    def test_static_wall_probe_is_dismissed(self):
        d=np.full((32,32),3000,dtype=np.uint16);c=Controller('probe_yield')
        rows=[c.act(d,1) for _ in range(14)]
        self.assertEqual([x[0] for x in rows[8:12]],[2,3,2,3])
        self.assertEqual(rows[12][1]['event'],'static_dismiss');self.assertEqual(rows[12][0],1)
    def test_unknown_ignores_valid_near_wall(self):
        d=np.full((32,32),3000,dtype=np.uint16);c=Controller('unknown_yield')
        self.assertTrue(all(c.act(d,1)[0]==1 for _ in range(100)))
    def test_dynamic_evidence_and_clearance(self):
        a=np.zeros((32,32),dtype=np.uint16);b=a.copy();b[6:25,8:24]=5000
        self.assertGreater(temporal_change(a,b),.02)
        c=Controller('probe_yield')
        for _ in range(10):c.act(a,1)
        for _ in range(2):c.act(b,1)
        _,r=c.act(b,1);self.assertEqual(r['event'],'dynamic_evidence')
        clear=np.full((32,32),40000,dtype=np.uint16)
        rows=[c.act(clear,1) for _ in range(12)]
        self.assertIn('clear_handoff',[r[1]['event'] for r in rows])
    def test_stop_preserved_and_no_future_input(self):
        d=np.zeros((32,32),dtype=np.uint16)
        for p in ['unknown_yield','probe_yield','probe_escape']:
            c=Controller(p)
            for _ in range(12):c.act(d,1)
            self.assertEqual(c.act(d,0)[0],0)

if __name__=='__main__':unittest.main()
