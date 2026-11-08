import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from bridge import Bridge, ObservationAdapter, DEPTH, GOAL
from derive import derive


class Tensor(np.ndarray):
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.asarray(self)


def t(value): return np.asarray(value).view(Tensor)


class Data:
    def __init__(self):
        self.actions=t([[1.,19.]])
        self.take_actions=t([[1.,23.]])
    @property
    def env_actions(self): return self.take_actions


class Tests(unittest.TestCase):
    def test_whitelist(self):
        adapter=ObservationAdapter()
        inputs={DEPTH:np.ones((1,8,8,1)),GOAL:np.ones((1,2))}
        adapter.frame(inputs,np.ones((1,512)),np.zeros((1,4)))
        with self.assertRaises(RuntimeError):
            adapter.frame({**inputs,'human_future':np.zeros(1)},np.ones((1,512)),np.zeros((1,4)))
        with self.assertRaises(RuntimeError):
            adapter.frame(inputs,np.ones((1,511)),np.zeros((1,4)))

    def test_override_and_reset(self):
        with tempfile.TemporaryDirectory() as folder:
            os.environ.update(FALCON_PROBE_MODE='override',FALCON_PROBE_OUTPUT=folder)
            head=SimpleNamespace(in_features=512,out_features=4,register_forward_hook=lambda callback:SimpleNamespace(remove=lambda:None))
            actor=SimpleNamespace(action_distribution=SimpleNamespace(linear=head),state_dict=lambda:{'w':t([1.,2.])})
            b=Bridge(SimpleNamespace(_agents=[SimpleNamespace(_actor_critic=actor)]))
            previous=t([[0.,0.]])
            for i in range(8):
                batch={DEPTH:t(np.ones((1,8,8,1))),GOAL:t(np.ones((1,2))),'human_oracle':t([999.])}
                episode=SimpleNamespace(scene_id='scene',episode_id='b' if i>=7 else 'a')
                b.before([episode],batch,previous,t([[i not in (0,7)]]),t(np.zeros((1,2,4))))
                b.capture(None,(t(np.zeros((1,512))),),t(np.zeros((1,4))))
                action=Data();b.commit(action,[1,1]);previous=action.actions.copy()
                b.after([action.env_actions[0]],[(dict(batch),0.,i==6,{})],previous)
            self.assertEqual(b.rows[3]['executed'],2)
            self.assertEqual(b.rows[6]['executed'],0)
            self.assertEqual(b.reset_count,2)
            self.assertEqual(b.previous_checks,6)
            b.close()
            self.assertTrue(json.loads((Path(folder)/'bridge_result.json').read_text())['weights_unchanged'])

    def test_derivation(self):
        root=Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as tmp:
            derive(root/'baselines/repos/Falcon',Path(tmp)/'derived')
            text=(Path(tmp)/'derived/probe_evaluator.py').read_text()
            self.assertLess(text.index('probe.commit('),text.index('prev_actions.copy_'))
            self.assertLess(text.index('outputs = envs.step('),text.index('probe.after('))
            self.assertIn('envs.close()',text)


if __name__=='__main__': unittest.main()
