import unittest
import numpy as np
from core import (H, N, Net, features, mask_for, softmax, prefix_loss, consistency_loss,
                  conditional_truth, components, cutoff_cost, evaluate, training_data, prefixes)


class Checks(unittest.TestCase):
    def test_capacity(self):
        for output, hidden in [(3,131),(N,32)]:
            net=Net(output,hidden,1)
            self.assertEqual(sum(x.size for x in net.p),1313)

    def test_resume_and_mass(self):
        truth=conditional_truth(np.array([[.2,-.3],[.7,.9]]))
        np.testing.assert_allclose(truth.sum(axis=-1),1,atol=1e-12)
        for age in range(H+1):
            self.assertTrue(np.all(truth[:,age,:,:2*age]==0))
        self.assertTrue(np.all(truth[:,H,:,-1]==1))
        np.testing.assert_allclose(components().sum(axis=-1),1,atol=1e-12)

    def test_true_kernel_prefix_identity(self):
        truth=conditional_truth(np.array([[.2,-.3],[.7,.9]]))
        for age in range(H):
            p=truth[:,age].copy()
            p[...,:2*(age+1)]=0
            p/=p.sum(axis=-1,keepdims=True)
            np.testing.assert_allclose(p,truth[:,age+1],atol=1e-12)

    def test_gradients(self):
        # Central finite differences independently check both gradient paths.
        net=Net(N,5,8)
        ages=np.array([0,3,7])
        x=features(np.array([[.2,.4],[-.5,.7],[.1,-.3]]),ages,np.array([0,1,2]))
        for objective in [lambda: prefix_loss(net,x,ages,np.array([3,19,32])),
                          lambda: consistency_loss(net,x,ages)]:
            _, grads=objective()
            for p,g in zip(net.p,grads):
                for index in [0,p.size//2,p.size-1]:
                    old=p.flat[index]
                    p.flat[index]=old+1e-5
                    plus=objective()[0]
                    p.flat[index]=old-1e-5
                    minus=objective()[0]
                    p.flat[index]=old
                    self.assertAlmostEqual((plus-minus)/2e-5,g.flat[index],places=7)

    def test_oracle_against_independent_step_dp(self):
        truth=conditional_truth(np.random.default_rng(5).uniform(-1,1,(31,2)))
        values=np.full((len(truth),3),17.)
        for age in reversed(range(H)):
            p=truth[:,age]
            g,b=p[...,2*age],p[...,2*age+1]
            values=np.minimum(17,1+g+60*b+(1-g-b)*values)
        result=evaluate(truth,truth)
        np.testing.assert_allclose(result['cost'],values.min(axis=-1),atol=1e-10)
        self.assertLess(result['consistency'],1e-25)
        self.assertAlmostEqual(result['kernel_kl'],0)

    def test_no_future_prefix_and_censoring(self):
        records=training_data(30,7)
        rows=prefixes(records)
        self.assertEqual(rows.shape[1],6)
        self.assertTrue(np.all(mask_for(rows[:,3])[np.arange(len(rows)),rows[:,4].astype(int)]))
        for c0,c1,m,t,k in records:
            group=rows[(rows[:,0]==c0)&(rows[:,1]==c1)&(rows[:,2]==m)]
            self.assertEqual(len(group),min(int(t),H))
            if t>H:
                self.assertTrue(np.all(group[:,4]==N-1))
                self.assertTrue(np.all(group[:,5]==2))


if __name__=='__main__':
    unittest.main()
