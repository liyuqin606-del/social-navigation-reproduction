"""Small geometric predictor from existing depth, pointgoal and executed actions.

Approximates a level pinhole camera attached rigidly to the robot base.
Unknown camera lever arm and quantization mean residuals are not certificates.
"""
from collections import deque
import numpy as np

def goal_vector(goal):
    r,phi=goal
    return np.array([-r*np.sin(phi),r*np.cos(phi)],dtype=np.float64)

def warp(depth,old_goal,goal,yaw):
    z=np.asarray(depth,dtype=np.float64)/65535*10
    v,u=np.indices(z.shape);x=(u-15.5)/16*z;y=(v-15.5)/16*z
    R=np.array([[np.cos(yaw),np.sin(yaw)],[-np.sin(yaw),np.cos(yaw)]])
    translation=goal_vector(old_goal)-R.T@goal_vector(goal)
    xz=R@(np.stack([x.ravel(),z.ravel()])-translation[:,None])
    zz=xz[1];safe=np.maximum(zz,.01)
    uu=np.rint(16*xz[0]/safe+15.5).astype(int)
    vv=np.rint(16*y.ravel()/safe+15.5).astype(int)
    valid=(z.ravel()>.01)&(zz>.05)&(uu>=0)&(uu<32)&(vv>=0)&(vv<32)
    target=np.full(1024,np.inf);np.minimum.at(target,(vv*32+uu)[valid],zz[valid])
    return target.reshape(32,32)

class Motion:
    def __init__(self,lag=4):
        self.history=deque(maxlen=lag);self.heading=0.;self.lag=lag

    def observe(self,depth,goal,previous_action):
        if self.history:
            self.heading += (1 if previous_action==2 else -1 if previous_action==3 else 0)*10/120
        current=np.asarray(depth,dtype=np.float64)/65535*10
        result={'moving_near':0.,'new_unknown':0.,'coverage':0.}
        if len(self.history)==self.lag:
            old,old_goal,heading=self.history[0]
            prediction=warp(old,old_goal,goal,self.heading-heading)
            mask=np.zeros((32,32),dtype=bool);mask[6:25,4:28]=True
            valid=mask&np.isfinite(prediction)
            count=max(1,int(valid.sum()))
            result={'moving_near':float(np.sum(valid&(current>.01)&(current<1.8)&(prediction-current>.25))/count),
                    'new_unknown':float(np.sum(valid&(prediction>.3)&(current<=.01))/count),
                    'coverage':float(np.sum(valid)/np.sum(mask))}
        self.history.append((np.asarray(depth).copy(),list(goal),self.heading))
        return result
