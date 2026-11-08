"""Intervention recorder. Oracle timing is diagnostic; actor remains frozen."""
import hashlib
import json
import os
import pickle
import random
from pathlib import Path
import numpy as np
from controller import Controller

DEPTH='agent_0_articulated_agent_jaw_depth'
GOAL='agent_0_pointgoal_with_gps_compass'


def require(ok,msg):
    if not ok: raise RuntimeError(msg)


def array(v):
    return v.detach().cpu().numpy().copy()


def digest(v):
    return hashlib.sha256(np.asarray(v).tobytes()).hexdigest()


def actor_digest(actor):
    h=hashlib.sha256()
    for k,v in sorted(actor.state_dict().items()):
        h.update(k.encode());h.update(array(v).tobytes())
    return h.hexdigest()


def rng_digest():
    import torch
    h=hashlib.sha256(pickle.dumps((random.getstate(),np.random.get_state()),protocol=4))
    h.update(array(torch.get_rng_state()).tobytes())
    for v in torch.cuda.get_rng_state_all(): h.update(array(v).tobytes())
    return h.hexdigest()


def selected_action(step,proposed,start,sequence):
    offset=step-start
    return int(sequence[offset]) if start>=0 and 0<=offset<len(sequence) else int(proposed)


class ProbeComplete(Exception): pass


class Bridge:
    def __init__(self,agent):
        self.out=Path(os.environ['FALCON_PROBE_OUTPUT'])
        self.start=int(os.environ['SCET_BRANCH_START'])
        self.sequence=json.loads(os.environ['SCET_BRANCH_ACTIONS'])
        self.policy=os.environ['SCET_POLICY']
        self.controller=Controller(self.policy)
        require(all(a in [1,2,3] for a in self.sequence),'illegal intervention action')
        self.actor=agent._agents[0]._actor_critic
        self.before_sha=actor_digest(self.actor)
        self.rows=[];self.features=[];self.logits=[];self.depths=[]
        self.current=None;self.pending=None
        self.hook=self.actor.action_distribution.linear.register_forward_hook(self.capture)

    def capture(self,module,inputs,output):
        require(self.current is not None and self.pending is None,'head forward order')
        self.pending=(array(inputs[0]),array(output))

    def before(self,episodes,batch,prev_actions,masks,recurrent):
        require(len(episodes)==1 and self.current is None and self.pending is None,'single transaction')
        prev=array(prev_actions)
        reset=not bool(array(masks)[0,0])
        require(reset==(not bool(self.rows)),'unexpected reset before terminal stop')
        if self.rows: require(int(prev[0,0])==self.rows[-1]['executed'],'actual prev action mismatch')
        obs_sha={k:digest(array(v)) for k,v in sorted(batch.items())}
        depth=array(batch[DEPTH])
        goal=array(batch[GOAL])
        require(depth.shape==(1,256,256,1) and goal.shape==(1,2),'observation contract')
        self.depths.append((np.clip(depth[0,::8,::8,0],0,1)*65535).astype(np.uint16))
        self.current=dict(step=len(self.rows),episode=[episodes[0].scene_id,str(episodes[0].episode_id)],
                          observations_sha=obs_sha,goal=goal[0].tolist(),reset=reset,previous_action=int(prev[0,0]),
                          recurrent_sha=digest(array(recurrent)),rng_sha=rng_digest())

    def commit(self,action_data,action_lengths):
        require(self.current is not None and self.pending is not None,'missing head')
        require(int(action_lengths[0])==1,'robot action slice')
        f,l=self.pending;self.pending=None
        require(f.shape==(1,512) and l.shape==(1,4),'head dimensions')
        require(np.isfinite(f).all() and np.isfinite(l).all(),'nonfinite head')
        raw=array(action_data.actions);taken=array(action_data.env_actions)
        proposed=int(taken[0,0]);require(taken[0,0]==proposed and 0<=proposed<=3,'action index')
        chosen,control=self.controller.act(self.depths[-1],proposed)
        self.current['control']=control
        if chosen!=proposed:
            action_data.actions[0,0]=chosen
            if action_data.take_actions is not None: action_data.take_actions[0,0]=chosen
        require(np.array_equal(array(action_data.actions)[:,1:],raw[:,1:]),'changed human stored actions')
        require(np.array_equal(array(action_data.env_actions)[:,1:],taken[:,1:]),'changed human executed actions')
        require(int(action_data.env_actions[0,0])==chosen,'override failed')
        self.current.update(proposed=proposed,executed=chosen,human_actions_sha=digest(taken[:,1:]),
                            feature_sha=digest(f),logit_sha=digest(l))
        self.features.append(f);self.logits.append(l)

    def after(self,step_data,outputs,prev_actions):
        require(self.current is not None,'missing transaction')
        require(int(np.asarray(step_data[0]).reshape(-1)[0])==self.current['executed'],'step action mismatch')
        require(int(prev_actions[0,0])==self.current['executed'],'prev action not actual')
        obs,reward,done,info=outputs[0]
        from habitat_baselines.utils.info_dict import extract_scalars_from_info
        metrics=extract_scalars_from_info(info)
        self.current.update(done=bool(done),reward=float(reward),metrics={k:float(v) for k,v in metrics.items()},
                            next_depth_sha=digest(obs[DEPTH]),terminal_observation_is_reset=bool(done))
        self.rows.append(self.current);self.current=None
        if done: raise ProbeComplete('TERMINAL_EPISODE_COLLECTED')
        if len(self.rows)>=500: raise RuntimeError('EXPECTED_TERMINAL_AT500_NOT_OBSERVED')

    def close(self):
        self.hook.remove()
        after=actor_digest(self.actor)
        result=dict(steps=len(self.rows),done=bool(self.rows and self.rows[-1]['done']),
                    pending=self.current is not None,weights_unchanged=after==self.before_sha,
                    actor_before_sha=self.before_sha,actor_after_sha=after,
                    intervention_start=self.start,sequence=self.sequence,policy=self.policy)
        (self.out/'trace.json').write_text(json.dumps(self.rows,indent=2)+'\n')
        (self.out/'bridge_result.json').write_text(json.dumps(result,indent=2)+'\n')
        if self.features:
            np.savez_compressed(self.out/'features.npz',features=np.concatenate(self.features),logits=np.concatenate(self.logits),depth_thumbnails=np.stack(self.depths))
        require(after==self.before_sha,'actor weights changed')
