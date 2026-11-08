"""Bounded Falcon interface probe. No learned navigation policy is deployed."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

DEPTH = 'agent_0_articulated_agent_jaw_depth'
GOAL = 'agent_0_pointgoal_with_gps_compass'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def array(value):
    return value.detach().cpu().numpy().copy()


def actor_digest(actor):
    result = hashlib.sha256()
    for name, value in sorted(actor.state_dict().items()):
        result.update(name.encode());result.update(array(value).tobytes())
    return result.hexdigest()


class ProbeComplete(Exception):
    pass


class ObservationAdapter:
    """Whitelist and validate deployed inputs; no full map or people oracle."""
    def frame(self, observations, features, logits):
        require(set(observations) == {DEPTH, GOAL}, 'unexpected adapter input')
        depth, goal = observations[DEPTH], observations[GOAL]
        require(depth.shape[0] == 1 and depth.ndim == 4 and depth.shape[-1] == 1, 'depth shape')
        require(goal.shape == (1, 2), 'pointgoal shape')
        require(features.shape == (1, 512) and logits.shape == (1, 4), 'Falcon head shape')
        require(all(np.isfinite(x).all() for x in (depth, goal, features, logits)), 'nonfinite deployment input')
        return dict(depth_shape=list(depth.shape), goal=goal[0].tolist(),
                    depth_range=[float(depth.min()), float(depth.max())],
                    feature_dim=features.shape[1], action_count=logits.shape[1])


class Bridge:
    def __init__(self, agent):
        self.output = Path(os.environ['FALCON_PROBE_OUTPUT'])
        self.mode = os.environ['FALCON_PROBE_MODE']
        require(self.mode in ('reference','shadow','override'), 'probe mode')
        self.actor = agent._agents[0]._actor_critic
        require(self.actor.action_distribution.linear.in_features == 512, 'feature width')
        require(self.actor.action_distribution.linear.out_features == 4, 'action width')
        self.before_digest = actor_digest(self.actor)
        self.adapter = ObservationAdapter()
        self.rows, self.features, self.logits = [], [], []
        self.pending, self.current = None, None
        self.reset_count, self.previous_checks = 0, 0
        self.hook = self.actor.action_distribution.linear.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        require(self.current is not None and self.pending is None, 'head hook count/order')
        self.pending = array(inputs[0]), array(output)

    def before(self, episodes, batch, prev_actions, masks, recurrent):
        require(len(episodes) == 1 and self.current is None and self.pending is None, 'single-env transaction required')
        previous = array(prev_actions)
        reset = not bool(array(masks)[0, 0])
        key = [episodes[0].scene_id, str(episodes[0].episode_id)]
        if self.rows:
            last = self.rows[-1]
            require(reset == bool(last['done']), 'reset mask/done mismatch')
            if not reset:
                require(last['episode'] == key, 'episode changed without reset')
                require(float(previous[0, 0]) == last['executed'], 'previous_action not actual execution')
                self.previous_checks += 1
        if reset:
            self.reset_count += 1
        require(DEPTH in batch and GOAL in batch, 'required policy observations absent')
        # Restrict before constructing a deployment frame. Full multi-agent batch
        # includes privileged observations for other agents; never forward it.
        allowed = {name: array(batch[name]) for name in (DEPTH, GOAL)}
        self.current = dict(step=len(self.rows), episode=key, reset=reset,
                            previous_action=float(previous[0,0]), observations=allowed,
                            recurrent_in_sha=hashlib.sha256(array(recurrent).tobytes()).hexdigest())

    def commit(self, action_data, action_lengths):
        require(self.current is not None and self.pending is not None, 'missing observation or forward')
        require(int(action_lengths[0]) == 1, 'robot action slice is not scalar')
        features, logits = self.pending
        self.pending = None
        raw_actions, raw_take = array(action_data.actions), array(action_data.env_actions)
        proposed = int(raw_take[0,0])
        require(raw_take[0,0] == proposed and 0 <= proposed <= 3, 'categorical action expected')
        chosen = proposed
        index = self.current['step']
        if self.mode == 'override' and index == 3:
            chosen = 3 if proposed == 2 else 2  # guaranteed different non-STOP turn
        if self.mode == 'override' and index == 6:
            chosen = 0  # explicit STOP/reset engineering test, not learned action
        if self.mode != 'reference':
            metadata = self.adapter.frame(self.current['observations'], features, logits)
        else:
            metadata = {'adapter_bypassed': True}
        if chosen != proposed:
            action_data.actions[0,0] = chosen
            if action_data.take_actions is not None:
                action_data.take_actions[0,0] = chosen
        require(np.array_equal(array(action_data.actions)[:,1:], raw_actions[:,1:]), 'other-agent stored action changed')
        require(np.array_equal(array(action_data.env_actions)[:,1:], raw_take[:,1:]), 'other-agent executed action changed')
        require(int(action_data.env_actions[0,0]) == chosen, 'env action not overwritten')
        self.current.update(proposed=proposed, executed=chosen, adapter=metadata,
                            depth_sha=hashlib.sha256(self.current['observations'][DEPTH].tobytes()).hexdigest(),
                            goal=self.current['observations'][GOAL][0].tolist())
        del self.current['observations']
        self.features.append(features);self.logits.append(logits)

    def after(self, step_data, outputs, prev_actions):
        require(self.current is not None, 'no pending transition')
        executed = float(np.asarray(step_data[0]).reshape(-1)[0])
        require(executed == self.current['executed'], 'actual env.step differs')
        require(float(prev_actions[0,0]) == executed, 'stored previous action differs after update')
        observation, reward, done, info = outputs[0]
        self.current.update(done=bool(done), received_next_observation=True,
                            next_depth_sha=hashlib.sha256(np.asarray(observation[DEPTH]).tobytes()).hexdigest(),
                            terminal_next_observation_is_reset=bool(done))
        self.rows.append(self.current);self.current = None
        if len(self.rows) >= 16:
            raise ProbeComplete('BOUNDED_16_STEP_ENGINEERING_PROBE')

    def close(self):
        self.hook.remove()
        after = actor_digest(self.actor)
        result = dict(mode=self.mode, scope='INTERFACE_ONLY_NO_TRAINING_NO_BENCHMARK',
                      steps=len(self.rows), reset_count=self.reset_count,
                      previous_action_checks=self.previous_checks,
                      actor_before_sha=self.before_digest, actor_after_sha=after,
                      pending_transition=self.current is not None,
                      weights_unchanged=after == self.before_digest)
        (self.output/'trace.json').write_text(json.dumps(self.rows,indent=2)+'\n')
        (self.output/'bridge_result.json').write_text(json.dumps(result,indent=2)+'\n')
        if self.features:
            np.savez_compressed(self.output/'features.npz', features=np.concatenate(self.features), logits=np.concatenate(self.logits))
        require(after == self.before_digest, 'actor state changed')
