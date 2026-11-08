#!/usr/bin/env python3
"""Official Falcon checkpoint inference and Habitat EGL smoke, NOT navigation evaluation."""
import argparse, hashlib, json, os, sys, time, subprocess
from pathlib import Path

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--repo',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('SLURM_JOB_ID'), 'Run only in managed Slurm allocation'
    a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=True)
    runtime={'python':sys.executable,'prefix':sys.prefix,'environment':{
        key:os.environ.get(key) for key in ['LD_LIBRARY_PATH','CONDA_PREFIX','CUDA_VISIBLE_DEVICES',
        '__EGL_VENDOR_LIBRARY_FILENAMES','__EGL_VENDOR_LIBRARY_DIRS']},'egl_vendor_files':{},'binding_ldd':{}}
    for directory in ['/usr/share/glvnd/egl_vendor.d','/etc/glvnd/egl_vendor.d',str(Path(sys.prefix)/'share/glvnd/egl_vendor.d')]:
        for f in Path(directory).glob('*.json'):
            runtime['egl_vendor_files'][str(f)]=f.read_text()
    for f in (Path(sys.prefix)/'lib/python3.9/site-packages').glob('habitat_sim*/habitat_sim/_ext/*.so'):
        result=subprocess.run(['ldd',str(f)],capture_output=True,text=True,timeout=30)
        runtime['binding_ldd'][str(f)]={'exit_code':result.returncode,'stdout':result.stdout,'stderr':result.stderr}
    (a.output/'runtime_environment.json').write_text(json.dumps(runtime,indent=2)+'\n')
    repo=a.repo.resolve()
    a.checkpoint=a.checkpoint.resolve()
    # Explicitly select the Falcon fork even when a shared env has other forks.
    sys.path[:0]=[str(repo),str(repo/'habitat-lab'),str(repo/'habitat-baselines')]
    os.chdir(repo)
    import numpy as np
    import torch
    import habitat_sim
    import falcon
    import habitat
    import habitat_baselines
    import habitat_baselines.run  # Match official custom configuration registration.
    module_paths={name:str(Path(module.__file__).resolve()) for name,module in
                  [('falcon',falcon),('habitat',habitat),('habitat_baselines',habitat_baselines)]}
    assert all(repo in Path(path).parents for path in module_paths.values()), module_paths
    from gym import spaces
    from habitat_baselines.config.default import get_config
    from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy
    from omegaconf import OmegaConf
    torch.set_num_threads(1)
    torch.manual_seed(100); np.random.seed(100)
    assert torch.cuda.device_count()==1
    expected='ba7f2824604639d934be2359226c9e9fb35d5b1b2fb39ffbcbbbd833252db6ca'
    assert digest(a.checkpoint)==expected
    cfg=get_config('social_nav_v2/falcon_hm3d.yaml')
    resolved=OmegaConf.to_yaml(cfg,resolve=True)
    (a.output/'resolved_config.yaml').write_text(resolved)
    obs_space=spaces.Dict({
        'articulated_agent_jaw_depth': spaces.Box(0.0,1.0,shape=(256,256,1),dtype=np.float32),
        'pointgoal_with_gps_compass': spaces.Box(-np.inf,np.inf,shape=(2,),dtype=np.float32),
    })
    model=PointNavResNetPolicy.from_config(cfg,obs_space,spaces.Discrete(4),agent_name='agent_0')
    # Official release contains its OmegaConf training metadata; no arbitrary source is accepted.
    ckpt=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    # Official MultiAgentAccessMgr indexes the trained robot by integer agent 0.
    # The frozen release has ckpt[0]['state_dict'], with net.* parameter keys.
    assert 0 in ckpt and 'state_dict' in ckpt[0]
    incompatible=model.load_state_dict(ckpt[0]['state_dict'],strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    model=model.cuda().eval()
    device=torch.device('cuda:0')
    initial=torch.zeros(2,*model.hidden_state_shape,device=device)
    actions=[]; values=[]; log_probs=[]; shapes=[]
    started=time.monotonic()
    with torch.inference_mode():
        hidden=initial.clone(); prev=torch.zeros(2,1,device=device,dtype=torch.long)
        for step in range(8):
            obs={'articulated_agent_jaw_depth':torch.rand(2,256,256,1,device=device),
                 'pointgoal_with_gps_compass':torch.tensor([[2.0,0.0],[5.0,0.5]],device=device)}
            masks=torch.full((2,1),step!=0,device=device,dtype=torch.bool)
            out=model.act(obs,hidden,prev,masks,deterministic=False)
            for tensor in (out.values,out.action_log_probs,out.rnn_hidden_states):
                assert torch.isfinite(tensor).all()
            assert out.actions.shape==(2,1) and ((out.actions>=0)&(out.actions<4)).all()
            actions.append(out.actions.cpu().tolist()); values.append(out.values.cpu().tolist())
            log_probs.append(out.action_log_probs.cpu().tolist());shapes.append(list(out.rnn_hidden_states.shape))
            hidden=out.rnn_hidden_states;prev=out.actions
        # A reset mask must erase the incoming recurrent history for identical inputs.
        reset=torch.zeros(2,1,device=device,dtype=torch.bool)
        left=model.act(obs,hidden,prev,reset,deterministic=True)
        right=model.act(obs,torch.randn_like(hidden),prev,reset,deterministic=True)
        reset_error=(left.values-right.values).abs().max().item()
        assert reset_error<1e-6 and torch.equal(left.actions,right.actions)
    torch.cuda.synchronize()
    inference_seconds=time.monotonic()-started
    simcfg=habitat_sim.SimulatorConfiguration()
    simcfg.scene_id='NONE';simcfg.gpu_device_id=0;simcfg.enable_physics=True
    sensor=habitat_sim.CameraSensorSpec()
    sensor.uuid='depth';sensor.sensor_type=habitat_sim.SensorType.DEPTH
    sensor.resolution=[64,64]
    agent=habitat_sim.agent.AgentConfiguration();agent.sensor_specifications=[sensor]
    with habitat_sim.Simulator(habitat_sim.Configuration(simcfg,[agent])) as sim:
        pixels=sim.get_sensor_observations()['depth']
        assert pixels.shape==(64,64) and np.isfinite(pixels).all()
    report={
        'status':'PASS','scope':'CHECKPOINT_AND_RENDERER_SMOKE_ONLY',
        'benchmark_episodes':0,'navigation_metrics':None,'synthetic_observations':True,
        'checkpoint_sha256':expected,'strict_checkpoint_load':True,
        'checkpoint_layout':'multi_agent_integer_0/state_dict',
        'repo':str(repo),'module_paths':module_paths,
        'missing_keys':[],'unexpected_keys':[], 'seed':100,'batch_size':2,'steps':8,
        'actions':actions,'values':values,'log_probabilities':log_probs,'hidden_shapes':shapes,
        'reset_value_error':reset_error,'inference_seconds':inference_seconds,
        'renderer':{'scene':'NONE','depth_shape':list(pixels.shape),'finite':True,'physics_enabled':True},
        'torch':torch.__version__,'habitat_sim':habitat_sim.__version__,
        'gpu':torch.cuda.get_device_name(0),'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'config_sha256':hashlib.sha256(resolved.encode()).hexdigest(),
    }
    (a.output/'smoke_result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
