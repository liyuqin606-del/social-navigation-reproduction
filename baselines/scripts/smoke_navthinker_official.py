#!/usr/bin/env python3
"""Official NavThinker config/model diagnostic with synthetic observations.

NO simulator, data episodes, training, checkpoint download, or benchmark metric.
Uses the frozen upstream source unchanged. Run in its Habitat-compatible env.
The optional DINO component uses its upstream constructor defaults and is NOT
the published YAML's default GRU world model. Failures are recorded, not patched.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dino-component", action="store_true")
    p.add_argument("--habitat-sim-first", action="store_true",
                   help="Explicit environment compatibility probe: import Habitat-Sim before Habitat-Lab/Magnum")
    args = p.parse_args()
    repo = args.repo.resolve()
    args.output = args.output.resolve()
    report = {"evidence_level": "synthetic-untrained-component-diagnostic",
              "benchmark_evaluated": False, "weights_loaded": False,
              "paper_reproduction": False, "diagnostic_complete": False,
              "config_overrides": [], "repo": str(repo), "stages": {},
              "runtime_environment_compatibility": {
                  "habitat_sim_first_requested": args.habitat_sim_first,
                  "habitat_sim_first_applied": False,
                  "scope": "Import order only; no upstream source or config change"}}
    overlay = repo / "compatibility_overlay_manifest.json"
    report["source_compatibility_overlay"] = json.loads(overlay.read_text()) if overlay.is_file() else None
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def stage(name, fn):
        start = time.monotonic()
        try:
            value = fn()
            report["stages"][name] = {"status": "passed", "seconds": time.monotonic() - start, "result": value}
            return True
        except Exception as error:
            report["stages"][name] = {"status": "failed", "seconds": time.monotonic() - start,
                                     "error_type": type(error).__name__, "error": str(error),
                                     "traceback": traceback.format_exc()}
            return False
        finally:
            save()

    for path in (repo, repo / "habitat-lab", repo / "habitat-baselines"):
        sys.path.insert(0, str(path))
    os.chdir(repo)
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("GLOG_minloglevel", "2")
    state = {}

    def runtime():
        native = {name: name in sys.modules for name in ("habitat_sim", "magnum", "corrade")}
        report["runtime_environment_compatibility"]["native_modules_before_runtime"] = native
        if args.habitat_sim_first:
            if any(native.values()):
                raise RuntimeError("Native Habitat/Magnum modules were already imported before the compatibility probe")
            import habitat_sim
            report["runtime_environment_compatibility"].update(
                habitat_sim_first_applied=True,
                habitat_sim_file=str(Path(habitat_sim.__file__).resolve()),
                habitat_sim_version=habitat_sim.__version__)
        import numpy as np
        import torch
        from gym import spaces
        from omegaconf import OmegaConf
        torch.manual_seed(0)
        np.random.seed(0)
        torch.set_num_threads(1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        state.update(torch=torch, np=np, spaces=spaces, OmegaConf=OmegaConf, device=device)
        return {"torch": torch.__version__, "numpy": np.__version__, "device": device,
                "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else None}

    if not stage("runtime", runtime):
        report["exit_code"] = 2
        save()
        return 2

    def compose():
        import habitat_baselines
        import habitat_baselines.run  # official custom sensor/config registration
        from habitat.config.default import patch_config
        from habitat_baselines.config.default import get_config
        resolved = Path(habitat_baselines.__file__).resolve()
        if repo not in resolved.parents:
            raise RuntimeError("Wrong habitat_baselines import: " + str(resolved))
        cfg = patch_config(get_config("navthinker_hm3d_eval.yaml", configs_dir=str(repo / "configs")))
        state["cfg"] = cfg
        b = cfg.habitat_baselines
        # Refuse a future auto-download path: do not silently alter its config.
        if b.policy_encoder.type != "resnet" or b.world_model.encoder_type != "navthinker":
            raise RuntimeError("Config now uses a different encoder; audit weight/download requirements before running")
        if b.world_model.navthinker_pretrained_path:
            raise RuntimeError("Evaluation WM config unexpectedly requests a checkpoint")
        state["world_model_config"] = state["OmegaConf"].to_container(b.world_model, resolve=True)
        cfg_path = repo / "configs/navthinker_hm3d_eval.yaml"
        return {"config_sha256": hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
                "habitat_baselines_import": str(resolved),
                "wm_encoder": b.world_model.encoder_type, "wm_dynamics": b.world_model.dynamics_type,
                "policy_encoder": b.policy_encoder.type, "rnn": b.rl.ddppo.rnn_type,
                "rnn_layers": b.rl.ddppo.num_recurrent_layers, "test_episode_count": b.test_episode_count}

    composed = stage("compose_unmodified_official_eval_config", compose)
    if composed:
        np, spaces, torch, device = (state[k] for k in ("np", "spaces", "torch", "device"))
        shape_map = {"articulated_agent_jaw_depth": (256, 256, 1), "pointgoal_with_gps_compass": (2,),
                     "localization_sensor": (4,), "human_num_sensor": (1,),
                     "human_state_goal": (6, 8), "oracle_humanoid_future_trajectory": (6, 4, 2)}
        # AgentAccessMgr removes agent_0_ before passing spaces to agent policy.
        obs_space = spaces.Dict({name: spaces.Box(low=0 if "depth" in name else -100,
                                                 high=1 if "depth" in name else 100,
                                                 shape=shape, dtype=np.float32)
                                 for name, shape in shape_map.items()})
        obs_space.spaces["human_num_sensor"] = spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32)
        obs = {name: torch.zeros((1,) + shape, device=device) for name, shape in shape_map.items()}
        obs["human_num_sensor"] = torch.zeros(1, 1, device=device, dtype=torch.int32)
        obs["articulated_agent_jaw_depth"].fill_(0.5)
        obs["human_num_sensor"].fill_(2)
        report["synthetic_observation_shapes"] = shape_map

        def world_model():
            from habitat_baselines.rl.world_model.models import SocialNavWorldModel
            model = SocialNavWorldModel(state["world_model_config"], observation_space=obs_space, device=device).to(device).eval()
            with torch.no_grad():
                embed = model.encoder(obs)
                action = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
                post, prior = model.dynamics.obs_step(None, action, embed, torch.ones(1, device=device), sample=False)
                feat = model.get_feat(post)
                assert torch.isfinite(feat).all()
                heads = {}
                for name, head in model.heads.items():
                    dist = head(feat, human_state_goal=obs["human_state_goal"]) if name == "human_traj" else head(feat)
                    mean = dist.mean() if callable(dist.mean) else dist.mean
                    assert torch.isfinite(mean).all(), name
                    heads[name] = list(mean.shape)
            return {"random_initialization": True, "config_path": "direct SocialNavWorldModel(composed world_model node)",
                    "encoder_shape": list(embed.shape), "feature_shape": list(feat.shape), "head_shapes": heads,
                    "parameters": sum(x.numel() for x in model.parameters()),
                    "dynamics_type": type(model.dynamics).__name__,
                    "policy_required_interfaces": {key: hasattr(model.dynamics, key) for key in
                        ("feat_size", "reset_imagination", "img_step_lookahead", "observe_post_action", "_hist_z")}}

        stage("world_model_direct_component_forward", world_model)

        def policy():
            from habitat_baselines.rl.world_model.navthinker_policy import NavThinkerPolicy
            model = NavThinkerPolicy.from_config(state["cfg"], obs_space, spaces.Discrete(4), agent_name="agent_0").to(device).eval()
            hidden = torch.zeros(1, model.net.num_recurrent_layers, model.net.recurrent_hidden_size, device=device)
            prev_actions = torch.zeros(1, 1, dtype=torch.long, device=device)
            masks = torch.zeros(1, 1, dtype=torch.bool, device=device)
            with torch.no_grad():
                output, new_hidden, _ = model.net(obs, hidden, prev_actions, masks)
                assert torch.isfinite(output).all()
            return {"feature_shape": list(output.shape), "hidden_shape": list(new_hidden.shape),
                    "parameters": sum(x.numel() for x in model.parameters()), "random_initialization": True}

        stage("official_policy_factory_and_forward", policy)

    if args.dino_component:
        def dino():
            torch, device = state["torch"], state["device"]
            file = repo / "habitat-baselines/habitat_baselines/rl/world_model/dino_wm_predictor.py"
            spec = importlib.util.spec_from_file_location("official_dino_wm_component", file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            model = module.DinoWMDynamics(device=device).to(device).eval()
            action = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
            with torch.no_grad():
                initial = model.initial(1)
                output = model.img_step(initial, action, sample=False)
                saved_hist = model._hist_z.clone()
                lookahead = model.img_step_lookahead(output, action)
                assert torch.equal(model._hist_z, saved_hist), "lookahead modified persistent history"
                assert torch.isfinite(lookahead["deter"]).all()
            return {"source_sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                    "settings": "upstream constructor defaults, batch=1, random weights; not default YAML dynamics",
                    "deter_shape": list(output["deter"].shape), "lookahead_preserves_history": True,
                    "parameters": sum(x.numel() for x in model.parameters())}
        stage("optional_standalone_dino_component", dino)
    if state["device"] == "cuda":
        report["cuda_peak_allocated_bytes"] = state["torch"].cuda.max_memory_allocated()
    report["diagnostic_complete"] = composed and all(name in report["stages"] for name in
        ("world_model_direct_component_forward", "official_policy_factory_and_forward"))
    report["all_requested_stages_passed"] = all(v["status"] == "passed" for v in report["stages"].values())
    report["exit_code"] = 0 if report["all_requested_stages_passed"] else 1
    save()
    print(json.dumps({"output": str(args.output), "stages": {k: v["status"] for k, v in report["stages"].items()}}))
    return report["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
