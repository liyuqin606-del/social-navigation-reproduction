#!/usr/bin/env python3
"""Read-only readiness/data audit. No simulator, model, downloads, or training.

Only explicitly requested output files are written. Episode records are never
filtered, fixed, or extracted from the source ZIP. Requires PyYAML for configs.
"""
import argparse
import ast
import collections
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import zipfile


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_vector(value, length):
    return (isinstance(value, list) and len(value) == length and
            all(isinstance(x, (float, int)) and not isinstance(x, bool)
                and math.isfinite(x) for x in value))


def audit_dped(path, scene_root=None):
    groups = {}
    keyed = collections.defaultdict(dict)
    issues = []
    with zipfile.ZipFile(path) as archive:
        crc_error = archive.testzip()
        for member in sorted(archive.namelist()):
            parts = member.split("/")
            if len(parts) != 3 or not member.endswith(".json.gz"):
                continue
            level, split, filename = parts
            group_name = level + "/" + split
            group = groups.setdefault(group_name, {
                "files": 0, "records": 0, "scenes": {},
                "field_presence": collections.Counter(),
                "action_values": collections.Counter(),
                "human_counts": collections.Counter(),
                "instruction_lengths": [], "action_lengths": [],
                "issue_counts": collections.Counter(),
            })
            group["files"] += 1
            data = json.loads(gzip.decompress(archive.read(member)))
            assert isinstance(data, dict) and isinstance(data.get("episodes"), list), member
            for offset, record in enumerate(data["episodes"]):
                group["records"] += 1
                group["field_presence"].update(record.keys())
                record_issues = []
                scene = record.get("scene_id")
                eid = record.get("episode_id")
                key = (scene, str(eid))
                if not isinstance(scene, str) or not scene.endswith(".basis.glb"):
                    record_issues.append("invalid_scene_id")
                elif Path(scene).name.removesuffix(".basis.glb") != filename.removesuffix(".json.gz"):
                    record_issues.append("scene_member_mismatch")
                if eid is None:
                    record_issues.append("missing_episode_id")
                if key in keyed[group_name]:
                    record_issues.append("duplicate_scene_episode_id")
                group["scenes"][scene] = group["scenes"].get(scene, 0) + 1
                if not finite_vector(record.get("start_position"), 3):
                    record_issues.append("invalid_start_position")
                rotation = record.get("start_rotation")
                if not finite_vector(rotation, 4):
                    record_issues.append("invalid_start_rotation")
                elif abs(sum(x * x for x in rotation) - 1.0) > 1e-4:
                    record_issues.append("nonunit_start_quaternion")
                goals = record.get("goals")
                if not isinstance(goals, list) or not goals:
                    record_issues.append("missing_goals")
                elif any(not isinstance(g, dict) or not finite_vector(g.get("position"), 3) for g in goals):
                    record_issues.append("invalid_goal_position")
                instruction = record.get("instruction")
                if not isinstance(instruction, str):
                    record_issues.append("instruction_not_string")
                else:
                    group["instruction_lengths"].append(len(instruction))
                    if not instruction.strip():
                        record_issues.append("empty_instruction")
                actions = record.get("gt_action")
                if not isinstance(actions, list) or not actions:
                    record_issues.append("empty_or_missing_gt_action")
                else:
                    group["action_lengths"].append(len(actions))
                    group["action_values"].update(map(str, actions))
                    if any(type(a) is not int or a not in range(4) for a in actions):
                        record_issues.append("gt_action_outside_0_3")
                    if actions[-1] != 0:
                        record_issues.append("gt_action_does_not_end_zero")
                info = record.get("info", {})
                distance = info.get("geodesic_distance")
                if not isinstance(distance, (float, int)) or not math.isfinite(distance) or distance < 0:
                    record_issues.append("invalid_geodesic_distance")
                human_num = info.get("human_num")
                group["human_counts"][str(human_num)] += 1
                if type(human_num) is not int or human_num < 0:
                    record_issues.append("invalid_human_count")
                else:
                    for human in range(human_num):
                        for waypoint in range(3):
                            prefix = f"human_{human}_waypoint_{waypoint}_"
                            if not finite_vector(info.get(prefix + "position"), 3):
                                record_issues.append("invalid_human_waypoint_position")
                            angle = info.get(prefix + "rotation")
                            if not isinstance(angle, (float, int)) or not math.isfinite(angle):
                                record_issues.append("invalid_human_waypoint_rotation")
                canonical = {k: v for k, v in record.items() if k != "instruction"}
                keyed[group_name][key] = {
                    "noninstruction_sha256": hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest(),
                    "instruction": instruction,
                }
                for code in sorted(set(record_issues)):
                    group["issue_counts"][code] += 1
                    issues.append({"code": code, "member": member, "offset": offset,
                                   "scene_id": scene, "episode_id": eid})
    for group in groups.values():
        group["scene_count"] = len(group["scenes"])
        for field in ("instruction_lengths", "action_lengths"):
            values = group.pop(field)
            group[field + "_summary"] = {"min": min(values), "max": max(values),
                                          "mean": sum(values) / len(values)} if values else None
        if scene_root is not None:
            group["scene_assets_present"] = sum((scene_root / s).is_file() for s in group["scenes"])
    pairs = {}
    for split in sorted({g.split("/")[1] for g in groups}):
        left, right = keyed.get("L1/" + split, {}), keyed.get("L2/" + split, {})
        common = left.keys() & right.keys()
        pairs[split] = {
            "shared_scene_episode_ids": len(common),
            "only_L1": [{"scene_id": s, "episode_id": e} for s, e in sorted(left.keys() - right.keys())],
            "only_L2": [{"scene_id": s, "episode_id": e} for s, e in sorted(right.keys() - left.keys())],
            "noninstruction_mismatch_count": sum(left[k]["noninstruction_sha256"] != right[k]["noninstruction_sha256"] for k in common),
            "identical_instruction_count": sum(left[k]["instruction"] == right[k]["instruction"] for k in common),
        }
    scene_sets = {name: set(group["scenes"]) for name, group in groups.items()}
    overlap = {level: sorted(scene_sets.get(level + "/train", set()) & scene_sets.get(level + "/val_full", set())) for level in ("L1", "L2")}
    return {"source": str(path), "bytes": path.stat().st_size, "sha256": digest(path),
            "md5": digest(path, "md5"), "crc_error_member": crc_error,
            "scene_root_checked": str(scene_root) if scene_root else None,
            "groups": groups, "paired_levels": pairs, "train_val_scene_overlap": overlap,
            "issues": issues, "records_modified_or_filtered": 0,
            "interpretation": "Data structural diagnostics only; action geometry/collisions and instruction correctness were not validated in a simulator."}


def audit_nav(repo):
    import yaml
    class UniqueKeyLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError("duplicate YAML key: " + str(key))
            result[key] = loader.construct_object(value_node, deep=deep)
        return result
    UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    configs = {}
    for name in ("navthinker_hm3d.yaml", "navthinker_hm3d_eval.yaml"):
        path = repo / "configs" / name
        data = yaml.load(path.read_text(), Loader=UniqueKeyLoader)
        b = data["habitat_baselines"]
        configs[name] = {"sha256": digest(path), "defaults": data["defaults"],
                         "habitat": data["habitat"], "habitat_baselines": b}
    modules = sorted((repo / "habitat-baselines/habitat_baselines/rl/world_model").glob("*.py"))
    modules += [repo / "habitat-baselines/habitat_baselines/rl/ppo/navthinker_trainer.py",
                repo / "habitat-baselines/habitat_baselines/rl/ppo/navthinker_evaluator.py"]
    syntax = []
    for path in modules:
        ast.parse(path.read_text(), filename=str(path))
        syntax.append({"path": str(path.relative_to(repo)), "sha256": digest(path), "ast_parse": "passed"})
    shell = {}
    for name in ("train.sh", "eval.sh"):
        result = subprocess.run(["bash", "-n", str(repo / "scripts" / name)], capture_output=True, text=True)
        shell[name] = {"exit_code": result.returncode, "stderr": result.stderr}
    requirements = {}
    for path in [repo / "requirements.txt", repo / "habitat-lab/requirements.txt", *sorted((repo / "habitat-baselines").rglob("requirements.txt"))]:
        requirements[str(path.relative_to(repo))] = {"sha256": digest(path), "lines": path.read_text().splitlines()}
    weights = sorted(str(p.relative_to(repo)) for suffix in ("*.pth", "*.pt", "*.ckpt", "*.safetensors") for p in repo.rglob(suffix))
    structured_file = repo / "habitat-baselines/habitat_baselines/config/default_structured_configs.py"
    structured = {}
    for cls in ast.parse(structured_file.read_text()).body:
        if isinstance(cls, ast.ClassDef) and cls.name in ("WorldModelConfig", "PolicyEncoderConfig"):
            structured[cls.name] = {}
            for node in cls.body:
                if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    try:
                        structured[cls.name][node.target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    networks = repo / "habitat-baselines/habitat_baselines/rl/world_model/networks.py"
    rssm = next(node for node in ast.parse(networks.read_text()).body if isinstance(node, ast.ClassDef) and node.name == "RSSM")
    rssm_attributes = {node.name for node in rssm.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(rssm):
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store) and isinstance(node.value, ast.Name) and node.value.id == "self":
            rssm_attributes.add(node.attr)
    return {"repo": str(repo), "configs": configs, "source_parse": syntax,
            "launcher_bash_syntax": shell, "dependency_sources": requirements,
            "checkpoint_files_in_snapshot": weights,
            "structured_config_literal_defaults": structured,
            "rssm_policy_interface_static_check": {key: key in rssm_attributes for key in ("feat_size", "reset_imagination", "img_step_lookahead", "observe_post_action", "_hist_z")},
            "local_runtime_modules": {name: importlib.util.find_spec(name) is not None for name in ("torch", "numpy", "yaml", "gym", "habitat_sim")},
            "hydra_composition": "not run: requires Habitat dependencies; YAML parsing alone is not composition",
            "simulator_evaluation": "not run"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument("--nav-repo", type=Path, default=root / "baselines/repos/NavThinker")
    parser.add_argument("--dped-zip", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"evidence_level": "source-and-data-diagnostics-only", "navthinker": audit_nav(args.nav_repo.resolve()),
              "dped": audit_dped(args.dped_zip.resolve(), args.scene_root)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "dped_records": {k: v["records"] for k, v in report["dped"]["groups"].items()},
                      "dped_issues": len(report["dped"]["issues"]), "nav_modules_parsed": len(report["navthinker"]["source_parse"])}))


if __name__ == "__main__":
    main()
