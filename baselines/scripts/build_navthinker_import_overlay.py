#!/usr/bin/env python3
"""Create a separately labelled source copy skipping three unrelated IL imports.

Every source file must match the frozen manifest. Only package __init__.py is
changed; no policy, world model, config, or Falcon implementation is substituted.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path
import shutil


TARGET = "habitat-baselines/habitat_baselines/__init__.py"
BASE_SHA = "42f1afc704716a67ada9b5b4f015e0e254328c15ec925e2b741fd496f7ef976f"
MODULES = {
    "habitat_baselines.il.trainers.eqa_cnn_pretrain_trainer",
    "habitat_baselines.il.trainers.pacman_trainer",
    "habitat_baselines.il.trainers.vqa_trainer",
}
NAMES = {"EQACNNPretrainTrainer", "PACMANTrainer", "VQATrainer"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    repo, output = args.repo.resolve(), args.output.resolve()
    assert not output.exists(), "Refusing to overwrite a pre-existing output"
    assert output != repo and repo not in output.parents, "Overlay must be outside source repo"
    manifest = json.loads(args.source_manifest.read_text())
    records = manifest["files"]
    assert len(records) == 520, "Unexpected upstream source manifest; review update first"
    for entry in records:
        rel = Path(entry["path"])
        assert not rel.is_absolute() and ".." not in rel.parts
        assert sha(repo / rel) == entry["sha256"], "Source hash mismatch: " + str(rel)
    assert sha(repo / TARGET) == BASE_SHA
    source = (repo / TARGET).read_text()
    tree = ast.parse(source)
    remove_lines, found = set(), set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module in MODULES:
            found.add(node.module)
            remove_lines.update(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            for value in node.value.elts:
                if isinstance(value, ast.Constant) and value.value in NAMES:
                    remove_lines.update(range(value.lineno, value.end_lineno + 1))
    assert found == MODULES, found
    modified = "".join(line for i, line in enumerate(source.splitlines(keepends=True), 1) if i not in remove_lines)
    ast.parse(modified)
    # __all__ changes only remove names whose imports were skipped.
    diff = "".join(difflib.unified_diff(source.splitlines(True), modified.splitlines(True),
                                      fromfile="original/" + TARGET, tofile="compatibility_overlay/" + TARGET))
    for entry in records:
        dest = output / entry["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / entry["path"], dest)
    (output / TARGET).write_text(modified)
    patch_name = "compatibility_overlay.patch"
    (output / patch_name).write_text(diff)
    changed = []
    for entry in records:
        if sha(output / entry["path"]) != entry["sha256"]:
            changed.append(entry["path"])
    assert changed == [TARGET], changed
    result = {
        "kind": "skip_unrelated_il_imports_only", "paper_reproduction": False,
        "upstream_commit": "fc3421b6991080a367a348f7f80fa899358f9a0b",
        "source_repo": str(repo), "source_manifest_sha256": sha(args.source_manifest),
        "source_file_count": len(records), "unchanged_source_files": len(records) - 1,
        "modified_path": TARGET, "base_sha256": BASE_SHA, "overlay_sha256": sha(output / TARGET),
        "skipped_modules": sorted(MODULES), "removed_all_exports": sorted(NAMES),
        "policy_world_model_configs_modified": False,
        "diff_file": patch_name, "diff_sha256": sha(output / patch_name),
    }
    (output / "compatibility_overlay_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
