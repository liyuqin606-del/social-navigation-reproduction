#!/usr/bin/env python3
"""Derive a logging-only evaluator from the pinned official Falcon source.

No policy, action, metric, termination, seeding, or environment code is replaced.
The original checkout is never written. A full derived module and unified diff
are saved in a separate output directory for review and reproducibility.
"""
import argparse
import ast
import difflib
import hashlib
import json
from pathlib import Path

OFFICIAL_COMMIT = 'd15ef512c4b1451f9daeaf6d4d7ecc702ed57430'
EVALUATOR_PATH = 'habitat-baselines/habitat_baselines/rl/ppo/falcon_evaluator.py'
OFFICIAL_EVALUATOR_SHA256 = 'd8bde7a4ed64917571f9125862e9446220a6576d3467e48159f78cfea47375bd'


def derive(repo, output):
    source = Path(repo) / EVALUATOR_PATH
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != OFFICIAL_EVALUATOR_SHA256:
        raise ValueError('Falcon evaluator differs from the audited official commit')
    original = raw.decode('utf-8')
    derived = original
    replacements = [
        ('        success_cal = 0 ## my added\n',
         '        # VLN audit: only observability and output location change.\n'
         '        audit_output = os.environ["VLN_RUN_OUTPUT"]\n'
         '        os.makedirs(audit_output, exist_ok=True)\n'
         '        episode_log = os.path.join(audit_output, "episode_metrics.jsonl")\n'
         '        if os.path.exists(episode_log):\n'
         '            raise RuntimeError("Refusing to append to an existing episode log")\n'
         '        success_cal = 0 ## my added\n'),
        ('                    stats_episodes[(k, ep_eval_count[k])] = episode_stats\n',
         '                    stats_episodes[(k, ep_eval_count[k])] = episode_stats\n'
         '                    # VLN audit: record the same scalars already used upstream.\n'
         '                    audit_row = {\n'
         '                        "scene_id": k[0], "episode_id": str(k[1]),\n'
         '                        "eval_index": ep_eval_count[k],\n'
         '                        "action_steps": len(actions_record[(k[0], k[1], ep_eval_count[k] - 1)]),\n'
         '                        "metrics": episode_stats,\n'
         '                    }\n'
         '                    with open(episode_log, "a", encoding="utf-8") as audit_file:\n'
         '                        audit_file.write(json.dumps(audit_row, allow_nan=False) + "\\n")\n'
         '                        audit_file.flush()\n'
         '                        os.fsync(audit_file.fileno())\n'),
        ('        result_path = os.path.join("output/", "result.json")',
         '        result_path = os.path.join(audit_output, "result.json")'),
        ('        actions_output_path = os.path.join("output/", "actions.json")',
         '        actions_output_path = os.path.join(audit_output, "actions.json")'),
    ]
    for before, after in replacements:
        if derived.count(before) != 1:
            raise ValueError('Expected exactly one audited source anchor: ' + before[:70])
        derived = derived.replace(before, after, 1)
    derived = derived.rstrip() + '\n\n' + (
        '        # VLN audit: unrounded aggregate and completion marker.\n'
        '        with open(os.path.join(audit_output, "aggregate_metrics.json"), "w") as audit_file:\n'
        '            json.dump(aggregated_stats, audit_file, indent=2, allow_nan=False)\n'
        '        with open(os.path.join(audit_output, "evaluation_complete.json"), "w") as audit_file:\n'
        '            json.dump({"completed_episode_records": len(stats_episodes),\n'
        '                       "requested_episode_count": number_of_eval_episodes,\n'
        '                       "evals_per_episode": evals_per_ep,\n'
        '                       "checkpoint_index": checkpoint_index, "training_step": step_id},\n'
        '                      audit_file, indent=2, allow_nan=False)\n'
    )
    ast.parse(derived)
    # Pinned source plus exact replacements preserve the policy call byte-for-byte.
    if derived.count('deterministic=False,') != original.count('deterministic=False,'):
        raise ValueError('Policy sampling mode changed')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    module = output / 'vln_falcon_evaluator.py'
    module.write_text(derived)
    diff_lines = difflib.unified_diff(original.splitlines(True), derived.splitlines(True),
                                     fromfile=EVALUATOR_PATH, tofile='vln_falcon_evaluator.py')
    patch = ''.join(line if line.endswith('\n') else line + '\n\\ No newline at end of file\n'
                    for line in diff_lines)
    (output / 'observability.patch').write_text(patch)
    metadata = {'official_commit': OFFICIAL_COMMIT, 'source_path': EVALUATOR_PATH,
                'source_sha256': digest, 'derived_sha256': hashlib.sha256(derived.encode()).hexdigest(),
                'patch_sha256': hashlib.sha256(patch.encode()).hexdigest(),
                'changes': ['per-episode scalar JSONL', 'output paths', 'unrounded aggregate', 'completion marker'],
                'policy_deterministic': False, 'original_checkout_modified': False}
    (output / 'observability_manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(derive(args.repo, args.output), indent=2))
