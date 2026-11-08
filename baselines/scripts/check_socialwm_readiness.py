#!/usr/bin/env python3
"""Offline audit gate for Social-WM inputs; does not import torch or run a model.

This checks recorded material completeness and source integrity. It cannot certify
that a new repository is official or that a new implementation matches the paper.
All null research parameters must be resolved with cited release evidence first.
Exit 2 means blocked/incomplete; exit 0 means material checks passed, not reproduced.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contract', type=Path, default=root / 'baselines/prep/socialwm/reproduction_contract.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode', choices=('eval', 'train'), default='eval')
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    blockers, checked = [], []
    for field, value in contract['official_release'].items():
        if field == ('training_entrypoint' if args.mode == 'eval' else 'evaluation_entrypoint'):
            continue
        if not value:
            blockers.append('official_release.' + field + ' missing')
    source_dir = contract['official_release']['source_dir']
    if source_dir and not Path(source_dir).is_dir():
        blockers.append('official source directory does not exist')
    for name, entry in contract['artifacts'].items():
        if name == ('training_rollout_manifest' if args.mode == 'eval' else 'socialwm_checkpoint'):
            continue
        for field in ('path', 'sha256', 'official_url'):
            if not entry[field]:
                blockers.append('artifacts.' + name + '.' + field + ' missing')
        if entry['path']:
            path = Path(entry['path'])
            if not path.is_file():
                blockers.append(name + ' file missing')
            elif not entry['sha256'] or sha256(path) != entry['sha256']:
                blockers.append(name + ' SHA256 mismatch or missing expected SHA')
            else:
                checked.append(name)
    training_only = {
        'collection_gate_geometry_radius_and_policy', 'collection_behavior_policy_and_seeds',
        'training_200_scene_ids_and_episode_ids', 'train_validation_split_ids',
        'absolute_learning_rates', 'batch_size_epochs_steps_schedule_weight_decay',
        'lambda_tf_lambda_ol_lambda_sig_lambda_id_beta', 'pretrained_trunk_identity_and_checkpoint'
    }
    for key, value in contract['unresolved_parameters'].items():
        if args.mode == 'eval' and key in training_only:
            continue
        if value is None:
            blockers.append('parameter unresolved: ' + key)
    integrity = []
    for manifest_name in ('sources.json', 'upstream_lewm_reference_manifest.json'):
        manifest = json.loads((root / 'baselines/prep/socialwm' / manifest_name).read_text())
        for item in manifest.get('sources', manifest.get('files', [])):
            if 'path' not in item:
                continue  # Recorded HTTP failures are evidence, not local assets.
            path = root / item['path']
            ok = path.is_file() and sha256(path) == item['sha256']
            integrity.append({'path': item['path'], 'sha256_matches': ok})
            if not ok:
                blockers.append('source snapshot integrity failed: ' + item['path'])
    result = {
        'work': 'Social-WM', 'mode': args.mode,
        'status': 'BLOCKED_MISSING_OFFICIAL_INPUTS' if blockers else 'MATERIAL_CHECKS_PASSED',
        'official_reproduction_executed': False, 'gpu_requested': False,
        'blockers': blockers, 'checked_runtime_artifacts': checked,
        'source_snapshot_checks': integrity,
        'warning': 'Passing material checks is not a runtime, correctness, or paper-result claim.'
    }
    text = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end='')
    return 2 if blockers else 0


if __name__ == '__main__':
    raise SystemExit(main())
