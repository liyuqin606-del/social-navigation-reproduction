#!/usr/bin/env python3
"""Generate a bounded Falcon managed-job specification; no SSH or GPU allocation.

Pass a subset/full/shard episode manifest, or a shard-plan manifest with
--shard-id. All inputs must be prepared remotely before the managed owner runs.
--deadline is interpreted by Slurm on the server; refresh it before freezing.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

LOCAL = Path(__file__).resolve().parents[2]
DEFAULT_REMOTE = '/mnt/home/liyuqin/USERDIR/vln_baselines_20261103'
SCRIPTS = ('run_falcon_subset.py', 'prepare_falcon_observed.py', 'validate_falcon_subset.py')
EVAL_OUTPUTS = (
    'run_status.json', 'run_manifest.json', 'source_episode_manifest.json', 'expected_episodes.json',
    'resolved_config.yaml', 'resolved_config.json',
    'instrumentation/vln_falcon_evaluator.py', 'instrumentation/observability.patch',
    'instrumentation/observability_manifest.json', 'episode_metrics.jsonl', 'actions.json',
    'result.json', 'aggregate_metrics.json', 'evaluation_complete.json', 'validation.json',
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def local_path(value):
    path = Path(value)
    return path if path.is_absolute() else LOCAL / path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--shard-id', help='Required when --manifest names the all-shards plan')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--remote-root', default=DEFAULT_REMOTE)
    p.add_argument('--remote-python')
    p.add_argument('--scenes-dir', help='Remote scene root override; full/shards default to isolated fullval_assets')
    p.add_argument('--data-path', help='Remote val.json.gz override; shards default to the prepared full-val data with exact scene filtering')
    p.add_argument('--deadline', required=True, help='Future server-local Slurm deadline YYYY-MM-DDTHH:MM:SS')
    p.add_argument('--num-envs', type=int, choices=(1, 2, 4, 8))
    p.add_argument('--cpus', type=int)
    p.add_argument('--wall-seconds', type=int, default=3600)
    p.add_argument('--science-timeout-seconds', type=int)
    p.add_argument('--output-budget-mib', type=int)
    p.add_argument('--env-json', type=Path, help='Optional extra nonsecret runtime environment variables')
    p.add_argument('--readiness-file', type=Path, action='append', default=[], help='Local env readiness / pip-freeze evidence to freeze; repeatable')
    p.add_argument('--extra-remote-inputs', type=Path, help='Optional JSON list of exact remote path/bytes/SHA records')
    a = p.parse_args()
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', a.deadline):
        p.error('Use an explicit server-local deadline YYYY-MM-DDTHH:MM:SS')
    datetime.strptime(a.deadline, '%Y-%m-%dT%H:%M:%S')
    if not 240 <= a.wall_seconds <= 3600:
        p.error('wall-seconds must be 240..3600')
    validation_timeout = 120
    science_timeout = a.science_timeout_seconds or a.wall_seconds - 180
    if science_timeout < 1 or science_timeout + validation_timeout + 30 > a.wall_seconds:
        p.error('Need science timeout + 120s validation + 30s finalization within wall time')
    selection = a.manifest.absolute()
    document = json.loads(selection.read_text())
    scientific = [selection, Path(__file__).resolve()]
    if 'shards' in document:
        if a.shard_id is None:
            p.error('A shard-plan manifest requires --shard-id (e.g. 00)')
        matches = [row for row in document['shards'] if row['shard_id'] == a.shard_id]
        if len(matches) != 1:
            p.error('shard-id must identify exactly one entry')
        selection = local_path(matches[0]['manifest'])
        document = json.loads(selection.read_text())
        scientific.append(selection)
    elif a.shard_id is not None and document.get('shard_id') != a.shard_id:
        p.error('shard-id does not match the supplied episode manifest')
    keys = [tuple(key) for key in document['episode_keys']]
    if document.get('source_split') != 'val' or len(keys) != len(set(keys)) or len(keys) != document['episode_count']:
        p.error('Invalid official val episode manifest')
    if document.get('scene_id_rewritten') or document.get('episode_content_rewritten'):
        p.error('This generator requires unmodified official episode content')
    if not document.get('mesh_file_entries_verified') or not document.get('source_mesh_package'):
        p.error('Verify the exact official scene-package entries before creating an executable spec')
    if 'shard_id' in document:
        mode = 'shard' + document['shard_id']
        default_envs = 1
    elif document['episode_count'] == 1087 and document['scene_count'] == 56:
        mode, default_envs = 'full1087', 8
    elif document['episode_count'] == 174 and document['scene_count'] == 9:
        mode, default_envs = 'subset174', 4
    else:
        p.error('Use the verified subset174, full1087, or a full-val shard manifest')
    num_envs = a.num_envs or default_envs
    if num_envs > document['scene_count']:
        p.error('num-envs exceeds the scene count')
    cpus = a.cpus or max(4, num_envs + 1)
    if cpus < num_envs:
        p.error('Allocate at least one CPU per environment')
    remote = Path(a.remote_root)
    if not remote.is_absolute():
        p.error('remote-root must be absolute')
    repo = remote / 'src/Falcon'
    full_geometry = mode != 'subset174'
    scenes_dir = Path(a.scenes_dir) if a.scenes_dir else (remote / 'fullval_assets/scene_datasets' if full_geometry else repo / 'data/scene_datasets')
    data_path = Path(a.data_path) if a.data_path else (repo / 'data/datasets/pointnav/social-hm3d-fullval/val/val.json.gz' if full_geometry else repo / document['dataset_root_path_in_bundle'])
    if not scenes_dir.is_absolute() or not data_path.is_absolute():
        p.error('Runtime scenes-dir and data-path must be absolute remote paths')
    py = a.remote_python or str(remote / 'env/falcon/bin/python')
    now = datetime.now(timezone.utc)
    run_id = 'vln_falcon_' + mode + '_' + now.strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8]
    assert re.fullmatch(r'[A-Za-z0-9_-]{1,63}', run_id)
    inputs = []

    def bind(local, destination, expected=None):
        path = local_path(local)
        row = {'path': str(destination), 'bytes': path.stat().st_size, 'sha256': sha(path)}
        if expected and any(row[k] != expected[k] for k in ('bytes', 'sha256')):
            raise ValueError('Local input no longer matches audited manifest: ' + str(path))
        inputs.append(row)
        return path

    for name in ('falcon_noaux_25.pth', 'pretrained_habitat3.pth'):
        bind(LOCAL / 'baselines/assets/falcon/pretrained_model' / name, remote / 'inputs' / name)
    for name in ('falcon_source.tar.gz', 'falcon_robot_assets_minimal.tar.gz'):
        bind(LOCAL / 'baselines/prep' / name, remote / 'inputs' / name)
    if full_geometry:
        # The transferred metadata bundle contains the full episode tar and all
        # shard manifests, not the eight redundant shard tar files. Filtering
        # the prepared full root uses byte-identical per-scene episode shards.
        full_manifest_path = local_path(document.get('parent_full_manifest', 'baselines/prep/falcon_full_val_20261103_manifest.json'))
        full_document = json.loads(full_manifest_path.read_text())
        scientific.append(full_manifest_path)
        bundle = full_document['bundle']
        bind(bundle['path'], remote / 'inputs/full_val_manifests' / Path(bundle['path']).name, bundle)
        geometry_path = LOCAL / 'baselines/prep/falcon_full_val_geometry_selected56_20261103_manifest.json'
        geometry = json.loads(geometry_path.read_text())
        if (geometry.get('status') != 'COMPLETE_DERIVED_SELECTION_INTEGRITY_RECORDED'
                or not geometry.get('file_bytes_preserved') or not geometry.get('paths_preserved')
                or geometry['source_mesh_package']['sha256'] != document['source_mesh_package']['sha256']):
            p.error('Derived geometry selection is not linked to the audited original source')
        members = {row['path'] for row in geometry['files']}
        for scene in document['scenes']:
            rel = Path(scene['scene_id'].removeprefix('hm3d/val/'))
            if str(rel) not in members or str(rel.with_suffix('.navmesh')) not in members:
                p.error('Derived geometry selection does not cover a required scene')
        scientific.append(geometry_path)
        mesh = geometry['bundle']
        mesh_name = Path(mesh['path']).name
        mesh_kind = 'DERIVED_SELECTION_NOT_OFFICIAL_ORIGINAL_ARCHIVE'
    else:
        bundle = document['bundle']
        bind(bundle['path'], remote / 'inputs' / Path(bundle['path']).name, bundle)
        mesh = document['source_mesh_package']
        mesh_name = Path(mesh['file']).name
        mesh_kind = 'OFFICIAL_ORIGINAL_ARCHIVE'
    # Large mesh bytes/SHA are taken from the prior archive audit, then verified
    # against the remote archive by the managed owner before sbatch.
    if not re.fullmatch(r'[0-9a-f]{64}', mesh['sha256']) or mesh['bytes'] <= 0:
        p.error('Missing audited mesh archive size/SHA')
    inputs.append({'path': str(remote / 'inputs' / mesh_name),
                   'bytes': mesh['bytes'], 'sha256': mesh['sha256']})
    if a.extra_remote_inputs:
        extra = json.loads(a.extra_remote_inputs.read_text())
        if not isinstance(extra, list):
            p.error('extra-remote-inputs must contain a list')
        inputs.extend(extra)
        scientific.append(a.extra_remote_inputs.absolute())
    unique = {}
    for row in inputs:
        if (not Path(row['path']).is_absolute() or type(row['bytes']) is not int or row['bytes'] < 0
                or not re.fullmatch(r'[0-9a-f]{64}', row['sha256'])):
            p.error('Invalid remote input digest record')
        if row['path'] in unique and unique[row['path']] != row:
            p.error('Conflicting remote input digest records')
        unique[row['path']] = row
    env = {'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'MAGNUM_LOG': 'quiet', 'HABITAT_SIM_LOG': 'quiet'}
    if a.env_json:
        env.update(json.loads(a.env_json.read_text()))
        scientific.append(a.env_json.absolute())
    if any(re.search(r'(TOKEN|PASSWORD|SECRET|API_KEY)', key, re.I) for key in env):
        p.error('Do not freeze secret environment values')
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        p.error('All runtime environment names and values must be strings')
    for path in a.readiness_file:
        if not path.is_file():
            p.error('Readiness evidence does not exist: ' + str(path))
        scientific.append(path.absolute())
    for path in ('baselines/audits/falcon_source_manifest.json', 'baselines/prep/falcon_robot_assets_manifest.json'):
        scientific.append(LOCAL / path)
    if document.get('mesh_qa_manifest'):
        scientific.append(local_path(document['mesh_qa_manifest']))
    uploads = [{'file': str(LOCAL / 'baselines/scripts' / name), 'remote_name': name} for name in SCRIPTS]
    uploads.append({'file': str(selection), 'remote_name': 'episode_manifest.json'})
    # Conservative planning estimate, not a measured result: at most 500 steps
    # per episode, 1KiB per action record plus 32MiB for metrics/logs/provenance.
    estimated = len(keys) * 500 * 1024 + (32 << 20)
    budget = (a.output_budget_mib << 20) if a.output_budget_mib else math.ceil(estimated / (64 << 20)) * (64 << 20)
    if budget < 16 << 20:
        p.error('Output budget must be at least 16MiB')
    wall = '%02d:%02d:%02d' % (a.wall_seconds // 3600, a.wall_seconds % 3600 // 60, a.wall_seconds % 60)
    spec = {
        'run_id': run_id, 'remote_root': str(remote / 'runs'), 'remote_python': py,
        'gpu_binding': 'slurm_linux_minor_uuid',
        'slurm': {'partition': 'tacc', 'gpus': 1, 'cpus': cpus, 'wall_time': wall, 'deadline': a.deadline},
        'science': {
            'cwd': str(repo),
            'argv': [py, '{source}/run_falcon_subset.py', '--repo', str(repo),
                     '--manifest', '{source}/episode_manifest.json',
                     '--checkpoint', str(remote / 'inputs/falcon_noaux_25.pth'),
                     '--pretrained', str(remote / 'inputs/pretrained_habitat3.pth'),
                     '--data-path', str(data_path),
                     '--scenes-dir', str(scenes_dir), '--num-envs', str(num_envs),
                     '--output', '{output}/eval'],
            'env': env, 'timeout_seconds': science_timeout,
            'validator_argv': [py, '{source}/validate_falcon_subset.py', '--output', '{output}/eval'],
            'validation_timeout_seconds': validation_timeout,
            'required_outputs': ['eval/' + name for name in EVAL_OUTPUTS],
        },
        'uploads': uploads, 'remote_inputs': list(unique.values()),
        'scientific_files': list(dict.fromkeys(str(path) for path in scientific)),
        'reserve_bytes': 4 << 30, 'output_budget_bytes': budget, 'temporary_transfer_bytes': budget,
        'remote_peak_additional_bytes': 2 * budget + (8 << 20), 'monitor_timeout_seconds': 10800,
        'evaluation_plan': {
            'mode': mode, 'scene_count': document['scene_count'], 'episode_count': len(keys),
            'geometry_archive_kind': mesh_kind, 'geometry_archive_sha256': mesh['sha256'],
            'original_mesh_archive_sha256': document['source_mesh_package']['sha256'],
            'runtime_scenes_dir': str(scenes_dir), 'runtime_data_path': str(data_path),
            'episode_storage_note': 'Shards use the prepared full-val root with exact content_scenes and original per-scene SHA checks' if 'shard_id' in document else 'Manifest-defined official episode collection',
            'num_environments': num_envs, 'status': 'SPEC_ONLY_NOT_FROZEN_NOT_SUBMITTED',
            'runtime_seconds_estimated': None, 'guaranteed_to_finish_within_one_hour': False,
            'output_estimate_rule': '500 steps/episode, 1KiB/action record, plus 32MiB metadata; rounded up to 64MiB',
            'preflight_note': 'Archives must be verified AND unpacked at exact runtime paths; GPU EGL/smoke must pass. No downloading/installing occurs in the job.',
            'scope_note': 'A shard or subset is not full-val coverage. Full1087 is one released checkpoint evaluation, not three independent training seeds.',
            'shard_aggregation_note': 'Require disjoint exact-key union; aggregate episode rows, never unweighted shard means; do not add the earlier subset run.',
            'failure_retention_note': 'Fixed required outputs and managed logs are collected when present. Python exception traceback is also in scientific.log. Dynamic tensorboard files remain remote.',
        },
    }
    a.output = a.output.absolute()
    if a.output.exists():
        p.error('Refusing to overwrite an existing spec; choose a fresh filename')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps({'status': 'SPEC_ONLY_NOT_FROZEN_NOT_SUBMITTED', 'path': str(a.output),
                      'run_id': run_id, 'mode': mode, 'episodes': len(keys), 'num_envs': num_envs,
                      'gpus': 1, 'wall_time': wall, 'required_outputs': len(EVAL_OUTPUTS),
                      'output_budget_mib': budget // (1 << 20)}, indent=2))


if __name__ == '__main__':
    main()
