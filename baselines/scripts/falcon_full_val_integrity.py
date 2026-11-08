#!/usr/bin/env python3
"""Read-only integrity checks for collected Falcon full-val shard results."""
from collections import Counter
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import tempfile

from validate_falcon_subset import validate as validate_subset

CHECKPOINT = 'ba7f2824604639d934be2359226c9e9fb35d5b1b2fb39ffbcbbbd833252db6ca'
INITIALIZATION = 'b73ca263e986d584b5c89073c8046e2b2cded757a31a2000a18f2774838fa83f'
COMMIT = 'd15ef512c4b1451f9daeaf6d4d7ecc702ed57430'
EVALUATOR = '3fbfcfb0baa881c57ceb1596c6e0284f0bd1bcfa81d92805e10a887e9ef23380'
REQUIRED_METRICS = ('success', 'spl', 'psc', 'human_collision')
INPUT_FILES = ('run_status.json', 'run_manifest.json', 'source_episode_manifest.json',
               'expected_episodes.json', 'resolved_config.json', 'resolved_config.yaml',
               'episode_metrics.jsonl', 'actions.json', 'result.json', 'aggregate_metrics.json',
               'evaluation_complete.json', 'validation.json',
               'instrumentation/observability_manifest.json',
               'instrumentation/vln_falcon_evaluator.py', 'instrumentation/observability.patch')
CONFIG_PATH_EXCEPTIONS = (
    'habitat.dataset.content_scenes', 'habitat.dataset.scenes_dir', 'habitat.dataset.data_path',
    'habitat_baselines.tensorboard_dir', 'habitat_baselines.video_dir',
    'habitat_baselines.checkpoint_folder', 'habitat_baselines.log_file',
    'habitat_baselines.eval_ckpt_path_dir', 'habitat_baselines.rl.ddppo.pretrained_weights')
LIMITATIONS = [
    'Each shard restarts seed 100; its scene assignment and random stream differ from one continuous full-val evaluation.',
    'This is one released checkpoint, not three independent training seeds or full-paper reproduction.',
    'The earlier 174-episode pilot is excluded; full coverage is accepted only as one disjoint union of declared full-val shards.']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    path = Path(path)
    return {'path': str(path.absolute()), 'bytes': path.stat().st_size, 'sha256': digest(path)}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def exact_keys(document):
    keys = document['episode_keys']
    require(all(isinstance(k, list) and len(k) == 2 and all(isinstance(v, str) for v in k)
                for k in keys), 'Episode keys must be exact [scene_id, string episode_id] pairs')
    keys = [tuple(k) for k in keys]
    require(len(keys) == len(set(keys)) == document['episode_count'], 'Manifest duplicate keys or count mismatch')
    return keys


def check_disjoint_coverage(expected, parts):
    expected = set(expected)
    counts = Counter(key for part in parts for key in part)
    duplicate = [list(key) for key, count in counts.items() if count != 1]
    missing = sorted(expected - set(counts))
    unexpected = sorted(set(counts) - expected)
    require(not duplicate, 'Repeated episode keys across shards: ' + repr(duplicate[:5]))
    require(not unexpected, 'Unexpected episode keys: ' + repr(unexpected[:5]))
    require(not missing, 'Missing full-val episode keys: %d' % len(missing))
    return {'expected_episodes': len(expected), 'observed_episodes': sum(counts.values()),
            'unique_episodes': len(counts), 'unique_scenes': len({key[0] for key in counts}),
            'missing': [], 'duplicate': [], 'unexpected': []}


def runtime_mapping(scene_root, original_keys):
    root = PurePosixPath(scene_root)
    require(root.is_absolute() and '..' not in root.parts, 'Runtime scene root must be an absolute normalized path')
    mapping = {}
    for original in original_keys:
        relative = PurePosixPath(original[0].removeprefix('data/scene_datasets/'))
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe original scene path')
        key = (str(root / relative), original[1])
        require(key not in mapping, 'Runtime scene mapping is not one-to-one')
        mapping[key] = original
    return mapping


def metric_columns(values):
    require(isinstance(values, dict) and all(type(v) in (int, float) and math.isfinite(v)
            for v in values.values()), 'Nonfinite/non-numeric episode metric')
    require(set(REQUIRED_METRICS) <= set(values), 'Missing official metric')
    require(all(0 <= values[name] <= 1 for name in REQUIRED_METRICS), 'Official metric is outside [0,1]')
    return set(values)


def successful_worker(directory, job_id, managed_plan=None):
    # The same CPU evidence check is used by the queue and final aggregation.
    # A timeout exception retains one completed child, never parent success.
    from falcon_shard_completion_evidence import completion_evidence
    return completion_evidence(directory, job_id, managed_plan)


def revalidate_without_writing_input(directory):
    stored = load(directory / 'validation.json')
    require(stored.get('status') == 'PASS_SUBSET_COMPLETE', 'Input shard was not previously validated')
    # The official project validator writes validation.json. Use links to inputs
    # with a fresh unlinked output filename, preserving collected evidence bytes.
    with tempfile.TemporaryDirectory(prefix='falcon-shard-revalidate-') as tmp:
        target = Path(tmp)
        for child in directory.iterdir():
            if child.name != 'validation.json':
                (target / child.name).symlink_to(child, target_is_directory=child.is_dir())
        result = validate_subset(target)
    require(result['status'] == 'PASS_SUBSET_COMPLETE', 'Shard revalidation failed: ' + repr(result['failures'][:5]))
    return result


def protocol(config, provenance):
    h, b = config['habitat'], config['habitat_baselines']
    require(h['seed'] == 100 and h['environment']['max_episode_steps'] == 500, 'Resolved seed/step protocol differs')
    require(h['environment']['iterator_options']['shuffle'] is False, 'Episode iterator shuffle changed')
    require(h['dataset']['split'] == 'val' and b['eval']['split'] == 'val', 'Resolved split differs')
    require(b['num_environments'] == 8 and b['test_episode_count'] == -1, 'Requires eight environments and all shard episodes')
    require(b['evaluate'] is True and b['eval']['evals_per_ep'] == 1 and
            b['eval']['should_load_ckpt'] is True and b['eval']['use_ckpt_config'] is False,
            'Resolved checkpoint evaluation protocol changed')
    require(b['evaluator']['_target_'] == 'vln_falcon_evaluator.FALCONEvaluator', 'Unexpected evaluator')
    require(provenance.get('official_commit') == COMMIT and provenance.get('seed') == 100 and
            provenance.get('policy_deterministic') is False and provenance.get('max_episode_steps') == 500 and
            provenance.get('num_environments') == 8, 'Requires official seed100/stochastic/500-step/eight-env provenance')
    require(provenance['inputs']['checkpoint']['sha256'] == CHECKPOINT, 'Released Falcon checkpoint SHA changed')
    require(provenance['inputs']['pointnav_initialization']['sha256'] == INITIALIZATION, 'PointNav initialization SHA changed')
    require(provenance['instrumentation']['derived_sha256'] == EVALUATOR, 'Evaluator instrumentation changed')
    require(b['eval_ckpt_path_dir'] == provenance['inputs']['checkpoint']['path'] and
            b['rl']['ddppo']['pretrained_weights'] == provenance['inputs']['pointnav_initialization']['path'] and
            h['dataset']['data_path'] == provenance['inputs']['episode_root']['path'],
            'Resolved model/data paths differ from hashed runtime input provenance')
    stripped = copy.deepcopy(config)
    for dotted in CONFIG_PATH_EXCEPTIONS:
        *parents, field = dotted.split('.')
        node = stripped
        for part in parents:
            node = node[part]
        del node[field]
    return {'config_excluding_only_physical_paths_and_scene_selection': stripped,
            'checkpoint_sha256': CHECKPOINT, 'initialization_sha256': INITIALIZATION,
            'official_commit': COMMIT, 'evaluator_sha256': EVALUATOR,
            'runner_sha256': provenance['runner_sha256'], 'validator_sha256': provenance['validator_sha256'],
            'base_config': provenance['base_config'], 'runtime_compatibility': provenance.get('runtime_compatibility', []),
            'torch_version': provenance['torch_version'], 'cuda_version': provenance['cuda_version']}


def collect(full_manifest, directories, *, single_shard=False, managed_plan=None):
    full_path = Path(full_manifest).resolve()
    full = load(full_path)
    expected = exact_keys(full)
    require(full['source_split'] == 'val' and len(expected) == 1087 and full['scene_count'] == 56,
            'Expected the official 1087-episode/56-scene full-val manifest')
    require(full.get('scene_id_rewritten') is False and full.get('episode_content_rewritten') is False and
            full['base_config'] == 'social_nav_v2/falcon_hm3d.yaml', 'Full manifest is not the original official Falcon protocol')
    full_scenes = {s['scene_id']: s for s in full['scenes']}
    require(len(full_scenes) == 56 and {key[0] for key in expected} == set(full_scenes), 'Full-val scene identity mismatch')
    directories = [Path(path).resolve() for path in directories]
    require((len(directories) == 1 if single_shard else len(directories) >= 2) and
            len(set(directories)) == len(directories), 'Supply one checkpoint shard or two or more distinct aggregate shard directories')
    rows, part_keys, shard_records, seen_ids = [], [], [], set()
    common_protocol, common_names = None, None
    for directory in directories:
        require(directory.is_dir(), 'Missing shard output: ' + str(directory))
        source = load(directory / 'source_episode_manifest.json')
        original_keys = exact_keys(source)
        shard_id = source.get('shard_id')
        require(isinstance(shard_id, str) and shard_id not in seen_ids and source.get('parent_full_manifest'),
                'A full-val shard ID/parent is required and must be unique; do not include the 174-episode pilot')
        seen_ids.add(shard_id)
        require(Path(source['parent_full_manifest']).name == full_path.name, 'Shard parent full manifest differs')
        require(source['source_split'] == 'val' and source.get('scene_id_rewritten') is False and
                source.get('episode_content_rewritten') is False, 'Shard is not original official val content')
        require(source['source_mesh_package']['sha256'] == full['source_mesh_package']['sha256'], 'Geometry source archive differs')
        require(source['base_config'] == full['base_config'], 'Shard base config differs')
        require(set(original_keys) <= set(expected), 'Shard includes an episode outside full val')
        for scene in source['scenes']:
            reference = full_scenes.get(scene['scene_id'])
            require(reference is not None and all(scene[k] == reference[k] for k in
                    ('scene_name', 'sha256', 'bytes', 'episode_count', 'episode_ids')), 'Shard scene content differs from full manifest')
        declared = [(s['scene_id'], ep) for s in source['scenes'] for ep in s['episode_ids']]
        require(Counter(declared) == Counter(original_keys), 'Shard scene list differs from its exact episode keys')
        require(len(source['scenes']) == source['scene_count'] and set(source['content_scenes']) ==
                {s['scene_name'] for s in source['scenes']}, 'Shard scene counts/names are inconsistent')
        provenance = load(directory / 'run_manifest.json')
        require(load(directory / 'run_status.json').get('status') == 'PASS_SUBSET_COMPLETE', 'Shard run did not finish successfully')
        worker = successful_worker(directory, provenance['slurm_job_id'], managed_plan)
        config = load(directory / 'resolved_config.json')
        current = protocol(config, provenance)
        if common_protocol is None:
            common_protocol = current
        require(current == common_protocol, 'Scientific configuration/protocol differs between shards')
        require(config['habitat']['dataset']['content_scenes'] == source['content_scenes'], 'Resolved scene selection differs from shard')
        require(provenance['episode_count'] == len(original_keys) and provenance['scene_count'] == source['scene_count'], 'Runtime counts differ from source manifest')
        source_sha = digest(directory / 'source_episode_manifest.json')
        require(provenance['inputs']['source_manifest']['sha256'] == source_sha, 'Source manifest provenance hash differs')
        require(provenance['inputs']['episode_root']['sha256'] == full['episode_root']['sha256'], 'Episode root bytes differ from full val')
        require(sorted(r['sha256'] for r in provenance['inputs']['shards']) == sorted(s['sha256'] for s in source['scenes']), 'Runtime episode shard digests differ')
        mapping = runtime_mapping(config['habitat']['dataset']['scenes_dir'], original_keys)
        document = load(directory / 'expected_episodes.json')
        require(Counter(map(tuple, document['episode_keys'])) == Counter(mapping.keys()) and
                Counter(map(tuple, document['original_episode_keys'])) == Counter(original_keys) and
                document['source_manifest_sha256'] == source_sha, 'Runtime-to-original key mapping differs from source input')
        original_validation = revalidate_without_writing_input(directory)
        require(digest(directory / 'instrumentation/vln_falcon_evaluator.py') == EVALUATOR,
                'Actual evaluator differs from fixed official-derived observation source')
        observed = []
        for line in (directory / 'episode_metrics.jsonl').read_text().splitlines():
            row = json.loads(line)
            runtime_key = (row['scene_id'], row['episode_id'])
            require(runtime_key in mapping, 'Observed episode cannot be normalized with exact runtime scene root')
            key = mapping[runtime_key]
            observed.append(key)
            values = row['metrics']
            names = metric_columns(values)
            if common_names is None:
                common_names = names
            require(names == common_names, 'Metric columns differ across episodes; no unequal metric denominators allowed')
            rows.append({**row, 'scene_id': key[0], 'episode_id': key[1],
                         'runtime_scene_id': runtime_key[0], 'source_shard_id': shard_id})
        require(Counter(observed) == Counter(original_keys), 'Actual shard episodes differ from the source manifest')
        part_keys.append(observed)
        shard_records.append({'shard_id': shard_id, 'output_directory': str(directory),
                              'slurm_job_id': provenance['slurm_job_id'], 'episode_count': len(observed),
                              'scene_count': source['scene_count'], 'completion_evidence': worker,
                              'interrupted_parent': worker['interrupted_parent'],
                              'files': [record(directory / name) for name in INPUT_FILES],
                              'revalidation_status': original_validation['status']})
    if single_shard:
        coverage = {'scope': 'ONE_SHARD_ONLY_NOT_FULL_VAL', 'completed_shard_episodes': len(part_keys[0]),
                    'completed_shard_scenes': len({key[0] for key in part_keys[0]}),
                    'full_val_expected_episodes': 1087, 'full_val_expected_scenes': 56}
    else:
        coverage = check_disjoint_coverage(expected, part_keys)
        require(coverage['unique_scenes'] == 56, 'Full-val scene union does not contain exactly 56 scenes')
    return {'rows': rows, 'coverage': coverage, 'shards': shard_records,
            'full_manifest': record(full_path), 'protocol': common_protocol,
            'protocol_sha256': hashlib.sha256(canonical(common_protocol).encode()).hexdigest()}
