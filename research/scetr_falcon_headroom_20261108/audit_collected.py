"""Post-collection audit, separate from the frozen experiment implementation."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(output):
    from validate import validate
    protocol = read(HERE / 'PROTOCOL.json')
    manifest = read(HERE / 'episodes.json')
    schedule = read(output / 'SCHEDULE.json')
    result = validate(output, write=False)
    assert result == read(output / 'VALIDATION.json'), 'remote/local result differs'
    assert [r['case'] for r in schedule['episodes']] == list(range(4))
    all_names = set()
    actor_hashes = set()
    total_steps = 0
    for case, entry in zip(manifest['cases'], schedule['episodes']):
        names = [entry['reference'], entry['repeat']]
        expected = []
        seen = set()
        if entry['first_collision'] is not None:
            for j, spec in enumerate(protocol['interventions']):
                start = max(0, entry['first_collision'] - spec['lead_steps'])
                identity = (start, tuple(spec['sequence']))
                if identity in seen:
                    continue
                seen.add(identity)
                expected.append(dict(directory=f"case{entry['case']}_branch{j}",
                                     start=start, sequence=spec['sequence'],
                                     lead_steps=spec['lead_steps']))
        assert entry['branches'] == expected, 'missing/extra/reordered candidate'
        names.extend(b['directory'] for b in expected)
        for name in names:
            assert name not in all_names
            all_names.add(name)
            directory = output / name
            trace = read(directory / 'trace.json')
            meta = read(directory / 'bridge_result.json')
            inputs = read(directory / 'INPUTS.json')
            cfg = read(directory / 'CONFIG.json')
            assert inputs['case'] == case
            assert inputs['manifest_sha256'] == sha(HERE / 'episodes.json')
            assert cfg['habitat']['seed'] == protocol['seed']
            assert cfg['habitat_baselines']['num_environments'] == 1
            assert cfg['habitat']['environment']['max_episode_steps'] == 500
            resolved_scene = str(Path(cfg['habitat']['dataset']['scenes_dir']) / case['scene_id'])
            assert all(r['episode'] == [resolved_scene, case['episode_id']] for r in trace)
            assert trace[-1]['metrics']['num_steps'] == len(trace)
            assert all(np.isfinite(list(r['metrics'].values())).all() for r in trace)
            arrays = np.load(directory / 'features.npz', allow_pickle=False)
            assert arrays['depth_thumbnails'].shape == (len(trace), 32, 32)
            for i, row in enumerate(trace):
                for array_key, hash_key in [('features', 'feature_sha'), ('logits', 'logit_sha')]:
                    assert hashlib.sha256(arrays[array_key][i].tobytes()).hexdigest() == row[hash_key]
            actor_hashes.add(meta['actor_before_sha'])
            total_steps += len(trace)
    assert len(actor_hashes) == 1
    assert len(all_names) == result['terminal_rollouts'] <= protocol['maximum_cases']
    assert {p.name for p in output.glob('case*') if p.is_dir()} == all_names
    result.update(total_environment_steps=total_steps,
                  independent_manifest_schedule_array_audit='PASS',
                  actor_sha256=next(iter(actor_hashes)),
                  files={str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                         if p.is_file() and p.name != 'LOCAL_AUDIT.json'})
    (output / 'LOCAL_AUDIT.json').write_text(json.dumps(result, indent=2) + '\n')
    return {k: v for k, v in result.items() if k not in ['results', 'files']}


def extract(archive, destination):
    destination.mkdir(exist_ok=False)
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        seen = set()
        for item in members:
            path = Path(item.name)
            assert not path.is_absolute() and '..' not in path.parts
            assert item.isfile() or item.isdir(), 'links and devices forbidden'
            assert item.name not in seen, 'duplicate archive member'
            seen.add(item.name)
        bundle.extractall(destination, members=members, filter='data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--archive', type=Path)
    args = parser.parse_args()
    if args.archive:
        extract(args.archive, args.output)
    print(json.dumps(audit(args.output), indent=2))
