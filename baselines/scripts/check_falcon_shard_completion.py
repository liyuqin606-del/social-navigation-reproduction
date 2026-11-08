#!/usr/bin/env python3
"""Read-only CPU completion check shared by queue checkpointing and aggregation."""
import argparse
import json
from pathlib import Path

from falcon_full_val_integrity import collect, digest, record, require


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--eval-output', type=Path, required=True)
    p.add_argument('--expected-manifest', type=Path, required=True)
    p.add_argument('--full-manifest', type=Path, required=True)
    p.add_argument('--managed-plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Refusing to overwrite an existing completion check')
    # A report may be written into the controller-owned temporary vm copy, but
    # never into the source shard evidence directory being checked.
    if a.eval_output.resolve() == a.output.resolve() or a.eval_output.resolve() in a.output.resolve().parents:
        p.error('Write the completion report outside the shard evidence directory')
    result = {'status': 'FAIL_SHARD_COMPLETION', 'complete': False,
              'full_paper_reproduction': False, 'full_val_complete': False,
              'interrupted_parent': None, 'failures': []}
    try:
        require(digest(a.eval_output / 'source_episode_manifest.json') == digest(a.expected_manifest),
                'Recovered shard differs from the exact expected manifest')
        data = collect(a.full_manifest, [a.eval_output], single_shard=True, managed_plan=a.managed_plan)
        shard = data['shards'][0]
        result.update(status='PASS_COMPLETE_SHARD', complete=True,
                      shard_id=shard['shard_id'], episode_count=shard['episode_count'], scene_count=shard['scene_count'],
                      interrupted_parent=shard['interrupted_parent'], completion_evidence=shard['completion_evidence'],
                      expected_manifest=record(a.expected_manifest), protocol_sha256=data['protocol_sha256'],
                      source_files=shard['files'], coverage=data['coverage'])
    except Exception as error:
        result['failures'].append(type(error).__name__ + ': ' + str(error))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if result['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
