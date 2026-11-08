#!/usr/bin/env python3
"""Finalize a completed finite Falcon queue locally; no SSH or GPU allocation.

--check is strictly read-only. Exit 3 means not ready, 4 means queue attention,
5 means an invalid finalization contract, and 6 means local aggregation failed.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import time


class GateError(Exception):
    def __init__(self, status, message, code):
        self.status, self.message, self.code = status, message, code


def require(condition, message):
    if not condition:
        raise GateError('INVALID_FINALIZATION_CONTRACT', message, 5)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def build_plan(queue_directory, output):
    queue, output = Path(queue_directory).resolve(), Path(output).resolve()
    checkpoint_path = queue / 'CHECKPOINT.json'
    state = read(checkpoint_path)
    require(state.get('schema') == 'continuous_reproduction_checkpoint_v1', 'Unexpected queue checkpoint schema')
    if state.get('status') == 'ATTENTION':
        raise GateError('QUEUE_REQUIRES_ATTENTION', state.get('error', 'Queue stopped for attention'), 4)
    if state.get('status') != 'COMPLETE':
        raise GateError('NOT_READY', 'Queue status is %s; completed shard IDs: %s' %
                        (state.get('status'), sorted(state.get('completed', {}))), 3)
    contract_path = Path(state['contract'])
    require(sha(contract_path) == state['input_files'][str(contract_path)], 'Queue contract differs from admitted hash')
    contract = read(contract_path)
    expected_ids = {f'balanced7_{i:02d}' for i in range(7)}
    units = {unit['unit_id']: unit for unit in contract['units']}
    require(set(state['completed']) == expected_ids and
            {cid for unit in units.values() for cid in unit['completion_ids']} == expected_ids,
            'Exactly the seven declared balanced7 completion IDs are required')
    require(all(row.get('status') == 'COMPLETE' for row in state['units'].values()), 'A queue unit is not complete')
    require(not output.exists(), 'Output directory must be new')
    request_path = output.with_name(output.name + '.finalization_request.json')
    require(not request_path.exists(), 'Finalization request already exists; use a new output name')
    require(not any(output.with_name(output.name + suffix).exists() for suffix in
                    ('.aggregate.log', '.independent_validation.log', '.finalization_receipt.json')),
            'Finalization sidecar evidence already exists; use a new output name')
    python = Path(contract['local_python'])
    require(python.is_absolute() and sha(python) == state['input_files'][str(python)], 'Local interpreter differs from admitted hash')
    project = Path('/Users/yuqinli/Documents/ChatGPT/VLN')
    full = project / 'baselines/prep/falcon_full_val_20261103_manifest.json'
    require(sha(full) == state['input_files'][str(full)], 'Full-val manifest differs from queue-bound input')
    selected = []
    proof_files = {}
    for cid in sorted(expected_ids):
        completion = state['completed'][cid]
        require(completion.get('independently_validated') is True and completion.get('proof_files'), 'Unvalidated queue completion: ' + cid)
        for path, value in completion['proof_files'].items():
            require(path not in proof_files or proof_files[path] == value, 'Conflicting completion proof hashes')
            proof_files[path] = value
        unit_id = completion['unit_id']
        plan = Path(completion['plan']).resolve()
        require(queue / 'attempts' in plan.parents, 'Completed plan lies outside this queue')
        matches = [row for row in state['units'][unit_id]['attempts']
                   if Path(row['plan']).resolve() == plan and row['run_id'] == completion['run_id']]
        require(len(matches) == 1, 'Completion does not identify one recorded attempt')
        index = matches[0]['variant_index']
        unit = units[unit_id]
        variant_ids = unit['completion_ids'] if index == 0 else unit['retry_specs'][index - 1]['completion_ids']
        require(cid in variant_ids, 'Completion ID is outside its actual primary/retry contract')
        worker, spec = read(plan / 'worker.json'), read(plan / 'SPEC.json')
        require(worker['run_id'] == spec['run_id'] == completion['run_id'], 'Completed attempt run ID differs')
        vm = plan / 'run/vm'
        remote_root = PurePosixPath(worker['output_directory'])
        require(remote_root.is_absolute(), 'Worker output root is not absolute')
        entry = Path(worker['argv'][1]).name
        if entry == 'run_falcon_parallel.py':
            pair, process = read(vm / 'parallel_plan.json'), read(vm / 'parallel_status.json')
            require(pair['run_id'] == completion['run_id'] and pair['remote_output_directory'] == str(remote_root), 'Executed pair identity differs')
            require(cid in {row['shard_id'] for row in pair['children']}, 'Shard was not planned in this pair')
            children = [row for row in process['children'] if row['shard_id'] == cid]
            require(len(children) == 1 and type(children[0].get('exit_code')) is int and children[0]['exit_code'] == 0,
                    'Completed child lacks an observed normal exit')
            remote_child = PurePosixPath(children[0]['output'])
        else:
            require(entry == 'run_falcon_subset.py' and worker['argv'].count('--output') == 1, 'Unexpected single-shard scientific entry')
            remote_child = PurePosixPath(worker['argv'][worker['argv'].index('--output') + 1])
        relative = remote_child.relative_to(remote_root)
        require(relative.parts and '..' not in relative.parts, 'Unsafe collected output mapping')
        directory = vm.joinpath(*relative.parts)
        require(directory.is_dir() and read(directory / 'source_episode_manifest.json')['shard_id'] == cid,
                'Discovered collected directory does not contain the completed shard')
        require(directory.resolve() not in output.parents, 'Final output cannot be inside shard evidence')
        selected.append({'shard_id': cid, 'owning_plan': str(plan), 'run_id': completion['run_id'],
                         'remote_output': str(remote_child), 'eval_output': str(directory),
                         'discovery': 'Completed attempt plus frozen worker/executed child output mapping'})
    # Verify each unique queue completion proof once; no model validator is run
    # during discovery or --check. The aggregate performs scientific validation.
    for path, digest in proof_files.items():
        require(sha(path) == digest, 'Queue completion evidence changed: ' + str(path))
    script_root = Path('/Users/yuqinli/Documents/ChatGPT/VLN/baselines/scripts')
    aggregator = script_root / 'aggregate_falcon_full_val.py'
    validator = script_root / 'validate_falcon_full_val_aggregate.py'
    args = ['--full-manifest', str(full)]
    for row in selected:
        args += ['--eval-output', row['eval_output']]
    args += ['--output', str(output)]
    return {'schema': 'falcon_continuous_finalization_plan_v1', 'status': 'READY_TO_AGGREGATE',
            'queue_directory': str(queue), 'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha(checkpoint_path),
            'full_manifest': str(full), 'selected_shards': selected, 'output': str(output),
            'request_path': str(request_path),
            'aggregate_argv': [str(python), str(aggregator)] + args,
            'validator_argv': [str(python), str(validator)] + args,
            'script_sha256': {str(path): sha(path) for path in (Path(__file__).resolve(), aggregator, validator)},
            'gpu_allocation': False, 'ssh': False, 'metrics_generated_by_planning': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--queue-directory', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true', help='Read-only discovery; never create outputs or run aggregation')
    p.add_argument('--timeout-seconds', type=int, default=600, help='Bound for each local CPU command')
    a = p.parse_args()
    if not 1 <= a.timeout_seconds <= 1800:
        p.error('timeout-seconds must be between 1 and 1800')
    try:
        plan = build_plan(a.queue_directory, a.output)
    except GateError as error:
        print(json.dumps({'status': error.status, 'message': error.message, 'metrics_generated': False}, indent=2))
        return error.code
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        print(json.dumps({'status': 'INVALID_FINALIZATION_CONTRACT', 'message': str(error), 'metrics_generated': False}, indent=2))
        return 5
    if a.check:
        print(json.dumps(plan, indent=2))
        return 0
    output, request = Path(plan['output']), Path(plan['request_path'])
    request.parent.mkdir(parents=True, exist_ok=True)
    try:
        with request.open('x') as stream:
            stream.write(json.dumps(plan, indent=2, allow_nan=False) + '\n')
    except OSError as error:
        print(json.dumps({'status': 'INVALID_FINALIZATION_CONTRACT', 'message': str(error), 'metrics_generated': False}))
        return 5
    receipt = {'status': 'LOCAL_FINALIZATION_STARTED', 'started_unix': time.time(),
               'request_sha256': sha(request), 'stages': [], 'gpu_allocation': False, 'ssh': False}
    for name, key in [('aggregate', 'aggregate_argv'), ('independent_validation', 'validator_argv')]:
        log_path = output.with_name(output.name + '.' + name + '.log')
        try:
            with log_path.open('xb') as stream:
                completed = subprocess.run(plan[key], stdout=stream, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, timeout=a.timeout_seconds)
            row = {'stage': name, 'exit_code': completed.returncode, 'timed_out': False,
                   'log': str(log_path), 'log_sha256': sha(log_path)}
        except (OSError, subprocess.TimeoutExpired) as error:
            row = {'stage': name, 'exit_code': None, 'timed_out': isinstance(error, subprocess.TimeoutExpired),
                   'error': str(error), 'log': str(log_path)}
        receipt['stages'].append(row)
        if row['exit_code'] != 0:
            receipt.update(status='LOCAL_FINALIZATION_FAILED', finished_unix=time.time())
            write(output.with_name(output.name + '.finalization_receipt.json'), receipt)
            print(json.dumps(receipt, indent=2))
            return 6
    receipt.update(status='FULL_VAL_AGGREGATE_VALIDATED', finished_unix=time.time(),
                   output=str(output), validation_sha256=sha(output / 'validation.json'))
    write(output.with_name(output.name + '.finalization_receipt.json'), receipt)
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
