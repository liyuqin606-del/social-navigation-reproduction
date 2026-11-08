"""Operator CLI for bounded editing sessions; no LLM or GPU is launched here."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from .io import ContractError, atomic_json, load_npz, read_json, require, sha256
from .replay import (_load_npz as load_replay_npz, _strings as replay_strings,
                     evaluate_head_replay)
from .session import Session, ROLES


def configuration(path):
    path = Path(path).absolute()
    cfg = read_json(path)
    require(cfg.get('reserve_free_bytes', 4 << 30) >= 4 << 30,
            'CLI_REQUIRES_AT_LEAST_FOUR_GIB_RESERVE')
    def absolute(value):
        p = Path(value)
        return str(p.absolute() if p.is_absolute() else (path.parent / p).absolute())
    cfg['base_tensors'] = absolute(cfg['base_tensors'])
    cfg['split_manifests'] = {role: absolute(value) for role, value in cfg['split_manifests'].items()}
    cfg['code_files'] = [absolute(p) for p in cfg.get('code_files', [])]
    return cfg


def replay(session, candidate, split, batch_path):
    """Bind an exact feature snapshot to a charged call before any head forward."""
    allowed = session.split_episode_keys(split)  # gate confirmation BEFORE reading its batch
    cfg = session.config
    batch, source = load_replay_npz(batch_path)  # array decoding only, no matrix computation
    require({'hidden', 'scene_ids', 'episode_ids', 'split'} <= set(batch), 'BATCH_METADATA_MISSING')
    require(batch['hidden'].ndim == 2, 'HIDDEN_MATRIX_REQUIRED')
    n = int(batch['hidden'].shape[0])
    require(batch['scene_ids'].shape == batch['episode_ids'].shape == (n,), 'BATCH_ID_SHAPE')
    scenes = replay_strings(batch['scene_ids'], 'scene_ids')
    episodes = replay_strings(batch['episode_ids'], 'episode_ids')
    keys = sorted(set(zip(scenes, episodes)))
    require(batch['split'].shape in ((), (n,)) and
            set(replay_strings(batch['split'], 'split')) == {split}, 'BATCH_SPLIT_MISMATCH')
    require(set(keys) <= {tuple(k) for k in allowed}, 'BATCH_EPISODE_OUTSIDE_SPLIT')
    request = {'episode_keys': [list(k) for k in keys], 'transition_steps': n,
               'backend': 'frozen_head_replay', 'source_batch_sha256': source['sha256'],
               'source_batch_bytes': source['bytes'], 'no_closed_loop_navigation': True}
    call = session.reserve_call('confirmation' if split == 'confirmation' else 'replay_forward',
                                candidate, split, request)
    directory = session.root / 'calls' / call['call_id']
    try:
        snapshot = directory / 'BATCH.npz'
        with Path(batch_path).open('rb') as stream:
            raw = stream.read(source['bytes'] + 1)
        require(len(raw) == source['bytes'], 'BATCH_CHANGED_AFTER_RESERVATION')
        with snapshot.open('xb') as stream:
            stream.write(raw)
        require(sha256(snapshot) == source['sha256'], 'BATCH_CHANGED_AFTER_RESERVATION')
        files = session.candidate_files(candidate)
        require(files['effective_npz']['sha256'] == call['candidate_tensor_sha256'],
                'CANDIDATE_CHANGED_AFTER_RESERVATION')
        base = session.candidate_files('base')['effective_npz']
        feedback = evaluate_head_replay(
            base['path'], None if candidate == 'base' else files['effective_npz']['path'],
            snapshot, directory / 'replay', allowed_keys=keys, expected_split=split,
            max_batch_size=cfg['max_transition_steps'])
        require(feedback['input_provenance']['batch']['sha256'] == source['sha256'], 'EXECUTED_BATCH_MISMATCH')
        require(feedback['input_provenance']['base']['sha256'] == base['sha256'], 'EXECUTED_BASE_MISMATCH')
        if candidate != 'base':
            require(feedback['input_provenance']['patch']['sha256'] == call['candidate_tensor_sha256'],
                    'EXECUTED_CANDIDATE_MISMATCH')
        bound = {'schema': 'weightlab_bound_replay_feedback_v1', 'call_id': call['call_id'],
                 'candidate': candidate, 'candidate_tensor_sha256': call['candidate_tensor_sha256'],
                 'split': split, 'episode_keys': request['episode_keys'], 'transition_steps': n,
                 'no_model_navigation_executed': True, 'replay': feedback}
        atomic_json(directory / 'BOUND_FEEDBACK.json', bound)
        result = session.complete_call(call['call_id'], bound, {'status': 'completed', 'files': [
            str(snapshot), str(directory / 'BOUND_FEEDBACK.json'),
            str(directory / 'replay/feedback.json'), str(directory / 'replay/paired_outputs.npz')]})
        return {'call_id': call['call_id'], 'status': result['status'], 'feedback': bound,
                'remaining_forward_calls': session.status()['remaining_forward_calls']}
    except BaseException as error:
        # A crash before this handler finishes leaves a reserved call for explicit
        # operator reconciliation. Never refund, rerun, or fabricate completion.
        outcome = 'failed' if isinstance(error, Exception) else 'unknown'
        failure = {'call_id': call['call_id'], 'error_type': type(error).__name__,
                   'error': str(error), 'budget_refunded': False,
                   'execution_stage_may_have_started': True}
        atomic_json(directory / 'EXECUTION_ERROR.json', failure)
        try:
            retained = [directory / 'EXECUTION_ERROR.json', directory / 'BATCH.npz',
                        directory / 'replay/feedback.json', directory / 'replay/paired_outputs.npz']
            session.complete_call(call['call_id'], failure, {'status': outcome,
                                  'files': [str(p) for p in retained if p.is_file()]})
        except Exception:
            # Preserve original failure and its pending debit if a binding or disk
            # error prevents a safe ledger update; the CLI does not repair evidence.
            pass
        raise


def inspect_candidate(session, candidate, tensor=None, values=False):
    files = session.candidate_files(candidate)
    arrays = load_npz(files['effective_npz']['path'])
    keys = session.config['editable_keys']
    require(tensor is None or tensor in keys, 'TENSOR_NOT_EDITABLE')
    selected = [tensor] if tensor else keys
    require(not values or sum(arrays[k].size for k in selected) <= 4096, 'PRINT_TENSOR_VALUE_CAP')
    result = {'candidate': candidate, 'effective_npz_sha256': files['effective_npz']['sha256'],
              'tensors': {k: {'shape': list(arrays[k].shape), 'dtype': str(arrays[k].dtype)} for k in selected},
              'model_forward_executed': False}
    if values:
        for key in selected:
            result['tensors'][key]['values'] = arrays[key].tolist()
    return result


DECISION_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'required': ['action', 'reason'],
    'properties': {
        'action': {'enum': ['propose_patch', 'replay', 'select', 'reject', 'freeze', 'confirm', 'stop']},
        'reason': {'type': 'string'}, 'candidate': {'type': 'string'}, 'parent': {'type': 'string'},
        'split': {'enum': list(ROLES)}, 'proposal_code': {'type': 'string'},
        'requested_probe': {'type': 'string'},
    },
}


def agent_prompt(session):
    """Curated LLM context, never the trusted config or held-out manifest."""
    state = session.status()
    summaries = []
    for call in state['calls']:
        item = {k: call[k] for k in ('call_id', 'kind', 'candidate', 'split', 'status')}
        feedback = call.get('feedback') or {}
        head = feedback.get('replay', {})
        if head.get('backend') == 'frozen_head_replay':
            item['feedback'] = {k: head.get(k) for k in (
                'backend', 'no_model_navigation_executed', 'samples', 'unique_episodes',
                'metrics', 'recorded_actions', 'supervised_labels', 'limitations')}
        elif feedback:
            item['feedback_summary'] = 'Operator must supply independently reviewed feedback for this backend.'
        summaries.append(item)
    safe_state = {k: state[k] for k in ('phase', 'checkpoint_sha256', 'editable_tensors',
        'forward_calls_used', 'max_forward_calls', 'remaining_forward_calls', 'max_batch_episodes',
        'max_transition_steps', 'max_code_runs', 'max_candidates', 'split_summary', 'selected',
        'frozen_candidate', 'candidates', 'pending_calls')}
    safe_state['calls'] = summaries
    return {'schema': 'weightlab_operator_agent_prompt_v1',
            'instructions': [
                'Return one JSON decision matching decision_schema. The operator reviews and executes it.',
                'Diagnose a concrete failure, propose a falsifiable hypothesis, compare controls, then revise or reject.',
                'Only supplied editable tensors may change; submit absolute float32 replacements over the stated parent.',
                'No backpropagation, autograd, optimizers, iterative loss fitting, external model calls, or benchmark lookup.',
                'Code execution is currently disabled in this CLI. proposal_code is reviewable text, never auto-executed.',
                'All actual head replay/full-forward/probe requests must be reserved by the host and debit the budget.',
                'Head replay uses fixed hidden states: its KL/label agreement is not navigation success or SOTA evidence.',
                'Use diagnosis and selection data only while OPEN. Confirmation identities/data are withheld until freeze.',
                'Freeze the selected candidate before confirmation. After confirmation, accept or decline; never retune.',
                'Keep base if evidence is insufficient. Official hidden benchmark data/results are unavailable here.',
            ], 'state': safe_state, 'decision_schema': DECISION_SCHEMA,
            'automated_llm_invocation': False, 'arbitrary_code_execution_enabled': False,
            'enforcement_boundary': 'Host/API protocol; exported prompts do not sandbox an external LLM or operator.'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    def command(name, help):
        s = subs.add_parser(name, help=help)
        s.add_argument('--session', required=True, type=Path)
        return s
    s = command('init', 'Create a new frozen-input session; paths resolve relative to config')
    s.add_argument('--config', required=True, type=Path)
    command('status', 'Read the accounting state; no model execution')
    s = command('inspect', 'Inspect editable tensor metadata or small tensors')
    s.add_argument('--candidate', default='base'); s.add_argument('--tensor')
    s.add_argument('--values', action='store_true')
    s = command('submit', 'Submit immutable absolute NPZ replacements, never deltas')
    s.add_argument('--name', required=True); s.add_argument('--npz', required=True, type=Path)
    s.add_argument('--rationale', required=True); s.add_argument('--parent', default='base')
    s = command('replay', 'Count and execute paired last-linear-head replay on a fixed NPZ batch')
    s.add_argument('--candidate', default='base'); s.add_argument('--split', choices=ROLES, required=True)
    s.add_argument('--batch', required=True, type=Path)
    for name in ('select', 'freeze'):
        s = command(name, name.capitalize() + ' a candidate through the session gate')
        s.add_argument('--candidate', required=True)
    s = command('reject', 'Reject a candidate, or decline a frozen result without unfreezing')
    s.add_argument('--candidate', required=True); s.add_argument('--reason', required=True)
    command('agent-prompt', 'Export curated JSON context/schema for an external LLM; invokes no LLM')
    s = command('resolve-call', 'Operator records a known failure/unknown execution, retaining its debit')
    s.add_argument('--call-id', required=True); s.add_argument('--outcome', choices=('failed', 'unknown'), required=True)
    s.add_argument('--reason', required=True)
    command('code', 'Disabled: fails closed until the code sandbox is independently validated')
    return p


def dispatch(args):
    if args.command == 'init':
        return Session.create(args.session, configuration(args.config))
    session = Session(args.session)
    if args.command == 'status': return session.status()
    if args.command == 'inspect': return inspect_candidate(session, args.candidate, args.tensor, args.values)
    if args.command == 'submit': return session.submit_patch(args.name, args.npz, args.rationale, args.parent)
    if args.command == 'replay': return replay(session, args.candidate, args.split, args.batch)
    if args.command == 'select': return session.select(args.candidate)
    if args.command == 'freeze': return session.freeze(args.candidate)
    if args.command == 'reject': return session.reject(args.candidate, args.reason)
    if args.command == 'agent-prompt': return agent_prompt(session)
    if args.command == 'resolve-call':
        require(args.reason.strip(), 'RECONCILIATION_REASON_REQUIRED')
        return session.complete_call(args.call_id, {'operator_reconciliation': True, 'reason': args.reason},
                                     {'status': args.outcome, 'files': []})
    if args.command == 'code':
        raise ContractError('CODE_EXECUTION_DISABLED_PENDING_VALIDATED_SANDBOX')
    raise ContractError('UNKNOWN_COMMAND')


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps({'status': 'refused_or_failed', 'error_type': type(error).__name__,
                          'error': str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
