"""Finite-budget, evidence-bound weight editing sessions (no model execution).

The host calls reserve_call BEFORE any external forward. Reservations are
permanent budget debits, including failed and unknown executions. This module
is an accounting/API boundary, not an OS sandbox around an arbitrary agent.
"""
import copy
import datetime as dt
import math
from pathlib import Path
import re
import secrets
import shutil

import numpy as np

from .io import (ContractError, atomic_json, atomic_npz, json_value, load_npz,
                 read_json, require, session_lock, sha256, tensor_record)
from .patches import delta_stats

ROLES = ('diagnosis', 'selection', 'confirmation')
KINDS = ('forward', 'replay_forward', 'activation_probe', 'confirmation')


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def reference(path):
    p = Path(path).absolute()
    return {'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)}


def verify_reference(row):
    p = Path(row['path'])
    require(p.is_file() and p.stat().st_size == row['bytes'] and sha256(p) == row['sha256'],
            'BOUND_FILE_CHANGED:' + p.name)


def episode_keys(document):
    if isinstance(document, dict):
        rows = document.get('records', document.get('episodes', document.get('episode_keys')))
    else:
        rows = document
    require(isinstance(rows, list) and rows, 'NONEMPTY_SPLIT_RECORDS_REQUIRED')
    result = []
    for row in rows:
        if isinstance(row, dict):
            pair = [row.get('scene_id'), row.get('episode_id')]
        else:
            require(isinstance(row, (list, tuple)) and len(row) == 2, 'INVALID_EPISODE_RECORD')
            pair = list(row)
        require(isinstance(pair[0], str) and pair[0] and isinstance(pair[1], (str, int))
                and not isinstance(pair[1], bool) and str(pair[1]), 'INVALID_EPISODE_KEY')
        result.append((pair[0], str(pair[1])))
    require(len(result) == len(set(result)), 'DUPLICATE_EPISODE_KEY')
    return result


def validated_config(config):
    cfg = json_value(config)
    require(isinstance(cfg, dict), 'CONFIG_OBJECT_REQUIRED')
    require(re.fullmatch(r'[0-9a-f]{64}', cfg.get('checkpoint_sha256', '')), 'CHECKPOINT_SHA256_REQUIRED')
    require('official_test_manifest' not in cfg, 'OFFICIAL_TEST_NOT_AVAILABLE_IN_AGENT_SESSION')
    cfg['base_tensors'] = str(Path(cfg['base_tensors']).absolute())
    require(isinstance(cfg.get('editable_keys'), list) and cfg['editable_keys']
            and all(isinstance(k, str) and k for k in cfg['editable_keys'])
            and len(cfg['editable_keys']) == len(set(cfg['editable_keys'])), 'EDITABLE_KEYS_REQUIRED')
    require(set(cfg.get('split_manifests', {})) == set(ROLES), 'EXACT_DIAGNOSIS_SELECTION_CONFIRMATION_ROLES_REQUIRED')
    cfg['split_manifests'] = {role: str(Path(cfg['split_manifests'][role]).absolute()) for role in ROLES}
    defaults = {'max_forward_calls': 50, 'max_batch_episodes': 8, 'max_transition_steps': 4096,
                'max_candidates': 50, 'max_code_runs': 300, 'max_confirmation_calls': 1,
                'reserve_free_bytes': 4 << 30, 'max_tensor_archive_bytes': 128 << 20}
    for key, value in defaults.items():
        cfg.setdefault(key, value)
        require(type(cfg[key]) is int and cfg[key] >= (0 if key == 'reserve_free_bytes' else 1), 'INVALID_LIMIT:' + key)
    require(cfg['max_forward_calls'] <= 100000 and cfg['max_candidates'] <= 10000
            and cfg['max_confirmation_calls'] <= cfg['max_forward_calls'], 'UNBOUNDED_SESSION_LIMIT')
    cfg.setdefault('max_relative_frobenius', 0.1)
    require(isinstance(cfg['max_relative_frobenius'], (int, float))
            and not isinstance(cfg['max_relative_frobenius'], bool)
            and math.isfinite(cfg['max_relative_frobenius']) and cfg['max_relative_frobenius'] > 0,
            'POSITIVE_FINITE_RELATIVE_FROBENIUS_LIMIT_REQUIRED')
    cfg.setdefault('norm_epsilon', 1e-12)
    require(isinstance(cfg['norm_epsilon'], (int, float)) and math.isfinite(cfg['norm_epsilon'])
            and cfg['norm_epsilon'] > 0, 'POSITIVE_NORM_EPSILON_REQUIRED')
    cfg['code_files'] = [str(Path(p).absolute()) for p in cfg.get('code_files', [])]
    return cfg


class Session:
    @classmethod
    def create(cls, root, config):
        """Create a new session, returning its JSON status; never run a model."""
        root = Path(root).absolute()
        cfg = validated_config(config)
        arrays = load_npz(cfg['base_tensors'], max_bytes=cfg['max_tensor_archive_bytes'])
        require(set(cfg['editable_keys']) <= set(arrays), 'EDITABLE_TENSOR_NOT_IN_BASE')
        split_docs, scene_sets = {}, {}
        for role, path in cfg['split_manifests'].items():
            doc = read_json(path)
            if isinstance(doc, dict) and 'role' in doc:
                require(doc['role'] == role, 'SPLIT_DECLARED_ROLE_MISMATCH')
            keys = episode_keys(doc)
            split_docs[role] = {'role': role, 'records': [{'scene_id': s, 'episode_id': e} for s, e in keys]}
            scene_sets[role] = {s for s, _ in keys}
        for i, a in enumerate(ROLES):
            for b in ROLES[i + 1:]:
                require(not scene_sets[a] & scene_sets[b], 'SPLITS_MUST_BE_SCENE_DISJOINT:' + a + ':' + b)
        require(not root.exists(), 'SESSION_ROOT_ALREADY_EXISTS_NO_OVERWRITE')
        root.parent.mkdir(parents=True, exist_ok=True)
        require(shutil.disk_usage(root.parent).free >= cfg['reserve_free_bytes'] + sum(a.nbytes for a in arrays.values()) * 2,
                'FREE_SPACE_RESERVE')
        root.mkdir()
        for directory in ('inputs', 'candidates', 'calls'):
            (root / directory).mkdir()
        atomic_npz(root / 'inputs/base_tensors.npz', arrays)
        for role, doc in split_docs.items():
            atomic_json(root / 'inputs' / (role + '.json'), doc)
        atomic_json(root / 'CONFIG.json', cfg)
        files = [Path(cfg['base_tensors']), root / 'CONFIG.json', root / 'inputs/base_tensors.npz']
        files += [Path(p) for p in cfg['split_manifests'].values()]
        files += [root / 'inputs' / (role + '.json') for role in ROLES]
        files += list(Path(__file__).absolute().parent.glob('*.py'))
        files += [Path(p) for p in cfg['code_files']]
        binding = {'schema': 'weightlab_input_binding_v1', 'created_utc': now(),
                   'files': [reference(p) for p in dict.fromkeys(files)],
                   'checkpoint_sha256': cfg['checkpoint_sha256'],
                   'base_tensors': {key: tensor_record(value) for key, value in arrays.items()}}
        atomic_json(root / 'INPUT_BINDING.json', binding)
        state = {'schema': 'weightlab_session_v1', 'created_utc': now(), 'updated_utc': now(),
                 'phase': 'OPEN', 'binding_sha256': sha256(root / 'INPUT_BINDING.json'),
                 'forward_calls_used': 0, 'candidates': {}, 'calls': {},
                 'selected': None, 'frozen': None, 'events': [{'event': 'created', 'utc': now()}]}
        atomic_json(root / 'LEDGER.json', state)
        return cls(root).status()

    def __init__(self, root):
        self.root = Path(root).absolute()
        require((self.root / 'LEDGER.json').is_file(), 'INITIALIZED_SESSION_REQUIRED')

    @property
    def config(self):
        """Trusted-host configuration. Do not put raw config in an agent prompt."""
        with session_lock(self.root):
            cfg, _, _ = self._load()
            return copy.deepcopy(cfg)

    def _load(self):
        cfg = read_json(self.root / 'CONFIG.json')
        state = read_json(self.root / 'LEDGER.json')
        require(state.get('schema') == 'weightlab_session_v1', 'LEDGER_SCHEMA')
        require(sha256(self.root / 'INPUT_BINDING.json') == state['binding_sha256'], 'INPUT_BINDING_CHANGED')
        binding = read_json(self.root / 'INPUT_BINDING.json')
        for row in binding['files']:
            verify_reference(row)
        require(state['forward_calls_used'] == len(state['calls']) <= cfg['max_forward_calls'], 'CALL_ACCOUNTING_INCONSISTENT')
        for candidate in state['candidates'].values():
            for key in ('manifest', 'patch_npz', 'effective_npz'):
                verify_reference(candidate[key])
        for call in state['calls'].values():
            verify_reference(call['reservation'])
            if call.get('completion'):
                verify_reference(call['completion'])
                for row in call.get('evidence_files', []):
                    verify_reference(row)
        if state['frozen']:
            for key in ('effective_npz', 'manifest'):
                verify_reference(state['frozen'][key])
        return cfg, state, binding

    def _save(self, state, event):
        state['updated_utc'] = now()
        state['events'].append({'event': event, 'utc': state['updated_utc']})
        atomic_json(self.root / 'LEDGER.json', state)

    def _space(self, cfg, additional=0):
        require(shutil.disk_usage(self.root).free >= cfg['reserve_free_bytes'] + additional, 'FREE_SPACE_RESERVE')

    def _candidate(self, state, candidate, usable=True):
        require(isinstance(candidate, str), 'CANDIDATE_NAME_REQUIRED')
        if candidate == 'base':
            return None
        require(candidate in state['candidates'], 'UNKNOWN_CANDIDATE')
        row = state['candidates'][candidate]
        if usable:
            require(row['status'] != 'REJECTED', 'CANDIDATE_REJECTED')
        return row

    def _arrays(self, cfg, state, candidate):
        row = self._candidate(state, candidate)
        path = self.root / 'inputs/base_tensors.npz' if row is None else Path(row['effective_npz']['path'])
        arrays = load_npz(path, max_bytes=cfg['max_tensor_archive_bytes'])
        return {key: arrays[key] for key in cfg['editable_keys']}

    def _status(self, cfg, state, binding):
        split_summary = {}
        for role in ROLES:
            keys = episode_keys(read_json(self.root / 'inputs' / (role + '.json')))
            split_summary[role] = {'episodes': len(keys), 'scenes': len({s for s, _ in keys}),
                                   'withheld': role == 'confirmation' and state['phase'] == 'OPEN'}
        return {'schema': state['schema'], 'phase': state['phase'], 'checkpoint_sha256': cfg['checkpoint_sha256'],
                'forward_calls_used': state['forward_calls_used'], 'max_forward_calls': cfg['max_forward_calls'],
                'remaining_forward_calls': cfg['max_forward_calls'] - state['forward_calls_used'],
                'max_code_runs': cfg['max_code_runs'], 'max_candidates': cfg['max_candidates'],
                'max_batch_episodes': cfg['max_batch_episodes'], 'max_transition_steps': cfg['max_transition_steps'],
                'split_summary': split_summary,
                'editable_tensors': {k: binding['base_tensors'][k] for k in cfg['editable_keys']},
                'selected': state['selected'], 'frozen_candidate': state['frozen']['candidate'] if state['frozen'] else None,
                'candidates': [{'name': name, 'status': row['status'], 'parent': row['parent'],
                                'rationale': row['rationale'], 'total_relative_frobenius': row['total_relative_frobenius']}
                               for name, row in state['candidates'].items()],
                'calls': [{'call_id': name, 'kind': row['kind'], 'candidate': row['candidate'], 'split': row['split'],
                           'status': row['status'], 'feedback': row.get('feedback')}
                          for name, row in state['calls'].items()],
                'pending_calls': [k for k, v in state['calls'].items() if v['status'] == 'reserved'],
                'official_test_available': False, 'paper_reproduction': False,
                'updated_utc': state['updated_utc']}

    def status(self):
        with session_lock(self.root):
            return self._status(*self._load())

    def candidate_files(self, name):
        with session_lock(self.root):
            cfg, state, binding = self._load()
            row = self._candidate(state, name, usable=False)
            if row is None:
                return {'candidate': 'base', 'effective_npz': reference(self.root / 'inputs/base_tensors.npz'),
                        'manifest': reference(self.root / 'INPUT_BINDING.json'), 'patch_npz': None,
                        'base_tensors': binding['base_tensors']}
            return json_value({'candidate': name, **{k: row[k] for k in ('effective_npz', 'patch_npz', 'manifest')}})

    def split_episode_keys(self, split):
        with session_lock(self.root):
            _, state, _ = self._load()
            require(split in ROLES, 'SPLIT_UNAVAILABLE')
            require(split != 'confirmation' or state['phase'] != 'OPEN', 'CONFIRMATION_WITHHELD_UNTIL_FREEZE')
            return [list(pair) for pair in episode_keys(read_json(self.root / 'inputs' / (split + '.json')))]

    def submit_patch(self, name, arrays_npz, rationale, parent='base'):
        with session_lock(self.root):
            cfg, state, binding = self._load()
            require(state['phase'] == 'OPEN', 'SESSION_FROZEN_NO_RETUNING')
            require(isinstance(name, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,62}', name)
                    and name != 'base', 'INVALID_PATCH_NAME')
            require(name not in state['candidates'] and not (self.root / 'candidates' / name).exists(), 'PATCH_NAME_ALREADY_USED_NO_OVERWRITE')
            require(len(state['candidates']) < cfg['max_candidates'], 'CANDIDATE_BUDGET_EXHAUSTED')
            require(isinstance(rationale, str) and 0 < len(rationale.strip()) <= 16000, 'PATCH_RATIONALE_REQUIRED')
            base = load_npz(self.root / 'inputs/base_tensors.npz', max_bytes=cfg['max_tensor_archive_bytes'])
            inherited = self._arrays(cfg, state, parent)
            submitted = load_npz(arrays_npz, max_bytes=cfg['max_tensor_archive_bytes'])
            require(set(submitted) <= set(cfg['editable_keys']), 'TENSOR_NOT_EDITABLE')
            for key, value in submitted.items():
                require(value.shape == base[key].shape, 'TENSOR_SHAPE_MISMATCH:' + key)
            effective = {key: np.array(value, copy=True) for key, value in inherited.items()}
            effective.update(submitted)
            changes, squared_delta, squared_base = {}, 0.0, 0.0
            for key in cfg['editable_keys']:
                direct = delta_stats(effective[key], inherited[key], cfg['norm_epsilon'])
                cumulative = delta_stats(effective[key], base[key], cfg['norm_epsilon'])
                require(cumulative['relative_frobenius'] <= cfg['max_relative_frobenius'], 'PER_TENSOR_RELATIVE_NORM_LIMIT:' + key)
                changes[key] = {'from_parent': direct, 'from_base': cumulative}
                squared_delta += cumulative['frobenius'] ** 2
                squared_base += cumulative['reference_frobenius'] ** 2
            total_relative = math.sqrt(squared_delta) / max(math.sqrt(squared_base), cfg['norm_epsilon'])
            require(total_relative <= cfg['max_relative_frobenius'], 'TOTAL_RELATIVE_NORM_LIMIT')
            require(any(changes[k]['from_parent']['changed_entries'] for k in submitted), 'PATCH_HAS_NO_ACTUAL_CHANGE')
            self._space(cfg, sum(v.nbytes for v in effective.values()) * 2 + sum(v.nbytes for v in submitted.values()))
            directory = self.root / 'candidates' / name
            directory.mkdir()
            atomic_npz(directory / 'patch.npz', submitted)
            atomic_npz(directory / 'effective.npz', effective)
            manifest = {'schema': 'weightlab_patch_v1', 'candidate_id': name, 'name': name, 'parent': parent,
                        'checkpoint_sha256': cfg['checkpoint_sha256'], 'created_utc': now(), 'rationale': rationale,
                        'semantics': 'absolute_replacements_over_parent; effective_npz_is_materialized_state',
                        'allowed_keys': cfg['editable_keys'],
                        'base_tensors': {k: binding['base_tensors'][k] for k in cfg['editable_keys']},
                        'patch_tensors': {k: tensor_record(v) for k, v in submitted.items()},
                        'effective_tensors': {k: tensor_record(v) for k, v in effective.items()},
                        'patch_npz': reference(directory / 'patch.npz'), 'effective_npz': reference(directory / 'effective.npz'),
                        'changes': changes, 'total_relative_frobenius': total_relative,
                        'max_relative_frobenius': cfg['max_relative_frobenius'],
                        'parent_manifest_sha256': state['candidates'][parent]['manifest']['sha256'] if parent != 'base' else state['binding_sha256']}
            atomic_json(directory / 'manifest.json', manifest)
            row = {'status': 'ACTIVE', 'parent': parent, 'rationale': rationale,
                   'manifest': reference(directory / 'manifest.json'), 'patch_npz': manifest['patch_npz'],
                   'effective_npz': manifest['effective_npz'], 'total_relative_frobenius': total_relative}
            state['candidates'][name] = row
            self._save(state, 'patch_submitted:' + name)
            return json_value({'candidate': name, **row, 'changes': changes})

    def reserve_call(self, kind, candidate, split, request):
        with session_lock(self.root):
            cfg, state, _ = self._load()
            require(kind in KINDS, 'UNKNOWN_COUNTED_FORWARD_KIND')
            require(split in ROLES, 'SPLIT_UNAVAILABLE')
            self._candidate(state, candidate)
            if kind == 'confirmation':
                require(state['phase'] in ('FROZEN', 'CONFIRMED', 'CONFIRMATION_FAILED') and state['frozen'], 'FREEZE_REQUIRED_FOR_CONFIRMATION')
                require(split == 'confirmation' and candidate == state['frozen']['candidate'], 'CONFIRM_ONLY_FROZEN_CANDIDATE')
                used = sum(c['kind'] == 'confirmation' for c in state['calls'].values())
                require(used < cfg['max_confirmation_calls'], 'CONFIRMATION_CALL_BUDGET_EXHAUSTED')
            else:
                require(state['phase'] == 'OPEN', 'NO_ANALYSIS_OR_RETUNING_AFTER_FREEZE')
                require(split in ('diagnosis', 'selection'), 'CONFIRMATION_WITHHELD_UNTIL_FREEZE')
            require(state['forward_calls_used'] < cfg['max_forward_calls'], 'FORWARD_CALL_BUDGET_EXHAUSTED')
            request = json_value(request)
            require(isinstance(request, dict), 'REQUEST_OBJECT_REQUIRED')
            keys = episode_keys(request.get('episode_keys'))
            require(len(keys) <= cfg['max_batch_episodes'], 'BATCH_EPISODE_CAP')
            allowed = set(episode_keys(read_json(self.root / 'inputs' / (split + '.json'))))
            require(set(keys) <= allowed, 'EPISODE_NOT_IN_REQUESTED_SPLIT')
            steps = request.get('transition_steps')
            require(type(steps) is int and 0 < steps <= cfg['max_transition_steps'], 'TRANSITION_STEP_CAP')
            self._space(cfg, 1 << 20)
            call_id = 'call_' + secrets.token_hex(16)
            directory = self.root / 'calls' / call_id
            directory.mkdir()
            row = {'call_id': call_id, 'call_token': call_id, 'kind': kind, 'candidate': candidate, 'split': split,
                   'request': request, 'status': 'reserved', 'reserved_utc': now(),
                   'budget_debit': 1, 'call_index': state['forward_calls_used'] + 1,
                   'candidate_tensor_sha256': sha256(self.root / 'inputs/base_tensors.npz') if candidate == 'base'
                       else state['candidates'][candidate]['effective_npz']['sha256']}
            atomic_json(directory / 'RESERVATION.json', row)
            row['reservation'] = reference(directory / 'RESERVATION.json')
            state['calls'][call_id] = row
            state['forward_calls_used'] += 1
            self._save(state, 'forward_reserved:' + call_id)
            return json_value(row)

    def complete_call(self, call_id, feedback, evidence):
        with session_lock(self.root):
            cfg, state, _ = self._load()
            require(call_id in state['calls'], 'UNKNOWN_CALL_TOKEN')
            row = state['calls'][call_id]
            require(row['status'] == 'reserved', 'CALL_ALREADY_FINAL_NO_OVERWRITE')
            feedback, evidence = json_value(feedback), json_value(evidence)
            require(isinstance(feedback, dict) and isinstance(evidence, dict), 'FEEDBACK_AND_EVIDENCE_OBJECTS_REQUIRED')
            outcome = evidence.get('status')
            require(outcome in ('completed', 'failed', 'unknown'), 'EXPLICIT_CALL_OUTCOME_REQUIRED')
            files = evidence.get('files', [])
            require(isinstance(files, list), 'EVIDENCE_FILES_MUST_BE_LIST')
            refs = []
            for item in files:
                ref = reference(item['path'] if isinstance(item, dict) else item)
                if isinstance(item, dict) and 'sha256' in item:
                    require(ref['sha256'] == item['sha256'], 'EVIDENCE_HASH_MISMATCH')
                refs.append(ref)
            require(outcome != 'completed' or refs, 'COMPLETED_FORWARD_REQUIRES_ARTIFACT_EVIDENCE')
            if row['kind'] != 'confirmation':
                require(state['phase'] == 'OPEN', 'ANALYSIS_FEEDBACK_AFTER_FREEZE_FORBIDDEN')
            else:
                require(state['frozen'] and row['candidate'] == state['frozen']['candidate']
                        and state['phase'] != 'DECLINED', 'CONFIRMATION_CANDIDATE_CHANGED_OR_DECLINED')
            self._space(cfg, 1 << 20)
            result = {'call_id': call_id, 'status': outcome, 'feedback': feedback, 'evidence': evidence,
                      'evidence_files': refs, 'completed_utc': now(), 'budget_refunded': False,
                      'external_execution_claim': 'host_reported_with_hash_bound_artifacts; not independently executed by Session'}
            destination = self.root / 'calls' / call_id / 'COMPLETION.json'
            require(not destination.exists(), 'ORPHAN_COMPLETION_REQUIRES_OPERATOR_RECONCILIATION')
            atomic_json(destination, result)
            row.update(status=outcome, feedback=feedback, completion=reference(destination), evidence_files=refs)
            if row['kind'] == 'confirmation':
                state['phase'] = 'CONFIRMED' if outcome == 'completed' else 'CONFIRMATION_FAILED'
            self._save(state, 'forward_' + outcome + ':' + call_id)
            return json_value(result)

    def select(self, candidate):
        with session_lock(self.root):
            cfg, state, binding = self._load()
            require(state['phase'] == 'OPEN', 'SESSION_FROZEN_NO_RETUNING')
            self._candidate(state, candidate)
            require(any(c['candidate'] == candidate and c['split'] == 'selection' and c['status'] == 'completed'
                        and c['kind'] != 'activation_probe' for c in state['calls'].values()), 'COMPLETED_SELECTION_FORWARD_REQUIRED')
            state['selected'] = candidate
            self._save(state, 'selected:' + candidate)
            return self._status(cfg, state, binding)

    def reject(self, candidate, reason):
        with session_lock(self.root):
            cfg, state, binding = self._load()
            require(isinstance(reason, str) and reason.strip(), 'REJECTION_REASON_REQUIRED')
            self._candidate(state, candidate, usable=False)
            if state['phase'] != 'OPEN':
                require(state['frozen'] and candidate == state['frozen']['candidate'], 'ONLY_FROZEN_RESULT_MAY_BE_DECLINED')
                require(not any(c['status'] == 'reserved' for c in state['calls'].values()), 'RESOLVE_RESERVED_CALLS_BEFORE_DECLINE')
                state['phase'] = 'DECLINED'
                state['decline_reason'] = reason
            else:
                require(candidate != 'base', 'IMMUTABLE_BASE_CONTROL_CANNOT_BE_REJECTED')
                state['candidates'][candidate].update(status='REJECTED', rejection_reason=reason)
                if state['selected'] == candidate:
                    state['selected'] = None
            self._save(state, 'rejected:' + candidate)
            return self._status(cfg, state, binding)

    def freeze(self, candidate):
        with session_lock(self.root):
            cfg, state, binding = self._load()
            require(state['phase'] == 'OPEN', 'SESSION_ALREADY_FROZEN')
            self._candidate(state, candidate)
            require(state['selected'] == candidate, 'FREEZE_REQUIRES_CURRENT_SELECTION')
            require(not any(c['status'] == 'reserved' for c in state['calls'].values()), 'RESOLVE_RESERVED_CALLS_BEFORE_FREEZE')
            arrays = self._arrays(cfg, state, candidate)
            self._space(cfg, sum(a.nbytes for a in arrays.values()) * 2)
            directory = self.root / 'frozen'
            require(not directory.exists(), 'FROZEN_ARTIFACT_ALREADY_EXISTS_NO_OVERWRITE')
            directory.mkdir()
            atomic_npz(directory / 'effective.npz', arrays)
            manifest = {'schema': 'weightlab_frozen_candidate_v1', 'candidate': candidate,
                        'checkpoint_sha256': cfg['checkpoint_sha256'], 'frozen_utc': now(),
                        'effective_npz': reference(directory / 'effective.npz'),
                        'effective_tensors': {k: tensor_record(v) for k, v in arrays.items()},
                        'allowed_keys': cfg['editable_keys'],
                        'base_tensors': {k: binding['base_tensors'][k] for k in cfg['editable_keys']},
                        'input_binding_sha256': state['binding_sha256'], 'forward_calls_used_at_freeze': state['forward_calls_used'],
                        'selection_call_ids': [k for k, c in state['calls'].items() if c['candidate'] == candidate
                                               and c['split'] == 'selection' and c['status'] == 'completed'],
                        'no_post_confirmation_retuning': True}
            atomic_json(directory / 'manifest.json', manifest)
            state['frozen'] = {'candidate': candidate, 'effective_npz': manifest['effective_npz'],
                               'manifest': reference(directory / 'manifest.json'), 'frozen_utc': manifest['frozen_utc']}
            state['phase'] = 'FROZEN'
            self._save(state, 'frozen:' + candidate)
            return json_value(state['frozen'])
