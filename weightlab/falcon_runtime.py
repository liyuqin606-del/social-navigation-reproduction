"""Falcon-only inference adapters for absolute tensor patches and capture.

Torch is imported only when applying a patch. Capture uses the tensor protocol
of the supplied actor; neither adapter creates a model, device or GPU allocation.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import weakref
import zipfile

import numpy as np

from .falcon_checkpoint import (
    ALLOWED_TENSOR_KEYS, CHECKPOINT_SHA256, HEAD_KEYS, MAX_ARCHIVE_BYTES,
    MAX_STORAGE_BYTES, MAX_UNCOMPRESSED_BYTES, file_sha256, tensor_sha256,
)


def _document(value):
    return json.loads(Path(value).read_text()) if isinstance(value, (str, Path)) else copy.deepcopy(value)


def _numpy_tensor(tensor):
    value = tensor.detach().cpu().numpy()
    if np.asarray(value).dtype != np.dtype("float32"):
        raise ValueError("ACTOR_TENSOR_DTYPE_MUST_BE_FLOAT32")
    array = np.array(value, dtype="<f4", copy=True, order="C")
    if not np.isfinite(array).all():
        raise ValueError("ACTOR_TENSOR_NONFINITE")
    return array


def _match(array, entry, label):
    if (entry.get("dtype") != "float32" or list(array.shape) != entry.get("shape")
            or tensor_sha256(array) != entry.get("sha256")):
        raise ValueError("TENSOR_METADATA_MISMATCH: " + label)


def _read_patch_npz(path, allowed):
    """Read bounded NPY payloads directly; reject objects before allocation."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("PATCH_ARCHIVE_SIZE_LIMIT")
    arrays, total = {}, 0
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if not infos or len(infos) > len(ALLOWED_TENSOR_KEYS):
            raise ValueError("PATCH_MEMBER_COUNT_LIMIT")
        for info in infos:
            key = info.filename.removesuffix(".npy")
            if (not info.filename.endswith(".npy") or key not in allowed or key in arrays
                    or info.flag_bits & 1 or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
                raise ValueError("UNKNOWN_OR_DUPLICATE_PATCH_TENSOR: " + key)
            total += info.file_size
            if info.file_size > MAX_STORAGE_BYTES + 16384 or total > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("PATCH_UNCOMPRESSED_SIZE_LIMIT")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=16384)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream, max_header_size=16384)
                else:
                    raise ValueError("UNSUPPORTED_NPY_VERSION")
                if (fortran or dtype.kind != "f" or dtype.itemsize != 4 or dtype.byteorder == ">"
                        or not shape or len(shape) > 4 or any(type(n) is not int or n <= 0 for n in shape)):
                    raise ValueError("PATCH_REQUIRES_CONTIGUOUS_LITTLE_ENDIAN_FLOAT32")
                count = math.prod(shape)
                if count * 4 > MAX_STORAGE_BYTES:
                    raise ValueError("PATCH_TENSOR_SIZE_LIMIT")
                payload = stream.read(count * 4 + 1)
                if len(payload) != count * 4:
                    raise ValueError("NPY_PAYLOAD_LENGTH_MISMATCH")
                arrays[key] = np.frombuffer(payload, dtype="<f4").reshape(shape).copy()
                if not np.isfinite(arrays[key]).all():
                    raise ValueError("PATCH_TENSOR_NONFINITE")
    return arrays


def apply_patch(actor_critic, patch_npz, manifest):
    """Apply verified absolute tensors to a fresh, matching Falcon actor.

    Accepts the core weightlab_patch_v1 manifest. Prefer `effective_npz` for
    candidate chains. A sparse patch is accepted only if every omitted effective
    tensor is still equal to base; inherited edits must never silently disappear.
    All validation precedes mutation, and no policy architecture is changed.
    """
    document = _document(manifest)
    if document.get("schema") != "weightlab_patch_v1" or document.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("UNVERIFIED_FALCON_PATCH_MANIFEST")
    base = document.get("base_tensors", {})
    allowed = document.get("allowed_keys", [])
    if (not isinstance(base, dict) or not base or not set(base) <= ALLOWED_TENSOR_KEYS
            or not isinstance(allowed, list) or not allowed or len(allowed) != len(set(allowed))
            or not set(allowed) <= set(base)):
        raise ValueError("INVALID_PATCH_ALLOWLIST_OR_BASE_METADATA")
    if not Path(patch_npz).is_file() or Path(patch_npz).stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("PATCH_ARCHIVE_SIZE_LIMIT")
    artifact_sha = file_sha256(patch_npz)
    if not any(isinstance(document.get(name), dict) and document[name].get("sha256") == artifact_sha
               for name in ("patch_npz", "effective_npz")):
        raise ValueError("PATCH_FILE_SHA256_MISMATCH")
    replacement_meta = document.get("patch_tensors", {})
    effective = document.get("effective_tensors", replacement_meta)
    if (not isinstance(effective, dict) or not effective or not set(effective) <= set(base)
            or not isinstance(replacement_meta, dict) or not set(replacement_meta) <= set(allowed)):
        raise ValueError("INVALID_EFFECTIVE_TENSOR_METADATA")
    # A materialized effective state can carry unchanged non-editable base
    # tensors. Verify these bytes but never write them to the actor.
    arrays = _read_patch_npz(patch_npz, set(base))
    params = dict(actor_critic.named_parameters())
    original = {}
    for key, entry in base.items():
        if key not in params:
            raise ValueError("ACTOR_PARAMETER_MISSING: " + key)
        original[key] = _numpy_tensor(params[key])
        _match(original[key], entry, "actor base " + key)
    for key, array in arrays.items():
        if key not in effective:
            raise ValueError("PATCH_KEY_NOT_IN_EFFECTIVE_STATE: " + key)
        _match(array, effective[key], "effective patch " + key)
        if array.shape != original[key].shape:
            raise ValueError("PATCH_SHAPE_DIFFERS_FROM_ACTOR: " + key)
        if key not in allowed and tensor_sha256(array) != base[key].get("sha256"):
            raise ValueError("NONEDITABLE_TENSOR_CHANGED: " + key)
    for key, entry in effective.items():
        if key not in arrays:
            if any(entry.get(field) != base[key].get(field) for field in ("shape", "dtype", "sha256")):
                raise ValueError("PATCH_MISSING_INHERITED_TENSOR: " + key)
    import torch
    with torch.no_grad():
        prepared = {key: torch.as_tensor(array, device=params[key].device, dtype=params[key].dtype)
                    for key, array in arrays.items() if key in allowed}
        try:
            for key, value in prepared.items():
                params[key].copy_(value)
        except BaseException:
            for key in prepared:
                params[key].copy_(torch.as_tensor(original[key], device=params[key].device, dtype=params[key].dtype))
            raise
        for param in actor_critic.parameters():
            param.requires_grad_(False)
        actor_critic.eval()
    return {"status": "PATCH_APPLIED_INFERENCE_ONLY", "candidate_id": document.get("candidate_id"),
            "patch_sha256": artifact_sha, "checkpoint_sha256": CHECKPOINT_SHA256,
            "applied_tensors": {key: {"shape": list(array.shape), "dtype": "float32",
                                       "sha256": tensor_sha256(array)} for key, array in arrays.items() if key in allowed},
            "architecture_changed": False, "model_forward_executed": False,
            "scientific_improvement_demonstrated": False}


_ACTIVE_CAPTURES = weakref.WeakKeyDictionary()
_REQUIRED_METADATA = {"scene_ids", "episode_ids", "prev_actions", "reset_masks", "split"}
_OPTIONAL_METADATA = {"step_ids"}


class FalconCapture:
    """Capture one robot actor's hidden/logit pair at its linear action head.

    The caller must pass the actual agent_0 actor (for example
    agent._agents[0].actor_critic), never a multi-agent manager. Call on_step()
    immediately before each act(), with current episode IDs, previous actions,
    reset_masks (True means a reset), and a declared split. The metadata is
    consumed exactly once. Ground-truth, oracle and future-label fields are
    rejected. A callback receives NumPy arrays; without one, storage is bounded.

    policy_id='base' requires a matching export manifest and emits base_logits.
    Other policies emit logits, which base replay must not silently reinterpret.
    """
    def __init__(self, actor_critic, recorder=None, *, policy_id="unverified", base_manifest=None, max_records=1024):
        if not hasattr(actor_critic, "action_distribution") or not hasattr(actor_critic.action_distribution, "linear"):
            raise ValueError("EXPLICIT_ROBOT_ACTOR_WITH_LINEAR_HEAD_REQUIRED")
        self.actor = actor_critic
        self.head = actor_critic.action_distribution.linear
        self.recorder = recorder
        if recorder is not None and not callable(recorder):
            raise TypeError("RECORDER_MUST_BE_CALLABLE")
        if type(max_records) is not int or max_records <= 0:
            raise ValueError("POSITIVE_CAPTURE_RECORD_BOUND_REQUIRED")
        self.policy_id, self.max_records = policy_id, max_records
        self.records, self._pending, self._handle = [], None, None
        self._record_count = 0
        params = dict(actor_critic.named_parameters())
        if not all(key in params for key in HEAD_KEYS):
            raise ValueError("FALCON_HEAD_PARAMETERS_MISSING")
        arrays = {key: _numpy_tensor(params[key]) for key in HEAD_KEYS}
        if arrays[HEAD_KEYS[0]].shape != (4, 512) or arrays[HEAD_KEYS[1]].shape != (4,):
            raise ValueError("FALCON_HEAD_SHAPE_REQUIRED")
        self.head_hashes = {key: tensor_sha256(value) for key, value in arrays.items()}
        if policy_id == "base":
            if base_manifest is None:
                raise ValueError("BASE_CAPTURE_REQUIRES_EXPORTED_MANIFEST")
            base = _document(base_manifest)
            if base.get("checkpoint_sha256") != CHECKPOINT_SHA256:
                raise ValueError("BASE_CAPTURE_CHECKPOINT_SHA_MISMATCH")
            for key in HEAD_KEYS:
                _match(arrays[key], base["base_tensors"][key], "capture base " + key)

    def __enter__(self):
        if self._handle is not None or self.head in _ACTIVE_CAPTURES:
            raise RuntimeError("DUPLICATE_FALCON_CAPTURE_HOOK")
        self._handle = self.head.register_forward_hook(self._hook)
        _ACTIVE_CAPTURES[self.head] = self
        return self

    def on_step(self, metadata=None, **values):
        if self._handle is None:
            raise RuntimeError("CAPTURE_CONTEXT_NOT_ACTIVE")
        if self._pending is not None:
            raise RuntimeError("PREVIOUS_CAPTURE_METADATA_NOT_CONSUMED")
        data = dict(metadata or {})
        if set(data) & set(values):
            raise ValueError("DUPLICATE_STEP_METADATA")
        data.update(values)
        if set(data) - (_REQUIRED_METADATA | _OPTIONAL_METADATA) or not _REQUIRED_METADATA <= set(data):
            raise ValueError("CAPTURE_METADATA_ALLOWLIST_OR_REQUIRED_FIELDS")
        if not isinstance(data["split"], str) or not data["split"]:
            raise ValueError("EXPLICIT_SPLIT_REQUIRED")
        for name in ("scene_ids", "episode_ids"):
            if not isinstance(data[name], (list, tuple, np.ndarray)) or not all(isinstance(x, str) for x in data[name]):
                raise ValueError("EXACT_STRING_EPISODE_IDENTIFIERS_REQUIRED")
            data[name] = np.asarray(data[name], dtype=str).copy()
        for name in ("prev_actions", "reset_masks", "step_ids"):
            if name in data:
                value = data[name]
                if hasattr(value, "detach"):
                    value = value.detach().cpu().numpy()
                data[name] = np.asarray(value).reshape(-1).copy()
        if data["reset_masks"].dtype.kind != "b":
            raise ValueError("RESET_MASKS_MUST_BE_BOOLEAN_TRUE_ON_RESET")
        if data["prev_actions"].dtype.kind not in "iu" or not np.isin(data["prev_actions"], [0, 1, 2, 3]).all():
            raise ValueError("PREVIOUS_ACTIONS_MUST_USE_FOUR_ACTION_SPACE")
        if "step_ids" in data and (data["step_ids"].dtype.kind not in "iu" or (data["step_ids"] < 0).any()):
            raise ValueError("STEP_IDS_MUST_BE_NONNEGATIVE_INTEGERS")
        self._pending = data

    def _hook(self, module, inputs, output):
        if module is not self.head or self._pending is None:
            raise RuntimeError("MISSING_OR_AMBIGUOUS_STEP_METADATA")
        pending, self._pending = self._pending, None
        if len(inputs) != 1:
            raise ValueError("UNEXPECTED_LINEAR_HEAD_INPUT")
        hidden, logits = _numpy_tensor(inputs[0]), _numpy_tensor(output)
        if hidden.ndim != 2 or hidden.shape[1] != 512 or logits.shape != (hidden.shape[0], 4):
            raise ValueError("CAPTURE_HIDDEN_LOGIT_SHAPE_MISMATCH")
        for name in ("scene_ids", "episode_ids", "prev_actions", "reset_masks", "step_ids"):
            if name in pending and pending[name].shape != (len(hidden),):
                raise ValueError("CAPTURE_METADATA_BATCH_MISMATCH: " + name)
        # Detect an edit made while a 'base' capture context is active.
        if self.policy_id == "base":
            current = dict(self.actor.named_parameters())
            if any(tensor_sha256(_numpy_tensor(current[key])) != self.head_hashes[key] for key in HEAD_KEYS):
                raise ValueError("BASE_HEAD_CHANGED_DURING_CAPTURE")
        record = {**pending, "hidden": hidden,
                  "base_logits" if self.policy_id == "base" else "logits": logits}
        if self.recorder is None:
            if len(self.records) >= self.max_records:
                raise BufferError("CAPTURE_RECORD_LIMIT_USE_STREAMING_CALLBACK")
            self.records.append(record)
        else:
            self.recorder(record)
        self._record_count += 1

    @property
    def provenance(self):
        return {"policy_id": self.policy_id, "head_tensor_sha256": dict(self.head_hashes),
                "recorded_batches": self._record_count, "reset_mask_semantics": "True means episode reset",
                "oracle_or_future_metadata_allowed": False, "model_architecture_changed": False}

    def close(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
            if _ACTIVE_CAPTURES.get(self.head) is self:
                del _ACTIVE_CAPTURES[self.head]
        self._pending = None

    def __exit__(self, *_):
        self.close()
