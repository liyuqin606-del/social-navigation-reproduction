"""CPU-only paired replay of Falcon's *last linear head* on frozen features.

This is neither a policy rollout nor a world model. The session controller must
reserve a counted call and verify its immutable input hashes before invoking it.
``allowed_keys`` and ``expected_split`` come from that session's frozen manifest,
never from the batch itself. Confirmation access is gated by the controller.

NPZ patches contain ABSOLUTE replacement tensors, not deltas. A chained candidate
must be materialized by the controller before reaching this function. No training,
autograd, parameter search, or optimizer is implemented here.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Iterable
import zipfile

import numpy as np


WEIGHT_KEY = "action_distribution.linear.weight"
BIAS_KEY = "action_distribution.linear.bias"
HEAD_KEYS = frozenset((WEIGHT_KEY, BIAS_KEY))
SPLITS = frozenset(("diagnosis", "selection", "confirmation"))
BASE_LOGITS_RTOL = 1e-5
BASE_LOGITS_ATOL = 1e-5
MAX_NPZ_BYTES = 64 * 1024 * 1024


class ReplayValidationError(ValueError):
    """Invalid, mismatched, or unsupported replay evidence."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayValidationError(message)


def _load_npz(path: str | Path) -> tuple[dict[str, np.ndarray], dict]:
    path = Path(path).resolve(strict=True)
    _require(path.is_file(), f"Not a regular file: {path}")
    _require(path.stat().st_size <= MAX_NPZ_BYTES, "NPZ exceeds the byte limit")
    raw = path.read_bytes()
    # Read a single byte snapshot so hashes describe exactly what was evaluated.
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        names = [x.filename for x in entries]
        _require(len(names) == len(set(names)), "Duplicate NPZ members")
        _require(all(n.endswith(".npy") and "/" not in n and "\\" not in n
                     for n in names), "Unsupported NPZ member names")
        _require(sum(x.file_size for x in entries) <= MAX_NPZ_BYTES,
                 "Uncompressed NPZ exceeds the byte limit")
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
    except (ValueError, TypeError) as exc:
        raise ReplayValidationError("NPZ contains unsupported or object/pickle data") from exc
    _require(all(not x.dtype.hasobject for x in arrays.values()),
             "Object/pickle arrays are forbidden")
    return arrays, {"path": str(path), "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest()}


def _float32(array: np.ndarray, shape: tuple[int, ...], name: str) -> None:
    _require(array.dtype == np.dtype("float32"), f"{name} must be float32")
    _require(array.shape == shape, f"{name} must have shape {shape}")
    _require(bool(np.isfinite(array).all()), f"{name} contains nonfinite values")


def _strings(array: np.ndarray, name: str) -> list[str]:
    _require(array.dtype.kind in "US", f"{name} must use string dtype, never object")
    try:
        values = [x.decode("utf-8") if isinstance(x, bytes) else str(x)
                  for x in array.reshape(-1).tolist()]
    except UnicodeDecodeError as exc:
        raise ReplayValidationError(f"{name} must contain UTF-8 strings") from exc
    _require(all(x and x.strip() == x and "\x00" not in x for x in values),
             f"{name} contains an empty or noncanonical identifier")
    return values


def _allowed_keys(keys: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for key in keys:
        _require(isinstance(key, (list, tuple)) and len(key) == 2,
                 "allowed_keys must contain (scene_id, episode_id) pairs")
        _require(all(isinstance(x, str) and x and x.strip() == x and "\x00" not in x
                     for x in key), "allowed_keys must contain canonical strings")
        result.add(tuple(key))
    _require(bool(result), "allowed_keys cannot be empty")
    return result


def _action_ids(array: np.ndarray, n: int, name: str) -> None:
    _require(array.shape == (n,), f"{name} must have shape ({n},)")
    _require(array.dtype.kind in "iu", f"{name} must use integer dtype")
    _require(bool(((array >= 0) & (array < 4)).all()), f"{name} outside [0, 3]")


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    # Statistics in float64 prevent artificial infinities for extreme finite logits.
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _tensor_record(array: np.ndarray) -> dict:
    value = np.ascontiguousarray(array, dtype="<f4")
    return {"shape": list(value.shape), "dtype": "float32",
            "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def evaluate_head_replay(
    base_tensors_npz: str | Path,
    patch_npz_or_none: str | Path | None,
    batch_npz: str | Path,
    output_dir: str | Path,
    *,
    allowed_keys: Iterable[tuple[str, str]],
    expected_split: str,
    max_batch_size: int = 256,
) -> dict:
    """Return and save finite, explicitly non-navigation head-replay feedback.

    Required batch arrays: hidden[N,512], scene_ids[N], episode_ids[N], and
    split (a string scalar or N strings). Optional: actions[N], target_actions[N],
    base_logits[N,4], step_ids[N], prev_actions[N], reset_masks[N]. All features,
    logits, and model tensors are float32. Action IDs refer to the unchanged
    checkpoint's action mapping; no STOP-as-WAIT reinterpretation is supported.

    Repeated episode keys are expected for multiple timesteps. If step_ids are
    supplied, (scene, episode, step) must be unique. Labels are supervision only;
    action agreement or cross-entropy cannot establish navigation improvement.
    Output must be a NEW directory. Input validation failures leave no output.
    """
    output = Path(output_dir).absolute()
    _require(not output.exists(), "output_dir must not already exist")
    _require(expected_split in SPLITS, "Unknown expected_split role")
    _require(isinstance(max_batch_size, int) and not isinstance(max_batch_size, bool)
             and max_batch_size > 0, "max_batch_size must be a positive integer")
    allowed = _allowed_keys(allowed_keys)
    base, base_record = _load_npz(base_tensors_npz)
    _require(set(base) == HEAD_KEYS,
             "Replay requires exactly the two head tensors; non-head changes need a full forward")
    _float32(base[WEIGHT_KEY], (4, 512), WEIGHT_KEY)
    _float32(base[BIAS_KEY], (4,), BIAS_KEY)
    effective = {key: value.copy() for key, value in base.items()}
    patch_record = None
    patch = {}
    if patch_npz_or_none is not None:
        patch, patch_record = _load_npz(patch_npz_or_none)
        _require(bool(patch) and set(patch) <= HEAD_KEYS,
                 "Patch must contain only absolute head tensor replacements")
        for key, value in patch.items():
            _float32(value, base[key].shape, key)
            effective[key] = value.copy()

    batch, batch_record = _load_npz(batch_npz)
    required = {"hidden", "scene_ids", "episode_ids", "split"}
    optional = {"actions", "target_actions", "base_logits", "step_ids",
                "prev_actions", "reset_masks"}
    _require(required <= set(batch), f"Missing required batch arrays: {sorted(required - set(batch))}")
    _require(set(batch) <= required | optional,
             f"Unknown batch arrays (including navigation claims): {sorted(set(batch) - required - optional)}")
    hidden = batch["hidden"]
    _require(hidden.ndim == 2, "hidden must be a matrix")
    n = hidden.shape[0]
    _require(0 < n <= max_batch_size, "Batch size is zero or exceeds the frozen call cap")
    _float32(hidden, (n, 512), "hidden")
    for name in ("scene_ids", "episode_ids"):
        _require(batch[name].shape == (n,), f"{name} must have shape ({n},)")
    scenes, episodes = (_strings(batch[name], name) for name in ("scene_ids", "episode_ids"))
    keys = list(zip(scenes, episodes))
    _require(set(keys) <= allowed, "Batch contains episode keys outside the allowed manifest")
    _require(batch["split"].shape in ((), (n,)), "split must be a scalar or N strings")
    roles = _strings(batch["split"], "split")
    _require(set(roles) == {expected_split}, "Mixed splits or unexpected split role")
    for name in ("actions", "target_actions", "prev_actions"):
        if name in batch:
            _action_ids(batch[name], n, name)
    if "reset_masks" in batch:
        masks = batch["reset_masks"]
        _require(masks.shape == (n,) and masks.dtype.kind in "biuf"
                 and bool(((masks == 0) | (masks == 1)).all()),
                 "reset_masks must be an N-vector of booleans or 0/1")
    if "step_ids" in batch:
        steps = batch["step_ids"]
        _require(steps.shape == (n,) and steps.dtype.kind in "iu"
                 and bool((steps >= 0).all()), "step_ids must be nonnegative integers")
        _require(len(set((s, e, int(t)) for (s, e), t in zip(keys, steps))) == n,
                 "Duplicate (scene, episode, step) samples")

    # These are the only model computations. No simulator or encoder is imported.
    with np.errstate(over="ignore", invalid="ignore"):
        base_logits = hidden @ base[WEIGHT_KEY].T + base[BIAS_KEY]
        candidate_logits = (hidden @ effective[WEIGHT_KEY].T + effective[BIAS_KEY]
                            if patch else base_logits.copy())
    _require(bool(np.isfinite(base_logits).all() and np.isfinite(candidate_logits).all()),
             "Head forward produced nonfinite logits")
    recorded_check = {"provided": "base_logits" in batch, "verified": False,
                      "rtol": BASE_LOGITS_RTOL, "atol": BASE_LOGITS_ATOL}
    if "base_logits" in batch:
        _float32(batch["base_logits"], (n, 4), "base_logits")
        recorded_check["max_abs_error"] = float(np.max(np.abs(
            base_logits.astype(np.float64) - batch["base_logits"].astype(np.float64))))
        _require(bool(np.allclose(base_logits, batch["base_logits"],
                                  rtol=BASE_LOGITS_RTOL, atol=BASE_LOGITS_ATOL)),
                 "Recorded base_logits do not match the verified base head/features")
        recorded_check["verified"] = True

    base_logp = _log_softmax(base_logits)
    candidate_logp = _log_softmax(candidate_logits)
    base_p, candidate_p = np.exp(base_logp), np.exp(candidate_logp)
    base_argmax, candidate_argmax = base_logits.argmax(axis=1), candidate_logits.argmax(axis=1)
    kl = (base_p * (base_logp - candidate_logp)).sum(axis=1)
    reverse_kl = (candidate_p * (candidate_logp - base_logp)).sum(axis=1)
    # Only sub-roundoff negative KL is possible; retain a material failure.
    _require(bool((kl >= -1e-10).all() and (reverse_kl >= -1e-10).all()),
             "Numerically invalid KL divergence")
    kl, reverse_kl = np.maximum(kl, 0), np.maximum(reverse_kl, 0)
    metrics = {
        "mean_kl_base_to_candidate": float(kl.mean()),
        "max_kl_base_to_candidate": float(kl.max()),
        "mean_kl_candidate_to_base": float(reverse_kl.mean()),
        "mean_total_variation": float((0.5 * np.abs(base_p - candidate_p).sum(axis=1)).mean()),
        "argmax_action_change_count": int((base_argmax != candidate_argmax).sum()),
        "argmax_action_change_fraction": float((base_argmax != candidate_argmax).mean()),
        "max_abs_logit_change": float(np.max(np.abs(
            candidate_logits.astype(np.float64) - base_logits.astype(np.float64)))),
    }
    action_observations = None
    if "actions" in batch:
        actions = batch["actions"]
        action_observations = {
            "interpretation": "likelihood of recorded actions, not correctness or navigation success",
            "base_mean_negative_log_likelihood": float(-base_logp[np.arange(n), actions].mean()),
            "candidate_mean_negative_log_likelihood": float(-candidate_logp[np.arange(n), actions].mean()),
        }
    supervised = None
    if "target_actions" in batch:
        targets = batch["target_actions"]
        base_ce = float(-base_logp[np.arange(n), targets].mean())
        candidate_ce = float(-candidate_logp[np.arange(n), targets].mean())
        supervised = {
            "interpretation": "provided label supervision only; not navigation gain",
            "base_cross_entropy": base_ce, "candidate_cross_entropy": candidate_ce,
            "candidate_minus_base_cross_entropy": candidate_ce - base_ce,
            "base_top1_label_agreement": float((base_argmax == targets).mean()),
            "candidate_top1_label_agreement": float((candidate_argmax == targets).mean()),
        }
    feedback = {
        "schema_version": 1, "status": "completed", "backend": "frozen_head_replay",
        "no_model_navigation_executed": True, "closed_loop_navigation_metrics": None,
        "interpretation": "paired last-head outputs on fixed features; no recurrent policy rollout",
        "split": expected_split, "samples": n, "unique_episodes": len(set(keys)),
        "counting": {"counted_tool_calls": 1, "head_batched_forward_computations": 2 if patch else 1,
                     "feature_rows_per_head": n, "closed_loop_episodes": 0,
                     "budget_owner": "session controller reserves before invocation, including failures"},
        "input_provenance": {"base": base_record, "patch": patch_record, "batch": batch_record,
                             "allowed_keys_sha256": hashlib.sha256(json.dumps(
                                 sorted(allowed), separators=(",", ":")).encode()).hexdigest()},
        "effective_head": {key: _tensor_record(value) for key, value in effective.items()},
        "patch_semantics": "absolute replacements; missing head tensor retains base",
        "recorded_base_logits_check": recorded_check,
        "metrics": metrics, "recorded_actions": action_observations,
        "supervised_labels": supervised,
        "limitations": [
            "No full-network forward, simulator, navigation metric, or world-model training was executed.",
            "Fixed hidden features become off-policy after interventions in closed-loop deployment.",
            "Argmax changes are diagnostic only; official Falcon evaluation samples actions.",
            "Repeated timesteps are not independent navigation episodes or confidence-interval units.",
            "Session must verify frozen input hashes, action mapping, dataset origin and split access.",
        ],
    }
    # Reject nonfinite values anywhere before producing a completed report.
    json.dumps(feedback, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    arrays = {"base_logits": base_logits, "candidate_logits": candidate_logits,
              "base_log_probs": base_logp, "candidate_log_probs": candidate_logp,
              "base_argmax": base_argmax, "candidate_argmax": candidate_argmax,
              "kl_base_to_candidate": kl, "scene_ids": np.asarray(scenes),
              "episode_ids": np.asarray(episodes)}
    for name in ("step_ids", "actions", "target_actions"):
        if name in batch:
            arrays[name] = batch[name]
    arrays_path = output / "paired_outputs.npz"
    np.savez(arrays_path, **arrays)
    artifact_bytes = arrays_path.read_bytes()
    feedback["artifacts"] = {"paired_outputs": {
        "path": str(arrays_path), "bytes": len(artifact_bytes),
        "sha256": hashlib.sha256(artifact_bytes).hexdigest()}}
    (output / "feedback.json").write_text(json.dumps(feedback, indent=2, allow_nan=False) + "\n")
    return feedback
