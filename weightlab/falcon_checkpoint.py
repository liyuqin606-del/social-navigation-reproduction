"""Bounded, CPU-only export of tensors from the verified Falcon checkpoint.

No torch import, tensor unpickling, upstream class import or model forward is
performed. A restricted metadata decoder preserves storage references; selected
contiguous float32 slices are then read directly from the checkpoint ZIP.
"""
from __future__ import annotations

import collections
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import pickle
import tempfile
import zipfile

import numpy as np

CHECKPOINT_SHA256 = "ba7f2824604639d934be2359226c9e9fb35d5b1b2fb39ffbcbbbd833252db6ca"
HEAD_KEYS = ("action_distribution.linear.weight", "action_distribution.linear.bias")
ALLOWED_TENSOR_KEYS = frozenset(HEAD_KEYS + (
    "net.visual_fc.1.weight", "net.visual_fc.1.bias",
    "net.tgt_embeding.weight", "net.tgt_embeding.bias", "net.prev_action_embedding.weight",
) + tuple("net.state_encoder.rnn." + stem + "_l" + str(layer)
          for layer in (0, 1) for stem in ("weight_ih", "weight_hh", "bias_ih", "bias_hh")))
MAX_ARCHIVE_BYTES = 128 << 20
MAX_UNCOMPRESSED_BYTES = 128 << 20
MAX_METADATA_BYTES = 4 << 20
MAX_STORAGE_BYTES = 64 << 20
MAX_MEMBERS = 1024


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value):
    """Hash exact C-contiguous little-endian float32 values, without conversion."""
    array = np.asarray(value)
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise ValueError("TENSOR_DTYPE_MUST_BE_FLOAT32")
    if not np.isfinite(array).all():
        raise ValueError("NONFINITE_TENSOR")
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes(order="C")).hexdigest()


class _Standin:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def _tensor_record(storage, offset, size, stride, *unused):
    return {"storage": storage, "offset": offset, "shape": size, "stride": stride}


class _MetadataReader(pickle.Unpickler):
    def __init__(self, stream):
        super().__init__(stream)
        self.classes = {}

    def find_class(self, module, name):
        known = {("collections", "OrderedDict"): collections.OrderedDict,
                 ("collections", "defaultdict"): collections.defaultdict,
                 ("torch._utils", "_rebuild_tensor_v2"): _tensor_record}
        if (module, name) in known:
            return known[module, name]
        primitives = {"bool": bool, "dict": dict, "float": float, "list": list,
                      "long": int, "unicode": str}
        if module in ("__builtin__", "builtins") and name in primitives:
            return primitives[name]
        allowed = module.startswith(("omegaconf.", "habitat.config.", "habitat_baselines.config.", "falcon."))
        allowed = allowed or (module, name) in {
            ("typing", "Any"), ("typing", "List"), ("typing", "Dict"), ("torch", "FloatStorage")}
        if not allowed:
            raise ValueError("UNSUPPORTED_PICKLE_GLOBAL: " + module + "." + name)
        key = (module, name)
        if key not in self.classes:
            self.classes[key] = type(name, (_Standin,), {"_source_module": module})
        return self.classes[key]

    def persistent_load(self, pid):
        if (not isinstance(pid, tuple) or len(pid) != 5 or pid[0] != "storage"
                or getattr(pid[1], "_source_module", None) != "torch"
                or getattr(pid[1], "__name__", None) != "FloatStorage"
                or not isinstance(pid[2], str) or not pid[2].isdigit()
                or type(pid[4]) is not int or not 0 < pid[4] * 4 <= MAX_STORAGE_BYTES):
            raise ValueError("UNSUPPORTED_OR_UNBOUNDED_STORAGE_REFERENCE")
        return {"key": pid[2], "numel": pid[4], "dtype": "float32"}


def _inspect_archive(archive):
    infos = archive.infolist()
    if len(infos) > MAX_MEMBERS:
        raise ValueError("ZIP_MEMBER_COUNT_LIMIT")
    names, total = set(), 0
    for info in infos:
        name = PurePosixPath(info.filename)
        if (info.filename in names or name.is_absolute() or ".." in name.parts
                or "\\" in info.filename or info.flag_bits & 1
                or info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED)):
            raise ValueError("UNSAFE_OR_DUPLICATE_ZIP_MEMBER")
        names.add(info.filename)
        total += info.file_size
        if info.file_size > MAX_STORAGE_BYTES or total > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP_UNCOMPRESSED_SIZE_LIMIT")
        if info.file_size > max(1, info.compress_size) * 1000:
            raise ValueError("ZIP_COMPRESSION_RATIO_LIMIT")
    metadata = [info for info in infos if info.filename.endswith("/data.pkl")]
    if len(metadata) != 1 or metadata[0].file_size > MAX_METADATA_BYTES:
        raise ValueError("EXPECTED_ONE_BOUNDED_METADATA_MEMBER")
    return metadata[0].filename, names


def _extract_array(archive, prefix, members, metadata):
    if not isinstance(metadata, dict) or set(metadata) != {"storage", "offset", "shape", "stride"}:
        raise ValueError("INVALID_TENSOR_METADATA")
    shape, stride = metadata["shape"], metadata["stride"]
    if (not isinstance(shape, (tuple, list)) or not shape or len(shape) > 4
            or any(type(n) is not int or n <= 0 for n in shape)
            or not isinstance(stride, (tuple, list)) or len(stride) != len(shape)):
        raise ValueError("INVALID_TENSOR_SHAPE_OR_STRIDE")
    expected, current = [], 1
    for size in reversed(shape):
        expected.insert(0, current)
        current *= size
    if tuple(stride) != tuple(expected):
        raise ValueError("ONLY_CONTIGUOUS_TENSORS_SUPPORTED")
    storage, offset = metadata["storage"], metadata["offset"]
    if (not isinstance(storage, dict) or storage.get("dtype") != "float32"
            or type(offset) is not int or offset < 0 or offset + current > storage["numel"]):
        raise ValueError("TENSOR_SLICE_OUTSIDE_STORAGE")
    member = prefix + "/data/" + storage["key"]
    if member not in members or archive.getinfo(member).file_size != storage["numel"] * 4:
        raise ValueError("STORAGE_LENGTH_MISMATCH")
    payload = archive.read(member)  # zipfile verifies CRC; bounded above.
    array = np.frombuffer(payload, dtype="<f4", count=current, offset=offset * 4).reshape(shape).copy()
    if not np.isfinite(array).all():
        raise ValueError("NONFINITE_TENSOR")
    return array


def export_tensors(checkpoint, output_npz, keys=None):
    """Export selected original tensors and a provenance JSON; refuse overwrite.

    NPZ arrays are absolute tensor values under exact state_dict keys. The
    returned JSON-compatible report is also written beside NPZ as `.json`.
    Only the verified official checkpoint is supported by this adapter.
    """
    checkpoint, output_npz = Path(checkpoint).absolute(), Path(output_npz).absolute()
    output_json = output_npz.with_suffix(".json")
    if output_npz.suffix != ".npz" or output_npz.exists() or output_json.exists():
        raise ValueError("FRESH_NPZ_AND_SIDECAR_REQUIRED")
    selected = list(HEAD_KEYS if keys is None else keys)
    if (not selected or len(selected) != len(set(selected)) or not set(selected) <= ALLOWED_TENSOR_KEYS):
        raise ValueError("UNKNOWN_DUPLICATE_OR_EMPTY_TENSOR_SELECTION")
    if not checkpoint.is_file() or checkpoint.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("CHECKPOINT_FILE_SIZE_LIMIT")
    actual_sha = file_sha256(checkpoint)
    if actual_sha != CHECKPOINT_SHA256:
        raise ValueError("UNVERIFIED_FALCON_CHECKPOINT_SHA256")
    with zipfile.ZipFile(checkpoint) as archive:
        metadata_name, members = _inspect_archive(archive)
        prefix = metadata_name.removesuffix("/data.pkl")
        if prefix + "/byteorder" in members and archive.read(prefix + "/byteorder").strip() != b"little":
            raise ValueError("ONLY_LITTLE_ENDIAN_CHECKPOINT_SUPPORTED")
        data = _MetadataReader(io.BytesIO(archive.read(metadata_name))).load()
        state = data[0]["state_dict"]
        arrays = {key: _extract_array(archive, prefix, members, state[key]) for key in selected}
    if HEAD_KEYS[0] in arrays and arrays[HEAD_KEYS[0]].shape != (4, 512):
        raise ValueError("UNEXPECTED_FALCON_HEAD_SHAPE")
    if HEAD_KEYS[1] in arrays and arrays[HEAD_KEYS[1]].shape != (4,):
        raise ValueError("UNEXPECTED_FALCON_HEAD_SHAPE")
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output_npz.name + ".", dir=output_npz.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez(stream, **arrays)
        os.replace(temporary, output_npz)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    tensors = {key: {"shape": list(value.shape), "dtype": "float32",
                     "sha256": tensor_sha256(value), "numel": int(math.prod(value.shape))}
               for key, value in arrays.items()}
    report = {"schema": "weightlab_falcon_base_v1", "checkpoint_sha256": actual_sha,
              "checkpoint": {"path": str(checkpoint), "bytes": checkpoint.stat().st_size, "sha256": actual_sha},
              "state_dict_location": "checkpoint[0]['state_dict']", "base_tensors": tensors,
              "base_npz": {"path": str(output_npz), "bytes": output_npz.stat().st_size,
                           "sha256": file_sha256(output_npz)},
              "hash_contract": "SHA256 of C-contiguous little-endian float32 raw tensor bytes",
              "method": "Restricted inert metadata decoding and bounded ZIP storage slicing; no Torch import or model forward",
              "scientific_result": False, "real_gpu_execution": False}
    output_json.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
