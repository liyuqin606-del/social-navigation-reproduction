"""Small durable, non-executing storage helpers for weight editing sessions."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


class ContractError(ValueError):
    """A session invariant or scientific accounting requirement was violated."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def json_value(value):
    # Reject NaN/Infinity and arbitrary Python objects in feedback and requests.
    return json.loads(json.dumps(value, allow_nan=False, ensure_ascii=False))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(array):
    return hashlib.sha256(np.asarray(array, dtype='<f4', order='C').tobytes(order='C')).hexdigest()


def tensor_record(array):
    return {'shape': list(array.shape), 'dtype': 'float32', 'sha256': tensor_sha256(array)}


def read_json(path):
    return json.loads(Path(path).read_text())


def _fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    payload = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_npz(path, arrays):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def session_lock(root):
    """One stable inode serializes reservation and completion transactions."""
    with (Path(root) / 'SESSION.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def load_npz(path, *, max_bytes=128 << 20):
    """Numeric float32 only, bounded expanded archive, never pickle or extract."""
    path = Path(path)
    require(path.is_file(), 'NPZ_FILE_MISSING')
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        require(entries and len(entries) <= 10000, 'NPZ_ENTRY_COUNT')
        require(sum(row.file_size for row in entries) <= max_bytes, 'NPZ_EXPANDED_SIZE_LIMIT')
        require(len({r.filename for r in entries}) == len(entries), 'NPZ_DUPLICATE_ENTRY')
        require(all(row.filename.endswith('.npy') for row in entries), 'NPZ_NON_ARRAY_ENTRY')
    with np.load(path, allow_pickle=False) as data:
        require(len(data.files) == len(set(data.files)), 'NPZ_DUPLICATE_TENSOR_KEY')
        arrays = {}
        for key in data.files:
            require(isinstance(key, str) and key, 'EMPTY_TENSOR_KEY')
            array = data[key]
            require(array.dtype.kind == 'f' and array.dtype.itemsize == 4, 'FLOAT32_REQUIRED:' + key)
            require(array.size > 0 and np.isfinite(array).all(), 'NONFINITE_OR_EMPTY_TENSOR:' + key)
            arrays[key] = np.array(array, dtype='<f4', order='C', copy=True)
    return arrays
