"""Pure NumPy edit recipes; every result is an absolute replacement tensor."""
import math
import numpy as np

from .io import require


def _array(value):
    a = np.asarray(value)
    require(a.dtype.kind == 'f' and a.dtype.itemsize == 4, 'FLOAT32_REQUIRED')
    require(a.size > 0 and np.isfinite(a).all(), 'FINITE_NONEMPTY_ARRAY_REQUIRED')
    return np.array(a, dtype=np.float32, copy=True)


def _strength(value):
    value = float(value)
    require(math.isfinite(value), 'FINITE_STRENGTH_REQUIRED')
    return value


def rank_one(weight, left, right, strength=1.0):
    """Return W + strength * outer(left, right), without vector normalization."""
    w, u, v = _array(weight), _array(left), _array(right)
    require(w.ndim == 2 and u.shape == (w.shape[0],) and v.shape == (w.shape[1],), 'RANK_ONE_SHAPE_MISMATCH')
    result = w.astype(np.float64) + _strength(strength) * np.outer(u.astype(np.float64), v.astype(np.float64))
    result = result.astype(np.float32)
    require(np.isfinite(result).all(), 'RECIPE_OVERFLOW')
    return result


def remove_direction(weight, direction, selected_columns=None, strength=1.0):
    """Remove a fraction of output-direction projection from chosen W columns.

    W[:,J] <- W[:,J] - strength*d*(d.T@W[:,J])/(d.T@d).
    Other columns are unchanged. Direction is length out_features; no tensor
    supplied to this function is modified in place.
    """
    w, d = _array(weight), _array(direction)
    require(w.ndim == 2 and d.shape == (w.shape[0],), 'DIRECTION_SHAPE_MISMATCH')
    columns = list(range(w.shape[1])) if selected_columns is None else list(selected_columns)
    require(columns and all(type(i) is int and 0 <= i < w.shape[1] for i in columns)
            and len(columns) == len(set(columns)), 'INVALID_SELECTED_COLUMNS')
    d64 = d.astype(np.float64)
    denominator = float(np.dot(d64, d64))
    require(denominator > 0, 'ZERO_DIRECTION')
    block = w[:, columns].astype(np.float64)
    w[:, columns] = (block - _strength(strength) * np.outer(d64, d64 @ block) / denominator).astype(np.float32)
    require(np.isfinite(w).all(), 'RECIPE_OVERFLOW')
    return w


def bias_shift(bias, shift, strength=1.0):
    """Return b + strength*shift; shift must be a finite scalar or exact vector."""
    b = _array(bias)
    require(b.ndim == 1, 'BIAS_VECTOR_REQUIRED')
    delta = np.asarray(shift, dtype=np.float64)
    require(delta.shape in ((), b.shape) and np.isfinite(delta).all(), 'BIAS_SHIFT_SHAPE_OR_VALUE')
    result = (b.astype(np.float64) + _strength(strength) * delta).astype(np.float32)
    require(np.isfinite(result).all(), 'RECIPE_OVERFLOW')
    return result


def delta_stats(value, reference, epsilon=1e-12):
    delta = value.astype(np.float64) - reference.astype(np.float64)
    norm = float(np.linalg.norm(delta.ravel()))
    ref_norm = float(np.linalg.norm(reference.astype(np.float64).ravel()))
    matrix = delta.reshape(delta.shape[0], -1) if delta.ndim >= 2 else delta.reshape(1, -1)
    return {'changed_entries': int(np.count_nonzero(delta)),
            'rank': int(np.linalg.matrix_rank(matrix)),
            'rank_convention': 'first_dimension_by_remaining_dimensions; vectors/scalars are one row',
            'frobenius': norm, 'reference_frobenius': ref_norm,
            'relative_frobenius': norm / max(ref_norm, epsilon)}
