"""CPU tensor-analysis worker. The parent must launch it under the OS sandbox."""
import ast
import contextlib
import io
import json
from pathlib import Path
import resource
import sys

import numpy as np


def main():
    root = Path(sys.argv[1]).resolve()
    resource.setrlimit(resource.RLIMIT_CPU, (25, 25))
    # Constrain output size; parent also enforces a wall-clock and directory cap.
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 << 20, 64 << 20))
    source = (root / 'proposal.py').read_text()
    tree = ast.parse(source)
    forbidden_names = {'eval', 'exec', 'compile', 'open', '__import__', 'input',
                       'breakpoint', 'globals', 'locals', 'getattr', 'setattr', 'delattr'}
    forbidden_attrs = {'backward', 'autograd', 'optim', 'optimizer', 'ctypeslib',
                       'load', 'save', 'savez', 'savez_compressed', 'fromfile', 'tofile',
                       'memmap', 'loadtxt', 'savetxt', 'genfromtxt'}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError('Imports are disabled; NumPy is already available as np')
        if isinstance(node, ast.Name) and (node.id in forbidden_names or node.id.startswith('_')):
            raise ValueError('Forbidden name: ' + node.id)
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in forbidden_attrs):
            raise ValueError('Forbidden attribute: ' + node.attr)
    with np.load(root / 'base.npz', allow_pickle=False) as data:
        base = {k: data[k].copy() for k in data.files}
    with np.load(root / 'active.npz', allow_pickle=False) as data:
        active = {k: data[k].copy() for k in data.files}
    edits = {}

    def load_tensor(name, original=False):
        return (base if original else active)[name].copy()

    def save_edit(name, value):
        if name not in base:
            raise ValueError('Tensor is not exposed by this workbench: ' + str(name))
        array = np.asarray(value)
        if array.dtype.kind != 'f' or array.shape != base[name].shape or not np.isfinite(array).all():
            raise ValueError('save_edit requires finite floating tensor with original shape')
        array = array.astype(base[name].dtype)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite values after deployment-dtype conversion')
        edits[name] = array.copy()

    def load_probe(name):
        # Parent copies only evidence from an allowed, completed feedback call.
        if Path(name).name != name or name not in json.loads((root / 'probe_files.json').read_text()):
            raise ValueError('Unknown probe artifact')
        with np.load(root / 'probes' / name, allow_pickle=False) as data:
            return {k: data[k].copy() for k in data.files}

    builtins = {k: v for k, v in vars(__import__('builtins')).items()
                if k in {'print', 'range', 'len', 'min', 'max', 'sum', 'abs', 'enumerate',
                         'zip', 'list', 'dict', 'tuple', 'set', 'float', 'int', 'str', 'bool',
                         'sorted', 'reversed', 'all', 'any', 'isinstance', 'ValueError', 'Exception'}}
    scope = {'__builtins__': builtins, 'np': np, 'load_tensor': load_tensor,
             'save_edit': save_edit, 'load_probe': load_probe, 'tensor_names': sorted(base)}
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        exec(compile(tree, 'proposal.py', 'exec'), scope, scope)
    (root / 'stdout.txt').write_text(captured.getvalue()[:100000])
    if edits:
        np.savez(root / 'edits.npz', **edits)
    (root / 'worker_result.json').write_text(json.dumps({
        'status': 'CODE_COMPLETED', 'edited_keys': sorted(edits),
        'model_forward_executed': False,
        'note': 'CPU tensor arithmetic only; semantic no-iterative-fitting rule is also reviewed in source.'
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
