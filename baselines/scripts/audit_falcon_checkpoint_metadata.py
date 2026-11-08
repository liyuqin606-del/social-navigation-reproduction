#!/usr/bin/env python3
"""Read Falcon checkpoint metadata with inert stand-ins; never import model code."""
import argparse
import collections
import hashlib
import io
import json
import pickle
from pathlib import Path
import zipfile


class Standin:
    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, *args, **kwargs):
        self.args = args

    def __setstate__(self, state):
        self.state = state


def tensor_metadata(storage, offset, size, stride, *args):
    return {"_tensor_metadata": True, "size": size, "stride": stride,
            "storage_offset": offset}


class MetadataReader(pickle.Unpickler):
    def __init__(self, stream):
        super().__init__(stream)
        self.classes = {}

    def find_class(self, module, name):
        known = {
            ("collections", "OrderedDict"): collections.OrderedDict,
            ("collections", "defaultdict"): collections.defaultdict,
            ("torch._utils", "_rebuild_tensor_v2"): tensor_metadata,
        }
        primitives = {"bool": bool, "dict": dict, "float": float,
                      "list": list, "long": int, "unicode": str}
        if (module, name) in known:
            return known[module, name]
        if module in ("__builtin__", "builtins") and name in primitives:
            return primitives[name]
        allowed = module.startswith((
            "omegaconf.", "habitat.config.", "habitat_baselines.config.", "falcon."
        )) or (module, name) in [
            ("typing", "Any"), ("typing", "List"), ("typing", "Dict"),
            ("torch", "FloatStorage")
        ]
        if not allowed:
            raise ValueError("Unsupported pickle global " + module + "." + name)
        key = (module, name)
        if key not in self.classes:
            self.classes[key] = type(name, (Standin,), {"_source_module": module})
        return self.classes[key]

    def persistent_load(self, pid):
        return {"_storage_reference": pid}


def plain(value):
    if isinstance(value, Standin):
        state = getattr(value, "state", {})
        if "_content" in state:
            return plain(state["_content"])
        if "_val" in state:
            return plain(state["_val"])
        return {"_standin": value.__class__._source_module + "." +
                value.__class__.__name__, "state_keys": list(state)}
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, type):
        return str(value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with zipfile.ZipFile(args.checkpoint) as archive:
        members = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(members) != 1:
            raise ValueError("Expected exactly one data.pkl")
        ckpt = MetadataReader(io.BytesIO(archive.read(members[0]))).load()
    policy = ckpt[0]["state_dict"]
    report = {
        "method": "Restricted local unpickler with inert stand-ins; no upstream classes/functions imported, no tensor payload reconstructed, no model forward. Metadata decoded without interpolation resolution.",
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_top_level_keys": [str(k) for k in ckpt],
        "extra_state": plain(ckpt.get("extra_state")),
        "config": plain(ckpt["config"]),
        "policy_state_dict_key_count": len(policy),
        "policy_aux_keys": [key for key in policy
                            if any(term in key for term in ("aux", "future", "human", "people"))],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "config"}))


if __name__ == "__main__":
    main()

