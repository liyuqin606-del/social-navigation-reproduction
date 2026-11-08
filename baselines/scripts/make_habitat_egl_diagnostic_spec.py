#!/usr/bin/env python3
"""Prepare a managed EGL diagnostic spec; never freeze, submit, or allocate."""
import argparse
import datetime
import json
from pathlib import Path
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline", help="Explicit Slurm deadline in cluster time; default local time +2h")
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refuse to overwrite a possibly frozen spec")
    here = Path(__file__).resolve().parent
    project = here.parents[1]
    root = "/mnt/home/liyuqin/USERDIR/vln_baselines_20261103"
    python = root + "/env/falcon/bin/python"
    now = datetime.datetime.now().astimezone()
    spec = {
        "run_id": "vln_egl_diag_" + now.strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8],
        "remote_root": root + "/runs", "remote_python": python,
        "gpu_binding": "slurm_linux_minor_uuid",
        "slurm": {"partition": "tacc", "gpus": 1, "cpus": 2, "wall_time": "00:10:00",
                  "deadline": args.deadline or (now + datetime.timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")},
        "science": {
            "cwd": root,
            "argv": [python, "{source}/diagnose_habitat_egl.py", "--output", "{output}/egl", "--case-timeout", "60"],
            "env": {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "MAGNUM_LOG": "verbose", "HABITAT_SIM_LOG": "warning"},
            "timeout_seconds": 480, "validation_timeout_seconds": 30,
            "validator_argv": [python, "{source}/validate_habitat_egl.py", "--output", "{output}/egl"],
            "required_outputs": ["egl/inventory.json", "egl/diagnostic_summary.json", "egl/validation.json"] +
                                ["egl/" + case + suffix for case in ("loader", "sim_first", "magnum_first") for suffix in (".json", ".log")]
        },
        "uploads": [{"file": str(here / name), "remote_name": name} for name in ("diagnose_habitat_egl.py", "validate_habitat_egl.py")],
        "remote_inputs": [],
        "scientific_files": [str(Path(__file__).resolve()), str(project / "baselines/prep/environment_evidence.json"),
                             str(project / "baselines/prep/egl_runtime_packages_20261103.json")],
        "reserve_bytes": 4 * 1024**3, "output_budget_bytes": 16 * 1024**2,
        "temporary_transfer_bytes": 16 * 1024**2, "remote_peak_additional_bytes": 64 * 1024**2,
        "monitor_timeout_seconds": 7200,
        "diagnostic_plan": {
            "status": "SPEC_ONLY_NOT_FROZEN_NOT_SUBMITTED", "benchmark_evaluated": False, "paper_reproduction": False,
            "weights_loaded": False, "scene": "NONE", "render_size": [32, 32], "node_selection": "scheduler chosen; record actual node",
            "exit_contract": "Zero means complete independently validated diagnostic capture. Native/Python renderer failures remain explicit renderer_status=FAIL; incomplete evidence or timeout returns nonzero.",
            "scope": "Inventory and independent cold-process sim-first/magnum-first renderer probes; no model, episode, real scene, driver repair, install, or visibility override."
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2) + "\n")
    print(str(args.output.resolve()))


if __name__ == "__main__":
    main()
