#!/usr/bin/env python3
"""Independently verify an EGL diagnostic capture; renderer FAIL is never PASS."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(output):
    summary = json.loads((output / "diagnostic_summary.json").read_text())
    inventory = json.loads((output / "inventory.json").read_text())
    assert digest(output / "inventory.json") == summary["inventory_sha256"]
    assert summary["scope"] == "EGL_NODE_DIAGNOSTIC_NO_BENCHMARK"
    assert summary["schema"] == 1
    for key in ("paper_reproduction", "benchmark_evaluated", "weights_loaded"):
        assert summary[key] is False
    assert set(summary["cases"]) == {"loader", "sim_first", "magnum_first"}
    assert inventory["hostname"] == summary["hostname"]
    assert inventory["environment"]["SLURM_JOB_ID"] == summary["slurm_job_id"]
    mask = summary["cuda_visible_devices"]
    assert re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", mask)
    assert inventory["environment"]["CUDA_VISIBLE_DEVICES"] == mask
    assert inventory["nvidia_smi"].get("returncode") == 0
    assert inventory["nvidia_smi"]["timed_out"] is False
    assert mask in inventory["nvidia_smi"]["stdout"]
    assert isinstance(inventory["vendor_files"], dict)
    assert isinstance(inventory["library_files"], dict)
    assert inventory["habitat_binding_ldd"], "Missing native binding linkage evidence"
    for linkage in inventory["habitat_binding_ldd"].values():
        assert linkage["timed_out"] is False and linkage.get("returncode") == 0
    outcomes = {}
    for name in ("loader", "sim_first", "magnum_first"):
        row = summary["cases"][name]
        assert row["case"] == name and row["status"] == "EXITED"
        assert row["timed_out"] is False, "Timed-out probes are incomplete"
        code = row["returncode"]
        assert type(code) is int and code not in (124, 127)
        assert 0 <= row["elapsed_seconds"] <= summary["case_timeout_seconds"] + 5
        log = output / (name + ".log")
        report_path = output / (name + ".json")
        assert log.stat().st_size == row["log_bytes"]
        assert digest(log) == row["log_sha256"]
        assert digest(report_path) == row["report_sha256"]
        report = json.loads(report_path.read_text())
        assert report["case"] == name and report["hostname"] == summary["hostname"]
        assert report["environment"]["SLURM_JOB_ID"] == summary["slurm_job_id"]
        assert report["environment"]["CUDA_VISIBLE_DEVICES"] == mask
        for key in ("weights_loaded", "benchmark_evaluated", "paper_reproduction"):
            assert report[key] is False
        if name == "loader":
            assert code == 0 and report["child_exit_code"] == 0
            assert report["status"] == "DIAGNOSTIC_COMPLETE"
            assert set(report["loads"]) == {"libEGL.so.1", "libOpenGL.so.0", "libGLdispatch.so.0", "libEGL_nvidia.so.0"}
            assert "loaded_library_paths" in report
            outcomes[name] = "CAPTURE_COMPLETE"
            continue
        assert report["scene_id"] == "NONE"
        assert report["native_modules_before"] == {"habitat_sim": False, "magnum": False, "corrade": False}
        if code == 0:
            assert report["child_exit_code"] == 0 and report["status"] == "PASS"
            assert report["stage"] == "renderer_closed"
            assert report["frame"] == {"shape": [32, 32], "finite": True}
            assert report["habitat_sim_version"] == "0.3.1"
            outcomes[name] = "PASS"
        elif code == 2:
            assert report["child_exit_code"] == 2 and report["status"] == "FAIL"
            assert report["traceback"] and report["error_type"]
            outcomes[name] = "FAIL_PYTHON_EXCEPTION"
        else:
            # An observed native abort is a FAILED renderer test. It can be a
            # complete capture only with a saved prelude and an intact explicit
            # native-error log, never merely because the wrapper wrote success.
            assert code in (-6, -11), "Unexpected signal/exit is not classifiable"
            assert report["status"] == "STARTED" and "child_exit_code" not in report
            assert report["stage"] in ("before_native_import", "magnum_imported", "habitat_sim_imported", "creating_none_renderer")
            text = log.read_text(errors="replace")
            assert any(marker in text for marker in ("EGL_BAD_", "Unable to create windowless context", "malloc_consolidate", "Segmentation fault", "Fatal Python error")), "No explicit native failure evidence"
            outcomes[name] = "FAIL_NATIVE_SIGNAL"
    status = "PASS" if all(outcomes[n] == "PASS" for n in ("sim_first", "magnum_first")) else "FAIL"
    return summary, {"diagnostic_complete": True, "renderer_status": status,
                     "case_classifications": outcomes,
                     "status": "DIAGNOSTIC_COMPLETE" if status == "PASS" else "DIAGNOSTIC_COMPLETE_WITH_RENDERER_FAILURES"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    try:
        summary, classification = verify(args.output)
        if args.finalize:
            summary.update(classification)
            (args.output / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        else:
            for key, value in classification.items():
                assert summary[key] == value, "Aggregate classification disagrees with evidence"
        result = {"integrity_status": "PASS", **classification,
                  "benchmark_evaluated": False, "paper_reproduction": False}
        code = 0
    except Exception as exc:
        result = {"integrity_status": "FAIL", "diagnostic_complete": False,
                  "renderer_status": "UNKNOWN", "error_type": type(exc).__name__,
                  "error": str(exc), "benchmark_evaluated": False, "paper_reproduction": False}
        code = 2
    (args.output / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
