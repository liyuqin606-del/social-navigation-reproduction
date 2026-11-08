#!/usr/bin/env python3
"""Bounded EGL inventory and empty-scene renderer probes inside an existing GPU job.

No checkpoint, navigation episode, real scene, install, driver repair, or visibility
override. Each native-library probe runs in a fresh child interpreter.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
import traceback

ENV_KEYS = ("SLURM_JOB_ID", "SLURMD_NODENAME", "SLURM_JOB_GPUS", "SLURM_STEP_GPUS",
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES",
            "LD_LIBRARY_PATH", "CONDA_PREFIX", "__EGL_VENDOR_LIBRARY_FILENAMES",
            "__EGL_VENDOR_LIBRARY_DIRS", "DISPLAY", "EGL_PLATFORM", "MAGNUM_LOG",
            "HABITAT_SIM_LOG", "PYTHONNOUSERSITE")
CASES = ("loader", "sim_first", "magnum_first")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def guard():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Existing managed Slurm GPU allocation required")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"GPU-[0-9a-fA-F-]{36}", mask):
        raise RuntimeError("Require the managed worker's verified single full GPU UUID mask")


def command(argv, timeout=10):
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return {"argv": argv, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "timed_out": False}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "timed_out": True}
    except OSError as exc:
        return {"argv": argv, "error": str(exc), "timed_out": False}


def inventory():
    roots = [Path("/usr/share/glvnd/egl_vendor.d"), Path("/etc/glvnd/egl_vendor.d"),
             Path(sys.prefix) / "share/glvnd/egl_vendor.d"]
    roots.extend(Path(p) for p in os.environ.get("__EGL_VENDOR_LIBRARY_DIRS", "").split(":") if p)
    files = {p for root in roots for p in root.glob("*.json")}
    files.update(Path(p) for p in os.environ.get("__EGL_VENDOR_LIBRARY_FILENAMES", "").split(":") if p)
    vendor = {}
    for p in sorted(files):
        try:
            if p.stat().st_size > 65536:
                raise ValueError("Vendor JSON unexpectedly exceeds 64 KiB")
            raw = p.read_text()
            vendor[str(p)] = {"text": raw, "parsed": json.loads(raw), "sha256": sha(p)}
        except (OSError, ValueError) as exc:
            vendor[str(p)] = {"error": str(exc)}
    libs = {}
    libdirs = [Path(sys.prefix) / "lib", Path("/usr/lib/x86_64-linux-gnu"),
               Path("/lib/x86_64-linux-gnu"), Path("/usr/lib64"),
               Path("/usr/local/nvidia/lib64"), Path("/usr/lib/x86_64-linux-gnu/nvidia/current")]
    for root in libdirs:
        for pattern in ("libEGL*", "libOpenGL*", "libGLdispatch*", "libnvidia-egl*", "libnvidia-glsi*"):
            for p in root.glob(pattern):
                if p.is_file():
                    libs[str(p)] = {"resolved": str(p.resolve()), "bytes": p.stat().st_size}
    binding = {}
    site_roots = list((Path(sys.prefix) / "lib").glob("python*/site-packages"))
    for root in site_roots:
        candidates = list(root.glob("habitat_sim/_ext/*.so")) + list(root.glob("habitat_sim*/habitat_sim/_ext/*.so"))
        for p in candidates:
            binding[str(p)] = command(["ldd", str(p)])
    driver = Path("/proc/driver/nvidia/version")
    return {"python": sys.executable, "python_version": sys.version, "prefix": sys.prefix,
            "hostname": socket.gethostname(), "environment": {k: os.environ.get(k) for k in ENV_KEYS},
            "nvidia_smi": command(["nvidia-smi", "--query-gpu=index,uuid,name,driver_version,minor_number,pci.bus_id", "--format=csv,noheader"]),
            "kernel_driver": driver.read_text() if driver.is_file() else None,
            "vendor_files": vendor, "library_files": libs, "habitat_binding_ldd": binding,
            "device_files": [str(p) for p in sorted(Path("/dev").glob("nvidia*"))]}


def mapped_libraries():
    p = Path("/proc/self/maps")
    if not p.is_file():
        return []
    return sorted({line.split()[-1] for line in p.read_text().splitlines()
                   if any(s in line for s in ("libEGL", "libOpenGL", "libGLdispatch", "libnvidia", "habitat_sim_bindings", "magnum"))
                   and "/" in line.split()[-1]})


def loader_probe(report, save):
    handles = {}
    report["loads"] = {}
    # This child is intentionally separate from both renderer children, so these
    # explicit loader calls cannot warm up or repair their library state.
    for name in ("libEGL.so.1", "libOpenGL.so.0", "libGLdispatch.so.0", "libEGL_nvidia.so.0"):
        try:
            handles[name] = ctypes.CDLL(name, mode=ctypes.RTLD_LOCAL)
            report["loads"][name] = {"loaded": True}
        except OSError as exc:
            report["loads"][name] = {"loaded": False, "error": str(exc)}
        save()
    egl = handles.get("libEGL.so.1")
    if egl is not None:
        egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
        egl.eglQueryString.restype = ctypes.c_char_p
        egl.eglGetError.restype = ctypes.c_uint
        egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
        egl.eglGetProcAddress.restype = ctypes.c_void_p
        extensions = egl.eglQueryString(None, 0x3055)
        report["client_extensions"] = extensions.decode() if extensions else None
        report["query_string_error"] = hex(egl.eglGetError())
        ext = set((report["client_extensions"] or "").split())
        report["magnum_device_path_extensions_available"] = bool(
            ext.intersection({"EGL_EXT_device_enumeration", "EGL_EXT_device_base"})
            and {"EGL_EXT_platform_base", "EGL_EXT_platform_device"}.issubset(ext))
        address = egl.eglGetProcAddress(b"eglQueryDevicesEXT")
        report["egl_devices"] = []
        if address:
            query = ctypes.CFUNCTYPE(ctypes.c_uint, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))(address)
            array = (ctypes.c_void_p * 32)()
            count = ctypes.c_int()
            ok = query(32, array, ctypes.byref(count))
            report["query_devices"] = {"ok": bool(ok), "count": count.value, "error": hex(egl.eglGetError())}
            string_ptr = egl.eglGetProcAddress(b"eglQueryDeviceStringEXT")
            if ok and string_ptr:
                strings = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)(string_ptr)
                for i in range(min(max(count.value, 0), 32)):
                    value = strings(array[i], 0x3055)
                    report["egl_devices"].append({"enumeration_index": i,
                                                  "extensions": value.decode() if value else None})
        else:
            report["query_devices"] = {"function_available": False}
    report["loaded_library_paths"] = mapped_libraries()
    report["loaded_egl_dependency_reports"] = {
        p: command(["ldd", p]) for p in report["loaded_library_paths"]
        if Path(p).name.startswith(("libEGL", "libnvidia-egl"))}
    report["status"] = "DIAGNOSTIC_COMPLETE"


def renderer_probe(report, save, case):
    report["native_modules_before"] = {key: key in sys.modules for key in ("habitat_sim", "magnum", "corrade")}
    report["stage"] = "before_native_import"
    save()
    if case == "magnum_first":
        import magnum  # noqa: F401 -- deliberate order comparison only
        report["stage"] = "magnum_imported"
        save()
    import habitat_sim
    import numpy as np
    report.update(stage="habitat_sim_imported", habitat_sim_version=habitat_sim.__version__,
                  habitat_sim_file=habitat_sim.__file__)
    report["loaded_library_paths"] = mapped_libraries()
    save()
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = "NONE"
    config.gpu_device_id = 0  # worker has already bound exactly the allocated UUID
    config.enable_physics = False
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "depth"
    sensor.sensor_type = habitat_sim.SensorType.DEPTH
    sensor.resolution = [32, 32]
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    report["stage"] = "creating_none_renderer"
    save()
    with habitat_sim.Simulator(habitat_sim.Configuration(config, [agent])) as sim:
        depth = sim.get_sensor_observations()["depth"]
        report["frame"] = {"shape": list(depth.shape), "finite": bool(np.isfinite(depth).all())}
        if depth.shape != (32, 32) or not report["frame"]["finite"]:
            raise RuntimeError("Invalid empty-scene depth frame")
    report["loaded_library_paths"] = mapped_libraries()
    report.update(stage="renderer_closed", status="PASS")


def child(case, output):
    report = {"case": case, "status": "STARTED", "pid": os.getpid(),
              "hostname": socket.gethostname(), "environment": {k: os.environ.get(k) for k in ENV_KEYS},
              "scene_id": "NONE" if case != "loader" else None, "weights_loaded": False,
              "benchmark_evaluated": False, "paper_reproduction": False}
    path = output / (case + ".json")
    save = lambda: dump(path, report)
    save()
    try:
        (loader_probe(report, save) if case == "loader" else renderer_probe(report, save, case))
        report["child_exit_code"] = 0
        save()
        return 0
    except Exception as exc:
        report.update(status="FAIL", child_exit_code=2, error_type=type(exc).__name__,
                      error=str(exc), traceback=traceback.format_exc())
        save()
        traceback.print_exc()
        return 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-timeout", type=int, default=60)
    p.add_argument("--probe", choices=CASES, help=argparse.SUPPRESS)
    args = p.parse_args()
    guard()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.probe:
        return child(args.probe, args.output)
    if not 5 <= args.case_timeout <= 120:
        raise ValueError("case timeout must be between 5 and 120 seconds")
    if any(args.output.iterdir()):
        raise ValueError("Use an empty output directory")
    dump(args.output / "inventory.json", inventory())
    summary = {"schema": 1, "scope": "EGL_NODE_DIAGNOSTIC_NO_BENCHMARK",
               "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "hostname": socket.gethostname(), "slurm_job_id": os.environ["SLURM_JOB_ID"],
               "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
               "inventory_sha256": sha(args.output / "inventory.json"),
               "diagnostic_complete": False, "renderer_status": "UNKNOWN",
               "paper_reproduction": False, "benchmark_evaluated": False, "weights_loaded": False,
               "case_timeout_seconds": args.case_timeout, "cases": {}}
    summary_path = args.output / "diagnostic_summary.json"
    dump(summary_path, summary)
    for case in CASES:
        log = args.output / (case + ".log")
        row = {"case": case, "status": "STARTED", "timed_out": False}
        summary["cases"][case] = row
        dump(summary_path, summary)
        argv = [sys.executable, str(Path(__file__).resolve()), "--output", str(args.output), "--probe", case]
        start = time.monotonic()
        with log.open("w") as stream:
            process = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=args.case_timeout)
            except subprocess.TimeoutExpired:
                row["timed_out"] = True
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        row.update(status="EXITED", returncode=code, elapsed_seconds=time.monotonic()-start,
                   log_bytes=log.stat().st_size, log_sha256=sha(log))
        child_path = args.output / (case + ".json")
        row["report_sha256"] = sha(child_path) if child_path.is_file() else None
        dump(summary_path, summary)
    # Completion and scientific failure classification are independently derived
    # from process exits, staged child reports, and hashed complete log files.
    validator = Path(__file__).with_name("validate_habitat_egl.py")
    result = subprocess.run([sys.executable, str(validator), "--output", str(args.output), "--finalize"], timeout=30)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
