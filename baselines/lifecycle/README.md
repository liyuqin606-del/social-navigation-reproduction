# Navigation baseline managed-run adapter

This directory prepares/owns one bounded scientific unit per Slurm job. No experiment has been submitted by building or testing it. The bundled `managed_run.py` and `manage_storage.py` retain the tacc-3090 skill's original source/license line unchanged.

`prepare_run.py` copies every runtime file into a new run-plan directory, gives storage its own ledger directory, generates both freezes, and validates the bindings **without connecting or allocating**. Only launch its printed managed argv after inputs/environment are ready. Do not run `owner.py` directly and do not reuse a historical plan directory.

Example specification (replace scientific paths, inputs, deadline and storage estimates):

```json
{
  "run_id": "vln_falcon_hm3d_shard000_UNIQUE",
  "remote_root": "/mnt/home/liyuqin/USERDIR/vln_baselines/runs",
  "remote_python": "/mnt/home/liyuqin/.Miniconda3/envs/falcon/bin/python",
  "slurm": {
    "partition": "tacc", "gpus": 1, "cpus": 4,
    "wall_time": "01:00:00", "deadline": "REPLACE_WITH_FUTURE_SERVER_TIME"
  },
  "science": {
    "cwd": "/mnt/home/liyuqin/USERDIR/vln_baselines/Falcon",
    "argv": ["/mnt/home/liyuqin/.Miniconda3/envs/falcon/bin/python", "{source}/evaluate_shard.py", "--output", "{output}"],
    "env": {},
    "timeout_seconds": 3420,
    "validator_argv": ["/mnt/home/liyuqin/.Miniconda3/envs/falcon/bin/python", "{source}/validate_shard.py", "--output", "{output}"],
    "validation_timeout_seconds": 120,
    "required_outputs": ["metrics.json", "episodes.json"]
  },
  "uploads": [
    {"file": "/ABS/evaluate_shard.py", "remote_name": "evaluate_shard.py"},
    {"file": "/ABS/validate_shard.py", "remote_name": "validate_shard.py"}
  ],
  "remote_inputs": [{"path": "/ABS/REMOTE/data_manifest.json", "bytes": 123, "sha256": "REPLACE"}],
  "scientific_files": ["/ABS/LOCAL/evaluation_protocol.json"],
  "output_budget_bytes": 67108864,
  "temporary_transfer_bytes": 67108864,
  "remote_peak_additional_bytes": 134217728,
  "monitor_timeout_seconds": 10800
}
```

The validator must enforce the **real** unit contract: exact episode IDs and count, official metrics, expected policy/checkpoint/config hashes, no duplicates or missing scenes. Generic subprocess exit zero is insufficient. `required_outputs` lists all result files to retrieve; video/raw-prediction production must fit the frozen budget or use a different approved plan. Scientific scripts must put results under `VLN_RUN_OUTPUT` (also available as `{output}` in argv). The source directory is accessible as `{source}`. All downloads and installations are completed before this owner begins.

```sh
python3 baselines/lifecycle/prepare_run.py --spec /ABS/spec.json --directory /ABS/fresh-plan
```

The printed argv starts the copied `managed_run.py`, whose owner publishes the bound DRIVER receipt, verifies keys/host, checks remote free space and explicit input hashes, uploads only its frozen manifest, then calls `sbatch` once. Ambiguous submission queries the exact unique name; it never re-submits. Every query/cancellation checks account/name/job ID. Release requires the main `sacct` job row to be terminal and `squeue` empty. Server wall time is capped at one hour, with a required deadline and worker timeout. A complete benchmark can span explicit disjoint episode shards; running one shard is not the full benchmark.

Worker checks its Slurm job name/ID and actual CUDA-visible GPU count/model before launching the scientific command. Final results and failed outputs get SHA manifests. Owner downloads into `.partial`, verifies bytes/SHA, atomically commits to `vm/`, and hard-links into `incremental_cache/`. A saver independently verifies collected units. Slurm logs are also hash-verified after release. No automatic cache cleanup is enabled.

If laptop/SSH monitoring fails, retain the original receipt. Inspect exact unique job with `squeue`/`sacct` and its persistent remote output. A job still present or an unknown submission is not released. Use a separate recovery receipt; do not re-run the same scientific unit merely because monitoring failed. This adapter deliberately does not auto-restart/requeue jobs or manufacture a successful managed receipt after interrupted monitoring.

`MANAGED_OFFLINE_CHECK.json`: eight original lifecycle fake-owner tests. `ADAPTER_CHECK.json`: identity, traversal, freeze/tamper and synthetic worker/validator tests. Both are **offline evidence only**, no TACC/model performance evidence.
