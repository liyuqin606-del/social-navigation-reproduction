# Finite reproduction queue

`continuous_reproduction_queue.py` prepares and runs a finite, ordered list of
already prepared scientific specifications. It implements the user's explicit
authorization to return to the normal Slurm queue after each allocation. It
does not reserve idle GPUs, call `sbatch`, bypass a partition limit, requeue an
existing job, or modify the managed lifecycle or scientific implementation.

Each new allocation follows the real project entry points:

1. Write a derived specification in a new attempt directory.
2. Invoke `baselines/lifecycle/prepare_run.py` to freeze that unique plan.
3. Invoke only the copied plan's `managed_run.py --config … --freeze …`.
4. Wait for the owner and manager to finish, a terminal **main** `sacct` row,
   and an empty `squeue` result for that exact job identity.
5. Rehash the collected manifest and worker identity, then run the independent
   local scientific validator. Only complete validated shard IDs are saved as
   completed. A subsequent unit is submitted only after these gates pass.

The queue exits when its finite requested completion IDs are complete. A local
checkpoint records orchestration and completed scientific units; it does not
restore an interrupted episode, simulator, or random-number-generator state.
The earlier 14/174 partial runs are not imported as completed full-val shards.

## Contract

All paths shown below should be absolute. Paths in the queue contract can be
relative to that contract, but upload/scientific-file paths in the existing
SPEC must already be absolute. Validator argv values are passed directly, with
no shell evaluation. Use an absolute script path and explicit dependency files.

```json
{
  "schema": "continuous_reproduction_queue_v1",
  "local_python": "/absolute/project/.venv/bin/python",
  "poll_seconds": 30,
  "units": [
    {
      "unit_id": "pair00_02",
      "spec": "/absolute/falcon_pair00_02_ready.json",
      "completion_ids": ["balanced4_00", "balanced4_02"],
      "deadline_after_seconds": 7200,
      "validator_timeout_seconds": 120,
      "local_validator_argv": [
        "{python}", "/absolute/validate_falcon_parallel.py",
        "--plan", "/absolute/falcon_pair00_02_plan.json",
        "--output", "{vm}"
      ],
      "validation_files": [
        "/absolute/validate_falcon_subset.py",
        "/absolute/check_falcon_shard_completion.py",
        "/absolute/full_val_manifest.json",
        "/absolute/shard00_manifest.json",
        "/absolute/shard02_manifest.json"
      ],
      "completion_checks": [
        {
          "completion_id": "balanced4_00",
          "argv": [
            "{python}", "/absolute/check_falcon_shard_completion.py",
            "--eval-output", "{vm}/shards/balanced4_00",
            "--expected-manifest", "/absolute/shard00_manifest.json",
            "--full-manifest", "/absolute/full_val_manifest.json",
            "--managed-plan", "{plan}",
            "--output", "{vm}/completion_check_00.json"
          ]
        }
      ],
      "retry_specs": [
        {
          "spec": "/absolute/falcon_retry_single_00.json",
          "completion_ids": ["balanced4_00"],
          "local_validator_argv": [
            "{python}", "/absolute/validate_falcon_subset.py",
            "--output", "{vm}/eval"
          ],
          "completion_checks": []
        },
        {
          "spec": "/absolute/falcon_retry_single_02.json",
          "completion_ids": ["balanced4_02"],
          "local_validator_argv": [
            "{python}", "/absolute/validate_falcon_subset.py",
            "--output", "{vm}/eval"
          ],
          "completion_checks": []
        }
      ]
    }
  ]
}
```

The example illustrates schema only and is not a runnable production contract.
For both shards to be retained after a timed-out pair, give both explicit
`completion_checks`. For a timed-out single-shard retry to retain a completed
child, give that retry's checker with `--eval-output {vm}/eval` as well.

Each unit's IDs must be disjoint from other units. Every primary and retry SPEC
must have its own unique existing `run_id`; IDs already present in another
project run plan are rejected. The controller **preserves** `run_id`, because
the scientific pair plan can bind it and its remote output directory.

`deadline_after_seconds` changes only `slurm.deadline`, immediately before the
first preparation of an attempt, using timezone-aware `Asia/Shanghai` time
(the same UTC+08:00 as HKT). The original SPEC is never overwritten. The exact
before/after value and both hashes are retained in `SPEC_DERIVATION.json`.
An already prepared attempt is never refreshed or refrozen on a later poll or
restart. The deadline window must exceed the allocation wall time. Each SPEC
must request 1–2 GPUs, at most one hour, and explicit
`gpu_binding: "slurm_linux_minor_uuid"`.

`{python}` is the configured local interpreter, `{plan}` is the frozen managed
plan directory, and `{vm}` is a **new disposable copy** of the collected VM
outputs for that single validator call. Validators may write reports in that
copy; the original collected files and their hashes remain unchanged. The
validator's full stdout/stderr and exit receipt are retained outside the copy.
Each call receives a fresh copy. Copies require the original SPEC's free-space
reserve plus their actual size. Scripts must print any report that should be
retained after the copy is removed. Imported helper modules and manifests
should be listed in `validation_files`; directly named existing argv files,
all SPEC uploads/scientific files, the interpreter/profile and lifecycle
scripts are also hash-bound at queue admission.

## Bounded retries and attention

No retry happens by default. Only the explicitly listed finite `retry_specs`
can be selected, and each candidate can run at most once. A controlled timeout
is either raw Slurm `TIMEOUT`, or the verified worker's exact
`TimeoutError / SCIENTIFIC_UNIT_TIMEOUT`. A recorded environment or validation
error takes priority over a scheduler timeout and stops at `ATTENTION`.

After terminal release, explicit per-shard checks can retain complete shards
from a timed-out pair. The controller then chooses the first unused candidate
whose IDs are all unfinished. Candidates wholly completed are skipped;
candidates overlapping a completed shard are also skipped. Two unfinished
single-shard candidates can therefore run sequentially. Exhausted candidates
with unfinished shards require attention. A failed validator, environment
error, corrupt evidence, missing release proof or unknown submission stops
new allocation and preserves the attempt directory and logs.

A successful single-shard retry is independently validated before another
candidate is selected. It starts that shard from its beginning with the
explicit SPEC settings; it does not claim lossless continuation of interrupted
episodes. Completed shards are never knowingly resubmitted. A failed parent
receipt remains failed even if all its children independently pass completion
checks. If all requested shards validate after a controlled timeout, the next
unit can proceed without inventing a successful parent run.

## Checking, starting and restarting

Read-only contract check, with no state creation, SSH or allocation:

```sh
/absolute/project/.venv/bin/python /absolute/project/baselines/scripts/continuous_reproduction_queue.py \
  --queue /absolute/queue.json --check
```

Start only when the finite queue is authorized and all input files are final:

```sh
/absolute/project/.venv/bin/python /absolute/project/baselines/scripts/continuous_reproduction_queue.py \
  --queue /absolute/queue.json --directory /absolute/project/baselines/runs/finite_queue_20261103
```

Restart with the **same** queue and directory. A nonblocking OS file lock rejects
a second controller, including one started in another process. JSON checkpoint
updates use write/fsync/atomic replace and directory fsync. A completed entry's
evidence is rehashed before it is skipped. Queue input changes after admission
are rejected; do not edit a running contract or its bound files.

The managed process starts in its own session. Stopping only the controller
leaves the existing managed owner to monitor and release its allocation. A
restart waits on that exact active manager/owner using process identity and
receipts. It never restarts `managed_run.py` for an existing launch intent.
An intent is durably written **before** process launch; if a crash leaves no
verifiable owner/manager receipt, the controller stops with
`SUBMISSION_OR_OWNER_STATE_UNKNOWN_NO_RESUBMIT`. Even a crash before the actual
launch may therefore need manual reconciliation: avoiding duplicate submission
takes precedence over guessing. A dead owner with an unresolved job likewise
requires attention; the controller does not impersonate the owner or cancel
jobs. `ATTENTION` is persistent and is not cleared automatically on restart.

Exit 0 means the finite requested units passed the queue's completion checks;
exit 2 means attention is required. Neither exit code is itself a claim of
matching a paper, SOTA, full1087 coverage, or a restored stochastic trajectory.
The separate full-val exact-union aggregator still establishes dataset coverage
and recomputes final episode-weighted metrics.

## Offline verification

```sh
/absolute/project/.venv/bin/python /absolute/project/baselines/scripts/test_continuous_reproduction_queue.py
```

Tests exercise competing process locks, completed-checkpoint restart, unknown
launch refusal, active-owner recovery, release evidence, log tampering,
environment-error stops, finite timeout candidate selection, independent copy
validation and the actual CPU-only `prepare_run.py` freeze. They never invoke
`managed_run.py`, SSH, Slurm, a GPU, model inference or scientific evaluation.
