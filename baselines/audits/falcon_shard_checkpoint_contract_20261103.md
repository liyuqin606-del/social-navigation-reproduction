# Falcon completed-shard checkpoint contract

The hourly continuation controller and final full-val aggregation share the same read-only CPU completion check. A timed-out two-GPU pair can retain one already completed shard without rerunning it. This is a single-shard checkpoint; the interrupted parent is never reported as a successful pair.

```sh
.venv/bin/python baselines/scripts/check_falcon_shard_completion.py \
  --eval-output /ABS/plan/run/vm/shards/balanced4_00 \
  --expected-manifest baselines/prep/falcon_full_val_balanced4_shard_00_20261103_manifest.json \
  --full-manifest baselines/prep/falcon_full_val_20261103_manifest.json \
  --managed-plan /ABS/plan \
  --output /ABS/NEW_completion_00.json
```

For a one-GPU retry, use its own managed plan and `run/vm/eval`. The controller may provide a disposable copy of `vm` while passing the real `--managed-plan`; the checker rechecks copied bytes against the original collected SHA manifest. Its output may be outside the shard directory in that temporary copy. It prints the complete JSON report as well as writing it, so the controller can retain its log. Exit 0 and `status=PASS_COMPLETE_SHARD` / `complete=true` indicate a completed shard. Any missing or inconsistent evidence returns exit 2. An existing report is never overwritten.

Every acceptance requires:

- Exact expected shard-manifest SHA, membership in the original full1087 manifest, original runtime-to-source episode mapping and a complete original subset-validator pass. The validator runs through temporary input links, leaving collected `validation.json` untouched.
- Seed 100, stochastic actions, 500-step limit, eight environments, the fixed checkpoint/initialization/evaluator hashes and the remaining scientific protocol checks shared with the final aggregate.
- Frozen `CONTROL.json`, worker/config and pair-plan identity; agreement among the driver, worker, raw main `sacct` row and released empty queue; every collected file must match the complete byte/SHA receipt.

Normally the owning scientific process and validator must both exit 0 under a completed parent. The only failure exception is an explicit worker `TimeoutError: SCIENTIFIC_UNIT_TIMEOUT`, or an independently recorded Slurm `TIMEOUT` accompanied by the worker's TERM/KILL scientific-process exit. The latter does not excuse an ordinary exit 1 or SIGABRT. Missing worker/collection receipts after a hard timeout are insufficient.

For that exception, the executed pair must match the frozen plan. The retained child must have an observed normal exit 0, the exact assigned CUDA UUID, correct process group, remote output/log paths and executed arguments, source manifest/counts and model inputs. Its own `run_status` and original validator must already be successful. Any explicit non-timeout wrapper failure, environment failure, child signal, unknown child exit, incomplete episode log or changed digest rejects the checkpoint.

Accepted interrupted-parent evidence records `interrupted_parent=true`, `parent_batch_success=false`, the precise timeout basis, and the original collection location. Final full-val aggregation still requires the disjoint union of all 1087 episodes/56 scenes; a single accepted checkpoint cannot produce a full-val score.

Verification so far is offline only: isolated process/collection fixtures exercise the timeout branch without creating metric values, and the original scientific validator rejects their intentionally empty scientific evidence. Counterexamples cover environment failure, SIGABRT or unknown child exit, wrong GPU/path/input, non-timeout wrapper failure, missing successful validation, wrong raw scheduler identity and collected-file tampering. No real timeout-retained shard result has yet been claimed. No frozen scientific runner, evaluator, lifecycle worker or existing result was modified for this change.
