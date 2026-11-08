# Falcon full-val shard aggregation readiness

Tools are prepared; **no full-val scores have been generated**. The current complete-results path has not been exercised because no complete successful 1087-episode collection was supplied.

Entry points:

- `baselines/scripts/aggregate_falcon_full_val.py`: validate inputs, normalize exact episode identities, aggregate per-episode values, and invoke independent validation.
- `baselines/scripts/validate_falcon_full_val_aggregate.py`: re-read the original shards, recheck coverage/protocol, and recompute the result independently. Rational summation of the exact binary metric values cross-checks the writer's floating `fsum` means.
- `baselines/scripts/falcon_full_val_integrity.py`: shared read-only input/protocol checks. It invokes the existing `validate_falcon_subset.py` inside a temporary directory of input links, writing its new validation report only there; the collected shard files remain unchanged.

Supply the official full1087 manifest and each verified completed shard evaluation directory exactly once. Both single-job `vm/eval` and paired-job `vm/shards/<shard_id>` layouts are supported. Normally, an ancestor `WORKER_RECEIPT.json` must agree with the shard job ID and have zero scientific/validation exits. A strict shared checkpoint check now also accepts a normally completed child of a timed-out pair, subject to the frozen-plan, actual exit/GPU/path, complete collection hashes and original-validator checks described in `falcon_shard_checkpoint_contract_20261103.md`. Its parent remains failed, and `interrupted_parent=true` is retained in aggregate provenance. A model's own success JSON cannot excuse an abnormal child exit.

```sh
.venv/bin/python baselines/scripts/aggregate_falcon_full_val.py \
  --full-manifest baselines/prep/falcon_full_val_20261103_manifest.json \
  --eval-output /ABS/pair_00_02/run/vm/shards/balanced4_00 \
  --eval-output /ABS/pair_01_03/run/vm/shards/balanced4_01 \
  --eval-output /ABS/pair_00_02/run/vm/shards/balanced4_02 \
  --eval-output /ABS/pair_01_03/run/vm/shards/balanced4_03 \
  --output /ABS/NEW_full_val_aggregate
```

The standalone validator takes the same `--full-manifest`, repeated `--eval-output`, and `--output` arguments. Replace the example paths with actual verified collections; do not run against pending jobs.

This version binds the absolute local evidence paths as well as hashes. Generate and revalidate at the same collection location; relocating the input tree requires a separately reviewed rebind. The aggregate output must be outside the immutable shard directories.

Acceptance requires exactly 1087 unique `(original scene_id, string episode_id)` keys across 56 scenes, disjoint shard IDs and episodes, original per-scene data digests, the fixed released checkpoint and initialization hashes, seed 100, stochastic actions, 500-step limit and 8 environments. The full resolved scientific configuration must be identical between shards; only physical model/data/output paths and declared scene selection are removed from that comparison. Those paths remain bound to the hashed runtime provenance. The actual observed evaluator source must match the frozen instrumentation hash. Different runner/validator hashes or runtime compatibility declarations are also rejected pending review.

Runtime scene IDs are mapped through the recorded resolved `scenes_dir` and the exact original scene ID. No suffix guessing, scene-name-only matching or filesystem symlink resolution is used to normalize an episode. The earlier 174-episode pilot lacks the required full-val shard parent/ID and is explicitly rejected.

All metric columns must be finite, numeric and present for every episode. Means use one equal weight per episode, never one weight per shard. Official outputs include SR, SPL, PSC, H-Coll and `Total = 0.4*SR + 0.3*SPL + 0.3*PSC`, with unrounded and four-decimal files. The package also includes normalized episode JSONL, coverage, input/protocol hashes and the independent validation report. Missing or inconsistent inputs produce a failure report without fabricated scores.

Validation performed: all three scripts parse; the actual 1087-key manifest passes exact coverage and runtime-mapping checks; missing, duplicate and unexpected keys are rejected; a real 8-environment resolved config is accepted as configuration evidence while a 4-environment mutation is rejected; NaN, infinity, booleans and numeric strings are rejected as metric values; the original 174 manifest and a real failed full-shard run are rejected. Missing run directories produce no score files. These are static and failure-guard checks, not successful full-val model evaluation.

Required interpretation remains attached to every aggregate: each shard restarts seed 100, changing scene assignment and stochastic streams relative to a continuous full-val run; the result covers one released checkpoint and does not reproduce three independently trained seeds or the entire paper.
