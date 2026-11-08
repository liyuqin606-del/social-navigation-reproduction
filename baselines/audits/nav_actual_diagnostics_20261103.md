# NavThinker actual diagnostics — 2026-11-03

Job **28089** completed with verified local outputs and released GPU resources. Both Nav subprocesses exited normally with code 1, and their independent classifiers returned 0: the known source failures were captured successfully, while the models did not pass all stages. All 12 collected output files were rechecked against their byte/SHA manifest.

| Variant | Observed result |
|---|---|
| Unmodified official source | Import failed: `No module named 'habitat_baselines.il.data'`. No world-model forward reached. |
| Documented unrelated-IL import overlay | Official config composed. Randomly initialized world model completed one synthetic component forward; policy construction failed: `'RSSM' object has no attribute 'feat_size'`. |

The component has 39,285,874 parameters; its encoder and latent feature shapes were `[1,896]` and `[1,1160]`, and its human trajectory head produced `[1,6,4,2]`. Peak allocated CUDA memory was 293,009,920 bytes. These are engineering observations on synthetic inputs with **random weights, no official pretrained checkpoint and zero benchmark episodes**. They do not provide SR/SPL/PSC results or establish paper reproduction.

This was an independent `habitat_sim`-first compatibility probe in the prepared Habitat-Sim 0.3.1 / Torch 2.1.2+cu118 environment. Both logs contain zero duplicate-static-plugin warnings and no allocator error. Job 28086's two Nav SIGABRT exits remain abnormal and are rejected by the corrected contract; its old validator success cannot make them valid completions. The old evidence was not rewritten.

Import order is not causally isolated: job 28089 also ran on a different node and NVIDIA driver (560.35.05 versus 530.30.02 in 28086). The supported conclusion is that this explicit compatibility variant completed normally on the observed node. The missing IL package, default RSSM/policy interface mismatch, and absent evaluated checkpoint remain unresolved reproduction blockers.

Evidence: `baselines/runs/nav_diagnostics_20261103_a/run/DRIVER_RECEIPT.json`, its `vm/WORKER_RECEIPT.json`, and `vm/diag/` reports/logs. The adjacent JSON audit records the exact collected-file hashes, stage observations and evidence limits. No further GPU diagnostic was started by this summary task.
