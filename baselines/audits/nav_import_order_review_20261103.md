# NavThinker import-order review — 2026-11-03

Read-only source review; no imports executed locally, no source or diagnostic script edited, no GPU job submitted.

The frozen NavThinker source can import Magnum before Habitat-Sim during `import habitat_baselines`, before it reaches the known missing IL data modules. A source-level path is:

1. `habitat_baselines/__init__.py` imports its common package.
2. `habitat_baselines/common/__init__.py:6` imports env_factory; `common/env_factory.py:4` imports Habitat.
3. `habitat/__init__.py:9` imports Benchmark; `core/benchmark.py:21` imports Env.
4. `core/env.py:30` imports datasets. Dataset registration eventually imports the pointnav dataset, then navigation task code.
5. `tasks/nav/nav.py:37` imports the habitat simulator actions package, initializing `habitat/sims` and its simulator package.
6. `sims/habitat_simulator/__init__.py:7` imports object_state_machine.
7. `sims/habitat_simulator/object_state_machine.py:12` imports Magnum; line 15 imports Habitat-Sim.

The Habitat simulator wrapper itself also has Magnum at line 20 and Habitat-Sim at line 26. The current Falcon smoke explicitly imports Habitat-Sim first.

[Official Habitat-Sim v0.3.1 package initializer](https://github.com/facebookresearch/habitat-sim/blob/v0.3.1/src_python/habitat_sim/__init__.py#L13) first imports the native habitat_sim_bindings extension; its Git blob is `142260c84d5ecfad2b6bcbb314981fc0c295dc3d`. This makes an explicitly recorded Habitat-Sim-first diagnostic a reasonable compatibility experiment. It does **not** prove that the observed SIGABRT was caused by import order.

[Official PR 1861](https://github.com/facebookresearch/habitat-sim/pull/1861) explains the inclusion of static plugins in Magnum's Python module. It does not establish an import-order rule or explain this particular allocator failure.

[Habitat-Lab issue 2082](https://github.com/facebookresearch/habitat-lab/issues/2082) reports a similar malloc_consolidate error. A non-maintainer comment says reinstalling Habitat-Sim 0.3.3 resolved their instance; no maintainer-confirmed import-order remedy was found. This is not grounds to change the experiment's pinned simulator version.

[Issue 2062](https://github.com/facebookresearch/habitat-lab/issues/2062) and [issue 2152](https://github.com/facebookresearch/habitat-lab/issues/2152) include duplicate-static-plugin warnings followed by further execution and other errors. Such warnings alone do not establish the cause of an allocator abort.

Suggested evidence boundary: preserve the original signal-aborted runs as runtime failures; label any explicit pre-import variant as runtime compatibility handling. A passed model component diagnostic remains synthetic and untrained, with paper_reproduction=false.

