# Falcon GPU binding review — 2026-11-03

Status: **static review passed; final runtime result pending**. The reviewer edited only the future eval-spec generator, inspected current lifecycle source and failed receipts, and did not change frozen plans, connect through SSH, allocate CUDA, or execute a model.

`make_falcon_eval_spec.py` now emits top-level `gpu_binding: slurm_linux_minor_uuid`. A real subset174 manifest was used to generate and parse a temporary spec: 174 episodes, 4 environments, 1 GPU, one-hour wall limit, 15 required outputs and 6 exact remote archive inputs. The temporary spec was removed without freezing or submission. Current generator, worker and prepare-run sources all parse successfully; their reviewed hashes are in the adjacent JSON audit.

The current worker obtains UUID/minor mappings from `nvidia-smi -q -x`, requires allocation and original mask agreement, verifies the character-device minor, applies the UUID mask before importing Torch, checks the visible RTX 3090 count, and compares CUDA driver UUIDs in the resulting device order. CUDA accepts UUID masks and assigns process-local ordinals in the selected order. [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)

Two concrete issues were resolved by the parent task before further freezing:

- Driver 530 rejected `minor_number` as a CSV query field in job 28085. XML supplies the required field without that unsupported query.
- Science environment overrides could replace the checked mask after worker validation. `prepare_run.py` now rejects `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`, `SLURM_JOB_GPUS`, and `SLURM_STEP_GPUS` in that environment.

The mode remains specific to this TACC deployment: interpreting Slurm numeric IDs as Linux minors is an explicit site assumption. Slurm can use cgroup-relative masks, and NVML indices need not equal Linux minors; ambiguous inputs therefore continue to fail closed. [Slurm GRES documentation](https://slurm.schedmd.com/gres.html)

Reviewed receipts show jobs 28084 and 28085 failed before scientific execution. The parent reports job 28086 passed binding and entered Falcon smoke; this review did not inspect its final result. A binding check or smoke pass does not establish episode evaluation or paper reproduction. Regenerate future specs with the updated generator; historical draft specs do not acquire the new binding automatically.
