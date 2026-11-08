# TACC 单卡 CUDA 可见性：独立核查

截至 job 28084 的已回收证据，失败发生在模型启动之前。证据支持 **Slurm 给出的编号与当前进程 NVML/CUDA 枚举编号不一致**，需要把已分配设备精确映射成 UUID；不能简单改成 `CUDA_VISIBLE_DEVICES=0`，也不能清空限制后使用所有可见设备。

## 当前实证

读取了 `baselines/runs/diagnostics_20261103_{a,b}/run/vm/WORKER_RECEIPT.json` 及 scheduler receipt：

| 项目 | job 28083 | job 28084 |
|---|---|---|
| Slurm `CUDA_VISIBLE_DEVICES` | `2` | `2` |
| `SLURM_JOB_GPUS` | 初次未保存 | `2` |
| NVML / nvidia-smi | 初次未保存 | 只有 index 0 和 1，两张 RTX 3090 |
| 进程可见设备文件 | 初次未保存 | `/dev/nvidia2`、`/dev/nvidia3` |
| Torch | 初次未保存 | `2.1.2+cu118`, CUDA build `11.8` |
| NVIDIA driver | 初次未保存 | `530.30.02` |
| Torch CUDA count | 0 | 0 |
| `torch.cuda.init()` | 初次未保存 | `No CUDA GPUs are available` |
| 模型运行 | 未启动 | 未启动 |

28084 的 UUID 为 `GPU-a769d98f-3e75-911d-19c6-04e97d1eede1` 和 `GPU-2a1ea953-f6f5-8f57-52ad-a0cbb1bada00`；该次日志尚未包含两者各自的 Linux minor，不能按两行顺序把其中一个擅自认定为分配设备。

## 为什么可以静默返回零

PyTorch 2.1.2 的 `device_count()` 优先使用 NVML，只有 NVML 返回负值时才退回 CUDA runtime；当 mask 首个整数 2 不小于 NVML 总数 2 时，`_device_count_nvml()` 直接返回 0，不会触发初始化警告。因此初次没有 warning 不代表 driver 已经正常初始化。[PyTorch 2.1.2 源码，683–740 行](https://github.com/pytorch/pytorch/blob/v2.1.2/torch/cuda/__init__.py#L683-L740)。

当前 28084 的显式 CUDA 初始化也失败，说明不只是 NVML admission 的假阴性。`CUDA_VISIBLE_DEVICES=2` 对“只枚举到 0、1”的空间是无效选择，与该失败相符；最终根因仍应由保存 minor→UUID 映射后的受管理运行确认。[CUDA 可见设备语义](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)。

## 历史成功不能覆盖当前差异

按 `tacc-3090/references/run.md` 找到并读取 2026-10-28 的 job 27819：当时 `/usr/bin/python3` 通过 ctypes 直接调用 `libcuda.so.1` Driver API，在 `CUDA_VISIBLE_DEVICES=0,1` 下对两张 3090 各验证 4096 个整数输出。它没有使用 PyTorch，没有测试非零起始编号 `2`，也不是当前 Conda 环境的通过证明。原始证据位于：

```text
/Users/yuqinli/Documents/ChatGPT/CVPR_New/research/tacc_access_20261023/
  ready_to_submit/gpu_check.py
  live_20261028/run01/job.sh
  live_20261028/run01/vm/result.json
```

## 对主任务 UUID 方案的审核

完整 GPU UUID 是 CUDA 官方允许的 `CUDA_VISIBLE_DEVICES` 值，可避免整数索引在容器/设备枚举变化后的歧义。[CUDA 官方说明](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html)。NVIDIA 将 minor number 定义为 `/dev/nvidia[minor number]` 对应的数字，适合把设备文件和 UUID 关联。[nvidia-smi 官方文档](https://docs.nvidia.com/deploy/nvidia-smi/index.html)。

建议并审阅的限制是：

1. 在 Torch/CUDA 首次初始化之前，保存原 Slurm mask 和 `SLURM_JOB_GPUS`；只有显式启用该站点的 `slurm_linux_minor_uuid` 模式才转换。
2. 查询 `index,uuid,name,driver_version,minor_number`；若驱动不支持该 CSV 字段，使用 `nvidia-smi -q -x` 中各 gpu 的 `uuid` / `minor_number`。查不到或多义就停止，不按输出行序猜测。
3. 该站点必须有“Slurm 分配 ID→设备文件”的依据，再按唯一 minor 选 UUID；核实 `/dev/nvidiaN` 是字符设备且实际 `st_rdev` minor 一致。不要普遍假设 Slurm GPU ID、NVML index、Linux minor 永远相同。[Slurm GRES 编号说明](https://slurm.schedmd.com/gres.html)。
4. 新 mask 只含本次分配的一张 GPU UUID；不扩张到同容器可见的另一张卡。
5. Torch 初始化后再次检查数量、型号，并用 CUDA Driver API `cuDeviceGetUuid` 验证实际 UUID 与所选相同。当前 worker 的该调用在 `get_device_name` 后执行，CUDA 已经由 Torch 初始化，调用顺序合理。
6. 保存原 mask、查询原文、minor 匹配、转换后 UUID 和 CUDA 实际 UUID，作为执行层修正的证据。失败时仍保留原 receipt。

## 启动环境补充

初始 worker 在 `import torch` / admission 检查之后，才把 `cfg['env']` 加到 science 子进程。因此子进程的 `PYTHONNOUSERSITE` / `LD_LIBRARY_PATH` 不影响 worker 自身。若需要禁用 user site，应在 job shell 启动 Python 前设置 `PYTHONNOUSERSITE=1` 或使用 `python -s`；运行中才设置不会移除 Python 已加入的 user-site 搜索路径。

28084 已记录真正加载的 Torch 为 cu118，排除了最初“可能错误导入 CPU-only wheel”的猜测。仍建议 worker 自身保存 `torch.__file__`、运行环境白名单和库解析，以免以后被同名包或环境路径干扰。

此独立核查没有提交 GPU 作业、修改远程环境、改 Slurm mask或调用任何设备初始化。实际修复与运行由主任务的 managed lifecycle 负责。
