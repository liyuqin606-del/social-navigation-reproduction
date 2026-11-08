# Habitat EGL 节点诊断与证据判读 — 2026-11-03

准备完成，未冻结、未提交、未申请 GPU。两个 renderer 探针只渲染空场景 NONE 的 32×32 depth 帧，不加载 Falcon/NavThinker 模型、检查点、导航 episode 或真实场景，不安装或修复节点。

## 已见失败与可推断范围

job28090 的 worker receipt 证明在 node 10-0-3-10 上 CUDA 的实际 UUID 与 Slurm 分配一致；驱动版本为 560.35.05。随后八个环境子进程在创建 windowless EGL context 时报告 EGL_BAD_PARAMETER，之后有原生 allocator 错误，主进程收到 ConnectionResetError。尚未进入有效导航评测。

此前 530.30.02 节点的成功 smoke 保存了 /usr/share/glvnd/egl_vendor.d/10_nvidia.json，ICD 路径为 libEGL_nvidia.so.0；Habitat native binding 的 ldd 显示 EGL/OpenGL dispatcher 来自当前 Conda 环境。它证明该节点的一次工作路径，不证明所有 530 节点都正常或所有 560 节点都失败。

已核对 [Habitat-Sim v0.3.1 的 Magnum submodule](https://github.com/facebookresearch/habitat-sim/tree/v0.3.1/src/deps/magnum)，固定为 4a456f2cefb141f31f8b7a2887c6b7190cab5116。[对应 WindowlessEglApplication.cpp](https://github.com/mosra/magnum/blob/4a456f2cefb141f31f8b7a2887c6b7190cab5116/src/Magnum/Platform/WindowlessEglApplication.cpp#L135) 在 EGL 客户端扩展未满足设备枚举及 platform-device 条件时，回退到默认 display 路径；line305 恰好输出本次错误。这使 **ICD 发现/加载、客户端扩展** 成为优先诊断项。尚未据此确定缺库、库版本不匹配或导入顺序中的哪一个是根因。

[NVIDIA 官方容器文档](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html#driver-capabilities)将 CUDA/compute、NVML/utility 与 EGL/graphics 能力分开。因此 CUDA 和 nvidia-smi 正常不能证明容器已提供 EGL 图形库；在进程中事后修改 capability 环境变量也不会补挂载缺失库。

[Habitat 官方 issue2099 的维护者反馈](https://github.com/facebookresearch/habitat-sim/issues/2099)说明只有 Mesa 软件 EGL 设备时需检查 NVIDIA driver/GLVND 发现。此处只借鉴定位方法，不能把该 issue 的具体根因套到 node 10-0-3-10。

## 可运行产物

- [diagnose_habitat_egl.py](../scripts/diagnose_habitat_egl.py)：现有 managed Slurm allocation 内的短诊断。
- [validate_habitat_egl.py](../scripts/validate_habitat_egl.py)：独立重建状态并验证输出哈希、进程退出与日志完整性。
- [make_habitat_egl_diagnostic_spec.py](../scripts/make_habitat_egl_diagnostic_spec.py)：只生成 spec，不提交。
- [本次 spec](../prep/egl_node_diagnostic_spec_20261103.json)：1 GPU、2 CPU、10 min walltime、science≤480s、16 MiB output；gpu_binding=slurm_linux_minor_uuid。无 nodelist 假设，记录实际节点。

要求已有 SLURM_JOB_ID 和 managed worker 已核验的单一完整 GPU UUID mask；不更改 CUDA_VISIBLE_DEVICES、LD_LIBRARY_PATH 或 EGL vendor 覆写。spec 只提高诊断日志详细度。

诊断包含：

1. NVIDIA 版本/设备枚举、内核驱动文本、受限环境变量、vendor JSON 内容及哈希、常见目录的 EGL/GLdispatch/NVIDIA EGL 文件与符号链接目标、Habitat binding 的 ldd。
2. 独立 loader 子进程：ctypes 尝试加载 EGL dispatcher、OpenGL、GLdispatch、NVIDIA EGL；记录错误、实际映射库、依赖，以及 EGL_NO_DISPLAY 客户端扩展/设备扩展。只枚举设备，不为其他设备创建 context。
3. 独立新 Python 进程：先 import habitat_sim，再创建 NONE renderer，gpu_device_id=0 对应已验证 UUID mask。
4. 另一个独立新 Python 进程：先 import magnum，再 import habitat_sim，运行相同 renderer。此项只隔离两模块的导入顺序，不模拟整个 Habitat-Lab import 链或八进程并发。

loader 的按 soname 动态加载可能不同于 Habitat binding 自己的 ELF RPATH；应结合 inventory 的 ldd 和 renderer 自己的 loaded_library_paths 判断，不能仅凭某次裸 CDLL 的失败断言 renderer 必然找不到 dispatcher。

## 判读规则

| 实际证据 | 可支持的结论 |
|---|---|
| NVIDIA vendor JSON 不存在，或 library_path 指向的 ICD 不能加载 | 支持 ICD 发现/挂载/依赖缺失方向；结合实际加载路径核对 |
| libEGL_nvidia 加载失败，明确缺失依赖或版本符号；库文件版本与内核驱动不一致 | 支持图形驱动用户态库不完整/不匹配方向，仍需具体错误 |
| EGL 客户端缺设备/platform扩展，sim-first 与 magnum-first 都失败 | 支持节点图形运行时问题；单凭两者失败不能区分安装与容器挂载 |
| sim-first 成功、magnum-first 失败 | 支持这套环境对导入顺序敏感；仍不是八进程实际评测的修复证明 |
| 两个独立单进程都成功 | 尚未解释原八环境失败；后续才需检查并发、子进程启动方式及资源差异 |

驱动版本差异本身不是已确认原因。选择已有成功证据的节点可以是运行避障策略，不能写成修复了 560 节点。

## 完成与失败合同

renderer 的 PASS 需要独立进程正常退出、child_exit_code=0、明确关闭 renderer，以及有效的 32×32 finite frame。

若 renderer 捕获 Python 异常，记录 FAIL；若发生 SIGABRT/SIGSEGV，仅在监督进程确实观察到 -6/-11、已有完整启动记录、日志哈希吻合且日志含明确原生错误时，归类 FAIL_NATIVE_SIGNAL。**这可以是完整的失败诊断证据，绝不是 renderer 成功。**

顶层进程返回 0 表示诊断证据通过独立完整性校验；可能同时有 renderer_status=FAIL 和 DIAGNOSTIC_COMPLETE_WITH_RENDERER_FAILURES。timeout、缺报告/日志、意外退出、信号与 PASS 自报矛盾等均返回非零，不能报诊断完成。所有结果固定 benchmark_evaluated=false、paper_reproduction=false。

本地只运行了 AST、CLI help 和合成证据合同检查：原生失败被标为 FAIL，信号+自报 PASS、timeout、缺日志均被拒绝。未调用 GPU、EGL 或 Habitat renderer。[检查记录](egl_diagnostic_contract_checks_20261103.json)中的 fixtures 不是实测结果。

