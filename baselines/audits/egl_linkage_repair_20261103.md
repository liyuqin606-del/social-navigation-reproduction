# Habitat-Sim EGL 动态链接缺项：最小修复输入

现有 `baselines/prep/cpu_import_failure.log` 显示，`import habitat_sim` 在载入 0.3.1 的 Python 扩展时因 `libEGL.so.1` 不存在而失败，尚未运行模型或创建 EGL context。主任务进一步报告远程 `ldd` 中 `libEGL.so.1`、`libOpenGL.so.0` 均为 `not found`，标准系统目录和当前环境均没有这些库；主任务报告宿主 glibc 为 2.27。

这个问题分两层：环境可以提供 GLVND 的 API loader / dispatch 库，让扩展完成动态链接；真正的 GPU 渲染仍需要计算节点的 NVIDIA vendor driver / ICD。安装环境库不等于安装显卡驱动，也不证明渲染已经成功。[GLVND 架构说明](https://github.com/NVIDIA/libglvnd/blob/master/README.md)、[EGL ICD 加载规则](https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)。

当前 conda-forge 已拆分包：`libglvnd` 只携带 `libGLdispatch.so.*`，`libegl` 携带 `libEGL.so.*`，`libopengl` 携带 `libOpenGL.so.*`。所以只安装名字叫 `libglvnd` 的包不能补齐这两个缺少的 soname。[官方打包 recipe](https://github.com/conda-forge/libglvnd-feedstock/blob/main/recipe/meta.yaml)。

## 已取得并校验的安装输入

清单：`baselines/prep/egl_runtime_packages_20261103.json`。本地包：`baselines/prep/egl_packages/`。三个包均为官方 conda-forge `linux-64`、版本 **1.7.0**、build **ha4b6fd6_5**，总计 **231,148 bytes**。已实际下载并核对官方 API 的长度和 SHA-256。

| 包 | SHA-256 | 依赖 |
|---|---|---|
| [libglvnd](https://conda.anaconda.org/conda-forge/linux-64/libglvnd-1.7.0-ha4b6fd6_5.conda) | `6eb77601a44b78c4631d886d54ef54f321c2bdc69fdefc4065a1e0491f030c76` | `__glibc >=2.17,<3.0.a0` |
| [libegl](https://conda.anaconda.org/conda-forge/linux-64/libegl-1.7.0-ha4b6fd6_5.conda) | `13d3fde9c1dcf254968cc70652222a43f25d04e69555ccf855553110e753288e` | 上述 glibc，以及相同版本/build 的 libglvnd |
| [libopengl](https://conda.anaconda.org/conda-forge/linux-64/libopengl-1.7.0-ha4b6fd6_5.conda) | `7bdca717c840e17d7dd050f925dd964da61086c2612b8579a4ff4147105f291f` | 上述 glibc，以及相同版本/build 的 libglvnd |

三包的声明依赖与 glibc 2.27 相容，完整依赖闭包不引入其它 Python 包。元信息来自 [libglvnd API](https://api.anaconda.org/package/conda-forge/libglvnd)、[libegl API](https://api.anaconda.org/package/conda-forge/libegl)、[libopengl API](https://api.anaconda.org/package/conda-forge/libopengl)。这是一组精确安装输入，不需要重新求解整个训练环境。

`libegl` 元信息标记 `has_prefix=true`，应使用 Conda 正常链接这些显式包并执行 prefix relocation；不要把 payload 简单解包覆盖到环境。此核查 agent 没有修改远程环境、安装驱动或提交 GPU 作业。

## 验证顺序

1. 在原环境完成精确包链接后，重跑对 Habitat 扩展的 `ldd`。应确认上述缺项消失，并保存解析到的实际绝对路径。若库存在但 loader 尚找不到，只在该任务进程中把已核实的 `<env>/lib` 加入 `LD_LIBRARY_PATH`，保留已有有效路径；不改系统 profile。
2. 再执行 CPU `import habitat_sim`。通过只说明 ELF 依赖与 Python 导入成功，不说明 GPU 可用。
3. 在同一个受管理的 GPU allocation 内检查 NVIDIA `libEGL_nvidia.so.0` 及对应 vendor JSON，然后执行既有 Falcon 空场景 renderer smoke。系统 loader 可以来自环境，vendor driver 应与计算节点驱动一致。

Conda-forge 的 libegl 包已有搜索系统 ICD 目录的 patch，不必提前设置 `__EGL_VENDOR_LIBRARY_FILENAMES` / `__EGL_VENDOR_LIBRARY_DIRS`；只有实际发现 vendor 目录不可见时，才依据存在的 NVIDIA JSON 指定路径。前者一旦设置会排除其它未列明的 ICD，不能随意猜路径。[官方 Conda patch](https://github.com/conda-forge/libglvnd-feedstock/blob/main/recipe/libglvnd_look_for_egl_icd_in_system_location.patch)、[NVIDIA ICD 规则](https://github.com/NVIDIA/libglvnd/blob/master/src/EGL/icd_enumeration.md)。

## combined diagnostics 应记录的内容

应记录 `LD_LIBRARY_PATH`。审阅时 `make_diagnostics_spec.py` 的 `science.env` 未冻结它，`run_reproduction_diagnostics.py` 的子进程继承父进程环境，因此仅保存脚本/config hash不足以复现这次链接行为。

建议把实际采用的路径设置写入 spec，并在**计算节点**保存以下白名单字段：`LD_LIBRARY_PATH`、`CONDA_PREFIX`、`CUDA_VISIBLE_DEVICES`、`DISPLAY`、`__EGL_VENDOR_LIBRARY_FILENAMES`、`__EGL_VENDOR_LIBRARY_DIRS`；同时保存 Habitat 扩展 `ldd` 的库解析、NVIDIA driver 版本、vendor JSON 路径与内容/hash。不要导出完整环境变量表。登录节点的检查应单独保存，不能代替计算节点证据。

本文件给出修复依据和已校验输入；远程安装成功、CPU import 成功、GPU renderer 成功三种状态应分别由主任务真实日志确认。
