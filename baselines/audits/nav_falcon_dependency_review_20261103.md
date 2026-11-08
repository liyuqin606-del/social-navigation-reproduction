# NavThinker / Falcon 导入依赖与 smoke 审阅

仅只读源代码、清单和 checkpoint pickle opcode；没有安装依赖或执行 GPU。Falcon smoke 仅补了输出目录创建，不修改官方源码。

## 依赖差异

两仓库下列 requirements 文本逐字相同：根目录、`habitat-lab/requirements.txt`、Habitat-Baselines 的 `il`、`rl`、`rl/ddppo`、`rl/ver` requirements。NavThinker 新增的 WM、DINO predictor 默认组件使用 Torch/NumPy，`wm_visualizer.py` 额外顶层导入 cv2；cv2 本来已是 Falcon 的导入依赖。**已有完整 Falcon 基础依赖可复用，Nav smoke 没有另一个大型模型包安装需求。** 默认 ResNet/GRU 与独立 DINO predictor 不需要下载 DINOv2/DepthAnything 权重，不需要安装 CLIP、timm 或 transformers。

在已具备 Py3.9 / Habitat-Sim 0.3.1 / Torch / Gym / Hydra 的基础上，仍应核实这些共同依赖，而不是盲目安装整个新环境：

| Python import | 发行包 / 原 requirements | 使用原因 |
|---|---|---|
| numpy | `numpy==1.23.5` | 两仓库根 requirements 固定 |
| torchvision | `torchvision`，版本须与已经选定的 Torch wheel 配套 | ResNet policy 顶层导入 |
| quaternion | `numpy-quaternion` | Habitat 几何/导航模块顶层导入 |
| attr | `attrs` | Habitat 数据/传感器结构 |
| cv2 | `opencv-python` 或环境现有兼容 cv2 提供者 | 地图/可视化/导航模块 |
| PIL / imageio | `Pillow` / `imageio` | 共同可视化依赖 |
| scipy / numba | `scipy>=1.10.1` / `numba>=0.44.0`，最终解依赖不能升级已固定 NumPy | 几何和 Habitat 核心模块 |
| yaml / omegaconf | `PyYAML` / `omegaconf>=2.2.3` | PDDL / 配置 |
| tqdm / ifcfg / threadpoolctl | 同名发行包 | trainer / DDP / VER 顶层导入 |
| torch.utils.tensorboard | `tensorboard==2.8.0`, `protobuf==3.20.1` | TensorboardWriter 无条件导入 SummaryWriter |
| faster_fifo | `faster_fifo>=1.4.2` | 包顶层载入 VER trainer；官方虽有 fallback，保留声明依赖更稳妥 |
| magnum / corrade / habitat_sim | Habitat-Sim Conda 栈提供 | 不应另装同名无关 PyPI 包 |
| lmdb / webdataset | `lmdb>=0.98`, `webdataset==0.1.40` | **Falcon** 顶层导入 IL trainer→IL data；Nav 的最小 skip-IL overlay 不走它们 |

`wandb`、OpenAI `clip`、`pygame`、`pybullet` 的相关 import 有 ImportError 分支，在此无模拟器策略诊断中不必补装。Python 3.9 不走 `pickle5` backport 分支。`moviepy`、`gitpython`、`matplotlib` 是官方声明依赖，但上述无视频组件诊断的静态顶层闭包未显示必须使用它们；若完整评测另有调用，应按运行路径补齐。

`nav_falcon_import_closure_20261103.json` 给出逐项源码路径/行号。它是 AST 闭包（Falcon 228 / Nav 234 个源码模块），包含条件与 try/except 分支；**不是真实 import 成功报告**。

## 新增上游缺件：IL data 包不在官方 NavThinker 发布中

官方 `habitat_baselines/__init__.py` 无条件导入三个 IL trainer；它们再导入：

```text
habitat_baselines.il.data.eqa_cnn_pretrain_data
habitat_baselines.il.data.nav_data
habitat_baselines.il.data.data
```

该缺件已经独立核验，**不是本地 tar 过滤推断**：

- 官方 [recursive tree](https://api.github.com/repos/hutslib/NavThinker/git/trees/fc3421b6991080a367a348f7f80fa899358f9a0b?recursive=1) 为完整树（`truncated=false`），520 个 blob，没有上述目录/文件。
- 2026-11-03 用 GitHub connector 的 `fetch_file` 对固定 commit `fc3421b6991080a367a348f7f80fa899358f9a0b` 逐个读取三个路径，三次均得到 GitHub 官方 API `404 NOT_FOUND`。
- 现有 local snapshot 同样为 520 文件，原工作树未改。上游 `.gitignore` 有裸 `data` 规则，但“为何漏发”只是推测；确定事实是当前官方树缺件。

pip 无法修复这个仓库内部缺包。应先记录原始 smoke 的真实失败，再用明确标注的兼容副本继续定位后续默认 RSSM/策略接口问题。

## 最小兼容副本

`build_navthinker_import_overlay.py` 要求原 snapshot 的 520 个文件全部匹配 `navthinker_source_manifest.json`；只在新目录复制源码后，修改 `habitat-baselines/habitat_baselines/__init__.py`，移除三个与导航无关的 IL trainer 导入及对应 `__all__` 导出。其余 **519 文件 SHA 完全不变**。没有借用 Falcon 实现，没有修改策略、WM、配置、损失或环境逻辑。

- 原 `__init__.py` SHA-256：`42f1afc704716a67ada9b5b4f015e0e254328c15ec925e2b741fd496f7ef976f`
- 派生文件 SHA-256：`eca37e2dbcd46b1590e508a2fb59bec4447576b77d4534707685776ad3feb226`
- patch SHA-256：`6f8e7e95140ed3e31d4dfedede3f3cd4ee4318971750e84a242ca8b8ff217be1`

本机已成功构建 `baselines/prep/navthinker_il_import_overlay/`，生成 `compatibility_overlay.patch` 和 `compatibility_overlay_manifest.json`。尚未执行其模型前向。

```bash
python build_navthinker_import_overlay.py \
  --repo /path/to/original/NavThinker \
  --source-manifest /path/to/navthinker_source_manifest.json \
  --output /path/to/separate/NavThinker-il-import-overlay
```

原始和兼容副本的 smoke 必须分开保存输出。smoke 将兼容 manifest 附在 JSON 中；validator 对兼容结果加 `compatibility_overlay_` 分类前缀，`paper_reproduction=false` 始终不变。原始真实缺包错误只有精确匹配已核实路径和 traceback 时，才可分类为“入口诊断已完成”。

## Falcon GPU smoke 的具体审阅结果

`falcon_gpu_smoke.py` 的策略 API 调用与官方实现一致：

- `PointNavResNetPolicy.from_config(..., agent_name='agent_0')` 是多智能体配置下正确入口；给出的 agent 前缀已移除，与官方 MultiAgentAccessMgr 的处理一致。
- 深度 `(256,256,1)`、pointgoal `(2,)` 和 `Discrete(4)` 对应当前 Falcon YAML 的网络输入/输出。
- `model.hidden_state_shape` 返回 `(num_recurrent_layers, hidden_size)`，前面增加 batch=2 正确；LSTM 内部的状态布局由官方 RNN encoder 属性处理，脚本没有手工错误拆分。
- `model.act` 返回的 `values/actions/action_log_probs/rnn_hidden_states` 均是官方 `PolicyActionData` 字段；bool mask 和 previous-action 形状正确。确定性 reset 对照调用合理。
- 现有 `falcon_noaux_25.pth` 的 `data.pkl` 经 **pickletools opcode 只读**检查，`state_dict` 键直接为 `net.*`、actor/critic 参数等，没有需要去掉的 `actor_critic.` 前缀。`load_state_dict(ckpt['state_dict'], strict=True)` 合理，不应改成宽松加载。

确认并已修正包装层问题：原脚本写 `resolved_config.yaml` 前没有创建输出目录。现在解析参数后将 `a.output` 转为绝对路径并 `mkdir(parents=True, exist_ok=True)`。未改其它逻辑。

仍需注意的结果边界/诊断限制：

1. `PointNavResNetPolicy` 直接构造绕过了官方 `SingleAgentAccessMgr` 的初始化 checkpoint 加载，因此这个 smoke 不能证明完整 evaluator 已就绪；其用途是实际发布权重的严格装载和合成观测推理。
2. 空 `scene_id='NONE'` 的 depth 渲染证明 EGL/renderer 创建成功；没有验证 HM3D 场景、Spot/humanoid 资产或社会导航 episode。`physics_enabled=true` 表示 simulator 配置开启，不能写成动态碰撞验证成功。
3. 目前只有所有阶段通过后才落盘 `smoke_result.json`。若 renderer 阶段失败，前面的成功推理没有独立 JSON，但 stderr/Slurm 日志会留下异常；建议最终依据实际日志报告阶段边界。
4. 计时包含随机观测、CPU 回传和多次推理，是 smoke 总耗时；不能作为论文推理延迟。
5. 应在启动器确保 PYTHONPATH 指向 Falcon 自己的 snapshot，并在结果汇总记录实际模块路径，避免同一共享环境里导入 NavThinker 的同名包。

以上为代码审阅结论，不是运行通过。最终以主任务的真实 Slurm 日志和结果 JSON 为准。
