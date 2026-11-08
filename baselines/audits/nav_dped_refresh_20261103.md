# NavThinker / DPet：2026-11-03 复现推进记录

本轮已完成官方发布状态刷新、完整 DPed 数据逐条检查、NavThinker 配置/源码接口诊断，并交付可以在 Habitat 环境运行的模型诊断脚本。**尚无这两项的论文 benchmark 分数，也没有完成其论文复现。** 本记录保留“数据已检查”“源代码能解析”“模型真实前向”“论文评测”之间的区别。

补充导入审阅确认：NavThinker 官方完整 tree 缺少被顶层导入的 `habitat_baselines.il.data` 包；三个路径通过官方 API 再读均为 404。已单独构建只跳过三项无关 IL trainer 导入的兼容副本，519/520 个源文件完全不变，算法与配置不变。证据、依赖清单及 Falcon smoke 审阅见 [追加报告](nav_falcon_dependency_review_20261103.md)。兼容副本结果须与原始 snapshot 分开记录。

## 官方材料实时核验

| 项目 | 当前官方 commit | 发布状态 | 本地处理 |
|---|---|---|---|
| NavThinker | `fc3421b6991080a367a348f7f80fa899358f9a0b` | 仓库 520 个文件，GitHub releases 为空；README 的 policy + world model 预训练权重仍未发布；项目页没有额外权重下载链接 | 现有快照与官方一致，工作树干净；未重克隆、未改上游 |
| DPet / DPed-VLN | `eaa8e4383cccb27fd4229f0a267ddd362bdbbbab` | 正式仓库仅 README；仍写代码清理中；GitHub releases 为空，无正式模型入口/权重 | 保留已有快照，复用现有数据 ZIP |

来源：[NavThinker 仓库](https://github.com/hutslib/NavThinker)、[NavThinker releases](https://api.github.com/repos/hutslib/NavThinker/releases)、[项目页实际目标](https://hutslib.github.io/NavThinker.dc.html)、[DPedVLN 正式仓库](https://github.com/tju-dhj/DPedVLN)、[DPedVLN releases](https://api.github.com/repos/tju-dhj/DPedVLN/releases)。响应详情和文件树保存在 `nav_dped_refresh_20261103.json`。同作者的 `DPed-VLN` 开发仓库不等同于论文声明的 `DPedVLN` 正式发布，不能任选开发配置声称复现论文。

## DPed：已真实执行的完整数据检查

数据沿用 `/Users/yuqinli/Documents/ChatGPT/Edit_Weight/data/DPed-VLN.zip`，未重复下载、解压、修改或过滤。当前 [Zenodo 官方 API](https://zenodo.org/api/records/22816620) 返回 21,367,584 bytes，MD5 `2d80b3e3b6bc79a1abbf7116d24d2b27`，与本地文件一致；SHA-256 为 `c44b8edaa7962e70d9db3c3eef593cf3b8be62c93670afc6f38b432798d79756`。

`audit_nav_dped_inputs.py` 已在项目现有 `.venv` 成功执行，逐条检查共 **66,166 条**记录：ZIP CRC、gzip/JSON、场景与文件名对应、复合 episode ID 唯一性、三维位置、四元数单位长度、目标、动作序列范围及结束动作、有限测地距离、人数、每人三个 waypoint 的位置和朝向、指令字段、跨级配对及训练/验证场景重叠。

| 划分 | 场景文件/唯一场景 | episode | 非空字符串指令 | 结构异常 |
|---|---:|---:|---:|---|
| L1 train | 560 | 32,172 | 32,172 | 0 |
| L2 train | 560 | 32,152 | 32,152 | 0 |
| L1 val_full | 54 | 921 | 920 | 1 条空指令 |
| L2 val_full | 54 | 921 | 920 | 1 条空指令 |

- L1/L2 的共有训练 32,152 对、验证 921 对，在移除 `instruction` 后，逐条 JSON 完全一致。L2 训练缺少的 20 个 `(scene_id, episode_id)` 已完整列在 JSON；没有丢弃 L1 多出的条目。
- 空指令是同一个验证 episode：`hm3d/val/00896-5hGmubEE6ZF/5hGmubEE6ZF.basis.glb`，`episode_id="10"`，两个级别各出现一次。保留原数据，不静默删掉它。
- 训练和验证场景交集均为空；两级合并共需 **614 个不同 HM3D 场景**。精确路径表见 `dped_required_scenes_20261103.json`。这张表表示所需场景，不表示场景已经取得。
- 17,310 对训练指令、332 对验证指令在 L1/L2 中完全相同；这是数据事实，不能直接解释为标注错误。训练里还有 0 人 episode（L1 18,420 / L2 18,405）；需保持发布分布，不自行删除。

上述检查只证明结构和配对关系。没有进入模拟器验证轨迹几何、可达性、碰撞或语言语义，也没有 DPet 推理结果。

重跑命令：

```bash
.venv/bin/python baselines/scripts/audit_nav_dped_inputs.py \
  --dped-zip /Users/yuqinli/Documents/ChatGPT/Edit_Weight/data/DPed-VLN.zip \
  --output baselines/audits/nav_dped_inputs_20261103.json
```

可追加 `--scene-root /path/to/data/scene_datasets` 检查这些精确 `scene_id` 对应的本地资产是否存在。源 ZIP 仍只读。

## NavThinker：新增的具体复现障碍

已对 10 个相关 Python 源文件执行 AST 解析，对官方 train/eval launcher 执行 `bash -n`，对两份 YAML 检查重复键并提取全部参数；这些检查通过。依赖原文和文件 SHA-256 保存在 `nav_dped_inputs_20261103.json`，没有以 import 成功或语法成功冒充模型结果。

默认 YAML 合并 structured config 后，WM 是 `encoder_type=navthinker` / `dynamics_type=gru`，策略视觉编码器为 `resnet`，策略记忆为两层 LSTM。当前 `NavThinkerNet` 却无条件读取 `world_model.dynamics.feat_size`，rollout 路径还依赖 `reset_imagination`、`img_step_lookahead`、`_hist_z` 等 DINO 动力学接口。官方 `RSSM` 类没有这些定义。**这是源代码可定位的接口不一致；是否在完整环境触发、最先在哪一步触发，须由附带 smoke 日志确认。** 未给上游打补丁，未把切换成自选 DINO 配置当作论文原配置。

相关代码：

- `habitat-baselines/habitat_baselines/config/default_structured_configs.py`：`WorldModelConfig.dynamics_type="gru"`、`PolicyEncoderConfig.type="resnet"`。
- `habitat-baselines/habitat_baselines/rl/world_model/navthinker_policy.py`：`NavThinkerNet.__init__` 读取 `dynamics.feat_size`，`forward` / `_run_lookahead` 使用上述接口。
- `habitat-baselines/habitat_baselines/rl/world_model/networks.py`：`RSSM` 实际接口。

官方文档依赖为 Python 3.9、Habitat-Sim 0.3.1 headless+Bullet，以及仓库内的 Habitat-Lab / Habitat-Baselines；附加 requirements 固定 NumPy 1.23.5、moviepy 1.0.3。PyTorch 只给下限，不能据此声称某个精确软件环境是作者原环境。TACC 共用基础环境由主任务统一准备，使用 NavThinker 自己的源码路径隔离导入。

## 已交付的可执行模型诊断

`smoke_navthinker_official.py` 在正确环境中执行以下步骤，并把各阶段结果和异常 traceback 写成 JSON：

1. 导入实际 Torch/Gym/NumPy，记录版本与设备；种子设为 0。
2. 通过官方 `get_config` / `patch_config` 组合未修改的 `navthinker_hm3d_eval.yaml`，记录实际导入位置和最终架构。
3. 使用该 WM 节点直接构造官方 `SocialNavWorldModel`，对明确标注的合成观测执行 encoder、RSSM posterior 及 depth / trajectory / reward 三头前向，检查输出有限性。
4. 使用官方 `NavThinkerPolicy.from_config` 初始化策略并尝试前向。失败保留原始异常，不补接口或替换架构。
5. 可选 `--dino-component`：独立调用上游 DINO-WM 默认构造器，测试一次预测和不改变历史的 lookahead。它是另一组件诊断，**不是默认 YAML 的 GRU 路径，更不是 benchmark**。

```bash
python baselines/scripts/smoke_navthinker_official.py \
  --repo /absolute/path/to/NavThinker \
  --output /absolute/path/to/navthinker_smoke.json \
  --dino-component

python baselines/scripts/validate_navthinker_smoke.py \
  /absolute/path/to/navthinker_smoke.json \
  --output /absolute/path/to/navthinker_smoke_validation.json
```

所有权重是随机初始化；不下载权重、不加载 checkpoint、不创建模拟器、不训练、不输出导航指标。脚本对实际失败返回非零并保存记录。validator 只有看到完整证据才标 `diagnostic_complete=true`；精确匹配的已知 RSSM `feat_size` 异常可分类为“已完成诊断且确认默认接口故障”。未知错误保留未决状态。`paper_reproduction` 始终为 `false`。

本机 `.venv` 缺 Torch，已实际验证“缺依赖→返回 2→保留失败 JSON”的路径，三份脚本 AST 解析通过；**本机未执行模型前向**。后续远端执行结果应单独追加真实日志，不把脚本已准备写成已运行。

## 接下来能执行的官方入口与缺项

NavThinker 官方评测入口是 `bash scripts/eval.sh navthinker_hm3d_eval.yaml <checkpoint.pth> 1`，但还需要匹配论文的检查点、准确配置和官方场景/episode/机器人/人体资产。默认评测 YAML 只设 50 episodes，不能直接充当论文全量评测。官方训练例子 `bash scripts/train.sh navthinker_hm3d.yaml 4` 请求 4 进程；若在 1–2 张 3090 上调整，须记录有效全局 batch、更新次数和配置差异。

DPet 当前无正式可执行入口和权重。可继续做的数据、场景和共享环境准备已完成到可审计程度；正式策略复现需要上游补齐实现或明确的论文版本开发快照/配置。此轮没有联系作者、发邮件、开 issue 或申请 GPU。
