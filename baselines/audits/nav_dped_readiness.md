# NavThinker 与 DPet 复现准备核查

日期：2026-11-03。只读公开代码、配置、数据元信息及已有数据包；没有访问 TACC、申请 GPU、安装依赖、载入权重或运行评测。没有重复下载 DPed 数据。

## 当前结论

|方法|官方代码|论文检查点|当前能完成什么|论文基线评测状态|
|---|---|---|---|---|
|NavThinker|部分发布，已固定并取得本地快照|README 明确仍待发布 policy + world model checkpoints；releases 空|源码/config 审查；在材料补齐后做明确标为开发版的运行验证|未就绪：缺检查点，且默认配置与论文架构不一致|
|DPet / DPed-VLN|论文正式仓库仅 README，仍说明代码清理中|未提供公开下载链接；releases 空|已复用并重新校验公开 episode 数据；开发仓库只能作旁证|未就绪：缺正式实现、匹配配置与论文检查点|

这两项不能以随机初始化、另选开发配置或缩减为若干回合的冒烟运行，替代论文结果复现。

## 已固定的本地材料

- NavThinker 官方仓库：`baselines/repos/NavThinker/`，commit `fc3421b6991080a367a348f7f80fa899358f9a0b`，当前工作树干净，占用约 5.8 MiB。来源：<https://github.com/hutslib/NavThinker>。
- DPed-VLN 正式仓库：`baselines/repos/DPedVLN/`，commit `eaa8e4383cccb27fd4229f0a267ddd362bdbbbab`，仅 `README.md`。来源：<https://github.com/tju-dhj/DPedVLN>。
- 重要配置与 README 的 bytes/SHA256、数据包校验和及独立计数：同目录 `nav_dped_readiness.json`。
- 同作者开发仓库 `tju-dhj/DPed-VLN` 与正式入口 `tju-dhj/DPedVLN` 分开看待。本轮未重复克隆开发仓库；前序只读核查记录在 `research/weight_edit_dped_audit.md`。

## NavThinker：真实入口、输入和预算

README 的实际入口为：

```sh
bash scripts/train.sh navthinker_hm3d.yaml 4
bash scripts/eval.sh navthinker_hm3d_eval.yaml /path/to/checkpoint.pth 1
```

训练脚本调用 DD-PPO 的 torch.distributed.launch；评测脚本要求明确检查点，默认写视频。训练配置为 `total_num_steps: 15000000`、`num_environments: 28`、PPO rollout `num_steps: 128`、`ppo_epoch: 2`、`num_mini_batch: 8`、hidden_size 512。world model 在线使用 replay buffer 20000、warmup 5000、batch 16、sequence_length 50、每次 15 epochs。这里只报告公开配置，不把它换算成未经测量的时长或 3090 显存需求。

当前 eval 配置 `test_episode_count: 50`，并非论文完整表格结果所需的自动保证；Social-MP3D 原文报告 300 回合。评测默认打开视频会产生额外存储。TACC 技能当前允许 1–2 张 GPU，不能直接运行脚本的 4 卡默认值；若将来从头训练，需要保留科学预算并明确并行度/批量变化，不能静默减步数后报告完整复现。

必需输入：

- Social-HM3D / Social-MP3D episode 文件，路径规则 `data/datasets/pointnav/social-hm3d/{split}/{split}.json.gz`；README 与 Falcon 指向同一个官方 episode Drive 目录。
- 与 episode `scene_id` 完全对应的 HM3D / Matterport3D 场景资产，不能用普通示例场景替代。
- Spot URDF 与附件、humanoid URDF / motion data、Habitat3.0 benchmark assets、`spot_walking_trajectory.csv`。
- 训练默认需要 `pretrained_model/pretrained_habitat3.pth`；这不是已发布 NavThinker 策略权重。正式评测需要与配置完全匹配的 policy + world model checkpoint。
- Linux Python 3.9、Habitat-Sim 0.3.1 headless/withbullet、该 fork 的 habitat-lab / habitat-baselines。根 requirements 只固定 numpy 1.23.5、moviepy 1.0.3 等，不能视为完整可复现环境锁。

**关键版本问题：** README/论文讲 DA-V2 patch world model，但当前默认 YAML 的 `world_model.encoder_type` 是 `navthinker`，选择 ResNet；策略使用两层 LSTM，world model 在线共同训练。原论文描述 DA-V2、因果 Transformer 与 GRU 的路径。代码存在 Depth Anything adapter 不代表默认 YAML 已切到论文方案。训练 `pred_horizon: 5`，评测 `pred_horizon: 4` 也需要与真实检查点核对。当前不应自行拼一个看似接近论文的 YAML，称其为原始 baseline。

固定源码证据：[README](https://github.com/hutslib/NavThinker/blob/fc3421b6991080a367a348f7f80fa899358f9a0b/README.md)、[train config](https://github.com/hutslib/NavThinker/blob/fc3421b6991080a367a348f7f80fa899358f9a0b/configs/navthinker_hm3d.yaml)、[eval config](https://github.com/hutslib/NavThinker/blob/fc3421b6991080a367a348f7f80fa899358f9a0b/configs/navthinker_hm3d_eval.yaml)。

## DPed-VLN：已有数据可复用，但模型仍缺材料

官方 DOI：`10.5281/zenodo.22816620`。网页工具打不开 Zenodo 时，官方 API `https://zenodo.org/api/records/22816620` 成功返回开放数据元信息：`DPed-VLN.zip`，21,367,584 bytes，MD5 `2d80b3e3b6bc79a1abbf7116d24d2b27`。

本轮直接校验并复用了已有文件：

`/Users/yuqinli/Documents/ChatGPT/Edit_Weight/data/DPed-VLN.zip`

其 bytes、MD5 与官方元信息完全一致；SHA256 为 `c44b8edaa7962e70d9db3c3eef593cf3b8be62c93670afc6f38b432798d79756`。已有更完整审查在同项目 `results/dped_dataset_audit.json`；本轮重新逐个解压 gzip JSON 到内存、计数，没有执行压缩包内容、没有改动原文件、没有展开重复副本。

|压缩包内原始分组|JSON文件|回合数|不同场景数|空指令数|
|---|---:|---:|---:|---:|
|L1/train|560|32172|560|0|
|L2/train|560|32152|560|0|
|L1/val_full|54|921|54|1|
|L2/val_full|54|921|54|1|

因此不能不加说明地把数据包说成“两级各 32172 训练回合、561 训练场景、验证指令全部有效”。`val_full` 是发布目录的原始名字，不能在数据层静默改成另一个划分名称。原始 921 回合不应因空指令直接过滤为 920 而仍与论文 921 回合结果对比。使用原始加载器的实际处理方式目前仍待正式代码确认。

正式仓库明确场景资产不在数据包内；仍需 HM3D 场景、Habitat3.0 动态人体/机器人资产。虽然同作者开发仓库含 PPO/IL、多个动作空间和本地 checkpoint 路径，但缺少与论文表 2/3 各行一一对应的公开模型和冻结配置。故本轮不把从开发仓库任选配置训练当作 DPet 论文复现，也不为这些缺项申请 GPU 空等。

## 下一步边界

共享的场景/人体/Spot 资产准备可以随 Falcon 推进。NavThinker 的精确论文配置和原始检查点、DPet 的正式实现和检查点需要发布材料补齐；目前没有为此联系作者、创建 issue 或发送邮件。所有本轮成果都是材料准备与静态核验，不是模型跑分。
